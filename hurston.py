#!/usr/bin/env python3
"""Hurston-1 inference (ClarityVoice family, Destined AI): accented English text-to-speech from a plain-language voice description.

    python hurston.py --text "Thanks for calling, how can I help?" \
        --accent "Nigerian English accent" --gender Female --out hello.wav

The weights are downloaded from Hugging Face (destined/hurston-1, gated with automatic approval: accept the terms on the model
page once and run `hf auth login`). Pass --model /path/to/weights, or set HURSTON_MODEL, to use a local copy.

Options: --check retries a take when Whisper cannot follow it or the voice changes partway through (needs
openai/whisper-large-v3-turbo and speechbrain). Text up to ~40 words is one take; longer text is split at sentence ends. Every output has long stalls shortened,
is trimmed, and is loudness-levelled so all accents play back at the same volume.

Phone mode: --phone writes what a caller hears on a phone line: the 300-3,400 Hz telephone band at 8 kHz, G.711 mu-law
(--codec alaw for A-law), ready for Twilio and other telephony systems. It is a channel effect applied after synthesis."""
import os, re, json, argparse, tempfile
import numpy as np, torch

ACCENTS = ['American accent', 'Southern American accent', 'Nigerian English accent', 'Nigerian (Yoruba) English accent',
           'Nigerian (Igbo) English accent', 'West African English accent', 'East African English accent',
           'East African (Swahili) English accent', 'Southern African English accent']
STYLES = ['clear conversational voice', 'casual conversational voice', 'clear audiobook narration voice']
PACES = ['natural pacing', 'measured pacing', 'slow pacing', 'fast pacing']   # absolute words/s bins (v4b): slow <2.0, measured 2.0-2.6, natural 2.6-3.4, fast >3.4
AGES = ['adult', 'middle-aged', 'older']

BOS, TEXT_EOT = 128000, 128009
SOS, EOS, SOH, EOH, SOA = 128257, 128258, 128259, 128260, 128261
CODE_OFFSET = 128266
CODE_MAX = CODE_OFFSET + 7 * 4096 - 1
SR = 24000
DEFAULT_MODEL = os.environ.get('HURSTON_MODEL', 'destined/hurston-1')


def describe(gender='Female', age='adult', accent='American accent', style='clear conversational voice', pace='natural pacing'):
    """The description format the model was trained on: 'Gender, age, accent, style, pace'."""
    return f'{gender}, {age}, {accent}, {style}, {pace}'


class Hurston:
    def __init__(self, model=DEFAULT_MODEL, device='cuda', check=False, snac='hubertsiuzdak/snac_24khz',
                 asr='openai/whisper-large-v3-turbo', ecapa='speechbrain/spkrec-ecapa-voxceleb'):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from snac import SNAC
        self.dev = device
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to(device).eval()
        if os.path.isdir(snac):   # local copy: config.json + pytorch_model.bin
            self.snac = SNAC(**json.load(open(f'{snac}/config.json'))); self.snac.load_state_dict(torch.load(f'{snac}/pytorch_model.bin', map_location='cpu'))
        else: self.snac = SNAC.from_pretrained(snac)
        self.snac = self.snac.to(device).eval()
        self.asr = None
        if check:
            from transformers import pipeline
            self.asr = pipeline('automatic-speech-recognition', asr, torch_dtype=torch.float16, device=device)
            from speechbrain.inference.speaker import EncoderClassifier   # voice-change check (Apache-2.0)
            save = tempfile.mkdtemp() if os.path.isdir(ecapa) else os.path.expanduser('~/.cache/hurston/ecapa')   # local copy: link into a scratch dir
            self.spk = EncoderClassifier.from_hparams(ecapa, savedir=save, run_opts={'device': 'cuda:0' if device == 'cuda' else device})

    def _prompt(self, description, text):
        body = self.tok.encode(f'<description="{description}"> {text}', add_special_tokens=False)
        return [BOS, SOH, BOS] + body + [TEXT_EOT, EOH, SOA, SOS]

    @torch.inference_mode()
    def _decode(self, ids):
        ids = [t for t in ids if CODE_OFFSET <= t <= CODE_MAX]
        l1, l2, l3 = [], [], []
        for i in range(len(ids) // 7):
            s = [(t - CODE_OFFSET) % 4096 for t in ids[7 * i:7 * i + 7]]
            l1.append(s[0]); l2 += [s[1], s[4]]; l3 += [s[2], s[3], s[5], s[6]]
        if not l1: return np.zeros(SR // 10, dtype='float32')
        codes = [torch.tensor(l, device=self.dev)[None] for l in (l1, l2, l3)]
        y = self.snac.decoder(self.snac.quantizer.from_codes(codes))[0, 0].float().cpu().numpy()
        return y[2048:] if len(y) > 4096 else y   # drop the decoder warm-up

    @torch.inference_mode()
    def _generate(self, description, text, temperature):
        p = self._prompt(description, text)
        g = self.lm.generate(torch.tensor([p], device=self.dev), max_new_tokens=2048, min_new_tokens=28, do_sample=True,
                             temperature=temperature, top_p=0.9, repetition_penalty=1.1, eos_token_id=EOS,
                             pad_token_id=self.tok.pad_token_id)[0, len(p):].tolist()
        return self._decode(g[:g.index(EOS)] if EOS in g else g)

    def _wer(self, y, text):
        from scipy.signal import resample_poly
        norm = lambda t: re.sub(r"[^a-z0-9' ]+", ' ', t.lower()).split()
        h = norm(self.asr({'raw': resample_poly(y, 2, 3).astype('float32'), 'sampling_rate': 16000}, generate_kwargs={'language': 'en'})['text'])
        r = norm(text); d = list(range(len(h) + 1))
        for i, rw in enumerate(r, 1):
            prev, d[0] = d[0], i
            for j, hw in enumerate(h, 1): prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
        return d[-1] / max(1, len(r))

    def steady(self, y):
        """Voice similarity between the first and second half of a take (1 = same voice); None for takes under 2 s."""
        if len(y) < 2 * SR: return None
        from scipy.signal import resample_poly
        h = len(y) // 2
        with torch.no_grad():
            e = [torch.nn.functional.normalize(self.spk.encode_batch(torch.from_numpy(resample_poly(z, 2, 3).astype('float32'))[None].to(self.dev))[0, 0], dim=0) for z in (y[:h], y[h:])]
        return float(e[0] @ e[1])

    def sentence(self, text, description, temperature=0.4, retries=2, max_wer=0.5, min_steady=0.36):
        """One sentence. With check=True, regenerates up to `retries` times when Whisper WER > max_wer or the voice changes
        mid-take (steady < min_steady); keeps the best take."""
        best = None
        for _ in range(1 + (retries if self.asr else 0)):
            y = trim(squeeze(self._generate(description, text, temperature)))
            if not self.asr: return y
            w, st = self._wer(y, text), self.steady(y)
            ok = w <= max_wer and (st is None or st >= min_steady)
            score = 2 * ok + (w <= max_wer) + (1.0 if st is None else st) - w
            if best is None or score > best[0]: best = (score, y)
            if ok: break
        return best[1]

    def __call__(self, text, description, temperature=0.4, retries=2, gap=0.25, max_words=40, phone=False):
        """Any length of text. Text is kept in one take (one voice) up to ~max_words words; longer text is split at sentence
        ends into chunks of that size, joined with short pauses. Then the loudness is levelled.
        Returns float32 audio at SR (24 kHz), or at PHONE_SR (8 kHz, phone channel) with phone=True."""
        parts = []
        for snt in (x for x in re.split(r'(?<=[.!?])\s+', text.strip()) if x):
            if parts and len(parts[-1].split()) + len(snt.split()) <= max_words: parts[-1] += ' ' + snt
            else: parts.append(snt)
        out = []
        for s in parts:
            out += [self.sentence(s, description, temperature, retries), np.zeros(int(SR * gap), dtype='float32')]
        y = level(np.concatenate(out[:-1]))
        return _phone_channel(y) if phone else y


def trim(y, top_db=40):
    """Drop leading/trailing silence, keeping ~50 ms."""
    f = int(SR * .02)
    e = 20 * np.log10(np.array([np.sqrt(np.mean(y[i:i + f] ** 2)) + 1e-9 for i in range(0, max(1, len(y) - f), f)]))
    on = np.where(e > max(e.max() - top_db, -55))[0]
    return y[max(0, (on[0] - 2) * f):min(len(y), (on[-1] + 3) * f)] if len(on) else y


def squeeze(y, max_pause=0.5, keep=0.35):
    """Shorten any pause inside a take that is longer than max_pause seconds to keep seconds (the model sometimes stalls)."""
    f = int(SR * .02); n = len(y) // f
    if n == 0: return y
    e = np.sqrt(np.mean(y[:n * f].reshape(n, f) ** 2, axis=1)); quiet = e < e.max() * 0.03
    out, i = [], 0
    while i < n:
        j = i
        while j < n and quiet[j] == quiet[i]: j += 1
        seg = y[i * f:j * f]
        if quiet[i] and (j - i) * .02 > max_pause: seg = seg[:int(SR * keep / 2)].tolist() + seg[-int(SR * keep / 2):].tolist()
        out.append(np.asarray(seg, dtype='float32')); i = j
    return np.concatenate(out + [y[n * f:].astype('float32')])


def level(y, rms_db=-20.0, peak=0.89):
    """Same loudness for every voice: scale speech frames to the target RMS, then cap the peak."""
    f = int(SR * .02)
    fr = np.array([np.sqrt(np.mean(y[i:i + f] ** 2)) for i in range(0, max(1, len(y) - f), f)])
    act = fr[fr > fr.max() * 0.1] if fr.size and fr.max() > 0 else fr
    if not act.size or act.mean() <= 0: return y
    y = y * (10 ** (rms_db / 20) / np.sqrt(np.mean(act ** 2)))
    m = np.abs(y).max()
    return (y * (peak / m) if m > peak else y).astype('float32')


PHONE_SR = 8000
_ULAW_BIAS, _ULAW_CLIP = 0x84, 32635


def ulaw_encode(y):
    """float audio in [-1, 1] -> G.711 mu-law bytes (ITU-T G.711, same tables as libsndfile and Twilio)."""
    x = np.clip(np.round(np.asarray(y, dtype='float64') * 32768), -32768, 32767).astype(np.int32)
    sign = np.where(x < 0, 0x80, 0); x = np.minimum(np.abs(x), _ULAW_CLIP) + _ULAW_BIAS
    e = np.clip(np.floor(np.log2(np.maximum(x >> 7, 1))), 0, 7).astype(np.int32)
    return (~(sign | (e << 4) | ((x >> (e + 3)) & 0x0F)) & 0xFF).astype(np.uint8).tobytes()


def ulaw_decode(b):
    """G.711 mu-law bytes -> float audio in [-1, 1]."""
    u = ~np.frombuffer(b, dtype=np.uint8).astype(np.int32) & 0xFF
    x = ((((u & 0x0F) << 3) + _ULAW_BIAS) << ((u >> 4) & 0x07)) - _ULAW_BIAS
    return (np.where(u & 0x80, -x, x) / 32768).astype('float32')


def phone(y, sr=SR, low=300, high=3400):
    """Telephone channel: band-limit to the 300-3,400 Hz voice band, resample to 8 kHz and pass through G.711 mu-law.
    Returns float32 audio at 8 kHz (write it with soundfile subtype='ULAW', or send ulaw_encode(...) to a phone API)."""
    from scipy.signal import butter, sosfiltfilt, resample_poly
    y = sosfiltfilt(butter(6, [low, high], btype='bandpass', fs=sr, output='sos'), np.asarray(y, dtype='float64'))
    y = resample_poly(y, PHONE_SR, sr)
    return ulaw_decode(ulaw_encode(np.clip(y, -1, 1)))


_phone_channel = phone   # Hurston.__call__'s phone= argument shadows the name
ClarityVoice = Hurston   # older name, kept so existing code keeps working

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=DEFAULT_MODEL, help='Hugging Face repo id or local folder with the weights'); ap.add_argument('--snac', default='hubertsiuzdak/snac_24khz'); ap.add_argument('--text', required=True); ap.add_argument('--out', default='out.wav')
    ap.add_argument('--description', help='full description; overrides the fields below')
    ap.add_argument('--gender', default='Female', choices=['Female', 'Male']); ap.add_argument('--age', default='adult', choices=AGES)
    ap.add_argument('--accent', default='American accent', choices=ACCENTS); ap.add_argument('--style', default=STYLES[0], choices=STYLES)
    ap.add_argument('--pace', default=PACES[0], choices=PACES); ap.add_argument('--temperature', type=float, default=0.4)
    ap.add_argument('--check', action='store_true', help='retry takes Whisper cannot follow or where the voice changes mid-take (recommended)'); ap.add_argument('--retries', type=int, default=2); ap.add_argument('--seed', type=int)
    ap.add_argument('--phone', action='store_true', help='phone-line output: 300-3,400 Hz, 8 kHz, G.711 (mu-law WAV)'); ap.add_argument('--codec', default='ulaw', choices=['ulaw', 'alaw'])
    a = ap.parse_args()
    if a.seed is not None: torch.manual_seed(a.seed)
    import soundfile as sf
    tts = Hurston(a.model, check=a.check, snac=a.snac)
    d = a.description or describe(a.gender, a.age, a.accent, a.style, a.pace)
    y = tts(a.text, d, a.temperature, a.retries)
    if a.phone: sf.write(a.out, phone(y), PHONE_SR, subtype=a.codec.upper())
    else: sf.write(a.out, y, SR)
    print(a.out, '|', d, '| phone (8 kHz G.711 %s)' % a.codec if a.phone else '')
