#!/usr/bin/env python3
"""Hurston-1 speech server: an OpenAI-compatible /v1/audio/speech endpoint, so voice-agent platforms that take an
OpenAI-style TTS URL with custom voice names (Dograh, via its Speaches provider) can use Hurston-1 on your own GPU.

    pip install -r requirements.txt -r requirements-serve.txt
    python serve.py                       # http://0.0.0.0:8890/v1

    curl -s localhost:8890/v1/audio/speech -H 'Content-Type: application/json' \
      -d '{"model":"hurston-1","voice":"southern-female","input":"Hey there, thanks for calling.","response_format":"wav"}' > hello.wav

Request fields (same names as OpenAI's speech API):
  input            text to speak
  voice            a preset name from GET /v1/audio/voices (e.g. southern-female, nigerian-male), or a full Hurston
                   description: "Female, adult, Southern American accent, clear conversational voice, natural pacing"
  model            "hurston-1" (any value is accepted)
  response_format  "pcm" (raw 16-bit 24 kHz mono, streamed as it is generated) or "wav"
  speed            0.25-4.0; picks the trained pace label (slow / measured / natural / fast) rather than time-stretching
  instructions     optional; if it reads like a Hurston description it replaces the voice's description

Backends: vLLM (default when installed) streams audio while it is generated, about 0.2-0.4 s to first audio on one
GPU. --backend transformers needs no vLLM but returns each sentence only after it is finished (about real time).

Hurston-1 voices come from a description, not a reference clip. Every request is a fresh voice that fits the
description, so in a voice agent, where each sentence is its own request, the speaker can change from sentence to
sentence (same accent and gender, different person). Requests are seeded from the voice name, so the same text and voice
give the same audio, which helps caching, but the seed does not hold one voice across different sentences.
"""
import os, re, io, sys, json, time, uuid, asyncio, argparse, hashlib, logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hurston import (describe, ACCENTS, STYLES, PACES, AGES, SR, BOS, TEXT_EOT, SOS, EOS, SOH, EOH, SOA,
                     CODE_OFFSET, CODE_MAX, DEFAULT_MODEL)

log = logging.getLogger('hurston.serve')

# ---------------------------------------------------------------- voices
_ACCENT_KEYS = {   # Southern American first, as everywhere in Hurston-1
    'southern': 'Southern American accent',
    'american': 'American accent',
    'nigerian': 'Nigerian English accent',
    'yoruba': 'Nigerian (Yoruba) English accent',
    'igbo': 'Nigerian (Igbo) English accent',
    'west-african': 'West African English accent',
    'east-african': 'East African English accent',
    'swahili': 'East African (Swahili) English accent',
    'southern-african': 'Southern African English accent',
}
VOICES = {}
for key, accent in _ACCENT_KEYS.items():
    for g in ('female', 'male'):
        VOICES[f'{key}-{g}'] = dict(gender=g.capitalize(), age='adult', accent=accent, style='clear conversational voice')
        VOICES[f'{key}-{g}-older'] = dict(gender=g.capitalize(), age='older', accent=accent, style='clear conversational voice')
# Presets all use the "clear conversational voice" style: in testing, "casual" takes were misheard about half the time.
DEFAULT_VOICE = 'southern-female'


def _pace(speed):
    if speed is None: return 'natural pacing'
    s = float(speed)
    return 'slow pacing' if s < 0.8 else 'measured pacing' if s < 0.95 else 'natural pacing' if s <= 1.15 else 'fast pacing'


def resolve_voice(voice, speed=None, instructions=None):
    """-> (description, seed). The seed makes the same text + voice reproducible."""
    v = (voice or DEFAULT_VOICE).strip()
    for text in (instructions, v):   # a full description wins: "Gender, age, accent, style, pace"
        if text and text.count(',') >= 2 and re.search(r'\b(male|female)\b', text, re.I):
            return text.strip().strip('"'), int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
    key = v.lower().replace('_', '-').replace(' ', '-')
    if key not in VOICES:
        raise KeyError(v)
    p = VOICES[key]
    return describe(p['gender'], p['age'], p['accent'], p['style'], _pace(speed)), int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)


def split_text(text, max_words=40):
    """Sentences, packed into chunks of up to max_words words (one take each, same rule as Hurston.__call__)."""
    parts = []
    for s in (x for x in re.split(r'(?<=[.!?])\s+', text.strip()) if x):
        if parts and len(parts[-1].split()) + len(s.split()) <= max_words: parts[-1] += ' ' + s
        else: parts.append(s)
    return parts


# ---------------------------------------------------------------- audio shaping for streamed chunks
class Shaper:
    """Streaming stand-in for hurston.trim/squeeze/level: drops leading silence and long stalls, and levels loudness
    to -20 dB RMS with a slowly adapting gain and a peak cap, so every accent plays at the same volume."""
    def __init__(self, target_db=-20.0, peak=0.89):
        self.target = 10 ** (target_db / 20); self.peak = peak
        self.gain = None; self.started = False; self.quiet = 0
        self.e2 = 0.0; self.n = 0

    def __call__(self, y):
        f = int(SR * .02); out = []
        for i in range(0, len(y), f):
            fr = y[i:i + f]; r = float(np.sqrt(np.mean(fr ** 2)) + 1e-9)
            voiced = r > 0.004
            if not self.started:
                if not voiced: continue
                self.started = True
            if voiced:
                self.quiet = 0; self.e2 += r * r; self.n += 1
                g = self.target / np.sqrt(self.e2 / self.n)
                g = min(max(g, 0.3), 6.0)
                self.gain = g if self.gain is None else 0.9 * self.gain + 0.1 * g
            else:
                self.quiet += len(fr)
                if self.quiet > int(SR * 0.4): continue   # shorten stalls inside a take
            out.append(fr * (self.gain or 1.0))
        if not out: return np.zeros(0, dtype='float32')
        z = np.concatenate(out)
        return np.clip(z, -self.peak, self.peak).astype('float32')


def pcm16(y):
    return (np.clip(y, -1, 1) * 32767).astype('<i2').tobytes()


def wav_bytes(y):
    import soundfile as sf
    b = io.BytesIO(); sf.write(b, y, SR, format='WAV', subtype='PCM_16'); return b.getvalue()


# ---------------------------------------------------------------- engines
class SnacDecoder:
    def __init__(self, snac, device):
        import torch
        from snac import SNAC
        self.torch = torch; self.dev = device
        if os.path.isdir(snac):
            self.m = SNAC(**json.load(open(f'{snac}/config.json')))
            self.m.load_state_dict(torch.load(f'{snac}/pytorch_model.bin', map_location='cpu'))
        else:
            self.m = SNAC.from_pretrained(snac)
        self.m = self.m.to(device).eval()

    def decode(self, codes):
        """codes: SNAC code tokens (multiple of 7) -> float32 audio, 2048 samples per frame."""
        torch = self.torch
        l1, l2, l3 = [], [], []
        for i in range(len(codes) // 7):
            s = [(t - CODE_OFFSET) % 4096 for t in codes[7 * i:7 * i + 7]]
            l1.append(s[0]); l2 += [s[1], s[4]]; l3 += [s[2], s[3], s[5], s[6]]
        if not l1: return np.zeros(0, dtype='float32')
        with torch.inference_mode():
            c = [torch.tensor(l, device=self.dev)[None] for l in (l1, l2, l3)]
            return self.m.decoder(self.m.quantizer.from_codes(c))[0, 0].float().cpu().numpy()


def _prompt_ids(tok, description, text):
    body = tok.encode(f'<description="{description}"> {text}', add_special_tokens=False)
    return [BOS, SOH, BOS] + body + [TEXT_EOT, EOH, SOA, SOS]


class VLLMEngine:
    name = 'vllm'

    def __init__(self, model, snac, gpu_mem, max_len=4096):
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.llm = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=model, dtype='bfloat16', max_model_len=max_len, gpu_memory_utilization=gpu_mem,
            enable_prefix_caching=True))
        self.snac = SnacDecoder(snac, 'cuda')
        self.lock = asyncio.Lock()

    # Streaming decode: frame s is decoded with LEFT frames of context before it and AHEAD frames after it, which
    # matches a full-take decode to ~32 dB SNR (a plain 4-frame window is ~20 dB, audible as roughness at the joins).
    # Each SNAC frame is 2048 samples (85 ms); frame 0 is decoder warm-up and is dropped, as in hurston.py.
    LEFT, AHEAD, HOP = 6, 2, 2048

    async def _dec(self, codes, a, b):
        async with self.lock:
            return await asyncio.to_thread(self.snac.decode, codes[a * 7:b * 7])

    async def sentence(self, description, text, seed, temperature):
        """Yields float32 audio pieces while the take is being generated."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        sp = SamplingParams(temperature=temperature, top_p=0.9, repetition_penalty=1.1, max_tokens=2048,
                            min_tokens=28, stop_token_ids=[EOS], seed=seed)
        codes, s, n_done = [], 1, 0
        async for out in self.llm.generate(TokensPrompt(prompt_token_ids=_prompt_ids(self.tok, description, text)),
                                           sp, request_id=uuid.uuid4().hex):
            ids = out.outputs[0].token_ids
            for t in ids[n_done:]:
                if CODE_OFFSET <= t <= CODE_MAX: codes.append(t)
            n_done = len(ids)
            frames = len(codes) // 7
            if frames >= s + 1 + self.AHEAD:   # emit every frame that now has its look-ahead
                last = frames - 1 - self.AHEAD
                a = max(0, s - self.LEFT)
                y = await self._dec(codes, a, last + 1 + self.AHEAD)
                yield y[(s - a) * self.HOP:(last + 1 - a) * self.HOP]; s = last + 1
        frames = len(codes) // 7
        if frames > s:   # tail: the frames that were waiting for look-ahead
            a = max(0, s - self.LEFT)
            y = await self._dec(codes, a, frames)
            yield y[(s - a) * self.HOP:]


class TransformersEngine:
    name = 'transformers'

    def __init__(self, model, snac):
        from hurston import Hurston
        self.h = Hurston(model, snac=snac)
        self.lock = asyncio.Lock()

    async def sentence(self, description, text, seed, temperature):
        import torch
        async with self.lock:
            torch.manual_seed(seed)
            y = await asyncio.to_thread(self.h._generate, description, text, temperature)
        yield y


# ---------------------------------------------------------------- app
def build_app(engine, api_key=None, model_name='hurston-1'):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse, Response
    app = FastAPI(title='Hurston-1 speech server')

    def denied(req):
        if not api_key: return False
        h = req.headers.get('authorization', '')
        return h != f'Bearer {api_key}'

    def err(status, msg):
        return JSONResponse({'error': {'message': msg, 'type': 'invalid_request_error'}}, status_code=status)

    @app.get('/health')
    async def health():
        return {'status': 'ok', 'model': model_name, 'backend': engine.name}

    @app.get('/v1/models')
    async def models():
        return {'object': 'list', 'data': [{'id': model_name, 'object': 'model', 'owned_by': 'destined'}]}

    @app.get('/v1/audio/voices')
    async def voices():
        return {'voices': [{'id': k, 'description': describe(v['gender'], v['age'], v['accent'], v['style'])}
                           for k, v in VOICES.items()],
                'custom': 'Any "Gender, age, accent, style, pace" description also works as the voice.',
                'accents': ACCENTS, 'ages': AGES, 'styles': STYLES, 'paces': PACES}

    @app.post('/v1/audio/speech')
    async def speech(req: Request):
        if denied(req): return err(401, 'invalid API key')
        try: body = await req.json()
        except Exception: return err(400, 'body must be JSON')
        text = (body.get('input') or '').strip()
        if not text: return err(400, 'input is empty')
        if len(text) > 4096: return err(400, 'input is longer than 4096 characters')
        fmt = (body.get('response_format') or 'pcm').lower()
        if fmt not in ('pcm', 'wav'): return err(400, f'response_format {fmt!r} is not supported; use pcm or wav')
        try:
            description, seed = resolve_voice(body.get('voice'), body.get('speed'), body.get('instructions'))
        except KeyError as e:
            return err(400, f'unknown voice {e.args[0]!r}; see GET /v1/audio/voices or pass a full description')
        temperature = float(body.get('temperature', 0.4))
        t0 = time.time(); rid = uuid.uuid4().hex[:8]
        log.info('%s %s | %s | %d chars', rid, fmt, description, len(text))

        async def audio():
            shaper, first = Shaper(), True
            for i, part in enumerate(split_text(text)):
                if i: yield np.zeros(int(SR * 0.2), dtype='float32')
                async for y in engine.sentence(description, part, seed, temperature):
                    z = shaper(y)
                    if len(z):
                        if first: log.info('%s first audio %.2fs', rid, time.time() - t0); first = False
                        yield z

        if fmt == 'wav':
            parts = [y async for y in audio()]
            y = np.concatenate(parts) if parts else np.zeros(SR // 10, dtype='float32')
            log.info('%s done %.2fs audio in %.2fs', rid, len(y) / SR, time.time() - t0)
            return Response(wav_bytes(y), media_type='audio/wav')

        async def stream():
            n = 0
            async for y in audio():
                b = pcm16(y); n += len(b); yield b
            log.info('%s done %.2fs audio in %.2fs', rid, n / 2 / SR, time.time() - t0)
        return StreamingResponse(stream(), media_type='audio/pcm',
                                 headers={'X-Sample-Rate': str(SR), 'X-Channels': '1', 'X-Sample-Format': 's16le'})

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=DEFAULT_MODEL, help='Hugging Face repo id or local folder with the weights')
    ap.add_argument('--snac', default='hubertsiuzdak/snac_24khz')
    ap.add_argument('--host', default='0.0.0.0'); ap.add_argument('--port', type=int, default=8890)
    ap.add_argument('--backend', default='auto', choices=['auto', 'vllm', 'transformers'])
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.3,
                    help='vLLM share of GPU memory (the 3B model needs ~7 GB plus KV cache)')
    ap.add_argument('--api-key', default=os.environ.get('HURSTON_API_KEY'),
                    help='if set, requests must send "Authorization: Bearer <key>" (also HURSTON_API_KEY)')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    backend = a.backend
    if backend == 'auto':
        try: import vllm  # noqa: F401
        except ImportError: backend = 'transformers'
        else: backend = 'vllm'
    engine = VLLMEngine(a.model, a.snac, a.gpu_memory_utilization) if backend == 'vllm' else TransformersEngine(a.model, a.snac)
    import uvicorn
    log.info('Hurston-1 (%s backend) on http://%s:%d/v1', backend, a.host, a.port)
    uvicorn.run(build_app(engine, a.api_key), host=a.host, port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
