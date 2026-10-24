# Hurston-1

**Accented English text-to-speech built for the phone, steered by a plain-language voice description.**

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-destined%2Fhurston--1-yellow)](https://huggingface.co/destined/hurston-1)
[![Listen](https://img.shields.io/badge/Listen-18%20synthetic%20clips-blue)](https://huggingface.co/spaces/destined/hurston-1)
[![Databricks Marketplace](https://img.shields.io/badge/Databricks-Marketplace-orange)](https://marketplace.databricks.com/details/8b876c68-5ab0-423c-919c-3f86ee793cdf)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

Hurston-1 introduces **NB-TTS (Narrowband TTS)**: text-to-speech made for the phone line, where most customer service
still happens, even when the other side is an AI agent. It learned in part from real phone calls between our
contributors and AI voice agents, and its phone mode delivers what a caller actually hears: 8 kHz G.711 audio,
ready for Twilio and other telephony systems. Full 24 kHz audio is still the default.

Hurston-1 is a fine-tune of [Maya1](https://huggingface.co/maya-research/maya1) (Apache-2.0) built by
[Destined AI](https://destined.ai) to speak English accents that most TTS systems handle poorly: Southern American,
Nigerian, West African, East African, Southern African, alongside broader American English. It is the first open model
in Destined AI's ClarityVoice family.

This repository holds the inference code, examples, phone-mode notes and synthetic demo clips. The weights (3B
parameters, 6.6 GB of safetensors) are on Hugging Face at [destined/hurston-1](https://huggingface.co/destined/hurston-1).

## Quick start

You need Python 3.10 or newer and a CUDA GPU with about 8 GB of memory (the model runs in bfloat16). Tested with
Python 3.12, PyTorch 2.11, Transformers 4.57 and snac 1.2.1.

```bash
git clone https://github.com/Destined-AI/hurston-1
cd hurston-1
pip install -r requirements.txt
```

The weights are gated with automatic approval: open the [model page](https://huggingface.co/destined/hurston-1) once,
accept the terms, then log in so the script can download them:

```bash
hf auth login        # older huggingface_hub: huggingface-cli login
```

Then:

```bash
python hurston.py --accent "Southern American accent" --gender Male \
    --text "Thanks for calling, how can I help you today?" --out hello.wav

# the same line as a caller hears it: 8 kHz G.711 mu-law WAV
python hurston.py --phone --accent "Nigerian English accent" --gender Female \
    --text "Good afternoon. Please take one tablet twice a day after meals." --out call.wav
```

Add `--check` to retry takes that Whisper cannot follow or where the voice changes partway through. It is recommended
for anything you ship and needs `pip install speechbrain`. Pass `--model /path/to/weights` (or set `HURSTON_MODEL`) to
use a local copy of the weights.

```python
from hurston import Hurston, describe
tts = Hurston(check=True)               # downloads destined/hurston-1 on first use
wav = tts("Well hey there, thanks for calling.", describe('Male', 'adult', 'Southern American accent'))   # float32, 24 kHz
```

More in [`examples/`](examples/):

| Script | What it does |
|---|---|
| [`quickstart.py`](examples/quickstart.py) | One line in all nine accents, female and male |
| [`phone_mode.py`](examples/phone_mode.py) | The same line at 24 kHz and through the phone channel (mu-law and A-law); also applies the channel to any WAV |
| [`twilio_media_stream.py`](examples/twilio_media_stream.py) | Frames phone-mode audio into Twilio Media Stream messages |
| [`accent_test_set.py`](examples/accent_test_set.py) | Builds an accent-coverage test set, with manifest, for your speech recognizer or IVR |

## Hear it

The [listening page](https://huggingface.co/spaces/destined/hurston-1) plays one clip per accent and gender, plus phone
mode. The same clips are in [`demo/`](demo/) (voice descriptions and text in [`demo/demos.csv`](demo/demos.csv),
phone-mode pairs in [`demo/phone/`](demo/phone/)). Everything is synthetic; every voice is generated from a description.

## The South got something to say

In 1995 André 3000 stood on stage at the Source Awards, in front of a crowd that booed OutKast, and said
*"the South got something to say."* We think that line still fits.

Hurston-1 was built in Atlanta by a small, independent team. Since 2021 we've been recording consented, paid
contributors from the American South and from across Africa. These are voices that most AI systems still flatten into
generic American English. Nobody handed us a funding round to do this, so we built it ourselves.
These voices belong in AI, and now they have an open model.

We named it for **Zora Neale Hurston**. She grew up in Eatonville, Florida, trained as an anthropologist, and traveled
the South recording how Black Southerners actually talked: their stories, songs and speech, at a time when few
institutions thought those voices were worth keeping. Our work is a small echo of hers. The name is a tribute; it does
not imply any endorsement by her estate.

A few lines the way people actually talk around Atlanta, in [`demo/south/`](demo/south/):

| Line | Voice | Clip |
|---|---|---|
| *"The South got somethin to say."* | Male, Southern American accent, clear conversational | [south_got_somethin_to_say.wav](demo/south/south_got_somethin_to_say.wav) |
| *"Meet me on Peachtree. No, not that Peachtree, the other one."* | Female, Southern American accent, casual conversational | [peachtree.wav](demo/south/peachtree.wav) |
| *"I'm fixin' to head out, traffic on 285 is already backed up."* | Male, Southern American accent, casual conversational | [traffic_285.wav](demo/south/traffic_285.wav) |
| *"What part of Atlanta you from?"* | Female, Southern American accent, casual conversational | [what_part_of_atlanta.wav](demo/south/what_part_of_atlanta.wav) |

## Voice descriptions

The model was trained on descriptions of the form **`Gender, age, accent, style, pace`**:

| Field | Values |
|---|---|
| Gender | `Female`, `Male` |
| Age | `adult`, `middle-aged`, `older` |
| Accent | `American accent`, `Southern American accent`, `Nigerian English accent`, `Nigerian (Yoruba) English accent`, `Nigerian (Igbo) English accent`, `West African English accent`, `East African English accent`, `East African (Swahili) English accent`, `Southern African English accent` |
| Style | `clear conversational voice`, `casual conversational voice`, `clear audiobook narration voice` |
| Pace | `natural pacing`, `measured pacing`, `slow pacing`, `fast pacing` |

Pace labels are absolute speaking rates learned from the natural voices in the training data: roughly slow is under
2 words per second, measured 2 to 2.6, natural 2.6 to 3.4, fast over 3.4. The model follows them approximately, not
exactly.

Each call samples a new voice that fits the description. Hurston-1 does not clone voices and was not trained to
reproduce any real person.

## Built for the phone (NB-TTS)

Most voice AI is demoed in studio quality and then deployed over a phone line. Hurston-1 is built for where it will
actually be heard:

- **Trained on real calls.** About 10 hours of the training audio (7,600 turns from 250 contributors) is phone-quality
  speech from real calls with full-duplex AI voice agents, alongside studio-style reading and conversation.
- **Phone mode.** `--phone` (or `phone=True`) filters the output to the 300 to 3,400 Hz telephone band, resamples it to
  8 kHz and encodes it as G.711 mu-law (`--codec alaw` for A-law), the formats phone networks, SIP and Twilio use.
  It is a channel effect applied after synthesis, so `--check` still judges the full-quality take.
- **Ready to stream.** `ulaw_encode()` returns raw mu-law bytes, the payload Twilio media streams expect.

```python
from hurston import Hurston, describe, ulaw_encode
tts = Hurston(check=True)
call = tts("Hi, I'm calling about my appointment on Tuesday.", describe('Female', 'older', 'Nigerian English accent'), phone=True)
payload = ulaw_encode(call)   # 8 kHz mu-law bytes; base64 it for a Twilio media stream
```

More in [docs/PHONE.md](docs/PHONE.md).

## What people build with it

Some places to start. At Destined AI we use accented TTS like this to build broad, high-quality test data, and that is
still the use we would point to first.

- **Test data for speech recognition.** Generate the same sentences in nine accents and see where your ASR, captioning
  or transcription system breaks. It makes a quick accent-coverage check or a regression set before a release
  (`examples/accent_test_set.py`).
- **Simulated callers for voice agents and IVR.** Pair Hurston-1 with an LLM that writes caller scripts (customers,
  patients, applicants) and run hundreds of synthetic calls through your agent before real people do. Use phone
  mode so the agent hears what it will hear in production. Good for catching where an agent mishears accents,
  names or numbers.
- **Training data augmentation.** Add accented synthetic speech to ASR, keyword-spotting or wake-word training sets
  where those accents are thin. Mix it with real speech instead of training on synthetic audio alone.
- **Voice prompts and announcements.** Phone menus (phone mode output drops straight into an IVR), appointment
  reminders, store and transit announcements, app onboarding, in a voice that sounds like the community you serve.
- **Education and training.** Listening practice across world Englishes for learners, and accent-familiarity practice
  for customer service, healthcare and call-center staff.
- **Accessibility and read-aloud.** Read articles, forms or messages aloud in a voice closer to home.
- **Games, animation and media drafts.** Background characters, temp voice tracks and podcast or audiobook drafts,
  labeled as synthetic.
- **Research.** Accent-fairness audits of speech systems, listening and perception studies, and benchmarks that cover
  voices most datasets leave out.

**Not a fit yet:** one consistent brand voice across sessions (each call is a new voice; see Limitations), cloning a
specific person, or anything safety-critical without a human listening first.

## Evaluation

Two questions: does it sound like the accent, and does it say the words?

**Accent strength vs base Maya1.** Base Maya1 documents five accents (American, Indian, Middle Eastern,
Asian American, British). For a fair comparison we prompted it in its own documented description style, naming each
accent, and compared it with Hurston-1 on the same 10 sentences and 18 voices (180 clips per model). An accent
classifier (African vs American English, trained on real speakers) scored each voice group:

| Accent (probability judged African) | Maya1 base | Hurston-1 |
|---|---|---|
| Southern African | 0.60 | **0.95** |
| West African | 0.60 | **0.84** |
| East African | 0.81 | **0.98** |
| East African (Swahili) | 0.81 | **0.97** |
| Nigerian | 0.82 | **0.94** |
| Nigerian (Igbo) | 0.82 | **0.92** |
| Nigerian (Yoruba) | 0.87 | **0.93** |
| American (should be low) | 0.12 | **0.02** |
| Southern American (should be low) | n/a | **0.01** |

The classifier cannot tell Southern American from general American English; that accent was checked by listening.

**Intelligibility.** Whisper large-v3-turbo word error rate (WER) against the input text, numbers normalized
(lower is better).

*Everyday sentences* (the same 180 clips as the accent test above, `natural pacing`, single take):

| | Maya1 base | Hurston-1 |
|---|---|---|
| Mean WER | 0.02 | 0.03 |

*Harder sentences* (30 unseen sentences × 17 voices, 510 clips):

| | Hurston-1, single take | Hurston-1, with `--check` |
|---|---|---|
| Mean WER | 0.145 | **0.062** |
| Clips over 20% WER | 17.1% | **8.4%** |

The harder set covers reading, health, customer service, names and numbers, expressive and long sentences.
Base Maya1 is very clear because it mostly stays in American English; Hurston-1 keeps nearly the same clarity while
actually doing the accent.

| Accent (510-clip set, WER with `--check`) | Hurston-1 |
|---|---|
| American | 0.021 |
| Southern American | 0.070 |
| Nigerian | 0.070 |
| Nigerian (Igbo) | 0.016 |
| Nigerian (Yoruba) | 0.078 |
| West African | 0.059 |
| East African | 0.043 |
| East African (Swahili) | 0.019 |
| Southern African | 0.053 |

Names and numbers are the hardest category (WER 0.17 with `--check`). Part of that is the recognizer misspelling
African names such as *Adaeze Okafor* rather than the model mispronouncing them.

**Speaking rate.** Median 2.84 words per second with `natural pacing`, against 3.76 for base Maya1. Hurston-1
speaks a little more deliberately; use `fast pacing` for quicker delivery.

**Blind listening.** Testers heard the same sentence from base Maya1 and Hurston-1 in random order, without knowing
which was which, and chose the one that sounded more like a real person with the requested accent. Hurston-1 was chosen
**78%** of the time and base Maya1 **6%**; the rest were rated a tie or neither (54 comparisons). Base Maya1 was never
chosen for an African or Southern American voice; its few wins were on plain American English.

## Training data

LoRA (rank 64) fine-tune of Maya1, 2,000 steps, about 190 hours of speech:

| Source | Hours | License |
|---|---|---|
| Destined AI accented English corpus: scripted reading, conversational speech and ~10 h of real phone calls with AI voice agents, from consented, paid contributors | ~136 (a slice of our 1,800+ hour corpus) | Owned by Destined AI |
| [LibriTTS-R](https://www.openslr.org/141/) (audiobook style, keeps general quality) | ~46 | CC BY 4.0 |
| [Mozilla Common Voice](https://commonvoice.mozilla.org/) English, African-accented speakers, quality-filtered | ~10.5 | CC0 |

Accent labels come from each contributor's self-reported background, checked against an audio accent classifier and
human listening; speakers whose audio and labels disagreed were relabeled or removed. Pace labels come from each clip's
measured speaking rate. All contributors were adults. The full Destined AI corpus (1,800+ hours, recorded since 2021) is
not included in this release; contact us about licensing it.

## Limitations

- **No voice consistency across calls.** Each call is a fresh voice that fits the description. Long text is split
  at sentence ends into chunks of about 40 words, so the voice can change between chunks. Consistent named voices are
  planned for a later release.
- Names, dates and amounts are the weakest content. Use `--check` and listen before you ship anything important.
- The generic `African English accent` label is weaker (WER 0.26); use a specific region instead.
- English only. No child voices.
- Like other autoregressive TTS models it occasionally produces a garbled or overlong take, or the voice shifts mid-take;
  `--check` catches most of these.
- Output is leveled to the same loudness for every voice, and pauses longer than half a second inside a take are shortened.
- Phone mode changes the channel, not the speaking style: the model does not yet choose call-style delivery
  (turn-taking, backchannels) on request. See What's next.

## What's next

Hurston-1 is the first step. What we are building toward:

- **Telephony-native TTS.** A model that learns to *talk* like a person on a call, not just sound like the phone line:
  a call style distinct from reading style, with natural turn-taking, "mm-hm"s and call pacing, learned from our
  recordings of real people talking with full-duplex AI agents.
- **Consistent voices.** The same named voice across every call and every sentence, for agents and brands.

Follow [Destined AI on Hugging Face](https://huggingface.co/destined) for the next release.

## Responsible use

Do not use Hurston-1 to impersonate real people, deceive listeners, or commit fraud. Tell listeners when speech is
synthetic. Audio in `demo/` is synthetic.

## Credits and licenses

- Code in this repository and the model weights: Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The weights
  are derived from **Maya1** by Maya Research (Apache-2.0).
- Audio codec: **SNAC** 24 kHz by Hubert Siuzdak (MIT), downloaded at run time. Phone mode uses ITU-T G.711 (public standard).
- Optional checking: **Whisper large-v3-turbo** by OpenAI (MIT) and **ECAPA speaker embeddings** from SpeechBrain
  (Apache-2.0), downloaded at run time.
- Training data: LibriTTS-R (Koizumi et al., 2023; CC BY 4.0), Mozilla Common Voice (CC0), and Destined AI's own corpus.

## Citation

```bibtex
@misc{hurston2026,
  title  = {Hurston-1: Accented English Narrowband Text-to-Speech (NB-TTS) for the Phone},
  author = {{Destined AI}},
  year   = {2026},
  url    = {https://github.com/Destined-AI/hurston-1}
}
```
