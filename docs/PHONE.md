# Phone mode and NB-TTS

Hurston-1 is the first model in a category we call **NB-TTS (Narrowband TTS)**: text-to-speech built for the phone
line, where most customer service still happens, even when an AI agent answers the call.

## Why narrowband

Voice AI is usually demoed in studio quality and then deployed over a phone line. The public telephone network, SIP
trunks and services such as Twilio carry a narrow band of audio: roughly 300 to 3,400 Hz, sampled at 8 kHz, compressed
with G.711 (mu-law in North America and Japan, A-law almost everywhere else). A voice that sounds fine at 24 kHz can
lose intelligibility and character on that channel, and a voice agent that is tested at 24 kHz is being tested on audio
it will never hear in production.

Hurston-1 was trained in part on about 10 hours of real phone calls between consented contributors and full-duplex AI
voice agents (7,600 turns from 250 contributors), alongside studio-style reading and conversation. Its phone mode
outputs what the caller actually hears.

## What phone mode does

`--phone` on the command line, or `phone=True` in Python, applies a channel effect after synthesis:

1. band-pass filter to the 300 to 3,400 Hz telephone band (6th-order Butterworth, zero-phase),
2. resample from 24 kHz to 8 kHz,
3. encode and decode through ITU-T G.711 mu-law, so the quantization is in the audio.

It returns float32 audio at 8 kHz. Because the channel is applied after synthesis, `--check` (Whisper and voice-change
retries) still judges the full-quality take, and the same take can be exported both ways.

The mu-law tables in `hurston.py` follow ITU-T G.711 and match libsndfile and Twilio byte for byte.

## Writing files

```python
import soundfile as sf
from hurston import Hurston, describe, phone, PHONE_SR, SR

tts = Hurston()
y = tts("Thanks for calling. How can I help you today?", describe('Male', 'adult', 'Southern American accent'))

sf.write('line_24k.wav', y, SR)                                  # full quality
sf.write('line_phone.wav', phone(y), PHONE_SR, subtype='ULAW')   # 8 kHz G.711 mu-law WAV, drops straight into an IVR
sf.write('line_phone_alaw.wav', phone(y), PHONE_SR, subtype='ALAW')
```

`phone(audio, sr)` works on any audio, not only Hurston-1 output. `examples/phone_mode.py --wav recording.wav` applies
it to an existing file, which is a quick way to hear what a recording or another TTS voice sounds like on a call.

## Streaming to a phone API

`ulaw_encode()` returns raw mu-law bytes, one byte per sample at 8 kHz. That is the payload Twilio Media Streams
expect, base64-encoded in JSON `media` messages:

```python
from hurston import Hurston, describe, ulaw_encode
tts = Hurston()
call = tts("Hi, I'm calling about my appointment on Tuesday.", describe('Female', 'older', 'Nigerian English accent'), phone=True)
payload = ulaw_encode(call)   # bytes; see examples/twilio_media_stream.py for the message framing
```

For SIP media servers (Asterisk, FreeSWITCH) and most CPaaS platforms, 8 kHz mu-law or A-law is the native codec, so
the same bytes go in without transcoding.

## Simulated callers

Pair Hurston-1 phone mode with an LLM that writes caller scripts and you have synthetic callers in nine accents for
testing an IVR or a voice agent before real people reach it. Run the phone-mode audio into the agent's inbound leg and
check where it mishears accents, names or numbers. `examples/accent_test_set.py --phone` builds the test audio.

## Limits

- Phone mode changes the channel, not the speaking style. The model does not yet switch to a call-style delivery
  (turn-taking, backchannels, call pacing) on request. That is the goal of the next release.
- Names, dates and amounts are the weakest content on any channel. Use `--check` and listen before shipping.
- Each call to the model is a fresh voice that fits the description. Consistent named voices are planned.
