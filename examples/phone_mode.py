#!/usr/bin/env python3
"""Phone mode (NB-TTS): the same line at full quality and as a caller hears it on a phone line.

    python examples/phone_mode.py
    python examples/phone_mode.py --wav some_recording.wav      # apply the phone channel to any existing audio file

Phone mode band-limits the audio to 300-3,400 Hz, resamples it to 8 kHz and passes it through G.711 mu-law (or A-law),
the format phone networks, SIP trunks and Twilio use. It is a channel effect applied after synthesis."""
import os, sys, argparse
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hurston import Hurston, describe, phone, SR, PHONE_SR, DEFAULT_MODEL

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--text', default="Thanks for calling. I can see your appointment is on Tuesday at two thirty.")
ap.add_argument('--accent', default='Southern American accent')
ap.add_argument('--gender', default='Female', choices=['Female', 'Male'])
ap.add_argument('--model', default=DEFAULT_MODEL)
ap.add_argument('--wav', help='skip synthesis and apply the phone channel to this file instead')
ap.add_argument('--out', default='out')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

if a.wav:
    y, sr = sf.read(a.wav, dtype='float32')
    if y.ndim > 1: y = y.mean(axis=1)
    stem = os.path.splitext(os.path.basename(a.wav))[0]
else:
    tts = Hurston(a.model)
    y, sr = tts(a.text, describe(a.gender, 'adult', a.accent)), SR
    stem = 'line'
    sf.write(os.path.join(a.out, f'{stem}_24k.wav'), y, SR)             # full quality, 24 kHz

narrow = phone(y, sr)                                                    # float32 at 8 kHz, already through mu-law once
sf.write(os.path.join(a.out, f'{stem}_phone_ulaw.wav'), narrow, PHONE_SR, subtype='ULAW')   # what Twilio and most US/Japan lines carry
sf.write(os.path.join(a.out, f'{stem}_phone_alaw.wav'), narrow, PHONE_SR, subtype='ALAW')   # A-law, used in Europe and most other regions
print(f'{len(narrow) / PHONE_SR:.1f} s written to {a.out}/: {stem}_phone_ulaw.wav, {stem}_phone_alaw.wav' + ('' if a.wav else f', {stem}_24k.wav'))
