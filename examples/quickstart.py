#!/usr/bin/env python3
"""Hurston-1 quick start: one line in every accent, female and male, written to out/.

    python examples/quickstart.py
    python examples/quickstart.py --text "Your prescription is ready for pickup." --check

The first run downloads the weights from Hugging Face (destined/hurston-1; accept the terms on the model page once and run
`hf auth login`). --check retries takes Whisper cannot follow or where the voice changes mid-take (needs speechbrain)."""
import os, sys, argparse
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hurston import Hurston, ACCENTS, describe, SR, DEFAULT_MODEL

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--text', default="Hi, thanks for calling. How can I help you today?")
ap.add_argument('--model', default=DEFAULT_MODEL, help='Hugging Face repo id or local folder with the weights')
ap.add_argument('--check', action='store_true')
ap.add_argument('--out', default='out')
a = ap.parse_args()

tts = Hurston(a.model, check=a.check)
os.makedirs(a.out, exist_ok=True)
for accent in ACCENTS:
    for gender in ('Female', 'Male'):
        description = describe(gender, 'adult', accent)          # 'Gender, age, accent, style, pace'
        name = accent.replace(' ', '_').replace('(', '').replace(')', '')
        path = os.path.join(a.out, f'{name}_{gender}.wav')
        sf.write(path, tts(a.text, description), SR)             # float32 at 24 kHz
        print(path, '|', description)
