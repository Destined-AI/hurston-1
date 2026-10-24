#!/usr/bin/env python3
"""Build an accent-coverage test set for a speech recognizer: the same sentences in every Hurston-1 accent.

    python examples/accent_test_set.py                                   # built-in sentences, 24 kHz
    python examples/accent_test_set.py --sentences my_lines.txt --phone --check

Writes one WAV per sentence x accent x gender into --out, plus manifest.csv (file, text, accent, gender, description, sample_rate)
to score your ASR, captioning or IVR against. --phone writes 8 kHz G.711 mu-law files, what the recognizer hears on a call.
Use --check for a cleaner reference set: it retries takes Whisper cannot follow, so what is left measures your recognizer, not the TTS."""
import os, sys, csv, argparse
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hurston import Hurston, ACCENTS, describe, SR, PHONE_SR, DEFAULT_MODEL

SENTENCES = ["Hi, thanks for calling. How can I help you today?",
             "Your prescription is ready for pickup at the front desk.",
             "I understand. Let me check what we can do about that.",
             "The meeting has moved to Thursday at two thirty.",
             "Take a deep breath and tell me when the pain started.",
             "That's wonderful news, congratulations to the whole team!"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--sentences', help='text file, one sentence per line (default: six built-in lines)')
ap.add_argument('--accents', nargs='*', default=ACCENTS, help='subset of accents (default: all nine)')
ap.add_argument('--model', default=DEFAULT_MODEL)
ap.add_argument('--phone', action='store_true', help='8 kHz G.711 mu-law output')
ap.add_argument('--check', action='store_true')
ap.add_argument('--out', default='asr_test')
a = ap.parse_args()

sentences = [l.strip() for l in open(a.sentences)] if a.sentences else SENTENCES
sentences = [s for s in sentences if s]
tts = Hurston(a.model, check=a.check)
os.makedirs(a.out, exist_ok=True)
rows = []
for accent in a.accents:
    for gender in ('Female', 'Male'):
        description = describe(gender, 'adult', accent)
        for i, text in enumerate(sentences):
            name = f"{accent.replace(' ', '_').replace('(', '').replace(')', '')}_{gender}_{i:03d}.wav"
            y = tts(text, description, phone=a.phone)
            if a.phone: sf.write(os.path.join(a.out, name), y, PHONE_SR, subtype='ULAW')
            else: sf.write(os.path.join(a.out, name), y, SR)
            rows.append({'file': name, 'text': text, 'accent': accent, 'gender': gender, 'description': description,
                         'sample_rate': PHONE_SR if a.phone else SR})
            print(name, flush=True)
with open(os.path.join(a.out, 'manifest.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'{len(rows)} clips and manifest.csv in {a.out}/')
