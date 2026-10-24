#!/usr/bin/env python3
"""Turn Hurston-1 phone-mode audio into Twilio Media Stream messages.

A Twilio <Stream> WebSocket expects 8 kHz G.711 mu-law audio, base64-encoded, inside JSON "media" messages. Hurston-1's
phone mode already produces that audio, and ulaw_encode() gives the raw bytes, so there is no transcoding step.

This example builds the messages without opening a network connection. In your WebSocket handler (FastAPI, websockets,
aiohttp...) send each message with `await ws.send(msg)` after Twilio's "start" event arrives with the streamSid.
To stop playback when the caller interrupts, send {"event": "clear", "streamSid": sid}."""
import os, sys, json, base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hurston import Hurston, describe, ulaw_encode, PHONE_SR


def media_messages(stream_sid, ulaw_bytes, frame_ms=20, mark='hurston-done'):
    """Yield Twilio media messages (JSON strings) for raw mu-law bytes, 20 ms per message (160 bytes at 8 kHz), then a mark."""
    n = PHONE_SR * frame_ms // 1000
    for i in range(0, len(ulaw_bytes), n):
        payload = base64.b64encode(ulaw_bytes[i:i + n]).decode()
        yield json.dumps({'event': 'media', 'streamSid': stream_sid, 'media': {'payload': payload}})
    yield json.dumps({'event': 'mark', 'streamSid': stream_sid, 'mark': {'name': mark}})   # Twilio echoes it when playback is done


if __name__ == '__main__':
    tts = Hurston()
    audio = tts("Hi, I'm calling about my appointment on Tuesday.", describe('Female', 'older', 'Nigerian English accent'), phone=True)
    ulaw = ulaw_encode(audio)                      # one byte per sample at 8 kHz
    msgs = list(media_messages('MZxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx', ulaw))
    print(f'{len(ulaw)} mu-law bytes = {len(ulaw) / PHONE_SR:.1f} s of audio -> {len(msgs) - 1} media messages of 20 ms + 1 mark')
    print(msgs[0][:120] + '...')
