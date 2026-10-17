"""
Standalone TTS diagnostic + playback test.

    python tts_test.py                    # diagnose, synthesise, verify, play
    python tts_test.py --text "Hello"     # custom sentence
    python tts_test.py --no-play          # synthesise + verify only

Verifies that real, non-silent audio is produced (by measuring the waveform),
then plays it on this machine's speaker.
"""

import argparse
import array
import contextlib
import os
import sys
import time
import wave

from agent.logging_setup import configure_logging
from agent.voice import TTSError, diagnostics, speak, synthesize


def measure(path: str):
    """Peak/RMS amplitude of a WAV, to prove the audio isn't silence."""
    if not path.lower().endswith(".wav"):
        return None  # mp3 (ElevenLabs) -- size check only
    with contextlib.closing(wave.open(path)) as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        raw = handle.readframes(frames)
    samples = array.array("h")
    samples.frombytes(raw[: len(raw) // 2 * 2])
    if not samples:
        return {"seconds": 0, "peak": 0, "rms": 0}
    peak = max(abs(s) for s in samples)
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
    return {"seconds": frames / rate, "peak": peak, "rms": rms}


def main() -> int:
    parser = argparse.ArgumentParser(description="Kelvin6K TTS test")
    parser.add_argument("--text", default="Hello, this is the Kelvin6K Robot Assistant.")
    parser.add_argument("--no-play", action="store_true")
    args = parser.parse_args()

    configure_logging()

    print("=== KELVIN6K TTS TEST ===\n")
    info = diagnostics()
    print(f"Provider              : {info['provider']}")
    print(f"ElevenLabs key set    : {info['elevenlabs_key_configured']}")
    print(f"Windows voices        : {info['voice_count']}")
    for name in info.get("pyttsx3_voices", []):
        print(f"  - {name}")
    print(f"Current voice         : {info.get('current_voice')}")
    print(f"Engine volume         : {info.get('volume')}   (0.0 = silent)")
    print(f"Engine rate           : {info.get('rate')}")
    if info.get("engine_error"):
        print(f"ENGINE ERROR          : {info['engine_error']}")
    print()

    print("--- 1. synthesise to file ---")
    try:
        path, mime = synthesize(args.text)
    except TTSError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1
    size = os.path.getsize(path)
    print(f"file : {path}")
    print(f"type : {mime}  ({size} bytes)")

    stats = measure(path)
    if stats:
        print(f"audio: {stats['seconds']:.2f}s  peak={stats['peak']}  rms={stats['rms']:.0f}")
        if stats["peak"] < 500:
            print("\nVERDICT: SILENT WAVEFORM -- the speech engine produced no sound.")
            return 1
        print("VERDICT: real, non-silent audio produced.")
    elif size < 1024:
        print("\nVERDICT: audio file suspiciously small.")
        return 1

    if args.no_play:
        return 0

    print("\n--- 2. play on this machine's speaker ---")
    print("You should HEAR the sentence now. If not, the audio is being")
    print("generated correctly but routed to the wrong output device:")
    print("  * check the Windows default playback device (speaker icon)")
    print("  * check Volume Mixer for python.exe")
    started = time.time()
    ok = speak(args.text)
    elapsed = time.time() - started
    print(f"\nspeak() -> {'OK' if ok else 'FAILED'} after {elapsed:.2f}s")
    if ok and elapsed < 0.4:
        print("WARNING: returned almost instantly -- playback probably did not run.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
