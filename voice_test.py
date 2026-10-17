"""
Voice test: ask a question out loud (or type one) and get a spoken answer built
from the CURRENT camera/AI state.

The camera runs in a background thread so answers reflect live conditions.

Usage:
    python voice_test.py                 # push-to-talk: press Enter, then speak
    python voice_test.py --text "Is printing running?"   # skip the microphone
    python voice_test.py --no-camera     # state-only (no camera needed)
    python voice_test.py --show-state    # print the JSON handed to the AI
"""

import argparse
import json
import sys
import threading
import time

from agent import config
from agent.print_monitor import (
    get_camera_status,
    get_current_layer_status,
    get_current_layer_width,
    get_current_printing_status,
)
from agent.qa import answer_question
from agent.rules import RuleEngine
from agent.state import PrinterState
from agent.vision_loop import VisionLoop
from agent.voice import speak

_SAMPLE_QUESTIONS = [
    "Is printing running?",
    "Is the layer being printed?",
    "What is the current layer width?",
    "Is the layer missing?",
    "Is there any problem with printing?",
]


def _print_state(state: PrinterState) -> None:
    print("\n--- current pipeline state (source of truth) ---")
    print("printing     :", get_current_printing_status(state))
    print("layer        :", get_current_layer_status(state))
    print("width        :", get_current_layer_width(state))
    print("camera       :", get_camera_status(state))


def main() -> int:
    parser = argparse.ArgumentParser(description="kelvin6k voice test")
    parser.add_argument("--text", type=str, default=None, help="ask this question without using the mic")
    parser.add_argument("--no-camera", action="store_true", help="don't start the camera")
    parser.add_argument("--show-state", action="store_true", help="print the JSON given to the AI")
    parser.add_argument("--warmup", type=float, default=4.0, help="seconds to let the camera settle")
    args = parser.parse_args()

    state = PrinterState()
    rules = RuleEngine()
    vision_loop = None

    if not args.no_camera:
        # Camera in the background so the answer reflects the live view.
        config.SHOW_CAMERA_WINDOW = False  # headless for this test
        vision_loop = VisionLoop(state=state, rules=rules)
        threading.Thread(target=vision_loop.run, daemon=True).start()
        print(f"Starting camera ({config.CAMERA_SOURCE.upper()}), warming up {args.warmup:.0f}s...")
        time.sleep(args.warmup)

    _print_state(state)

    if args.show_state:
        print("\n--- JSON handed to the AI ---")
        print(json.dumps(state.snapshot().get("print_status", {}), indent=2, default=str))

    if not config.GEMINI_API_KEY:
        print("\nNOTE: GEMINI_API_KEY is not set, so the assistant will reply with a")
        print("configuration error instead of a real answer. The state above is still live.")

    try:
        if args.text:
            questions = [args.text]
        else:
            print("\nSample questions:")
            for q in _SAMPLE_QUESTIONS:
                print(f"  - {q}")
            print("\nPress Enter, then speak your question. Ctrl+C to stop.")
            questions = None

        if questions:
            for question in questions:
                print(f"\nOperator: {question}")
                answer = answer_question(question, state)
                print(f"AI: {answer}")
                state.record_interaction(question, answer)
                speak(answer)
        else:
            from agent.listener import listen_for_question

            while True:
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    break
                try:
                    question = listen_for_question()
                except Exception as exc:
                    print(f"[LISTEN ERROR] {exc}", file=sys.stderr)
                    continue
                if not question:
                    print("Didn't catch that -- try again.")
                    continue
                print(f"Operator: {question}")
                answer = answer_question(question, state)
                print(f"AI: {answer}")
                state.record_interaction(question, answer)
                speak(answer)
                _print_state(state)
    except KeyboardInterrupt:
        pass
    finally:
        if vision_loop is not None:
            vision_loop.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
