"""
Safe diagnostic for the Gemini API configuration used by the voice assistant.

Reports whether a key is present and whether its shape looks plausible, then
sends one minimal live request. Never prints, logs, or stores the key itself.

Usage:
    python check_gemini_key.py
"""

from agent import config
from agent.qa import test_connection


def _key_format_looks_valid(key: str) -> bool:
    # Shape check only, not a validity check. Google AI Studio keys start with
    # "AIza" and are ~39 characters.
    return key.startswith("AIza") and len(key) >= 30


_ACTIONS = {
    "no_key": (
        "Create a free API key at https://aistudio.google.com/apikey\n"
        "and paste it into GEMINI_API_KEY in your .env file."
    ),
    "invalid_api_key": (
        "This key is being rejected by Gemini.\n"
        "Create a new key at https://aistudio.google.com/apikey\n"
        "and paste it into GEMINI_API_KEY in your .env file."
    ),
    "invalid_model": (
        f"'{config.GEMINI_MODEL}' is not available to this key.\n"
        "Set a model your key can use in .env, e.g. GEMINI_MODEL=gemini-2.5-flash\n"
        "(list what your key can access at https://aistudio.google.com/)."
    ),
    "rate_limit": "Free tier quota/rate limit hit. Wait a moment and try again.",
    "server_error": "Gemini is having a server-side problem. Try again shortly.",
    "network": "Check your internet connection and try again.",
}


def main():
    key = config.GEMINI_API_KEY

    print("=== Gemini API configuration diagnostic ===")
    print(f"API key present: {'yes' if key else 'no'}")
    print(f"API key format looks valid: {'yes' if key and _key_format_looks_valid(key) else 'no'}")
    print(f"Model configured: {config.GEMINI_MODEL}")

    if not key:
        print("\nResult: FAILURE -- no API key configured.")
        print(f"Action: {_ACTIONS['no_key']}")
        return

    print("\nSending a minimal test request to the Gemini API...")
    result = test_connection()

    if result["success"]:
        print(f"Result: SUCCESS -- {result['detail']}")
        return

    print(f"Result: FAILURE ({result['category']}) -- {result['detail']}")
    action = _ACTIONS.get(result["category"])
    if action:
        print(f"\nAction: {action}")


if __name__ == "__main__":
    main()
