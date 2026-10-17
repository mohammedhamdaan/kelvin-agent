"""
Diagnostic script to verify Gemini API key, environment loader, and SDK.
"""

import os
import pathlib
import sys
import traceback
from dotenv import load_dotenv


def main():
    print("=" * 65)
    print("KELVIN6K GEMINI CONFIGURATION & SDK DIAGNOSTIC")
    print("=" * 65)

    # 1. Check and print .env file existence
    env_file = pathlib.Path(".env")
    env_exists = env_file.is_file()
    print(f"\n[1] Checking .env file:")
    print(f"    - .env path: {env_file.resolve()}")
    print(f"    - .env exists: {'YES' if env_exists else 'NO'}")

    load_dotenv(override=True)

    # 2. Check and print GEMINI_API_KEY and GEMINI_MODEL
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    configured_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    print(f"\n[2] Checking Environment Variables:")
    if api_key:
        masked_key = (
            f"{api_key[:6]}...{api_key[-4:]}"
            if len(api_key) >= 10
            else "***"
        )
        print(f"    - GEMINI_API_KEY: Present (Length: {len(api_key)} chars, Masked: {masked_key})")
    else:
        print("    - GEMINI_API_KEY: NOT FOUND / EMPTY")

    print(f"    - GEMINI_MODEL: {configured_model}")

    # 3. Check and print installed SDKs
    print(f"\n[3] Checking Installed SDKs:")
    has_new_sdk = False
    new_sdk_ver = "unknown"
    try:
        from google import genai
        has_new_sdk = True
        new_sdk_ver = getattr(genai, "__version__", "installed")
        print(f"    - google-genai (New SDK): INSTALLED (version: {new_sdk_ver})")
    except ImportError:
        print("    - google-genai (New SDK): NOT INSTALLED")

    has_legacy_sdk = False
    legacy_sdk_ver = "unknown"
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import google.generativeai as legacy_genai
        has_legacy_sdk = True
        legacy_sdk_ver = getattr(legacy_genai, "__version__", "installed")
        print(f"    - google-generativeai (Legacy SDK): INSTALLED (version: {legacy_sdk_ver})")
    except ImportError:
        print("    - google-generativeai (Legacy SDK): NOT INSTALLED")

    if not api_key:
        print("\n[!] Cannot perform live API test: GEMINI_API_KEY is not set.")
        print("=" * 65)
        return

    # 4. Make a test call to Gemini
    print(f"\n[4] Making Test Call to Gemini:")
    prompt = "Hello Kelvin, reply with OK"
    print(f"    - Prompt: \"{prompt}\"")

    models_to_test = [configured_model]
    for alt in ["gemini-flash-lite-latest", "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-1.5-flash"]:
        if alt not in models_to_test:
            models_to_test.append(alt)

    test_succeeded = False
    successful_model = None

    if has_new_sdk:
        print("\n--> Testing via `google-genai` SDK:")
        from google import genai

        try:
            client = genai.Client(api_key=api_key)
            for m in models_to_test:
                print(f"    Attempting model: '{m}'...")
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=prompt,
                    )
                    reply_text = (response.text or "").strip()
                    print(f"    >>> SUCCESS with model '{m}'!")
                    print(f"    >>> Gemini Response: {reply_text}")
                    test_succeeded = True
                    successful_model = m
                    break
                except Exception as model_err:
                    print(f"    [X] Model '{m}' failed: {type(model_err).__name__}: {model_err}")
        except Exception as client_err:
            print("    [X] Client initialization error:")
            traceback.print_exc()

    elif has_legacy_sdk:
        print("\n--> Testing via legacy `google-generativeai` SDK:")
        import google.generativeai as legacy_genai

        try:
            legacy_genai.configure(api_key=api_key)
            for m in models_to_test:
                print(f"    Attempting model: '{m}'...")
                try:
                    model_inst = legacy_genai.GenerativeModel(m)
                    response = model_inst.generate_content(prompt)
                    reply_text = (response.text or "").strip()
                    print(f"    >>> SUCCESS with model '{m}'!")
                    print(f"    >>> Gemini Response: {reply_text}")
                    test_succeeded = True
                    successful_model = m
                    break
                except Exception as model_err:
                    print(f"    [X] Model '{m}' failed: {type(model_err).__name__}: {model_err}")
        except Exception as init_err:
            print("    [X] SDK initialization error:")
            traceback.print_exc()
    else:
        print("    [X] Neither `google-genai` nor `google-generativeai` is available.")

    print("\n" + "=" * 65)
    if test_succeeded:
        print(f"RESULT: GEMINI API CALL SUCCEEDED (Model: {successful_model})")
    else:
        print("RESULT: GEMINI API CALL FAILED (See exact tracebacks above)")
    print("=" * 65)


if __name__ == "__main__":
    main()
