"""
Safe diagnostic for the OPTIONAL Anthropic API configuration.

NOTE: the live voice assistant now uses Google Gemini (see check_gemini_key.py).
This script and agent/anthropic_qa.py are kept so the Anthropic backend remains
available and testable.

Reports whether a key/workspace ID is present and whether the key's *shape*
looks like a real key, then sends one minimal live request to confirm it
actually works. Never prints, logs, or stores the key or workspace ID
values themselves.

Usage:
    python check_api_key.py
"""

from agent import config
from agent.anthropic_qa import discover_workspaces, test_connection


def _key_format_looks_valid(key: str) -> bool:
    # Shape check only, not a validity check. Real Anthropic keys start
    # with "sk-ant-" and are much longer than any placeholder text.
    return key.startswith("sk-ant-") and len(key) >= 40


def _workspace_id_format_looks_valid(workspace_id: str) -> bool:
    # Shape check only, not a validity check. Anthropic workspace IDs are
    # of the form "wrkspc_01..." (a "wrkspc_" prefix followed by an
    # identifier, per Anthropic's own SDK/CLI documentation).
    return workspace_id.startswith("wrkspc_") and len(workspace_id) >= 20


_ACTIONS = {
    "invalid_api_key": (
        "This key is being rejected by the Anthropic API.\n"
        "Generate a new key at https://console.anthropic.com/settings/keys\n"
        "and paste it into ANTHROPIC_API_KEY in your .env file."
    ),
    "missing_workspace_id": (
        "This is an identity-linked API key, which requires a workspace ID on every request.\n"
        "See the Workspace ID section below for how to find it."
    ),
    "invalid_workspace_id": (
        "The configured ANTHROPIC_WORKSPACE_ID was rejected -- it's either wrong or this key\n"
        "doesn't have access to that workspace. See the Workspace ID section below."
    ),
    "billing": "Check your account balance at https://console.anthropic.com/settings/billing",
    "permission": "Check this key's workspace/model permissions in the Anthropic Console.",
    "invalid_model": (
        f"'{config.ANTHROPIC_MODEL}' is not available to this account.\n"
        "Try a different model in .env, e.g. ANTHROPIC_MODEL=claude-sonnet-5"
    ),
    "network": "Check your internet connection and try again.",
    "rate_limit": "Wait a moment and try again.",
    "no_key": "Add ANTHROPIC_API_KEY to your .env file.",
}

_WORKSPACE_CATEGORIES = {"missing_workspace_id", "invalid_workspace_id"}


def _print_workspace_help():
    print("\n--- Finding your Workspace ID ---")
    print("Attempting to safely list workspaces visible to this key...")
    workspaces = discover_workspaces()
    if workspaces:
        print(f"Found {len(workspaces)} workspace(s) this key can see:")
        for ws in workspaces:
            print(f"  - {ws['name']}  (id: {ws['id']})")
        print("\nCopy the id of the workspace you want this agent to act in into")
        print("ANTHROPIC_WORKSPACE_ID in your .env file.")
    else:
        print("Could not auto-discover workspaces with this key (expected for most")
        print("identity-linked keys -- listing workspaces needs org-admin access).")
        print("\nTo find it manually:")
        print("  1. Go to https://console.anthropic.com/ and sign in.")
        print("  2. Use the workspace switcher (top of the page) or go to")
        print("     Settings -> Workspaces, and open the workspace this key should act in.")
        print("  3. That workspace's page/settings shows its Workspace ID -- it looks like")
        print("     'wrkspc_' followed by a long identifier, e.g. wrkspc_01AbCdEfGhIjKlMnOpQrSt.")
        print("  4. Copy that entire ID into ANTHROPIC_WORKSPACE_ID in your .env file.")


def main():
    key = config.ANTHROPIC_API_KEY
    workspace_id = config.ANTHROPIC_WORKSPACE_ID

    print("=== Anthropic API configuration diagnostic ===")
    print(f"API key present: {'yes' if key else 'no'}")
    print(f"API key format looks valid: {'yes' if key and _key_format_looks_valid(key) else 'no'}")
    print(f"Workspace ID present: {'yes' if workspace_id else 'no'}")
    if workspace_id:
        print(f"Workspace ID format looks valid: {'yes' if _workspace_id_format_looks_valid(workspace_id) else 'no'}")
    print(f"Model configured: {config.ANTHROPIC_MODEL}")

    if not key:
        print("\nResult: FAILURE -- no API key configured.")
        print(f"Action: {_ACTIONS['no_key']}")
        return

    print("\nSending a minimal test request to the Anthropic API...")
    result = test_connection()

    if result["success"]:
        print(f"Result: SUCCESS -- {result['detail']}")
        return

    print(f"Result: FAILURE ({result['category']}) -- {result['detail']}")
    action = _ACTIONS.get(result["category"])
    if action:
        print(f"\nAction: {action}")

    if result["category"] in _WORKSPACE_CATEGORIES:
        _print_workspace_help()


if __name__ == "__main__":
    main()
