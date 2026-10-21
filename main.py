"""
Kelvin6K Voice Assistant CLI entrypoint.

Run interactive console:
    python main.py
    python main.py --voice
    python main.py --ask "What is the machine status?"
"""

import sys
from assistant_cli import main

if __name__ == "__main__":
    sys.exit(main())
