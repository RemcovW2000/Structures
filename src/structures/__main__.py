# Module execution entrypoint.
# The previous CLI has been removed. This keeps python -m structures from failing.
from __future__ import annotations

import sys

if __name__ == "__main__":  # pragma: no cover
    sys.stdout.write("The structures CLI has been removed. Nothing to run.\n")
