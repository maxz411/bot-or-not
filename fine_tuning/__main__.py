from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))

    from fine_tuning.cli import main
else:
    from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
