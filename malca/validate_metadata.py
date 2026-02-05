"""Small type aliases shared across pipeline modules.

This is intentionally lightweight: it exists mainly to make allowed values
discoverable in function signatures (for agentic refactors and internal APIs).
"""

from __future__ import annotations

from typing import Literal, TypeAlias


EventKind: TypeAlias = Literal["dip", "jump"]
