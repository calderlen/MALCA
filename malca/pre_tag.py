"""Compatibility shim for pre-tagging semantics.

`pre_filter` remains available, but `pre_tag` communicates that this stage
primarily annotates rows with failure flags for downstream gating.
"""

from __future__ import annotations

from malca.pre_filter import (
    apply_pre_filters,
    apply_pre_tags,
    filter_camera_medians,
    main,
)

__all__ = [
    "apply_pre_filters",
    "apply_pre_tags",
    "filter_camera_medians",
    "main",
]
