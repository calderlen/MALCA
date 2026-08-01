"""Multi-survey point-source point-lens fitting for MALCA.

The package initializer stays light because Review imports the schema during
database startup. Scientific functions live in the explicit submodules.
"""

from .schema import MICROLENSING_JOINT_VERSION

__all__ = ["MICROLENSING_JOINT_VERSION"]
