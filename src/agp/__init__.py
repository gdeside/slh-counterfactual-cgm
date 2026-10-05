"""
src/agp/__init__.py
===================

AGP (Ambulatory Glucose Profile) computation module.

Exposes functions to compute the median daily glucose profile and
episode-recurrence profiles, used across analysis and visualization scripts.

    from src.agp import compute_agp_profile
    from src.agp import compute_recurrence_by_duration, compute_recurrence_by_classification
"""

from .compute_agp import (
    compute_agp_profile,
    compute_recurrence_by_duration,
    compute_recurrence_by_classification,
)

__all__ = [
    "compute_agp_profile",
    "compute_recurrence_by_duration",
    "compute_recurrence_by_classification",
]
