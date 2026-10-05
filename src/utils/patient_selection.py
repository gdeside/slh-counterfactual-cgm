"""
src/utils/patient_selection.py
==============================

Choose which patient(s) an illustration script should show, without hard-coding
participant IDs.

Rule (``resolve_patient_id`` / ``resolve_patient_pair``)
--------------------------------------------------------
1. An explicit ID (``--patient_id`` argument, or the environment variables
   ``CGM_PATIENT_HIGH`` / ``CGM_PATIENT_LOW``) is used if that patient has an episode file.
2. Otherwise the patient is chosen automatically from the (filtered) cohort:
   ``"most"``   -> the patient with the most SLH episodes,
   ``"fewest"`` -> the patient with the fewest SLH episodes (at least one).

Patient IDs differ between datasets (original study codes, ``A001``… for the sample
data, ``OD_001``… for the open dataset), so scripts must not assume a given ID exists.
To reproduce the manuscript's patient-specific figures on the original data, pass the
two patients explicitly::

    CGM_PATIENT_HIGH=<id> CGM_PATIENT_LOW=<id> make fig3 fig4 fig5
"""
import os

from src.config import EPISODES_PATH, FILTER_STATUS_CSV
from src.episodes.episode_config import LABEL_SLH
from src.episodes.load_episodes import discover_episode_files, load_patient_episodes
from src.filtering.load_filtering import load_filtered_patient_list

ENV_HIGH = "CGM_PATIENT_HIGH"
ENV_LOW = "CGM_PATIENT_LOW"


def cohort_patient_ids() -> list[str]:
    """Patients with an episode file, restricted to the filtered cohort when available."""
    available = discover_episode_files(str(EPISODES_PATH))
    filtered = load_filtered_patient_list(str(FILTER_STATUS_CSV))
    return sorted(p for p in available if p in filtered) if filtered else sorted(available)


def slh_counts() -> dict[str, int]:
    """Number of SLH episodes per cohort patient."""
    counts = {}
    for pid in cohort_patient_ids():
        ep = load_patient_episodes(pid)
        if ep is not None and len(ep):
            counts[pid] = int((ep["classification"] == LABEL_SLH).sum())
    return counts


def _has_episodes(pid: str) -> bool:
    return (EPISODES_PATH / f"{pid}_episodes.csv").exists()


def _auto(role: str, counts: dict[str, int] | None = None) -> str:
    counts = counts if counts is not None else slh_counts()
    if role == "fewest":
        counts = {p: n for p, n in counts.items() if n > 0}
    if not counts:
        raise RuntimeError("No patient with SLH episodes found — run `make episode_detection` first.")
    pick = max if role == "most" else min
    pid = pick(sorted(counts), key=lambda p: counts[p])
    print(f"  [auto] patient with {role} SLH episodes: {pid} (n_SLH = {counts[pid]})")
    return pid


def resolve_patient_id(requested: str | None = None, role: str = "most") -> str:
    """Return ``requested`` if it exists in the data, else an automatically chosen patient."""
    if requested:
        if _has_episodes(requested):
            return requested
        print(f"  [warning] patient '{requested}' not found in {EPISODES_PATH} — selecting automatically.")
    return _auto(role)


def resolve_patient_pair(high: str | None = None, low: str | None = None) -> tuple[str, str]:
    """(patient with high SLH burden, patient with low SLH burden).

    Explicit IDs, then ``CGM_PATIENT_HIGH`` / ``CGM_PATIENT_LOW``, then automatic selection
    (most / fewest SLH episodes)."""
    high = high or os.environ.get(ENV_HIGH)
    low = low or os.environ.get(ENV_LOW)
    counts = None
    if not (high and _has_episodes(high)):
        counts = slh_counts()
        high = _auto("most", counts)
    if not (low and _has_episodes(low)) or low == high:
        counts = counts if counts is not None else slh_counts()
        low = _auto("fewest", {p: n for p, n in counts.items() if p != high})
    return high, low
