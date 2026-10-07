"""Counterfactual scripts: patient (or the patient with the most SLH episodes) and parameters from the command line."""
import pytest

from scripts.counterfactual import (
    lininterp_45days, lininterp_offset_comp, lininterp_stepbystep, visualize_cdf_sensitivity,
)
from src.counterfactual.counterfactual_config import OFFSET_VALUES, SELECTED_OFFSET_MIN


@pytest.mark.parametrize("module", [lininterp_45days, lininterp_stepbystep, lininterp_offset_comp])
def test_patient_defaults_to_automatic_and_can_be_given(module):
    assert module.parse_arguments([]).patient_id is None            # None -> patient with the most SLH episodes
    assert module.parse_arguments(["--patient_id", "P7"]).patient_id == "P7"
    assert module.parse_arguments(["--out_dir", "x"]).out_dir == "x"


@pytest.mark.parametrize("module", [lininterp_45days, lininterp_stepbystep])
def test_single_offset_option(module):
    assert module.parse_arguments([]).offset == SELECTED_OFFSET_MIN
    assert module.parse_arguments(["--offset", "45"]).offset == 45


def test_stepbystep_number_of_episodes():
    assert lininterp_stepbystep.parse_arguments([]).n_episodes == lininterp_stepbystep.N_EPISODES_TO_PLOT
    assert lininterp_stepbystep.parse_arguments(["--n_episodes", "5"]).n_episodes == 5


def test_offset_comparison_offsets():
    assert lininterp_offset_comp.parse_arguments([]).offsets == sorted(OFFSET_VALUES)
    assert lininterp_offset_comp.parse_arguments(["--offsets", "60", "0", "30", "30"]).offsets == [0, 30, 60]
    assert lininterp_offset_comp.parse_arguments(["--n_episodes", "2"]).n_episodes == 2
