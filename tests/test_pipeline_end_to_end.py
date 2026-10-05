"""End-to-end test on a synthetic trace: detection -> classification -> counterfactual -> delta metrics."""
import numpy as np
import pytest

from src.counterfactual.build_counterfactual import build_full_counterfactual_signal
from src.episodes.criteria import classify_episodes
from src.episodes.detection import detect_hyperglycemia_episodes
from src.episodes.episode_config import H_HYPER, LABEL_LLH, LABEL_SLH
from src.metrics.metrics import compute_time_in_range_metrics

TIMESTEP_MIN = 5.0
OFFSET_MIN = 30


@pytest.fixture
def classified(synthetic_trace):
    episodes = detect_hyperglycemia_episodes(synthetic_trace)
    return episodes, classify_episodes(synthetic_trace, episodes)


def test_three_episodes_detected(classified):
    episodes, _ = classified
    assert len(episodes) == 3


def test_exactly_one_slh_and_two_llh(classified):
    _, table = classified
    assert (table["classification"] == LABEL_SLH).sum() == 1
    assert (table["classification"] == LABEL_LLH).sum() == 2


def test_slh_is_the_daytime_short_isolated_spike(classified):
    _, table = classified
    slh = table[table["classification"] == LABEL_SLH].iloc[0]
    assert slh["onset_time"].hour == 12
    assert slh["duration_min"] <= 60
    assert slh["C3"] and slh["C4"] and slh["C5"]


def test_night_spike_fails_only_the_daytime_criterion(classified):
    _, table = classified
    night = table[table["onset_time"].dt.hour == 23].iloc[0]
    assert night["C1"] and not night["C3"]


def test_long_excursion_fails_duration(classified):
    _, table = classified
    long_ep = table[table["duration_min"] > 60].iloc[0]
    assert not long_ep["C1"]
    assert long_ep["classification"] == LABEL_LLH


@pytest.fixture
def counterfactual(synthetic_trace, classified):
    _, table = classified
    slh = [{"onset_time": r.onset_time, "offset_time": r.offset_time}
           for r in table[table["classification"] == LABEL_SLH].itertuples()]
    observed = synthetic_trace["glycemie"].to_numpy(dtype=float)
    return observed, build_full_counterfactual_signal(synthetic_trace, slh, OFFSET_MIN, TIMESTEP_MIN)


def test_counterfactual_only_changes_the_slh_zone(synthetic_trace, counterfactual):
    observed, cf = counterfactual
    changed = np.flatnonzero(cf != observed)
    assert len(changed) > 0
    day2 = synthetic_trace["time"].dt.day.to_numpy() == 2     # the SLH day
    assert day2[changed].all()


def test_counterfactual_removes_the_slh_excursion(counterfactual):
    observed, cf = counterfactual
    assert (observed > H_HYPER).sum() > (cf > H_HYPER).sum()
    assert cf.max() <= observed.max()


def test_counterfactual_keeps_llh_episodes_untouched(synthetic_trace, counterfactual):
    observed, cf = counterfactual
    long_zone = (synthetic_trace["time"].dt.day.to_numpy() == 3)
    assert np.array_equal(observed[long_zone], cf[long_zone])


def test_delta_metrics_have_expected_sign(counterfactual):
    observed, cf = counterfactual
    obs_m, cf_m = compute_time_in_range_metrics(observed), compute_time_in_range_metrics(cf)
    tar = [k for k in obs_m if "tar" in k.lower() and "180" in k]
    assert tar, f"no TAR metric in {list(obs_m)}"
    for k in tar:                       # removing SLH can only lower time above range
        assert cf_m[k] <= obs_m[k]
    assert any(cf_m[k] < obs_m[k] for k in tar)
    assert np.median(cf) <= np.median(observed)


def test_offset_zero_changes_no_more_than_larger_offset(synthetic_trace, classified):
    _, table = classified
    slh = [{"onset_time": r.onset_time, "offset_time": r.offset_time}
           for r in table[table["classification"] == LABEL_SLH].itertuples()]
    obs = synthetic_trace["glycemie"].to_numpy(dtype=float)
    n0 = (build_full_counterfactual_signal(synthetic_trace, slh, 0, TIMESTEP_MIN) != obs).sum()
    n60 = (build_full_counterfactual_signal(synthetic_trace, slh, 60, TIMESTEP_MIN) != obs).sum()
    assert n0 <= n60
