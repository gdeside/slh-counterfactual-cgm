"""Automatic / explicit patient selection (no hard-coded participant IDs)."""
import pandas as pd
import pytest

import src.episodes.load_episodes as load_episodes
import src.utils.patient_selection as ps


@pytest.fixture
def episodes_dir(tmp_path, monkeypatch):
    """Cohort of three patients with 5, 1 and 3 SLH episodes (+ one LLH each)."""
    for pid, n_slh in {"P1": 5, "P2": 1, "P3": 3}.items():
        t = pd.date_range("2000-01-01 12:00", periods=n_slh + 1, freq="1D")
        pd.DataFrame({"onset_time": t, "offset_time": t + pd.Timedelta(minutes=30),
                      "classification": ["SLH"] * n_slh + ["LLH"]}
                     ).to_csv(tmp_path / f"{pid}_episodes.csv", index=False)
    monkeypatch.setattr(ps, "EPISODES_PATH", tmp_path)
    monkeypatch.setattr(load_episodes, "EPISODES_PATH", tmp_path)
    monkeypatch.setattr(ps, "FILTER_STATUS_CSV", tmp_path / "missing_filter.csv")
    monkeypatch.delenv(ps.ENV_HIGH, raising=False)
    monkeypatch.delenv(ps.ENV_LOW, raising=False)
    return tmp_path


def test_slh_counts(episodes_dir):
    assert ps.slh_counts() == {"P1": 5, "P2": 1, "P3": 3}


def test_automatic_choice_is_most_and_fewest(episodes_dir):
    assert ps.resolve_patient_id() == "P1"
    assert ps.resolve_patient_id(role="fewest") == "P2"
    assert ps.resolve_patient_pair() == ("P1", "P2")


def test_explicit_id_is_respected(episodes_dir):
    assert ps.resolve_patient_id("P3") == "P3"


def test_unknown_id_falls_back_to_automatic(episodes_dir, capsys):
    assert ps.resolve_patient_id("A174") == "P1"
    assert "not found" in capsys.readouterr().out


def test_environment_variables_select_the_pair(episodes_dir, monkeypatch):
    monkeypatch.setenv(ps.ENV_HIGH, "P3")
    monkeypatch.setenv(ps.ENV_LOW, "P2")
    assert ps.resolve_patient_pair() == ("P3", "P2")


def test_pair_never_returns_the_same_patient_twice(episodes_dir):
    high, low = ps.resolve_patient_pair(high="P2", low="P2")
    assert high != low


def test_no_episode_files_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "EPISODES_PATH", tmp_path)
    monkeypatch.setattr(load_episodes, "EPISODES_PATH", tmp_path)
    monkeypatch.setattr(ps, "FILTER_STATUS_CSV", tmp_path / "missing.csv")
    with pytest.raises(RuntimeError):
        ps.resolve_patient_id()
