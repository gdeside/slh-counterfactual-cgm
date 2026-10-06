"""Fake patients re-assembled from real traces: mixing, noise, relative time (tested on synthetic inputs)."""
import numpy as np
import pandas as pd
import pytest

from scripts.sample_data import generate_sample_data_from_real as gen


def _trace(start, days, level, seed):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=days * gen.SLOTS_PER_DAY, freq="5min")
    return pd.DataFrame({"time": t, "glycemie": level + rng.normal(0, 5, len(t))})


def test_split_days_keeps_complete_days_on_a_regular_grid():
    df = _trace("2024-05-06 00:00", 5, 120, 0)
    days = gen.split_days(df)
    assert len(days) == 4                                  # 5 calendar days -> 4 full 03:00-03:00 segments
    assert all(d.shape == (gen.SLOTS_PER_DAY,) and np.isfinite(d).all() for d in days)


def test_split_days_drops_gappy_days():
    df = _trace("2024-05-06 03:00", 3, 120, 1)
    df = df[~((df["time"] >= "2024-05-07 06:00") & (df["time"] < "2024-05-07 14:00"))]   # 8 h gap in day 2
    assert len(gen.split_days(df)) == 2


def test_assign_pools_are_disjoint_and_ordered():
    pools = gen.assign_pools([f"P{i:02d}" for i in range(20)], 4, 5)
    flat = [p for pool in pools for p in pool]
    assert len(pools) == 4 and len(flat) == len(set(flat)) == 20
    assert pools[0][0] == "P00" and pools[-1][-1] == "P19"


def test_assign_pools_requires_enough_sources():
    with pytest.raises(ValueError):
        gen.assign_pools(["a", "b", "c"], 4, 5)


@pytest.fixture
def pool():
    days = []
    for k in range(6):                                      # six "real patients" with distinct levels
        days += gen.split_days(_trace("2024-01-01 03:00", 12, 100 + 20 * k, k))
    return days


def test_fake_patient_uses_relative_time_and_regular_grid(pool):
    fake = gen.build_fake_patient(pool, 20, np.random.default_rng(0))
    assert fake["time"].iloc[0] == gen.ORIGIN and gen.ORIGIN.year == 2000
    assert (fake["time"].diff().dropna() == pd.Timedelta(minutes=5)).all()
    assert len(fake) == 20 * gen.SLOTS_PER_DAY


def test_fake_patient_mixes_sources_and_adds_noise(pool):
    fake = gen.build_fake_patient(pool, 20, np.random.default_rng(1), noise_std=2.0)
    day_means = fake["glycemie"].to_numpy().reshape(20, -1).mean(axis=1)
    assert day_means.max() - day_means.min() > 30           # days come from several different source levels
    assert fake["glycemie"].between(gen.GLUCOSE_MIN, gen.GLUCOSE_MAX).all()
    for d in pool:                                          # no real day is reproduced verbatim
        assert not any(np.array_equal(d, fake["glycemie"].to_numpy()[i * 288:(i + 1) * 288]) for i in range(20))


def test_each_real_day_is_used_at_most_once(pool):
    sentinel = [np.full(gen.SLOTS_PER_DAY, 100.0 + i) for i in range(30)]
    fake = gen.build_fake_patient(sentinel, 30, np.random.default_rng(2), noise_std=0.0)
    levels = np.round(fake["glycemie"].to_numpy().reshape(30, -1).mean(axis=1) / fake["glycemie"].mean() * 100, 3)
    assert len(set(levels)) == 30


def test_fake_patient_is_reproducible_and_seed_dependent(pool):
    a = gen.build_fake_patient(pool, 10, np.random.default_rng(3))
    b = gen.build_fake_patient(pool, 10, np.random.default_rng(3))
    c = gen.build_fake_patient(pool, 10, np.random.default_rng(4))
    assert a.equals(b) and not a.equals(c)


def test_pool_too_small_is_rejected(pool):
    with pytest.raises(ValueError):
        gen.build_fake_patient(pool[:5], 10, np.random.default_rng(0))


def test_main_writes_four_fake_patients_without_leaking_source_ids(tmp_path, monkeypatch, capsys):
    pids = [f"SRC{i:03d}" for i in range(20)]
    monkeypatch.setattr(gen, "load_ranked_cohort", lambda: pids)
    monkeypatch.setattr(gen, "load_days", lambda pid: gen.split_days(_trace("2024-03-01 03:00", 15, 100 + int(pid[3:]) * 3, int(pid[3:]))))
    out = tmp_path / "fake"
    gen.main(["--out", str(out), "--n_days", "30", "--min_sources", "5"])
    files = sorted(p.name for p in (out / "Data" / "Raw").glob("*.xlsx"))
    assert files == ["A001.xlsx", "A002.xlsx", "A003.xlsx", "A004.xlsx"]
    df = pd.read_excel(out / "Data" / "Raw" / "A001.xlsx")
    assert list(df.columns) == ["time", "glycemie"] and df["time"].iloc[0] == gen.ORIGIN
    assert "SRC" not in capsys.readouterr().out                  # real IDs are never printed
