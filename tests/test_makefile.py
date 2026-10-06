"""The Makefile is the documented entry point: every target must be declared, documented and point to a real module."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAKEFILE = (ROOT / "Makefile").read_text()


def _targets():
    return re.findall(r"^([A-Za-z0-9_-]+):(?!=)", MAKEFILE, re.M)


def _phony():
    block = re.search(r"^\.PHONY:((?:.|\n)*?)\n(?=\S)", MAKEFILE, re.M).group(1)
    return set(re.findall(r"[\w-]+", block.replace("\\", " ")))


def test_every_target_is_phony_and_documented():
    targets = _targets()
    assert set(targets) <= _phony()
    assert _phony() <= set(targets)
    for t in targets:
        assert re.search(rf"^{re.escape(t)}:.*##", MAKEFILE, re.M), f"target '{t}' has no '##' help text"


def test_no_duplicate_targets():
    targets = _targets()
    assert len(targets) == len(set(targets))


def test_every_module_called_by_make_exists():
    modules = re.findall(r"\$\(RUN\)\s+(scripts[\w.]+)", MAKEFILE)
    assert modules
    missing = [m for m in modules if not (ROOT / (m.replace(".", "/") + ".py")).exists()]
    assert not missing, f"Makefile calls missing modules: {missing}"


def test_removed_targets_stay_removed():
    assert not re.search(r"^(clock_lock|slh_schema|fig5_determinants):", MAKEFILE, re.M)


def test_clinical_targets_check_the_table_first():
    for t in ("table1", "fig5", "fig6", "suppfig_phenotype", "regression_analysis"):
        assert re.search(rf"^{t}:\s*check_clinical_table", MAKEFILE, re.M), t
    assert "check_clinical_table" not in re.search(r"^paper:.*$", MAKEFILE, re.M).group(0)
