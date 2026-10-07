"""
scripts/preprocessing/convert_data_units.py
===========================================

Publish mmol/L copies of the project's data files.

The pipeline computes and stores glucose in **mg/dL** everywhere: window
signals, metrics, counterfactual signals and every threshold in
``src.config``.  This script does not change that.  It reads the produced
data files, converts the columns that hold a glucose *concentration*, and
writes the result to a separate tree so the mg/dL originals stay intact and
remain the pipeline's input.

Columns that are percentages (CV, TIR / TAR / TBR), counts, durations,
coverage, timestamps or HbA1c are copied unchanged — the classification is
in ``src.utils.unit_conversion.is_glucose_valued_column`` and is reported
per file so it can be checked.

Targets
-------
windows       Data/Preprocessed/<PID>/<PID>_win*.xlsx      (glycemie)
metrics       Data/Metrics/metrics_all_patients.csv        (mean/median/SD/...)
distribution  Data/Filtered/filtered_cohort_metrics_distribution.csv
              (long format: converted row-wise, glucose metrics only)
episodes      Data/Episodes/<PID>_episodes.csv             (no glucose column)
signals       Results/Sensitivity_Analysis/Signals/<PID>_signals.csv
sensitivity   Results/Sensitivity_Analysis/Data/cohort_sensitivity.csv
opendata      Results/Open_Data/**/*.csv                   (glucose_mgdl ->
              glucose_mmoll; prefer regenerating the release with
              ``generate_open_data.py --unit mmol/L``, which also rewrites
              DATACARD.md)
cohort        Data/Final cohort SLH.csv                    (';' + decimal ',')

Usage
-----
    # See what would be converted, and which columns, without writing
    python3 -m scripts.preprocessing.convert_data_units --dry-run

    # Write mmol/L copies under <BASE_PATH>/Converted_mmoll/
    python3 -m scripts.preprocessing.convert_data_units

    # Only some targets, to a chosen directory
    python3 -m scripts.preprocessing.convert_data_units \
        --targets metrics signals --output-dir ~/Desktop/mmol

    # Convert a mmol/L tree back to mg/dL
    python3 -m scripts.preprocessing.convert_data_units \
        --to mg/dL --source-dir <BASE_PATH>/Converted_mmoll --output-dir /tmp/back

Author
------
Guillaume Deside

Date
----
2026-09-10

Output
------
    <CGM_DATA_ROOT>/Converted_mmoll/  (mirror of the converted data files; mg/dL originals untouched)
"""

# =============================================================================
# STANDARD LIBRARIES
# =============================================================================
import argparse
import shutil
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional

# =============================================================================
# THIRD-PARTY LIBRARIES
# =============================================================================
import pandas as pd

# =============================================================================
# PROJECT IMPORTS
# =============================================================================
from src.config.paths import BASE_PATH
from src.utils.unit_conversion import (
    MGDL_PER_MMOLL,
    UNIT_MGDL,
    UNIT_MMOLL,
    convert_dataframe_units,
    convert_metric_table,
    normalize_unit,
    unit_tag,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_OUTPUT_NAME = "Converted_{tag}"
REPORT_NAME = "CONVERSION_REPORT.md"

#: Rounding applied to converted values (0.01 mmol/L ≈ 0.18 mg/dL).
DEFAULT_DECIMALS = 2


@dataclass
class Target:
    """
    One data product to convert.

    Attributes
    ----------
    name : str
        Short selector used by ``--targets``.
    pattern : str
        Glob relative to the source root.
    kind : {"wide", "long"}
        ``"wide"`` converts glucose-valued columns; ``"long"`` converts
        glucose-valued *rows* of a metric table.
    read_kwargs, write_kwargs : dict
        Extra arguments for the pandas reader / writer (separator, decimal
        mark, ...).
    metric_column : str, optional
        Name of the metric-name column for ``kind="long"``.
    """

    name: str
    pattern: str
    kind: str = "wide"
    read_kwargs: dict = field(default_factory=dict)
    write_kwargs: dict = field(default_factory=dict)
    metric_column: Optional[str] = None


TARGETS = [
    Target("windows", "Data/Preprocessed/*/*.xlsx"),
    Target("metrics", "Data/Metrics/metrics_all_patients.csv"),
    Target(
        "distribution",
        "Data/Filtered/filtered_cohort_metrics_distribution.csv",
        kind="long",
        metric_column="metric",
    ),
    Target("episodes", "Data/Episodes/*_episodes.csv"),
    Target("signals", "Results/Sensitivity_Analysis/Signals/*_signals.csv"),
    Target("sensitivity", "Results/Sensitivity_Analysis/Data/cohort_sensitivity.csv"),
    Target("opendata", "Results/Open_Data/**/*.csv"),
    Target(
        "cohort",
        "Data/Final cohort SLH.csv",
        read_kwargs={"sep": ";", "decimal": ","},
        write_kwargs={"sep": ";", "decimal": ",", "index": False},
    ),
]

TARGETS_BY_NAME = {t.name: t for t in TARGETS}


# =============================================================================
# FILE I/O
# =============================================================================
def _read(path: Path, target: Target) -> pd.DataFrame:
    """Read one data file according to its extension and target settings."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, **target.read_kwargs)

    return pd.read_csv(path, **target.read_kwargs)


def _write(df: pd.DataFrame, path: Path, target: Target) -> None:
    """Write one converted data file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(path, index=False, **target.write_kwargs)
        return

    write_kwargs = {"index": False, **target.write_kwargs}
    df.to_csv(path, **write_kwargs)


# =============================================================================
# CONVERSION
# =============================================================================
def convert_file(
    path: Path,
    target: Target,
    to_unit: str,
    decimals: Optional[int],
) -> tuple:
    """
    Convert one data file.

    Parameters
    ----------
    path : Path
        File to read.
    target : Target
        Product description (kind, reader/writer options).
    to_unit : str
        Unit to convert to.
    decimals : int or None
        Rounding applied to converted values.

    Returns
    -------
    df : pd.DataFrame
        Converted table.
    converted : list of str
        Columns (``kind="wide"``) or metrics (``kind="long"``) converted.
    """
    df = _read(path, target)

    if target.kind == "long":
        return convert_metric_table(
            df,
            metric_column=target.metric_column,
            to_unit=to_unit,
            decimals=decimals,
        )

    return convert_dataframe_units(df, to_unit=to_unit, decimals=decimals)


def run(
    source_root: Path,
    output_root: Path,
    targets: list,
    to_unit: str,
    decimals: Optional[int],
    dry_run: bool,
    in_place: bool,
    log: Callable[[str], None] = print,
) -> list:
    """
    Convert every file of every selected target.

    Parameters
    ----------
    source_root : Path
        Root the target globs are resolved against (normally ``BASE_PATH``).
    output_root : Path
        Root of the converted tree; ignored when ``in_place`` is set.
    targets : list of Target
        Products to convert.
    to_unit : str
        Unit to convert to.
    decimals : int or None
        Rounding applied to converted values.
    dry_run : bool
        Report what would happen without writing anything.
    in_place : bool
        Overwrite the source files instead of writing to ``output_root``.
    log : callable
        Where progress lines are sent.

    Returns
    -------
    list of dict
        One record per file: target, source, destination, converted names.
    """
    records = []

    for target in targets:
        paths = sorted(source_root.glob(target.pattern))

        if not paths:
            log(f"  [{target.name:12}] no file matched {target.pattern}")
            continue

        n_converted_files = 0
        columns_seen = set()

        for path in paths:
            try:
                df, converted = convert_file(path, target, to_unit, decimals)
            except Exception as exc:                      # noqa: BLE001
                log(f"  [{target.name:12}] FAILED {path.name}: {exc}")
                records.append({
                    "target": target.name,
                    "source": path,
                    "destination": None,
                    "converted": [],
                    "error": str(exc),
                })
                continue

            destination = (
                path if in_place
                else output_root / path.relative_to(source_root)
            )

            if converted:
                n_converted_files += 1
                columns_seen.update(converted)
                if not dry_run:
                    _write(df, destination, target)
            elif not dry_run and not in_place:
                # Nothing to convert: copy verbatim so the tree stays complete.
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)

            records.append({
                "target": target.name,
                "source": path,
                "destination": destination,
                "converted": converted,
                "error": None,
            })

        names = ", ".join(sorted(columns_seen)) if columns_seen else "none"
        log(
            f"  [{target.name:12}] {len(paths):4d} file(s), "
            f"{n_converted_files} converted — {names}"
        )

    return records


# =============================================================================
# REPORT
# =============================================================================
def write_report(
    records: list,
    output_root: Path,
    to_unit: str,
    from_unit: str,
    decimals: Optional[int],
) -> Path:
    """
    Write a Markdown provenance report next to the converted files.

    Parameters
    ----------
    records : list of dict
        Records returned by :func:`run`.
    output_root : Path
        Directory receiving the report.
    to_unit, from_unit : str
        Units involved in the conversion.
    decimals : int or None
        Rounding that was applied.

    Returns
    -------
    Path
        Path of the report that was written.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / REPORT_NAME

    lines = [
        f"# Glucose unit conversion: {from_unit} -> {to_unit}",
        "",
        f"Generated on {date.today().isoformat()} by "
        "`scripts/preprocessing/convert_data_units.py`.",
        "",
        f"Conversion factor: 1 mmol/L = {MGDL_PER_MMOLL} mg/dL "
        "(glucose molar mass 180.156 g/mol).",
        f"Converted values rounded to {decimals} decimals."
        if decimals is not None else "Converted values not rounded.",
        "",
        "Only columns holding a glucose **concentration** were converted.",
        "Percentages (CV, TIR/TAR/TBR), counts, durations, coverage,",
        "timestamps and HbA1c were copied unchanged.",
        "",
        "| Target | Files | Converted columns / metrics |",
        "| --- | --- | --- |",
    ]

    for name in dict.fromkeys(r["target"] for r in records):
        target_records = [r for r in records if r["target"] == name]
        converted = sorted({c for r in target_records for c in r["converted"]})
        lines.append(
            f"| `{name}` | {len(target_records)} | "
            f"{', '.join(f'`{c}`' for c in converted) if converted else '—'} |"
        )

    failures = [r for r in records if r["error"]]
    if failures:
        lines += ["", "## Failures", ""]
        lines += [f"- `{r['source'].name}`: {r['error']}" for r in failures]

    lines += [
        "",
        "## Reading these files back",
        "",
        "The analysis pipeline expects mg/dL. Do not point `BASE_PATH` at",
        "this tree; it is for sharing, publication and reporting only.",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert the project's data files between mg/dL and mmol/L.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--to", dest="to_unit", default=UNIT_MMOLL,
        help="Unit to convert to ('mmol/L' or 'mg/dL').",
    )
    parser.add_argument(
        "--targets", nargs="+", choices=sorted(TARGETS_BY_NAME), default=None,
        help="Data products to convert (default: all).",
    )
    parser.add_argument(
        "--source-dir", type=Path, default=BASE_PATH,
        help="Root the target patterns are resolved against.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Root of the converted tree "
             "(default: <source-dir>/Converted_<unit>).",
    )
    parser.add_argument(
        "--decimals", type=int, default=DEFAULT_DECIMALS,
        help="Rounding applied to converted values; -1 disables rounding.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be converted without writing anything.",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite the source files instead of writing a copy. "
             "Destroys the mg/dL originals the pipeline reads.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip the confirmation prompt required by --in-place.",
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Entry point."""
    args = parse_args(argv)

    to_unit = normalize_unit(args.to_unit)
    from_unit = UNIT_MGDL if to_unit == UNIT_MMOLL else UNIT_MMOLL
    decimals = None if args.decimals < 0 else args.decimals

    source_root = args.source_dir.expanduser()
    if not source_root.is_dir():
        print(f"Source directory not found: {source_root}")
        return 1

    tag = unit_tag(to_unit)
    output_root = (
        args.output_dir.expanduser() if args.output_dir
        else source_root / DEFAULT_OUTPUT_NAME.format(tag=tag)
    )

    targets = (
        [TARGETS_BY_NAME[n] for n in args.targets] if args.targets else TARGETS
    )

    if args.in_place and not args.dry_run and not args.force:
        print(
            f"--in-place will overwrite the {from_unit} source files under\n"
            f"  {source_root}\n"
            "The pipeline reads these files and expects mg/dL."
        )
        if input("Type 'yes' to continue: ").strip().lower() != "yes":
            print("Aborted.")
            return 1

    print(f"\nConverting {from_unit} -> {to_unit}")
    print(f"  source : {source_root}")
    print(f"  output : {'(in place)' if args.in_place else output_root}")
    if args.dry_run:
        print("  mode   : dry run, nothing will be written")
    print()

    records = run(
        source_root=source_root,
        output_root=output_root,
        targets=targets,
        to_unit=to_unit,
        decimals=decimals,
        dry_run=args.dry_run,
        in_place=args.in_place,
    )

    n_files = len(records)
    n_converted = sum(1 for r in records if r["converted"])
    n_failed = sum(1 for r in records if r["error"])

    print(
        f"\n{n_files} file(s) processed, {n_converted} with converted columns, "
        f"{n_failed} failed."
    )

    if not args.dry_run and not args.in_place and records:
        report_path = write_report(
            records, output_root, to_unit, from_unit, decimals
        )
        print(f"Report: {report_path}")

    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
