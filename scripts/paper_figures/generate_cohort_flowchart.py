"""
python3 -m scripts.paper_figures.generate_cohort_flowchart
============================================================

Cohort selection flow diagram (CONSORT-style), from the pediatric diabetes
convention roster at CUSL down to the analysis cohort.

The figure is hand-authored vector artwork, kept as
``scripts/paper_figures/assets/cohort_flowchart.svg`` (edit counts there).
This script publishes it to ``Figures/Supplementary/cohort_flowchart.svg`` and
renders a PDF copy with ``rsvg-convert`` (brew install librsvg).

Author: Guillaume Deside
Date: 2026-09-29
"""

import shutil
import subprocess
from pathlib import Path

from src.config import FIG_DIR

SOURCE_SVG = Path(__file__).parent / "assets" / "cohort_flowchart.svg"


def main() -> None:
    output_dir = Path(FIG_DIR) / "Supplementary"
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "cohort_flowchart.svg"
    pdf_path = output_dir / "cohort_flowchart.pdf"

    shutil.copyfile(SOURCE_SVG, svg_path)

    if shutil.which("rsvg-convert") is None:
        print("rsvg-convert not found: SVG written, PDF skipped.")
    else:
        subprocess.run(
            ["rsvg-convert", "-f", "pdf", "-o", str(pdf_path), str(SOURCE_SVG)],
            check=True,
        )

    print("Cohort flow diagram saved:")
    print(f"  Location: {svg_path}")


if __name__ == "__main__":
    main()
