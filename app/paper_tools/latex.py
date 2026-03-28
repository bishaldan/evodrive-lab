from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from app.paper_tools.common import PAPER_LATEX_DIR


def _build_command() -> list[str] | None:
    latexmk = shutil.which("latexmk")
    if latexmk:
        return [latexmk, "-pdf", "-interaction=nonstopmode", "main.tex"]
    tectonic = shutil.which("tectonic")
    if tectonic:
        return [tectonic, "main.tex"]
    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        return [pdflatex, "-interaction=nonstopmode", "main.tex"]
    return None


def build_latex(paper_dir: Path) -> int:
    command = _build_command()
    if command is None:
        print("No LaTeX toolchain found. Skipping build.")
        return 0
    first = subprocess.run(command, cwd=paper_dir, check=False)
    if "latexmk" not in command[0] and first.returncode == 0:
        second = subprocess.run(command, cwd=paper_dir, check=False)
        return second.returncode
    return first.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paper LaTeX draft if a TeX toolchain is available.")
    parser.add_argument("--paper-dir", default=str(PAPER_LATEX_DIR))
    args = parser.parse_args()
    raise SystemExit(build_latex(Path(args.paper_dir)))


if __name__ == "__main__":  # pragma: no cover
    main()
