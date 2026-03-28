from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from app.paper_tools.common import PAPER_LATEX_DIR


def _docker_command(paper_dir: Path) -> list[str] | None:
    docker = shutil.which("docker")
    if not docker:
        return None
    return [
        docker,
        "run",
        "--rm",
        "-v",
        f"{paper_dir.resolve()}:/work",
        "-w",
        "/work",
        "texlive/texlive:latest",
        "sh",
        "-lc",
        "pdflatex -interaction=nonstopmode main.tex && bibtex main && "
        "pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex",
    ]


def _build_commands(paper_dir: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    latexmk = shutil.which("latexmk")
    if latexmk:
        commands.append([latexmk, "-pdf", "-interaction=nonstopmode", "main.tex"])
    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        commands.append([pdflatex, "-interaction=nonstopmode", "main.tex"])
    docker_command = _docker_command(paper_dir)
    if docker_command:
        commands.append(docker_command)
    tectonic = shutil.which("tectonic")
    if tectonic:
        # Cached mode avoids the macOS-specific reqwest/system-configuration panic path
        # when a local cache already exists.
        commands.append([tectonic, "--only-cached", "--keep-logs", "--keep-intermediates", "main.tex"])
        commands.append([tectonic, "main.tex"])
    return commands


def build_latex(paper_dir: Path) -> int:
    commands = _build_commands(paper_dir)
    if not commands:
        print("No LaTeX toolchain found. Skipping build.")
        return 0
    last_code = 1
    for command in commands:
        print(f"Trying LaTeX build command: {' '.join(command)}")
        first = subprocess.run(command, cwd=paper_dir, check=False)
        if first.returncode == 0:
            if command[0].endswith("latexmk") or (len(command) > 2 and command[2] == "--rm"):
                return 0
            second = subprocess.run(command, cwd=paper_dir, check=False)
            return second.returncode
        last_code = first.returncode
    return last_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paper LaTeX draft if a TeX toolchain is available.")
    parser.add_argument("--paper-dir", default=str(PAPER_LATEX_DIR))
    args = parser.parse_args()
    raise SystemExit(build_latex(Path(args.paper_dir)))


if __name__ == "__main__":  # pragma: no cover
    main()
