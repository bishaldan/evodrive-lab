# LaTeX Draft

This directory contains the first arXiv-ready manuscript draft for EvoDrive Lab.

## Build

If a local LaTeX toolchain is installed:

```bash
python -m app.paper_tools.latex
```

If no LaTeX toolchain is installed, the build helper exits cleanly and prints a skip message.

## Notes

- The manuscript references generated figures under `../results/figures/`.
- The draft is written to stay honest about the benchmark scope and the lightweight PPO baseline.
- Final arXiv submission should include only the files needed for the manuscript, bibliography, and figures.
