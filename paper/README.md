# EvoDrive Lab Paper Workspace

This folder now contains the paper-facing experiment and manuscript workflow for EvoDrive Lab.

The paper is framed around:

`generalization of learned driving agents on unseen procedural 2D tracks`

## Recommended Paper Direction

The strongest paper we can support with the current codebase is:

**A comparative study of GA, NEAT, and a lightweight PPO-style baseline on procedural 2D driving tracks**

That framing fits the repo honestly because the project already has:

- deterministic procedural train, validation, and test track suites
- shared evaluation metrics across algorithms
- replay export for qualitative analysis
- train and held-out test performance reporting

It also avoids overclaiming. Right now the PPO implementation is a lightweight in-repo baseline, not a heavyweight external benchmark stack, so the paper should present it as:

- a baseline comparison
- a systems-and-benchmark paper
- an empirical study of generalization trends

It should not claim state-of-the-art performance.

## Candidate Titles

1. `EvoDrive Lab: A Procedural 2D Driving Benchmark for Comparing Neuroevolution and Lightweight Policy Optimization`
2. `EvoDrive Lab: A Reproducible Procedural 2D Driving Benchmark for Neuroevolution and Lightweight Policy Optimization`
3. `Generalization on Procedural 2D Driving Tracks: A Comparative Study of GA, NEAT, and PPO-Style Training`
4. `Learning to Drive on Unseen Procedural Tracks: A Small-Scale Benchmark with Evolutionary and Policy-Based Methods`

## Current Thesis

The current thesis to test is:

**Different training methods may achieve similar training performance while showing meaningfully different generalization behavior on unseen procedural tracks.**

## Paper Workflow

Start the Docker stack first:

```bash
docker compose up -d --build
```

Queue the canonical paper matrices:

```bash
make paper-status
make paper-queue-main
make paper-queue-sensors
make paper-queue-tracks
```

Or queue everything at once:

```bash
make paper-status
make paper-queue-full
```

Aggregate the current registry into paper outputs:

```bash
docker compose exec api python -m app.paper_tools.report aggregate-all
```

This writes:

- `paper/results/paper_run_manifest.csv`
- `paper/results/paper_status.md`
- `paper/results/paper_main_runs.csv`
- `paper/results/paper_main_summary.csv`
- `paper/results/paper_ablation_sensors_summary.csv`
- `paper/results/paper_ablation_tracks_summary.csv`
- `paper/results/paper_qualitative_cases.csv`

Build paper figures after the matrices are complete:

```bash
docker compose exec api python -m app.paper_tools.plots build-all
```

The plotting command is intentionally strict. It fails loudly if the benchmark matrix is incomplete, which helps prevent accidental use of partial results in the manuscript.

Build the manuscript locally if a LaTeX toolchain is installed:

```bash
python -m app.paper_tools.latex
```

## What This Folder Contains

- `configs/`
  frozen experiment matrices for the main comparison and the two quick ablations
- `outline.md`
  paper structure, section goals, and draft abstract
- `experiment_plan.md`
  exact experiment matrix, evaluation metrics, and run protocol
- `results_template.md`
  tables and figure placeholders for the first draft
- `arxiv_checklist.md`
  submission-readiness notes for authorship, licensing, and disclosure
- `latex/`
  the first complete LaTeX manuscript scaffold
- `results/`
  generated CSV summaries, markdown tables, and paper figures

## Current Boundaries

The paper should stay within the evidence the repo can support today:

- procedural 2D tracks only
- current observation and action space only
- GA, NEAT, and `numpy_ppo_lite` only
- train, validation, and test suites defined in code

If we later add stronger benchmarking infrastructure, better PPO, or more metrics, we can widen the paper scope.
