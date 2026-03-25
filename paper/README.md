# EvoDrive Lab Paper Workspace

This folder holds the first paper-planning materials for EvoDrive Lab.

The goal is to turn the current project into a small but credible paper centered on:

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
2. `Generalization on Procedural 2D Driving Tracks: A Comparative Study of GA, NEAT, and PPO-Style Training`
3. `Learning to Drive on Unseen Procedural Tracks: A Small-Scale Benchmark with Evolutionary and Policy-Based Methods`

## Current Thesis

The current thesis to test is:

**Different training methods may achieve similar training performance while showing meaningfully different generalization behavior on unseen procedural tracks.**

## What This Folder Contains

- `outline.md`:
  paper structure, section goals, and draft abstract
- `experiment_plan.md`:
  exact experiment matrix, evaluation metrics, and run protocol
- `results_template.md`:
  tables and figure placeholders for the first draft

## Current Boundaries

The paper should stay within the evidence the repo can support today:

- procedural 2D tracks only
- current observation and action space only
- GA, NEAT, and `numpy_ppo_lite` only
- train, validation, and test suites defined in code

If we later add stronger benchmarking infrastructure, better PPO, or more metrics, we can widen the paper scope.
