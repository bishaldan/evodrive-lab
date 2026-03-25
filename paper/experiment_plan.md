# Experiment Plan

## Main Research Question

**Which training method generalizes best to unseen procedural 2D driving tracks in EvoDrive Lab?**

## Hypothesis

A method can perform well on training tracks while still showing weaker generalization on held-out tracks. We expect the algorithms to differ not only in mean completion and reward, but also in crash rate and stability across seeds.

## Benchmark Definition

Current suites defined in code:

- train:
  `11, 17, 23, 29`
- validation:
  `101, 109`
- test:
  `211, 223, 227`

Current environment metrics available from the codebase:

- `mean_reward`
- `mean_completion`
- `crash_rate`
- `mean_steps`

These should be the primary reported metrics in the first paper draft.

## Algorithms To Compare

1. `GA`
   Custom neuroevolution baseline and the main visible training workflow.

2. `NEAT`
   Topology-evolving baseline via `neat-python`.

3. `PPO`
   Lightweight in-repo `numpy_ppo_lite` baseline.

## Honest Framing

The paper should explicitly state:

- PPO is a lightweight baseline, not a large-scale production RL benchmark
- the comparison is useful for trend analysis and reproducibility
- the goal is benchmark clarity, not state-of-the-art claims

## Experiment Matrix

Recommended first matrix:

- algorithms:
  `GA`, `NEAT`, `PPO`
- seeds:
  `5 runs per algorithm`
- total:
  `15 full runs`

Suggested seeds:

- `7`
- `11`
- `23`
- `42`
- `101`

## Default Run Budgets

These should be fixed before collecting results.

Proposed first-pass budgets:

- `GA`
  `total_iterations = 12`
- `NEAT`
  `total_iterations = 12`
- `PPO`
  `total_timesteps = 6000` to `10000`

These numbers are not final research claims. They are starting points for pilot experiments.

## Primary Reported Metrics

For each algorithm, report:

- mean and standard deviation of test `mean_completion`
- mean and standard deviation of test `mean_reward`
- mean and standard deviation of test `crash_rate`
- mean and standard deviation of test `mean_steps`

## Secondary Analysis

Add qualitative analysis with:

- representative replay from the best seed
- representative replay from the median seed
- one failure case on a held-out track

## Selection Policy

To avoid cherry-picking:

- use validation metrics to choose the representative checkpoint
- report final test metrics only after the run protocol is fixed

## Threats To Validity

- small number of train and test seeds
- limited environment complexity
- PPO baseline is intentionally lightweight
- environment reward design may bias algorithm behavior

## First Experiment Deliverables

1. Raw run outputs in `runs/`
2. CSV summaries in `reports/`
3. one aggregate comparison table
4. one train-vs-test completion plot
5. one crash-rate comparison plot
6. one qualitative replay figure

## Immediate Next Engineering Tasks For The Paper

1. add a script that runs the 5-seed experiment matrix automatically
2. add a script that aggregates run summaries into one CSV
3. add a script that produces paper-ready plots
4. freeze the experiment config before collecting final numbers
