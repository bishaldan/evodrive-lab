# Paper Outline

## Working Title

`EvoDrive Lab: A Procedural 2D Driving Benchmark for Comparing Neuroevolution and Lightweight Policy Optimization`

## Draft Abstract

We present EvoDrive Lab, a Docker-first Python benchmark for studying learning and generalization in procedural 2D driving tasks. The benchmark couples deterministic track generation, shared observation and action interfaces, and unified evaluation metrics with three training approaches: a custom genetic algorithm, NEAT, and a lightweight PPO-style baseline. Our main research question is whether methods that learn effective driving behavior on training tracks generalize equally well to unseen procedural tracks. We evaluate each method on shared train, validation, and held-out test suites using completion, reward, crash rate, and episode length metrics, and pair quantitative results with replay-based qualitative analysis. Rather than aiming for state-of-the-art results, EvoDrive Lab provides a compact, reproducible environment for comparing learning dynamics and generalization behavior across algorithm families.

## 1. Introduction

Goal:
- motivate why visible, reproducible driving benchmarks are useful
- explain the difference between training success and generalization success
- introduce EvoDrive Lab as both a simulator and a benchmark

Key points:
- many demos only show one trained agent
- procedural tracks let us test generalization, not only memorization
- lightweight, reproducible benchmarks are valuable for small labs and independent researchers

## 2. Benchmark Setup

Goal:
- describe the environment clearly enough to reproduce it

Include:
- procedural 2D track generation
- observation space
- action space
- reward design
- train, validation, and test suites
- episode termination conditions

## 3. Compared Methods

Goal:
- explain the algorithm families at the level needed for fair comparison

Subsections:
- Genetic Algorithm
- NEAT
- PPO-Style Baseline

Be explicit that:
- GA is a custom implementation
- NEAT is integrated through `neat-python`
- PPO is an in-repo lightweight baseline, not a large external training stack

## 4. Experimental Protocol

Goal:
- make the study reproducible and defensible

Include:
- fixed seeds for repeated runs
- number of runs per algorithm
- how checkpoints are selected
- which metrics are reported on validation and test
- how figures and tables are generated

## 5. Results

Goal:
- answer the central question directly

Core result categories:
- train performance
- validation performance
- unseen test performance
- variance across seeds
- qualitative replay differences

## 6. Discussion

Goal:
- interpret what the results mean without overselling

Discuss:
- methods that learn quickly may still generalize poorly
- diversity-based approaches may show different failure modes
- lightweight RL baselines can still be informative even if not state of the art

## 7. Limitations

Must include:
- simplified 2D setting
- small benchmark scale
- lightweight PPO baseline
- no claim of real-world driving relevance
- current metrics are limited to the benchmark’s supported outputs

## 8. Conclusion

Goal:
- summarize what EvoDrive Lab contributes
- describe how the benchmark can grow

## Figures To Target

1. Hero figure showing the environment and multi-car replay
2. Train vs test completion by algorithm
3. Crash rate by algorithm
4. Mean reward by algorithm
5. Example replay comparison on held-out tracks

## Tables To Target

1. Benchmark setup summary
2. Main aggregate results on test tracks
3. Variance across seeds
