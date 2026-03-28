# arXiv Submission Notes

## Recommended Title

EvoDrive Lab: A Reproducible Procedural 2D Driving Benchmark for Neuroevolution and Lightweight Policy Optimization

## Recommended Author

Bishal Mahatchhetri

## Recommended Primary Category

`cs.LG`

## Optional Cross-list

`cs.RO`

Use the cross-list only if you want the paper surfaced to robotics readers as well. The safer default is to submit under `cs.LG` only.

## Suggested Comments

`Small-scale procedural driving benchmark with 5 figures and open-source code: https://github.com/bishaldan/evodrive-lab`

## Recommended Abstract

EvoDrive Lab is a Docker-first benchmark for studying learning and generalization in procedural 2D driving tasks. The benchmark combines deterministic track generation, a shared observation and action interface, and unified evaluation metrics with three training approaches: a custom genetic algorithm, NEAT, and a lightweight PPO-style baseline implemented directly in the repository. The central question is whether methods that achieve strong behavior on optimization-adjacent tracks generalize similarly to unseen procedural tracks. Across five benchmark seeds, NEAT achieved the strongest held-out test completion in the main comparison (0.879), closely followed by the custom genetic algorithm (0.866), while the lightweight PPO baseline failed to solve the task reliably (0.002). A seven-ray sensor configuration improved the genetic algorithm but did not improve NEAT, and longer procedural tracks substantially reduced held-out completion for both evolutionary methods. We position EvoDrive Lab as a small-scale, reproducible benchmark and open-source research artifact rather than a realistic autonomous-driving system, and we focus on completion, reward, crash rate, and episode length as the primary comparison metrics.

## Final Reminder

Before submitting:

- verify the arXiv compile preview from the source tarball
- confirm the listed author name and any affiliation text
- choose the submission license intentionally
- make sure the uploaded source matches the exact repo revision you want to cite
