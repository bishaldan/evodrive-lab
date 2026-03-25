# Results Template

## Table 1. Benchmark Setup

| Item | Value |
| --- | --- |
| Environment | Procedural 2D top-down driving |
| Train seeds | 11, 17, 23, 29 |
| Validation seeds | 101, 109 |
| Test seeds | 211, 223, 227 |
| Observation | Ray sensors + motion state |
| Action | Steering + throttle |
| Algorithms | GA, NEAT, PPO-style baseline |

## Table 2. Main Test Results

| Algorithm | Completion Mean | Completion Std | Reward Mean | Reward Std | Crash Rate Mean | Crash Rate Std | Steps Mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GA | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NEAT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Table 3. Validation Results

| Algorithm | Completion Mean | Reward Mean | Crash Rate Mean | Steps Mean |
| --- | --- | --- | --- | --- |
| GA | TBD | TBD | TBD | TBD |
| NEAT | TBD | TBD | TBD | TBD |
| PPO | TBD | TBD | TBD | TBD |

## Figure Checklist

- Figure 1:
  environment overview and hero replay
- Figure 2:
  completion on validation vs test
- Figure 3:
  crash rate by algorithm
- Figure 4:
  reward by algorithm
- Figure 5:
  qualitative replay comparison

## Narrative Prompts

Use these prompts when writing the results section:

- Which algorithm had the highest held-out completion?
- Which algorithm had the lowest crash rate?
- Which algorithm was most stable across random seeds?
- Did the ranking on training resemble the ranking on test?
- Which qualitative failure modes appeared most often?
