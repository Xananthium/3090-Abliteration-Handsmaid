# REDCELL: the broader accuracy check

Three existing checkpoints were tested twice on the same frozen 224-question suite: 200 cybersecurity knowledge questions and six each for reasoning, instructions, consistency, and recovery.

The reference checkpoint is the original, already-abliterated pilot W6A8. These runs compare stronger edits and quantized exports; they include no BF16 weight evaluation.

![Models ranked by the accuracy measured in each panel](accuracy-ranked.png)

| Model | All correct: run 1 / run 2 | Mean accuracy | Cyber correct: run 1 / run 2 | Other checks: run 1 / run 2 |
|---|---:|---:|---:|---:|
| Pilot W6A8 | 209/224 / 207/224 | 92.86% | 192/200 / 191/200 | 17/24 / 16/24 |
| 1.5× W8A8 | 206/224 / 208/224 | 92.41% | 191/200 / 191/200 | 15/24 / 17/24 |
| 1.5× W6A8 | 205/224 / 207/224 | 91.96% | 191/200 / 192/200 | 14/24 / 15/24 |

## Category results

Counts below combine two runs on the same questions; the runs are not independent samples. Cybersecurity supplies 200 of 224 questions, so the total score can obscure changes in the smaller authored categories.

| Category | Questions | Pilot W6A8 | 1.5× W8A8 | 1.5× W6A8 |
|---|---:|---:|---:|---:|
| consistency | 6 | 8/12 | 8/12 | 8/12 |
| cybersecurity | 200 | 383/400 | 382/400 | 383/400 |
| instruction | 6 | 10/12 | 10/12 | 7/12 |
| reasoning | 6 | 3/12 | 2/12 | 2/12 |
| recovery | 6 | 12/12 | 12/12 | 12/12 |

W8A8 passed 7/7 basic native Anthropic tool and vision checks. These include a supplied PDF-error recovery fixture and simple color-image probes.

## What this establishes

The comparison reports useful capability at a fixed edit strength. It does not alter the decision recorded by the earlier study. Per-question gains and losses are retained in the comparison JSON.

- The cybersecurity set was curated from pinned CyberMetric sources; this is not an official CyberMetric score.
- Exact and lexical near duplicates were excluded before inference; shared topics and some recovery mechanisms remain.
- Published cybersecurity answer keys were retained after bounded ambiguity review; this does not certify every key.
- Authored capability checks score the requested exact text or JSON structure; a failure can reflect output-format compliance as well as reasoning.
- Temperature was zero and thinking was disabled. Both scheduled runs are reported despite output variability.
- The serving profile used BF16 KV, 8,192-token context, and eager execution.
- All scored responses completed within the declared budgets; unresolved outputs are checked separately.
- Private test results do not establish deployment or publication.

Evidence: [manifest](../manifest-v1.json), [frozen plan](../precision-comparison-plan-v1.json), [completed comparison](../precision-comparison-result-v1.json), [CSV](accuracy.csv).
