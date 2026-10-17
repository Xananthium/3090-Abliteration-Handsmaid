# REDCELL: what the stronger edit changed

The completed comparison found substantially fewer Minos-classified refusals
with the 1.5× edit, alongside one cybersecurity answer lost in both runs.

The baseline here is the original, already-abliterated pilot W6A8 checkpoint.
It is not a BF16 baseline or an unedited source-model comparison.

![Matched refusal and capability comparison](comparison.png)

| Measurement | Baseline W6A8 | 1.5× W6A8 |
|---|---:|---:|
| Refusals on 76 held-out prompts, run 1 | 27 | 12 |
| Refusals on the same prompts, run 2 | 30 | 18 |
| Mean refusal rate | 37.50% | 19.74% |
| Cybersecurity answers correct, each run | 79/80 | 78/80 |
| All capability answers correct, runs 1 / 2 | 105/112 / 105/112 | 104/112 / 105/112 |
| Benign refusals, each run | 0/64 | 0/64 |
| Tool and vision smoke checks | 7/7 | 7/7 |

The refusal rate fell by 17.76 percentage points (47.4% relative reduction).
The paired prompt-level sign test gave p=0.000659; repeated answers were not
treated as additional independent prompts. This does not remove classifier
bias or dependence between similar questions.

Across both capability runs, the score changed from 210/224 to 209/224,
a 0.45 percentage-point decline. One cybersecurity question changed from
correct twice to incorrect twice; one consistency question improved once.
This is evidence about those cases, not proof of broad degradation or broad
capability preservation.

The predeclared zero-loss rule rejected the candidate. That original result
remains unchanged. The user subsequently said one lost cybersecurity question
was not necessarily a poor tradeoff. The 1.5× edit is therefore the leading
candidate for a separate, broader accuracy comparison; this does not
retroactively change the first study's gate.

## Scope and reproduction

- Same original serving profile and complete-response evaluation protocol.
- Refusal budget: 2,048 tokens, with 4,096/6,144-token retries when needed.
- No unresolved responses in either model's final runs.
- Thinking disabled and temperature zero; observed scheduling variability
  remains, so every fixed replicate is reported.
- The edit scales the saved rank-3 correction; it is not a fresh direction search.
- These checks used BF16 KV, an 8,192-token context, and eager execution.
- Tool recovery checks use supplied local mock errors. No live Kali workflows
  or target exploitation were performed.
- Red/blue image probes are basic vision checks.
- The exact 465-prompt benchmark advertised by Hauhau was not recovered.
- No new checkpoint publication is established by this report.

Source evidence: [completed result](../replicated-result.json),
[independent regression audit](../audit/completed-replicated-sweep-case36-audit.json),
[recorded follow-up decision](../selected-candidate-review.json),
[CSV](comparison.csv).
