# History and model lineage

This chapter records the REDCELL experiments of October 14–16, 2026. A result belongs to a particular checkpoint, precision, runtime, prompt set, and scoring method. The word **baseline** changed meaning between studies; those references must stay separate.

| Reference used in the research | What it actually was |
|---|---|
| Reconstructed BF16 source | REDCELL's upstream F16 GGUF language tensors converted to BF16, with verified official Gemma vision tensors restored; no local abliteration merged |
| Initial NF4 baseline | A four-bit analysis view of that reconstructed source, used for the first small Heretic pilot |
| Pilot W6A8 reference | The first locally edited and quantized checkpoint; already edited before the October 15 comparison |
| Native W8A16 baseline | A later, single conversion of the reconstructed source to INT8 weights with BF16 activations, with no local adapter merged |

“No local edit” does not mean original Google pretrained weights. REDCELL was already an upstream specialized model. None of these names establishes that its language tensors are an untouched Google checkpoint.

## Where the source came from

The retained source receipt identifies `terrorswift/REDCELL-26B-A4B-OSINT-Cyber-APEX-GGUF`, revision `ba311d958b4ed191a7291b38a6d805662415a570`, and `REDCELL-26B-A4B-OSINT-Cyber-f16.gguf`. Its independently verified SHA-256 was `26cc25c909a26f8ed08a78444295ac174e7f329f8fb39ebae98f42d4115a7011`. The receipt records a reproduced conversion that matched the prior conversion byte for byte. [Source receipt](../evidence/related/full-round-20261006/redcell-source-receipt.json)

The reconstructed source contained 1,013 canonical tensors and approximately 48.07 GiB of tensor payload. Its language tensors underwent **F16-to-BF16 rounding**, which is not lossless. The vision tensors came from the previously verified official `google/gemma-4-26B-A4B-it` BF16 source. Calling the reconstruction “BF16” describes its stored dtype; it does not recover precision discarded before or during conversion. [Retained source provenance](../evidence/related/redcell-abliteration-20261014/export-receipt.json)

## October 14: the first small pilot

The first experiment used Heretic commit `66a952b49b00bfd1854bdad769fee5fb225b53be`, with local Gemma expert compatibility work. It was an eight-trial, low-rank pilot. Language linear layers were viewed in NF4; router, embeddings, and vision remained BF16. The response allowance was only 48 tokens, and refusal estimates used keyword matching. These are construction and measurement facts, not a general recipe or a claim that the same result holds at another precision. [Experiment plan](../evidence/related/redcell-abliteration-20261014/experiment-plan.json), [pilot settings](../evidence/related/redcell-abliteration-20261014/pilot-settings.json)

| Initial NF4 pilot measurement | Source reference | Selected candidate |
|---|---:|---:|
| Capability answers correct | 19/20 | 19/20 |
| Held-out responses flagged by refusal keywords | 16/16 | 3/16 |
| Optimization responses flagged | — | 2/12 |

The selected trial's recorded first-token KL was 0.04427, satisfying its configured constraint. That statistic and the 20-question score are narrow checks: neither proves preserved reasoning, reliable tools, or unchanged behavior over long answers. The pilot itself records keyword false-positive/false-negative risk and leaves final-precision validation open. [Pilot summary](../evidence/related/redcell-abliteration-20261014/pilot-summary.json)

Two exports followed. The W6A8 artifact stored packed INT6 group-32 weights with dynamic per-token INT8 activations and retained BF16 exclusions, including vision. Its receipt records 3,900 edited projections, 11,725 quantized linears, and 22,538,035,788 bytes. A separate merged BF16 artifact used FP32 merging followed by BF16 rounding. Its receipt verified tensor coverage, finite values, serialization, and preserved excluded tensors, but explicitly states that **the merged BF16 artifact did not receive a generation benchmark**. Attached NF4 or W6A8 results cannot be relabeled as BF16 results. [W6A8 export receipt](../evidence/related/redcell-abliteration-20261014/export-receipt.json), [BF16 merge receipt](../evidence/related/redcell-abliteration-20261014/bf16-verified-support/merge-receipt.json)

## October 15: longer answers changed the measurement

The next investigation recovered 98 unique markers from Reaper 2.5.2 and 90 from Reaper 2.6.0, alongside a separate Heretic comparator. The exact 465-prompt benchmark advertised by Hauhau was not recovered. Marker matching and the pinned Minos classifier remained different measurements; neither is a human-adjudicated ground truth for refusal, usefulness, or correctness. Prompt contents are intentionally omitted from this archive chapter. [Marker-source provenance](../evidence/early/marker-source-provenance.json)

An initial 300-token diagnostic truncated all 76 new held-out responses, all 16 original held-out responses, all 12 optimization responses, and 49/64 benign responses. Its refusal counts should not be treated as a complete-response estimate. The replacement protocol allowed 2,048 tokens, with 4,096/6,144-token retries when needed. The replicated comparison finished without unresolved final responses. [Short-budget summary](../evidence/early/pilot-w6a8-300-summary.json), [long-response summary](../evidence/early/pilot-w6a8-long-v2-summary.json)

The reference below is the **already edited pilot W6A8**, not the reconstructed source or BF16. The historical candidate label `1.5×` identifies a saved edit variant; it was not a new direction search.

| Matched complete-response measurement | Pilot W6A8 | Candidate W6A8, historical label `1.5×` |
|---|---:|---:|
| Minos-classified refusals, 76 prompts, repeat 1 | 27/76 | 12/76 |
| Same prompts, repeat 2 | 30/76 | 18/76 |
| Combined classified observations | 57/152 (37.50%) | 30/152 (19.74%) |
| Cybersecurity knowledge, two repeats | 158/160 | 156/160 |
| All capability checks, two repeats | 210/224 | 209/224 |
| Benign refusal labels, each repeat | 0/64 | 0/64 |
| Basic tool/vision smoke checks | 7/7 | 7/7 |

The classified rate fell by 17.76 percentage points. One cybersecurity case changed from correct in both repeats to incorrect in both; a consistency case improved once. The predeclared zero-loss rule rejected the candidate. Later willingness to tolerate a small loss led to a separate study; it did not rewrite that earlier decision. Repeated responses are observations of the same prompts, not 152 independent refusal questions. [Completed study report and scope](../evidence/early/study-report-v1/README.md), [comparison CSV](../evidence/early/study-report-v1/comparison.csv)

## Broader capability follow-up

Three existing exports were tested twice on the same 224-question suite: 200 curated cybersecurity knowledge questions plus six each for reasoning, instruction following, consistency, and recovery. It was a local study derived from pinned CyberMetric sources, **not an official CyberMetric score**. Exact-format grading also means an authored-case failure can reflect output compliance rather than lack of underlying knowledge. [Broader study report](../evidence/early/broader-validation/report-v1/README.md)

| Checkpoint | Correct, repeat 1 | Correct, repeat 2 | Combined observations | Cybersecurity observations | Other authored observations |
|---|---:|---:|---:|---:|---:|
| Pilot W6A8 | 209/224 | 207/224 | 416/448 (92.86%) | 383/400 | 33/48 |
| Candidate `1.5×` W8A8 | 206/224 | 208/224 | 414/448 (92.41%) | 382/400 | 32/48 |
| Candidate `1.5×` W6A8 | 205/224 | 207/224 | 412/448 (91.96%) | 383/400 | 29/48 |

The cybersecurity majority dominates the aggregate. On the six reasoning cases repeated twice, the scores were only 3/12, 2/12, and 2/12 respectively. All three obtained 12/12 on the small recovery set. These scores do not establish reliable open-ended assessment workflows. W8A8 additionally passed 7/7 basic native Anthropic tool/vision checks, including a supplied error-recovery fixture and simple color images. [Category counts](../evidence/early/broader-validation/report-v1/accuracy.csv)

The runs used temperature zero, thinking disabled, BF16 KV, an 8,192-token context, and eager execution. All scored final responses completed. Neither the small score difference between exports nor successful smokes establishes a general precision winner, long-context quality, or merged BF16 behavior.

The repeated-study and broader-suite counts are also preserved in [historical metrics CSV](../data/historical-metrics.csv), with unique-case counts, repeat counts, denominators, and source references kept separate.

## BF16 practicality and the later W8A16 reference

A separate native BF16 effort established that the reconstructed source could generate and that reversible adapter routing could run. The capability comparison was stopped at the user's request after **32 source observations**; its status explicitly records `complete_comparison: false`. There is no completed BF16 candidate comparison to infer from that partial run. Runtime limitations and the slow observed offload sample are detailed in [runtime and precision](04-runtime-and-precision.md). [BF16 stop receipt](../evidence/bf16/integrity-batch-status-v1.json)

The next working baseline was constructed once from the reconstructed source as native compressed-tensors **INT8 W8A16**, with BF16 activations and no adapter merged. The completed export recorded zero edited projections, 17 weight files totaling 28,663,631,516 bytes, 356 unchanged vision tensors, and 15 independent numeric checks. Its export receipt initially marked runtime validation pending; later native routing and development evaluations are separate evidence. [W8A16 completion receipt](../evidence/w8a16/export-completion-summary.json)

The subsequent five-branch comparison used this new reference, 24 authored cases repeated twice per thinking mode, and seven tool/vision smokes per branch and mode. It must not be pooled with the older W6A8 or cybersecurity-heavy suites. A later branch rebuilt factors against W8 weights while retaining an NF4-origin direction; that is not a fresh W8-activation-derived or BF16 direction search. See [measured results](02-measured-results.md) for the later scores and token costs. [Independent final audit](../evidence/w8a16/final-five-branch-independent-audit-v1.json), [factor-construction receipt](../evidence/w8a16/w8-weight-rebase-completion-v1.json)

## What the publication receipts establish

These are historical verified revisions, not a fresh check of current Hub contents.

| Artifact | Recorded public repository | Verified revision |
|---|---|---|
| Original pilot, merged BF16 with vision | `Xananthium/REDCELL-26B-A4B-OSINT-Cyber-Abliterated-Pilot-BF16-Vision` | `91af35c07db9ef6b570673405a89411bcb41f951` |
| Original pilot, W6A8 with vision | `Xananthium/REDCELL-26B-A4B-OSINT-Cyber-Abliterated-Pilot-W6A8-Vision` | `d27d2f66f97900539652a726a94537b24bb65caa` |
| Upstream REDCELL conversion, W6A8 with vision, without the local pilot edit | `Xananthium/REDCELL-26B-A4B-OSINT-Cyber-W6A8-Vision` | `224d29d960498e9202d7c5462f5e298556df3cfd` |

The BF16 upload receipt compares pinned Hub weight hashes and sizes against the release evidence and verifies downloaded support files. Publication receipts also preserve the pilot model cards. Later cleanup verified uploaded copies before removing local checkpoints and retained support metadata. The October 15 audit identified later strength variants as unpublished at that time; the receipts examined here do not establish their subsequent publication. [BF16 upload verification](../evidence/related/redcell-abliteration-20261014/bf16-upload-verification.json), [model-card publication receipt](../evidence/related/redcell-abliteration-20261014/model-card-publication.json), [uploaded-model cleanup audit](../evidence/bf16/uploaded-model-cleanup-summary.json)

The enduring gaps are explicit: no original upstream REDCELL BF16 language checkpoint was recovered; no complete BF16 candidate comparison exists in this phase; the exact advertised Hauhau refusal benchmark was unavailable; and local marker/classifier outcomes do not establish universal refusal behavior. Historical checkpoint paths in receipts can refer to files later removed after backup verification.
