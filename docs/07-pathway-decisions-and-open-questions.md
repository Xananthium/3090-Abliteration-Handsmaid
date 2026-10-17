# Pathways tried and decisions still open

The practical result is a usable W8A16 reference, completed comparisons of four edited branches, a bounded investigation of Hauhau's published Gemma weights, and a better evaluation harness. No tested branch established the full desired combination of near-zero refusals, broad quality retention, and acceptable generation cost.

## What worked and what did not establish a winner

| Pathway | Evidence reached | What worked | Limitation or outcome |
|---|---|---|---|
| Earlier Heretic-derived pilot and W6A8 comparisons | Completed repeated detector/capability evaluations | A stronger candidate reduced classifier-estimated refusals on that study's prompts | A closed-answer cybersecurity loss and other tradeoffs remained; its reference was an already-edited pilot, not the later unchanged W8A16 baseline |
| Direct BF16 iteration | Loading, smoke checks, partial source evaluation | Source and adapter feasibility were investigated | The comparison was operationally slow and incomplete; no completed BF16 candidate ranking followed |
| One unchanged W8A16 base with reversible native LoRA branches | Completed five-branch development comparison | Avoided repeatedly storing a full checkpoint; preserved a matched reference and separate branches | Unmerged runtime costs differ from a final merged/requantized deployment |
| Reuse saved NF4-derived directions on W8A16 | Completed three original branch comparisons | Thinking-enabled aggregate accuracy matched the baseline on the small suite | Generated-token overhead ranged from about 53% to 100%; thinking-disabled scores regressed |
| Recompute weight-dependent factors from W8 weights | Completed fourth edited branch comparison | Reached 47/48 thinking-enabled versus 46/48 baseline | About 102% more output tokens for one schema-case gain; still used NF4-derived directions, and construction changed in other ways |
| Recover numerical changes from Hauhau Gemma | Completed bounded F16 and matched-Q8 investigations | Found repeatable low-dimensional weight-edit structure in sampled late tensors with useful controls | No demonstrated causal refusal direction, REDCELL transfer, or behavior-preserving donor edit |
| Reuse original REDCELL specialist adapter or corpus | Historical availability audit | Clarified lineage and what would be needed for an exact reproduction | Original adapter, corpus, and exact training-base revision were not found in the checked public sources |
| Native W8 hidden-state capture | Installed-source audit and memory arithmetic | Identified a possible measurement route | No capture run, behavioral equivalence test, or validated runtime fit was performed |
| Improve and test authorized workflows without a new edit | Completed 16 synthetic observations | Unchanged model engaged, changed approach, and reported evidence-backed outcomes | Guided mock tools and a planted flag; real CTF performance remains unknown |

The details and denominators live in [history](01-history-and-model-lineage.md), [measured results](02-measured-results.md), [Hauhau findings](03-hauhau-gemma-investigation.md), [runtime](04-runtime-and-precision.md), and [workflow results](06-authorized-workflow-baseline.md).

## Iteration does not require making every candidate the next parent

In the later comparison, all candidates were independently attached to the same unchanged W8A16 base. The experiments accumulated knowledge rather than silently compounding changes into an increasingly hard-to-explain source model.

That distinction answers an earlier question: an iteration can improve the next hypothesis, measurement, or construction while keeping the reference fixed. Promoting a candidate to become the next source is a separate decision. None of the later branches earned that promotion from the completed evidence.

This also keeps provenance legible. “Recomputed from W8 weights” does not mean “fresh directions measured on W8 activations.” “The donor and target share ancestry” does not mean their behavior or internal representations remain interchangeable after specialist fine-tuning.

## The quality target remains a target

The preferred goal was approximately two or three refusals with 98% quality retention, with a fallback around five or six refusals and 94% retention. Those requests were aspirations. They were not converted into an independently validated, universal quality measure.

The later five-branch study did not run a refusal evaluation. Its 47/48 high score is a small development-set score. The earlier classifier counts came from different prompts and a different baseline. Combining those figures into a claim of “near-zero refusals and 98% quality” would manufacture a result that no experiment produced.

## Why the direction of the work changed

There were two separate concerns. First, blanket refusal removal treats all refusal decisions as the same defect; pursuing a zero count regardless of the request goes beyond improving reliability for legitimate work. The work continued as evaluation of CTFs and explicitly authorized security tasks, rather than further blanket refusal-removal optimization.

Second, the measurements exposed substantial efficiency and quality tradeoffs. Some branches needed nearly twice the output for the same aggregate score. Some introduced exact-format or reasoning failures. These technical findings matter independently of the change in scope.

The archive does not attribute malicious intent to the operator, claim a hidden conscious realization by the assistant, or turn a performance result into an ethical proof. The README's fictional narrator dramatizes the documented change; this section states it directly.

## Questions that would be worth answering next

- Can the unchanged model complete held-out local CTFs with real tools, including recovery from unavailable tools and misleading leads?
- Do tool schemas, history handling, or task instructions explain practical failures before any additional weight edit is considered?
- How do token work and wall time change under a controlled deployment comparison with matched settings and no overlapping background work?
- Which observed failures reproduce on fresh, unambiguous capability questions?
- Does any donor-derived hypothesis preserve target capability when evaluated independently? The current weight study alone cannot answer that.
- Can native W8 measurement run without changing the model behavior being measured? Source inspection alone cannot answer that either.

These are unresolved research questions, not completed experiments or queued background jobs. No new model was trained, converted, or deployed while assembling this archive.
