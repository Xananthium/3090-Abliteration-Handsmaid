# Evaluation lessons and mistakes worth preserving

The most useful outcome of this work is a better distinction between willingness, correctness, tool reliability, and the cost of reaching an answer. A change can improve one and damage another. A single refusal counter cannot measure all four.

## The marker list was not the benchmark

The archived Reaper packages yielded **98 unique markers in version 2.5.2** and **90 in version 2.6.0**. The separate 2.5.2 lecture list contained 22 entries and its refusal-pattern list contained 30. These are different detector inputs, not an interchangeable list of “a hundred and something” evaluation questions. The exact Hauhau 465-prompt benchmark was not recovered.

The earlier evaluation reported substring matches and the pinned NousResearch/Minos-v1 classifier separately. Neither establishes whether a response is accurate, useful, safe, or complete. A phrase can appear in a quotation or explanation; a refusal can avoid the expected phrases. A lower marker count is evidence about that detector on that corpus, not proof of a universally more capable model.

Sources: [marker provenance](../evidence/early/marker-source-provenance.json), [classifier provenance](../evidence/early/classifier-provenance.json), [300-token pilot](../evidence/early/pilot-w6a8-300-summary.json).

## A short refusal does not justify a short answer budget

The initial 300-token setting could observe short refusals, but it could also cut off a legitimate answer before the model finished. Comparing a completed refusal with an unfinished answer introduces a completion confound.

The later replicated study used marker-response budgets of 2,048, 4,096, and 6,144 tokens, and capability budgets of 512, 1,024, and 2,048, constrained by an 8,192-token context. The native thinking-enabled workflow suite used a 4,096-token output budget and one context-safe retry up to 6,144. The budget includes thinking and tool/final output; it is not a promise of that much user-visible text.

Retries cost tokens. Preserve the exhausted attempt, retry the same pre-turn history, and include both attempts in cost accounting. A timeout, parser error, empty result, or exhausted budget belongs in its own category rather than being silently counted as a refusal.

Sources: [replicated protocol](../evidence/early/replicated-plan.json), [workflow configuration](../evidence/w8a16/authorized-workflow-eval-v1/baseline-thinking-workflows-v1/configuration.json).

## Same score does not mean same answer

The later capability study had **24 previously seen development questions, repeated twice**. That is 48 observations, not 48 independent questions. It was run separately with thinking off and on. Results from those modes must not be pooled.

Compare exact case IDs and repeats. In the thinking-enabled study, the original strength-1.5 branch matched the baseline's 46/48 total while losing one baseline success and gaining a different one. The W8-weight-rebased branch reached 47/48: its one gain was a JSON-schema case in one repeat. All six reasoning cases already passed both baseline repeats. Calling this a new reasoning breakthrough would misdescribe the result.

The unedited baseline generated 20,336 tokens. The mixed branch generated 40,664 for the same 46/48 total; the W8-weight-rebased branch generated 41,063 for 47/48. Those totals include thinking and retries. They establish extra generation work, not identical wording and not a causal twofold change in GPU kernel speed. See [the measured results](02-measured-results.md) for the full comparison.

## Format failures and reasoning failures both matter

The thinking-disabled diagnostic found several different mechanisms:

- All three original edited branches introduced the same substantive rounding-order error in both repeats.
- Some correct conclusions failed because the answer added explanation around an integer, a JSON object, or the required word.
- Some route and scheduling mistakes were inherited from the baseline, rather than newly introduced by editing.
- The JSON-types prompt had wording ambiguity. The existing frozen scores were retained; clearer wording belongs in a new version.
- All 192 answers in the original four-branch diagnostic completed. A larger cap alone was therefore not supported as the explanation or remedy for those final failures.

For an agent, output format is functional behavior: a correct number inside an unparsable tool argument can still break a workflow. Keep the strict score and a separate qualitative diagnosis. Do not quietly award semantic credit after seeing which candidate would benefit.

Source: [response-format diagnosis](../evidence/w8a16/response-format-diagnostic-v2.json).

## Runtime identity is part of the experiment

The served model alias deliberately stayed the same for client compatibility. The alias is consequently not enough to identify the actual model. Record the checkpoint root, quantization configuration, adapter identity, chat template, thinking mode, sampling settings, context, and runtime profile.

Native LoRA routing probes restored the baseline answer after adapter use. An initial exact-log-probability equality check failed, but repeated baseline-only controls showed ordinary numerical variation of comparable size. The evidence supported repeated development comparisons; it did not establish bitwise determinism. Temperature zero does not make all observed schedules numerically identical.

Source: [routing review](../evidence/w8a16/routing-review-v1.json).

## Native tool APIs need completion checks

The installed vLLM implementation could label a parsed response as `tool_use` even when generation had reached its output ceiling. The workflow runner therefore checked both the native stop reason and output usage. Calls from an exhausted attempt were never dispatched.

The runner preserved full native assistant blocks, tool IDs, and matching tool results. It counted the same model/system/messages/tools/template that it would send for generation. Tool errors were explicit in the result body, because accepting an `is_error` field did not guarantee that the model's rendered prompt represented it.

A second subtle problem was evidence visibility. Independent calls may be generated in one assistant turn, but their results have not yet been seen by that assistant. A report in the same batch cannot legitimately cite a guessed ID from a sibling call. The fixture and replay auditor enforce that boundary.

Sources: [frozen workflow runner](../evidence/w8a16/authorized-workflow-eval-v1/run_workflows_v1.py), [fixture](../evidence/w8a16/authorized-workflow-eval-v1/fixtures_v1.py), [preflight validation](../evidence/w8a16/authorized-workflow-eval-v1/baseline-launch-receipt-v1.json).

## Fix the evaluator before interpreting the model

Before workflow inference, offline checks caught four scoring problems: counting a successful request after login as the same failed request before login; treating negated findings as affirmative claims; accepting an explicit URL port of zero as a default port; and permitting citation of unseen sibling-call evidence. These were corrected before freezing the run. The combined fixture/runner suite passed 38 offline tests, and an independent reviewer reproduced the fixes.

After inference, a separate reporter re-created each fixture, replayed the accepted tool batches, checked histories and hashes, and reconciled token work. A separate agent reviewed the traces qualitatively. The distinction matters: independently replaying a deterministic grader can establish consistency, while a qualitative review can notice that a supported narrow result is being described too broadly.

## What the percentages mean

There are at least three different quantities:

1. **Absolute score:** correct responses divided by evaluated responses.
2. **Score ratio:** candidate correct divided by baseline correct on the matched set.
3. **Retention of baseline successes:** baseline-correct observations that remain correct under the candidate.

For the thinking-enabled W8-weight-rebased branch these are 47/48, 47/46, and 46/46 respectively. None establishes “98% of the model's intelligence retained.” The user's preferred and fallback targets were research goals, not certified outcomes.

Seven successful tool/vision smoke checks likewise establish those seven predicates. The 16 successful synthetic workflow observations establish those guided tasks. Neither result replaces held-out capability tests or real local CTF execution.

Source: [independent five-branch audit](../evidence/w8a16/final-five-branch-independent-audit-v1.json).
