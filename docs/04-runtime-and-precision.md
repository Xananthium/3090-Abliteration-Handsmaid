# Runtime and precision

The experiments were conducted on two RTX 3090 GPUs, with 24 GiB nominal memory each and an NVLink connection. The decisive runtime distinction was between a large BF16 model that required host weight offload and a native INT8 W8A16 baseline that supported reversible edits without that offload in the small diagnostic profile. This is a record of the tested setup and its limits, not a universal speed ranking. [BF16 offload audit](../evidence/bf16/offload-performance-audit.json)

## Precision labels describe different things

| Label in these records | Weight representation | Activation handling | Scope |
|---|---|---|---|
| Reconstructed BF16 | BF16 language tensors converted from upstream F16 GGUF; restored BF16 vision | BF16 in the native diagnostic | Source and attempted full-model comparison |
| NF4 analysis | Four-bit analysis representation of language linears; BF16 exclusions | Analysis-framework handling | Initial pilot construction and small checks |
| W6A8 | Packed INT6 group-32 language weights; BF16 exclusions | Dynamic per-token INT8 in the recorded Humming path | Earlier exports and completed comparisons |
| W8A8 | Eight-bit integer weight export; BF16 exclusions | Eight-bit integer activation path | Later stronger-candidate export and broader comparison |
| Native W8A16 | Symmetric INT8 group-32 compressed-tensors weights; BF16 exclusions | BF16; activation quantization entries are null | Later native Marlin baseline and LoRA comparisons |
| FP8 | A separate floating-point representation | Depends on its runtime | Considered in source audit; not the chosen or tested W8 baseline |

Eight bits does not make INT8 and FP8 interchangeable. Likewise, W8A8 and W8A16 share a weight-width label but use different activation handling. BF16 activations do not imply BF16 model weights. Packed six-bit storage does not by itself establish native six-bit 3090 arithmetic or a measured speed benefit. The recorded export and runtime receipts establish a format and working path; a controlled performance comparison is still required to establish efficiency. [Original export](../evidence/related/redcell-abliteration-20261014/export-receipt.json), [W8A16 precision manifest](../evidence/w8a16/precision-manifest-v1.json), [installed-runtime audit](../evidence/w8a16/runtime-feasibility-audit.json)

KV-cache precision is a third, separate choice. A checkpoint's W/A label does not identify its KV cache. Nor does a global BF16 dtype option guarantee BF16 activations if an independent backend setting quantizes those activations. The recorded W8A16 launch explicitly cleared inherited Humming and Marlin activation-override variables. [W8A16 launch manifest](../evidence/w8a16/launch-v1.json)

## Why the BF16 attempt was impractical for this comparison

The reconstructed source contains approximately **48.07 GiB of tensor payload**, already greater than the two cards' combined nominal 48 GiB before KV, workspaces, vision replication, adapter buffers, and allocator overhead. Disk size divided by two is therefore not a valid GPU fit estimate. Offloading preserved BF16 values but introduced host-to-GPU weight traffic. NVLink helps communication between GPUs; it does not eliminate the host-memory transfer path. [BF16 feasibility audit](../evidence/bf16/runtime-feasibility-audit.json)

The recorded BF16 configuration used nominally 8 GiB of UVA weight offload per GPU, with approximately 8.43 GiB actually offloaded per GPU. The audit reports about 17.14 GiB per GPU for the loaded model with LoRA, and around 20.5 GiB with KV. A parent-recorded sequential sample produced 212 output tokens in 256 seconds, approximately 0.83 tokens/s. This single sample is evidence of a practical problem in that setup, not a controlled sustained-throughput benchmark. The audit itself performed no new GPU test. [Offload performance audit](../evidence/bf16/offload-performance-audit.json)

The source-generation smoke returned the expected arithmetic answer, and the base/edit/base routing smoke passed. Neither is a full capability assessment. The main BF16 comparison ended after 32 source observations when the working baseline changed to W8A16. Selective prefetch was examined as a possible alternative, but no measured prefetch improvement is established by these receipts. [Source smoke](../evidence/bf16/source-generation-smoke.json), [routing smoke](../evidence/bf16/lora-switch-smoke-v1.json), [incomplete comparison status](../evidence/bf16/integrity-batch-status-v1.json)

## The W8A16 baseline and the runtime that was actually tested

The one-time CPU export applied no local edit. It produced 17 weight files totaling **26.695 GiB**, preserved 748 BF16 tensors including all 356 vision tensors, and passed 15 independent packing/numeric checks. The export took 451.28 seconds on eight CPU threads, with a measured peak process RSS of 7.68 GiB and no GPU use. Those are conversion measurements, not inference performance. [Export completion receipt](../evidence/w8a16/export-completion-summary.json)

The installed runtime audited for this phase was vLLM 0.31.0. Its native compressed-tensors/Marlin path supported the chosen INT8 weights with BF16 activations on SM86. The recorded diagnostic launch was:

| Setting | Recorded value |
|---|---|
| Tensor parallelism | 2 GPUs |
| Context limit | 8,192 tokens |
| KV-cache dtype | BF16 |
| Weight offload | 0 GiB per GPU |
| GPU-memory utilization setting | 0.85 |
| Concurrent sequences | At most 4 |
| Batched-token limit | 4,096 |
| Execution and attention | Eager; Triton attention |
| Linear and MoE kernels | Marlin |
| Prefix caching | Disabled for fresh prefill on each branch |
| Native parsers | Gemma tool and reasoning parsers |

These values describe the private development comparison, not a claim that a 200k context, 80 GiB offload, 98% GPU-memory target, or production quantized KV was validated by it. [Frozen launch manifest](../evidence/w8a16/launch-v1.json)

The API model name remained `qwen3.6-35b-hauhau-w4a16-vision` for existing-client compatibility. That intentionally retained alias does **not** identify the actual architecture, lineage, or precision. The pinned checkpoint and precision manifest identify the REDCELL/Gemma W8A16 baseline.

## Native LoRA compatibility lessons

Gemma's per-expert loader expected gate/up/down adapter entries, while the historical edit affected down/output projections. Separate compatibility copies supplied zero gate/up factors; they did not alter the original saved factors or vision tensors. A runtime rank capacity of eight accommodated the smaller saved rank rather than changing its mathematical rank. [BF16 runtime feasibility audit](../evidence/bf16/runtime-feasibility-audit.json)

The INT8 Marlin audit found a geometry constraint: the logical per-GPU expert intermediate width was 352, padded by the kernel to 384. Group size 32 satisfied the relevant partitions; the audited larger groups did not. The down-projection adapter path used logical width 352 while respecting activation row stride 384. An isolated CPU wrapper probe checked that argument handling, and later native runtime checks supplied separate generation evidence. The supported scope remained non-fully-sharded adapters with zero expert gate/up factors. Nonzero gate/up edits were not validated by that work. [W8 runtime and export audit](../evidence/w8a16/runtime-feasibility-audit.json)

The same audit found that the inspected Humming MoE implementation lacked the required native expert-LoRA support and that the inspected AWQ MoE path accepted four-bit weights only. Those findings explain the chosen development backend; they are version-specific, not permanent claims about every release. FP8 alternatives and a Transformers eight-bit fallback were not exercised here.

Vision preservation depended on explicit exclusions. The installed multimodal wrapper passed quantization configuration into the towers, so an assumed automatic fallback would not have been sufficient evidence. The exporter checked unchanged vision tensors; basic image smokes later tested only a small part of vision behavior.

The first routing probe restored answers but failed exact baseline log-probability equality after a reset. A follow-up found comparable variation between baseline-only requests. Development comparisons proceeded with that variability recorded, rather than claiming bitwise determinism. [Initial routing smoke](../evidence/w8a16/routing-smoke-v1.json), [routing review](../evidence/w8a16/routing-review-v1.json)

## Production KV and diagnostic KV must stay separate

The historical native-KV probe recorded a TurboQuant Q4 vision error: `partial multimodal token full attention not supported`. The probe's TurboQuant-Q4 path without that vision mask and its native-INT4 vision path had empty error lists. These are compatibility observations only. An empty error list does not establish long-context accuracy, equivalent vision behavior, or a sustained-load result. [Native KV validation receipt](../evidence/related/redcell-abliteration-20261014/native-kv-validation.json)

The weight/edit studies used BF16 KV and an 8,192-token context to avoid mixing their comparison with a KV-format change. Production configurations using quantized KV are a different runtime profile. Results from these diagnostic runs cannot establish how that production cache behaves near its long-context or memory limits.

The older native sampling audit also verified that explicit request temperature zero reached the effective sampler despite a nonzero server default. Repeated outputs still varied. Thus “temperature zero” is a verified request property here, not a promise of identical outputs across schedules. [Sampling-parameter verification](../evidence/early/native-sampling-parameter-verification.json)

## Activation-capture feasibility: audited, not exercised

The separate native-W8 activation research receipt has status **`source_audit_complete_runtime_capture_not_tested`**. It inspected installed code, identified a possible hidden-state extraction path, and recorded limitations including a BF16-cache requirement and unresolved instrumented-generation equivalence. No activation capture, LoRA-plus-capture compatibility trial, or capture-instrumented behavioral trial ran in that phase. It performed no model calls, GPU allocations, or service changes. [Activation research audit](../evidence/w8a16/native-w8-activation-research-v1.json)

Its source-derived staging estimates ranged from 4.14 MiB per tensor-parallel rank for a small example to 682.67 MiB for a larger example. They exclude live auxiliary tensors, copies, cache pages, weights, KV, and workspace. They are **not measured peak memory or proof that a launch fits**. The audited extraction requirements also differ from the quantized-KV production profile. This archive preserves that feasibility result without presenting an untested optimization procedure.

## Boundaries of the measurements

Runtime adapter addition used BF16-cast factors. It is not claimed numerically identical to FP32 merging followed by BF16 rounding, nor to merging and requantizing a final export. Some later branches also overlapped bounded CPU research jobs. Their wall times are observed development timings; output-token increases can be compared separately without calling those timings controlled deployment speed. [Launch precision note](../evidence/w8a16/launch-v1.json), [final report completion](../evidence/w8a16/finish-development-report-v2-completion.json)

The completed studies establish specific functionality and case-level scores under frozen profiles. They do not establish a universal precision winner, zero-refusal behavior, broad capability preservation, or equivalence between diagnostic and production settings. The measured score and token comparisons are recorded separately in [measured results](02-measured-results.md).
