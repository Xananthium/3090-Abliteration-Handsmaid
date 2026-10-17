# Evidence and preservation

This repository is the portable research record for the work summarized in the README. It preserves completed results and the distinctions needed to interpret them. It is not a full model backup or a deployment package.

## Where things live

| Directory | Contents |
|---|---|
| `docs/` | Eight research chapters, with relative links to the evidence used |
| `data/` | Capability, paired-outcome, historical, and Hauhau measurement tables; source mapping and manifests |
| `figures/` | Byte-identical copies of the previously verified charts |
| `evidence/w8a16/` | Five-branch results, per-run evidence, relevant code/receipts, Hauhau diagnostics, and the synthetic workflow run |
| `evidence/bf16/` | Retained BF16 comparison/runtime/cleanup evidence |
| `evidence/early/` | Earlier comparison reports, detector provenance, and aggregate extracts |
| `evidence/related/` | Earlier source, pilot, export, and publication provenance |
| `evidence/checkpoint-metadata/` | Small checkpoint metadata where required by the source inventory |
| `scripts/verify_archive.py` | Offline integrity and document-link verification |

The local workspace remains `~/Abliteration-Research/`; the public repository is named `3090-Abliteration-Handsmaid`.

## What is copied and what is referenced

The [evidence manifest](../data/evidence-manifest.json) records each original path, original digest, role, preservation status, and, where bundled, archive path, size, and digest. The [source map](../data/source-map.json) maps original citation paths to their portable copies.

`copied_unchanged` means the archived bytes match the source. `copied_aggregate_extract` means a historical response summary was reduced to aggregate fields; response records and text were omitted, and the extract records the original digest and removed keys. `copied_provenance_note` identifies a source whose public copy contains provenance instead of the complete third-party text. The original local files were not changed by these transformations.

Tensor payloads and large binaries are not duplicated. Their previously recorded hashes and original locations remain references. A historical weight hash is not represented as a fresh rehash of that weight during documentation work. Installed upstream implementation files likewise remain hash references rather than redistributed copies.

This keeps the archive small while preserving the provenance needed to distinguish models, formats, scripts, and measurements. It does not make a deleted checkpoint recoverable without its model backup. No guarantee about current Hub availability is inferred from an old upload receipt.

## The original snapshots stay original

Bundled JSON/CSV/code evidence can contain the original machine paths, retired checkpoint locations, and old status labels. Those contents are retained to preserve the historical record and any existing checksums. They are not instructions to start services, repeat an experiment, or treat an abandoned plan as current.

The new research chapters use portable links. For example, a proposal written before the Hauhau pilots remains a proposal in its original file; the chapter explains which later observations superseded it. The older 300-token diagnostic is retained as an aggregate extract with its truncation limitation. The frozen report scores are never silently adjusted to make a later narrative stronger.

The five-branch evidence manifest includes source files and adapter identities beyond the small set of files needed to reproduce summary tables. Those dependencies are recorded even when their tensor payloads remain outside this repository.

## Verification

With Python 3.11 or newer, from the repository root:

```bash
python3 scripts/verify_archive.py
```

The verifier checks archive hashes and sizes, checks the authored document links and image references, and validates manifest counts. It performs no network requests, model loading, inference, service operations, or source-file mutation. Its successful result establishes consistency of the bundled archive, not independent scientific validation of every conclusion.

Separate research audits already checked the underlying results: the five-branch audit validated 480 capability observations and 70 smoke predicates; the workflow reporter replayed the saved native histories and fixture results; an additional agent reviewed the 16 workflow traces qualitatively. Those are separate from this archive's file-integrity check.

## Scope of publication

The public repository includes documentation, aggregate metrics, benign capability answers, synthetic workflow records, and bounded research evidence. It does not publish a real engagement's captured data, credentials, full model weights, or a new trained model. The old refusal-response summaries linked from the history are aggregate extracts rather than a republished corpus of generated answers.

This archive was assembled on 17 October 2026. No new inference, training, conversion, donor transfer, or activation capture was performed while writing it. Source and result timestamps describe when the recorded work happened, not a fresh replication during publication.
