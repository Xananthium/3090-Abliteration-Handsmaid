#!/usr/bin/env python3
"""Prepare the selected adapter for installed vLLM without touching base weights.

Run with the research environment Python. This script uses NumPy safetensors and
CPU-only PyTorch to execute the installed MoE packer in isolation. It never
imports vLLM, loads a model, initializes CUDA, or starts services. The destination
must be absent or empty; original files are always read-only.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from safetensors.numpy import load_file, save_file

ROOT = Path("/home/isnotreal/model-tools/redcell-marker-eval-20261015")
SOURCE = Path("/home/isnotreal/models/redcell-abliteration-source-bf16")
ORIGINAL = ROOT / "strength-1.5/adapter"
WORK = ROOT / "bf16-iteration"
OUTPUT = WORK / "adapters/strength-1.5-vllm"
RECEIPT = WORK / "vllm-adapter-validation.json"
VLLM = Path("/home/isnotreal/vllm-env/lib/python3.12/site-packages/vllm")
PEFT_UTILS = Path(
    "/home/isnotreal/model-tools/redcell-abliteration-20261014/env/"
    "lib/python3.12/site-packages/peft/tuners/tuners_utils.py"
)
BASE_PREFIX = "base_model.model."
LAYER_PREFIX = "model.language_model.layers."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path):
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        raw = stream.read(size)
    require(len(raw) == size, f"Incomplete header: {path}")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def keys(module):
    return tuple(BASE_PREFIX + module + f".lora_{part}.weight" for part in ("A", "B"))


def fingerprint(tensors, names):
    """Bind names, dtype, shape and exact C-order tensor bytes."""
    digest = hashlib.sha256()
    for name in sorted(names):
        array = tensors[name]
        descriptor = json.dumps(
            [name, array.dtype.str, list(array.shape)], separators=(",", ":")
        ).encode()
        digest.update(struct.pack("<Q", len(descriptor)))
        digest.update(descriptor)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def source_inventory():
    tensors = {}
    files = []
    for path in sorted(SOURCE.glob("*.safetensors")):
        values, digest = header(path)
        values.pop("__metadata__", None)
        require(not tensors.keys() & values.keys(), "Duplicate source tensor")
        tensors.update(values)
        files.append({"path": str(path), "header_sha256": digest,
                      "file_bytes": path.stat().st_size, "tensor_count": len(values)})
    index = json.loads((SOURCE / "model.safetensors.index.json").read_text())
    require(set(index["weight_map"]) == set(tensors), "Source index/header mismatch")
    require(len(tensors) == 1013, "Unexpected source tensor count")
    require({v["dtype"] for v in tensors.values()} == {"BF16"}, "Source is not all BF16")
    return tensors, files


def expected_modules(config, source_tensors):
    text = config["text_config"]
    layers, experts = text["num_hidden_layers"], text["num_experts"]
    require((layers, experts) == (30, 128), "Unexpected model architecture")
    old, added = {}, {}
    per_layer = []
    for layer in range(layers):
        prefix = f"{LAYER_PREFIX}{layer}"
        gate = source_tensors[prefix + ".experts.gate_up_proj"]["shape"]
        down = source_tensors[prefix + ".experts.down_proj"]["shape"]
        require(len(gate) == len(down) == 3 and gate[1] % 2 == 0, "Bad expert shape")
        e, doubled_intermediate, hidden = gate
        intermediate = doubled_intermediate // 2
        require([e, hidden, intermediate] == down, "Expert source dimensions disagree")
        require((e, hidden, intermediate) ==
                (experts, text["hidden_size"], text["moe_intermediate_size"]),
                "Expert shape differs from config")
        for expert in range(experts):
            module = f"{prefix}.experts.{expert}"
            old[module + ".down_proj"] = (hidden, intermediate)
            for part in ("gate", "up"):
                added[module + f".{part}_proj"] = (intermediate, hidden)
        for suffix in ("mlp.down_proj", "self_attn.o_proj"):
            module = prefix + "." + suffix
            shape = source_tensors[module + ".weight"]["shape"]
            require(len(shape) == 2, "Expected dense matrix")
            old[module] = tuple(shape)
        per_layer.append({"layer": layer, "experts": experts,
                          "gate_up_source_shape": gate, "down_source_shape": down})
    return old, added, per_layer


def compile_nodes(nodes, path, namespace):
    tree = ast.Module(body=[ast.ImportFrom(module="__future__",
                        names=[ast.alias(name="annotations")], level=0), *nodes],
                      type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)


def isolated_loader_checks(tensors, target_modules, rank, alpha):
    """Execute installed parsing, PEFT matching and MoE packing code on CPU.

    Only the needed function/class methods are extracted with AST. Package
    imports and model construction are deliberately omitted. The mapping stub
    supplies the inspected Gemma4 backbone prefix rule to the actual parser.
    """
    import torch

    require(not torch.cuda.is_initialized(), "CUDA unexpectedly initialized")
    torch.set_num_threads(2)
    weights_path = VLLM / "lora/lora_weights.py"
    namespace = {"torch": torch}
    nodes = []
    for node in ast.parse(weights_path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name in {
            "LoRALayerWeights", "PackedLoRALayerWeights"
        }:
            node.body = [part for part in node.body
                         if isinstance(part, ast.FunctionDef)
                         and part.name in {"__init__", "pack_moe"}]
            nodes.append(node)
    require(len(nodes) == 2, "Installed packer layout changed")
    compile_nodes(nodes, weights_path, namespace)

    parser_path = VLLM / "lora/utils.py"
    parser = next(node for node in ast.parse(parser_path.read_text()).body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "parse_fine_tuned_lora_name")
    compile_nodes([parser], parser_path, namespace)

    class BackboneMapper:
        def _map_name(self, name):
            require(name.startswith("model.language_model."), "Unexpected adapter tower")
            return name.replace("model.language_model.", "language_model.model.", 1)

    mapped = {}
    expected_suffixes = {f"experts.{i}.{part}_proj"
                         for i in range(128) for part in ("gate", "down", "up")}
    expected_suffixes.update({"down_proj", "o_proj"})
    for name in tensors:
        module, is_a = namespace["parse_fine_tuned_lora_name"](name, BackboneMapper())
        suffix = module[module.index(".experts") + 1:] if ".experts" in module else module.rsplit(".", 1)[-1]
        require(suffix in expected_suffixes, f"Unexpected vLLM module: {module}")
        require((module, is_a) not in mapped, "Mapped name collision")
        mapped[module, is_a] = name

    peft_fn = next(node for node in ast.parse(PEFT_UTILS.read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name == "check_target_module_exists")
    peft_ns = {"re": re}
    compile_nodes([peft_fn], PEFT_UTILS, peft_ns)
    peft_config = SimpleNamespace(target_modules=set(target_modules), exclude_modules=None,
                                  modules_to_save=None, target_parameters=None)
    for module in target_modules:
        require(peft_ns["check_target_module_exists"](peft_config, module),
                f"PEFT exact target does not match: {module}")
    negative_targets = [f"{LAYER_PREFIX}{layer}.mlp.{part}_proj"
                        for layer in range(30) for part in ("gate", "up")]
    negative_targets += ["model.vision_tower.encoder.layers.0.mlp.down_proj.linear",
                         "model.embed_vision", "lm_head"]
    for module in negative_targets:
        require(not peft_ns["check_target_module_exists"](peft_config, module),
                f"PEFT target unexpectedly includes: {module}")

    weights_cls, packed_cls = namespace["LoRALayerWeights"], namespace["PackedLoRALayerWeights"]
    layer_checks = []
    for layer in range(30):
        loras = []
        for expert in range(128):
            for part in ("gate", "down", "up"):
                module = f"language_model.model.layers.{layer}.experts.{expert}.{part}_proj"
                arrays = [tensors[mapped[module, is_a]] for is_a in (True, False)]
                loras.append(weights_cls(module, rank, alpha,
                                         torch.from_numpy(arrays[0]), torch.from_numpy(arrays[1])))
        packed = packed_cls.pack_moe(loras, f"language_model.model.layers.{layer}.experts")
        require(packed.rank == rank and packed.scaling == [1.0, 1.0, 1.0], "Packed scaling changed")
        for slot in range(3):
            require(packed.lora_a[slot].device.type == packed.lora_b[slot].device.type == "cpu",
                    "Non-CPU packing tensor")
        for slot in (0, 2):
            a, b = packed.lora_a[slot], packed.lora_b[slot]
            require(torch.count_nonzero(a).item() == torch.count_nonzero(b).item() == 0,
                    "Gate/up padding is nonzero")
            # All 128 experts: signed finite probe plus exact-zero factor proof.
            x = torch.linspace(-1.0, 1.0, a.shape[-1], dtype=torch.float32).reshape(1, -1, 1)
            x = x.expand(a.shape[0], -1, -1)
            delta = torch.bmm(b, torch.bmm(a, x))
            require(torch.count_nonzero(delta).item() == 0, "Nonzero gate/up probe output")
        for expert in range(128):
            module = f"{LAYER_PREFIX}{layer}.experts.{expert}.down_proj"
            for slot, name in zip((packed.lora_a[1], packed.lora_b[1]), keys(module)):
                require(slot[expert].numpy().tobytes() == tensors[name].tobytes(),
                        "Packing changed an expert down factor")
        layer_checks.append({"layer": layer, "complete_expert_triplets": 128,
                             "gate_up_zero_probe_projections": 256,
                             "packed_A_shapes": [list(v.shape) for v in packed.lora_a],
                             "packed_B_shapes": [list(v.shape) for v in packed.lora_b]})
    require(not torch.cuda.is_initialized(), "CUDA unexpectedly initialized after CPU checks")
    return {"method": "AST-isolated installed functions, real CPU PyTorch; no vLLM package import/model/kernel execution",
            "torch_version": torch.__version__, "cuda_initialized": False,
            "parsed_tensor_keys": len(mapped), "peft_exact_targets": len(target_modules),
            "peft_negative_targets": len(negative_targets), "packed_layers": layer_checks,
            "code": {"pack_moe": artifact(weights_path), "parser": artifact(parser_path),
                     "peft_target_matcher": artifact(PEFT_UTILS)}}


def main():
    require(OUTPUT.resolve() != ORIGINAL.resolve(), "Output aliases original")
    require(not OUTPUT.exists() or not any(OUTPUT.iterdir()), "Destination is not empty")
    require(not RECEIPT.exists(), "Validation receipt already exists")
    original_config_path = ORIGINAL / "adapter_config.json"
    original_tensor_path = ORIGINAL / "adapter_model.safetensors"
    originals_before = [artifact(original_config_path), artifact(original_tensor_path)]
    source_before = [artifact(SOURCE / "config.json"), artifact(SOURCE / "model.safetensors.index.json")]
    config = json.loads(original_config_path.read_text())
    rank, alpha = config["r"], config["lora_alpha"]
    require((rank, alpha) == (3, 3), "Unexpected rank/alpha")
    for field in ("rank_pattern", "alpha_pattern", "modules_to_save", "target_parameters",
                  "use_dora", "use_rslora", "lora_bias", "fan_in_fan_out"):
        require(not config.get(field), f"Unsupported adapter feature: {field}")
    require(config["bias"] == "none" and config["peft_type"] == "LORA", "Unexpected adapter type")
    source_tensors, source_headers = source_inventory()
    model_config = json.loads((SOURCE / "config.json").read_text())
    old_modules, added_modules, per_layer = expected_modules(model_config, source_tensors)
    original = load_file(str(original_tensor_path))
    old_names = {name for module in old_modules for name in keys(module)}
    require(set(original) == old_names and len(old_names) == 7800, "Original coverage differs")
    for module, (out_size, in_size) in old_modules.items():
        a, b = (original[name] for name in keys(module))
        require(a.shape == (rank, in_size) and b.shape == (out_size, rank), "Original shape mismatch")
        require(a.dtype == b.dtype == np.dtype("float32"), "Original factor is not FP32")
        require(np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite original factor")
    original_fingerprint = fingerprint(original, old_names)
    derived = dict(original)
    for module, (out_size, in_size) in added_modules.items():
        a_name, b_name = keys(module)
        require(a_name not in derived and b_name not in derived, "Padding collides with original")
        derived[a_name] = np.zeros((rank, in_size), dtype=np.float32)
        derived[b_name] = np.zeros((out_size, rank), dtype=np.float32)
    target_modules = sorted(old_modules.keys() | added_modules.keys())
    derived_config = dict(config, target_modules=target_modules)
    require({k:v for k,v in config.items() if k != "target_modules"} ==
            {k:v for k,v in derived_config.items() if k != "target_modules"}, "Config changed beyond targets")
    require(len(derived) == 23160 and len(target_modules) == 11580, "Bad derived coverage")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config_output = OUTPUT / "adapter_config.json"
    tensors_output = OUTPUT / "adapter_model.safetensors"
    save_file(derived, str(tensors_output), metadata={"format": "pt"})
    config_output.write_text(json.dumps(derived_config, indent=2, sort_keys=True) + "\n")
    reloaded = load_file(str(tensors_output))
    require(set(reloaded) == set(derived), "Saved tensor key mismatch")
    require(fingerprint(reloaded, old_names) == original_fingerprint, "Original bits changed")
    added_names = set(reloaded) - old_names
    for name in added_names:
        value = reloaded[name]
        require(value.shape == derived[name].shape and value.dtype == np.float32,
                "Padding shape/dtype changed")
        require(np.count_nonzero(value) == 0 and not np.signbit(value).any(), "Padding is not exact positive zero")
    loader_checks = isolated_loader_checks(reloaded, target_modules, rank, alpha)
    require(originals_before == [artifact(original_config_path), artifact(original_tensor_path)],
            "Original adapter files changed")
    require(source_before == [artifact(SOURCE / "config.json"), artifact(SOURCE / "model.safetensors.index.json")],
            "Source config/index changed")
    require(source_headers == source_inventory()[1], "Source headers changed")
    result = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_cpu_preparation_and_installed_packer_validation",
        "scope": "No GPU, inference, services, vLLM installation changes, source model changes, or original adapter changes.",
        "runtime_validation": "Pending root-owned full BF16 vLLM loading and legitimate capability/vision/tool checks.",
        "script": artifact(Path(__file__).resolve()),
        "original_artifacts": originals_before, "source_artifacts": source_before,
        "source_header_inventory": source_headers, "source_dimensions": per_layer,
        "derived_artifacts": [artifact(config_output), artifact(tensors_output)],
        "rank": rank, "lora_alpha": alpha, "alpha_over_rank": alpha / rank,
        "strength": "Selected 1.5 strength remains encoded in unchanged original factors; no second multiplier applied.",
        "original_projections": 3900, "original_tensors_preserved": len(old_names),
        "original_factor_dtype": "F32", "original_tensor_bytes": sum(original[k].nbytes for k in old_names),
        "original_tensor_fingerprint_before_and_after": original_fingerprint,
        "added_projections": len(added_modules), "added_zero_tensors": len(added_names),
        "added_zero_tensor_bytes": sum(reloaded[k].nbytes for k in added_names),
        "derived_projections": len(target_modules), "derived_tensors": len(reloaded),
        "derived_tensor_bytes": sum(v.nbytes for v in reloaded.values()),
        "target_modules": "Exact qualified PEFT module names from original factor keys plus per-expert gate/up names; all other adapter_config fields unchanged.",
        "coverage": {"layers": 30, "experts_per_layer": 128, "complete_gate_down_up_triplets": 3840,
                     "dense_down": 30, "attention_o": 30, "vision_adapter_modules": 0},
        "zero_addition_proof": "Every added A and B is finite positive FP32 zero with source-derived shapes. Thus (alpha/r)*(B@A) is the exact zero matrix; for every finite input x, B@(A@x)=0. BF16 conversion preserves zeros. CPU batched signed-input probes also passed for all 7680 added projections. This proves the added factors have zero algebraic effect; it does not assert identical engine rounding or behavior.",
        "installed_loader_checks": loader_checks,
        "originals_unchanged_after_validation": True,
        "minimum_additional_startup_flags": ["--enable-lora", "--max-lora-rank", "8", "--max-loras", "1",
            "--lora-dtype", "bfloat16", "--lora-modules", "strength-1.5-vllm=" + str(OUTPUT)],
        "optional_target_restriction": ["--lora-target-modules", "o_proj", "down_proj", "experts"],
        "base_startup_requirements": "Full BF16 source and dtype, current CPU offload budget; no weight/activation quantization or Humming settings. Native LoRA must be enabled at engine startup. No tower-connector LoRA is needed.",
        "precision_caveat": "Disk factors remain FP32. vLLM lora_dtype=bfloat16 casts factors and computes unmerged low-rank additions, which differs from FP32 W + (B@A)*scale rounded once to BF16. Once a candidate is selected, merge once from the untouched BF16 source and preserved FP32 factors and check that merged BF16 model before quantization.",
        "feasibility_limits": ["Installed packer and mapping conventions passed CPU checks; GPU kernels and serving are not validated by this script.",
            "TP=2 gives expert intermediate partition 352; installed unquantized LoRA oracle selects Triton fallback.",
            "Max rank 8 is the smallest supported engine capacity for rank 3. Expert LoRA buffers alone are estimated at 556.875 MiB per GPU for TP2/BF16/max_loras1; dense buffers and workspaces add more.",
            "CPU weight offload retains BF16 but adds transfer bandwidth cost; performance must be measured by the runtime owner."],
    }
    RECEIPT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "adapter": str(OUTPUT),
                      "receipt": artifact(RECEIPT), "derived_artifacts": result["derived_artifacts"],
                      "original_tensors_preserved": len(old_names), "zero_tensors_added": len(added_names),
                      "packed_layers": len(loader_checks["packed_layers"])}, indent=2))


if __name__ == "__main__":
    main()
