#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build one NF4-direction / W8-weight adapter, entirely on CPU.

The saved pilot directions, selected trial, rank, alpha, target modules and zero
padding are immutable inputs. Only active weight-dependent factors are rebuilt.
No model is instantiated; no API, service, GPU or checkpoint export is used.

First run --preflight. Then run --build with the same Python and thread count.
Both modes refuse existing output paths. A failed build is deliberately not
resumable: its incomplete directory is retained, and no completed receipt exists.
The existing precision and adapter-identity manifests are never modified.

The projection algorithm follows Philipp Emanuel Weidmann and contributors'
local Heretic Abliteration.modify_model, which is pinned and executed separately
on CPU fixtures for validation. CPU SVD rounding/RNG differs from the historical
GPU construction; this is not a claim of bitwise cross-device equivalence.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import struct
import sys
import time
from types import SimpleNamespace

ROOT = Path('/home/isnotreal/model-tools/redcell-marker-eval-20261015')
WORK = ROOT / 'w8a16-iteration'
OLD = Path('/home/isnotreal/model-tools/redcell-abliteration-20261014')
BASE = Path('/home/isnotreal/models/redcell-baseline-w8a16-native')
ORIGINAL = OLD / 'candidate-adapter'
STARTER = WORK / 'adapters/strength-1.0-vllm'
REFERENCE = OLD / 'heretic/src/heretic/modifiers/abliteration.py'
LOADER_CHECKS = ROOT / 'bf16-iteration/prepare_vllm_adapter.py'
VLLM = Path('/home/isnotreal/vllm-env/lib/python3.12/site-packages/vllm')
PEFT_UTILS = OLD / 'env/lib/python3.12/site-packages/peft/tuners/tuners_utils.py'
OUTPUT = WORK / 'adapters/nf4-direction-w8-weight-strength-1.0-vllm-v1'
PREFLIGHT = WORK / 'w8-weight-rebase-preflight-v1.json'
MODEL_ID = 'redcell-w8a16-nf4-direction-w8-weight-strength-1.0-v1'
LAYER_PREFIX = 'model.language_model.layers.'
BASE_PREFIX = 'base_model.model.'
RANK = ALPHA = 3
SEED = 20261014
GROUP = 32

PINNED = {
    OLD / 'pilot-summary.json': '39ce2d9baabaee5b9ad70cffc14e95ad2ca91faf0b4ca6fde14bf557c9b214ab',
    OLD / 'pilot-settings.json': '7bed47264a1352e5cf045df57a84809e21a4e697d4cfd475a592aef889b0ac8e',
    OLD / 'trials.json': '69eb673b3d8c5cf59f1f9cb4f0b755be48600750713bd02e6ca7f555884d0b93',
    OLD / 'refusal-directions.pt': 'c53e4a69845092099a096a9673893fce609a9b1dbbb022b5968f9492d218cda7',
    ORIGINAL / 'adapter_config.json': '26bdc03fe2ca3614c71ca8ba6bfb2eb25f54113f67bb7d3bb24ef6fa402904b9',
    ORIGINAL / 'adapter_model.safetensors': 'a46bafac4789ecb97bc94d2c3d6a106692f9694676d80604ccf9aa472d4f7389',
    REFERENCE: '66bd3ff495db689494d8facbe2c08509915f97168460da802248ff71d217f10d',
    OLD / 'heretic/src/heretic/model.py': '2031efcd1980d2734dfc2daa343c3b0588d1d25bf0062e8026fe35f4bd9d4b10',
    LOADER_CHECKS: 'e57818219ca7e11fb3ccf22c4743b1f37f3d32b5323162fa17c13942f8921d40',
    WORK / 'precision-manifest-v1.json': 'a8806a05173567d3f86df3781c3feaa8845fe5e1872fc8c7c6a45361ea7f931c',
    WORK / 'adapter-identities-v1.json': 'b6c8059c4e3c6fefc0c736b223740baf4130ce194a84dd283eefd49ab1687929',
    STARTER / 'adapter_config.json': '8f404726dab2bbd3ddbe99630a37ed3ded557ab077659c556f03e0ec36804891',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def artifact(path):
    return {'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def stat_identity(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def json_read(path):
    return json.loads(path.read_text())


def write_new(path, payload):
    # O_EXCL prevents accidental overwrite, including a dangling symlink.
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def json_new(path, data):
    write_new(path, (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def keys(module):
    return tuple(BASE_PREFIX + module + f'.lora_{part}.weight' for part in ('A', 'B'))


def cpu_setup(threads):
    # Applied before importing any numerical package. No device availability query.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    global torch, np, F, LA
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torch.linalg as LA
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(not torch.cuda.is_initialized(), 'CUDA was initialized')


class TensorReader:
    """Small read-only safetensors reader; reads individual tensors, not shards.

    Avoids retaining mmap-backed pages or importing a model-loading framework.
    Only the few dtypes needed for this controlled builder are decoded.
    """
    def __init__(self, paths):
        self.stack = ExitStack()
        self.entries, self.files = {}, {}
        for path in sorted(set(paths)):
            stream = self.stack.enter_context(path.open('rb'))
            prefix = stream.read(8)
            require(len(prefix) == 8, f'Invalid safetensors prefix: {path}')
            length = struct.unpack('<Q', prefix)[0]
            require(2 <= length < 64 * 1024 * 1024, f'Invalid header length: {path}')
            raw = stream.read(length)
            require(len(raw) == length, f'Short header: {path}')
            head = json.loads(raw)
            offset = 8 + length
            end = path.stat().st_size
            intervals = []
            for name, meta in head.items():
                if name == '__metadata__':
                    continue
                require(name not in self.entries, f'Duplicate tensor: {name}')
                start, stop = meta['data_offsets']
                require(0 <= start <= stop <= end - offset, f'Invalid offsets: {name}')
                self.entries[name] = (stream, offset, meta, path)
                intervals.append((start, stop))
            intervals.sort()
            require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])),
                    f'Overlapping safetensors entries: {path}')
            self.files[str(path)] = {'header_sha256': hashlib.sha256(raw).hexdigest(),
                                     'stat': stat_identity(path)}

    def close(self):
        self.stack.close()

    def meta(self, name):
        return self.entries[name][2]

    def raw(self, name):
        stream, offset, meta, path = self.entries[name]
        start, stop = meta['data_offsets']
        stream.seek(offset + start)
        data = stream.read(stop - start)
        require(len(data) == stop - start, f'Short tensor read: {path}:{name}')
        return data

    def array(self, name):
        meta = self.meta(name)
        dtypes = {'F32': '<f4', 'BF16': '<u2', 'I32': '<i4', 'I64': '<i8'}
        require(meta['dtype'] in dtypes, f'Unsupported tensor dtype: {name}')
        array = np.frombuffer(self.raw(name), dtype=dtypes[meta['dtype']]).copy()
        require(array.size == math.prod(meta['shape']), f'Tensor byte/shape mismatch: {name}')
        return array.reshape(meta['shape'])

    def tensor(self, name):
        value = torch.from_numpy(self.array(name))
        return value.view(torch.bfloat16) if self.meta(name)['dtype'] == 'BF16' else value


def coefficient(layer, distribution):
    distance = abs(layer - distribution['max_weight_position'])
    if distance > distribution['min_weight_distance']:
        return 0.0
    return distribution['max_weight'] + distance / distribution['min_weight_distance'] * (
        distribution['min_weight'] - distribution['max_weight'])


def load_inputs():
    inputs = {}
    for path, expected in PINNED.items():
        got = artifact(path)
        require(got['sha256'] == expected, f'Pinned input changed: {path}')
        inputs[str(path)] = got
    identities = json_read(WORK / 'adapter-identities-v1.json')
    for identity in identities['adapters'].values():
        for name, field in [('adapter_model.safetensors', 'adapter_model_sha256'),
                            ('adapter_config.json', 'adapter_config_sha256')]:
            path = Path(identity['path']) / name
            got = artifact(path)
            require(got['sha256'] == identity[field], f'Source candidate changed: {path}')
            inputs[str(path)] = got
    precision = json_read(WORK / 'precision-manifest-v1.json')
    require(precision['expected_base_checkpoint'] == str(BASE), 'Wrong W8 baseline')
    require(precision['weight_bits'] == 8 and precision['activation_dtype'] == 'bfloat16',
            'Wrong working precision')
    for path, field in [(BASE / 'config.json', 'config_sha256'),
                        (BASE / 'model.safetensors.index.json', 'index_sha256'),
                        (BASE / 'conversion-receipt.json', 'conversion_receipt_sha256'),
                        (WORK / 'launch-v1.json', 'launch_profile_sha256')]:
        got = artifact(path)
        require(got['sha256'] == precision[field], f'Baseline provenance changed: {path}')
        inputs[str(path)] = got
    config = json_read(BASE / 'config.json')
    sections = {'root': config, 'text_config': config.get('text_config', {}),
                'vision_config': config.get('vision_config', {})}
    actual_quant = {k: v['quantization_config'] for k, v in sections.items()
                    if v.get('quantization_config') is not None}
    require(actual_quant == precision['quantization_configs'], 'Quantization config mismatch')
    for quant in actual_quant.values():
        require(quant['format'] == 'pack-quantized' and quant['quant_method'] == 'compressed-tensors',
                'Wrong quantization format')
        require(quant.get('kv_cache_scheme') is None, 'Unexpected KV quantization')
        for group in quant['config_groups'].values():
            weight = group['weights']
            require(weight['num_bits'] == 8 and weight['type'] == 'int' and
                    weight['group_size'] == GROUP and weight['strategy'] == 'group' and
                    weight['symmetric'] is True and not weight.get('actorder') and
                    not weight.get('dynamic'), 'Unsupported W8 scheme')
            require(group.get('input_activations') is None and group.get('output_activations') is None,
                    'Activation quantization is prohibited')
    conversion = json_read(BASE / 'conversion-receipt.json')
    require(conversion['complete'] and not conversion['adapter_merge'] and
            conversion['edited_projection_count'] == 0 and conversion['vision_unchanged'] and
            conversion['quantized_linears'] == 11725, 'Baseline is not the unedited W8 export')
    summary, settings = json_read(OLD / 'pilot-summary.json'), json_read(OLD / 'pilot-settings.json')
    require(summary['adapter_path'] == str(ORIGINAL) and summary['selected_trial'] == 5,
            'Pilot identity changed')
    trial = next(t for t in json_read(OLD / 'trials.json') if t['number'] == summary['selected_trial'])
    parameters = trial['parameters']
    require(parameters['direction_index'] is None, 'This builder requires original per-layer directions')
    require(set(parameters['weight_distributions']) == {'attn.o_proj', 'mlp.down_proj'},
            'Unexpected component set')
    modifier = settings['modifier']['Abliteration']
    require(settings['quantization'] == 'bnb_4bit' and settings['seed'] == SEED and
            modifier['orthogonalize_direction'] and modifier['row_normalization'] == 'full' and
            modifier['full_normalization_lora_rank'] == RANK, 'Pilot method changed')
    adapter_config = json_read(STARTER / 'adapter_config.json')
    original_config = json_read(ORIGINAL / 'adapter_config.json')
    require({k: v for k, v in adapter_config.items() if k not in {'target_modules', 'base_model_name_or_path'}} ==
            {k: v for k, v in original_config.items() if k not in {'target_modules', 'base_model_name_or_path'}},
            'Starter adapter differs beyond normalized targets/base identity')
    require(adapter_config['base_model_name_or_path'] == str(BASE), 'Starter base is not W8')
    require(adapter_config['r'] == RANK and adapter_config['lora_alpha'] == ALPHA,
            'Rank or LoRA scaling changed')
    for field in ('rank_pattern', 'alpha_pattern', 'modules_to_save', 'target_parameters',
                  'use_dora', 'use_rslora', 'lora_bias', 'fan_in_fan_out'):
        require(not adapter_config.get(field), f'Unsupported LoRA option: {field}')
    require(adapter_config['bias'] == 'none' and adapter_config['peft_type'] == 'LORA',
            'Unsupported adapter type')
    directions = torch.load(OLD / 'refusal-directions.pt', map_location='cpu', weights_only=True)
    require(isinstance(directions, torch.Tensor) and directions.device.type == 'cpu' and
            directions.dtype == torch.float32 and tuple(directions.shape) == (31, 2816),
            'Unexpected saved direction tensor')
    require(torch.isfinite(directions).all().item(), 'Nonfinite saved directions')
    require(torch.count_nonzero(directions[0]).item() == 0, 'Unused embedding direction changed')
    require(torch.max(torch.abs(LA.vector_norm(directions[1:], dim=1) - 1)).item() < 2e-5,
            'Saved decoder directions are not unit vectors')
    for path in (VLLM / 'lora/lora_weights.py', VLLM / 'lora/utils.py', PEFT_UTILS, Path(__file__).resolve()):
        inputs[str(path)] = artifact(path)
    return inputs, config, conversion, parameters, adapter_config, directions


def inventory(reader, config, parameters, adapter_config):
    text = config['text_config']
    require((text['num_hidden_layers'], text['num_experts'], text['hidden_size'],
             text['moe_intermediate_size']) == (30, 128, 2816, 704), 'Architecture changed')
    projections, padding = {}, {}
    for layer in range(30):
        prefix = f'{LAYER_PREFIX}{layer}'
        modules = [(prefix + '.self_attn.o_proj', 'attn.o_proj'),
                   (prefix + '.mlp.down_proj', 'mlp.down_proj')]
        modules.extend((f'{prefix}.experts.{expert}.down_proj', 'mlp.down_proj')
                       for expert in range(128))
        for module, component in modules:
            shape = tuple(int(x) for x in reader.array(module + '.weight_shape'))
            require(len(shape) == 2 and shape[0] == 2816, f'Invalid projection shape: {module}')
            projections[module] = {'shape': shape, 'layer': layer, 'direction_row': layer + 1,
                                   'component': component,
                                   'coefficient': coefficient(layer, parameters['weight_distributions'][component])}
        for expert in range(128):
            for part in ('gate', 'up'):
                module = f'{prefix}.experts.{expert}.{part}_proj'
                shape = tuple(int(x) for x in reader.array(module + '.weight_shape'))
                require(shape == (704, 2816), f'Expert padding shape changed: {module}')
                padding[module] = shape
    all_shapes = {m: row['shape'] for m, row in projections.items()} | padding
    for module, (out_size, in_size) in all_shapes.items():
        expected = {'weight_packed': ('I32', [out_size, in_size // 4]),
                    'weight_scale': ('BF16', [out_size, in_size // GROUP]),
                    'weight_shape': ('I64', [2])}
        require(in_size % GROUP == 0, f'Invalid group boundary: {module}')
        for suffix, (dtype, shape) in expected.items():
            meta = reader.meta(module + '.' + suffix)
            require(meta['dtype'] == dtype and meta['shape'] == shape,
                    f'Packed tensor metadata mismatch: {module}.{suffix}')
        require(module + '.weight_zero_point' not in reader.entries and
                module + '.weight_g_idx' not in reader.entries, f'Unexpected quantizer: {module}')
    require(len(projections) == 3900 and len(padding) == 7680, 'Wrong target count')
    require(len(adapter_config['target_modules']) == len(all_shapes) and
            set(adapter_config['target_modules']) == set(all_shapes), 'Target-module coverage changed')
    active = Counter(row['component'] for row in projections.values() if row['coefficient'] != 0)
    require(dict(active) == {'attn.o_proj': 15, 'mlp.down_proj': 3096}, 'Active target range changed')
    return projections, padding


def load_starter(projections, padding):
    from safetensors.numpy import load_file
    arrays = load_file(str(STARTER / 'adapter_model.safetensors'))
    original_reader = TensorReader([ORIGINAL / 'adapter_model.safetensors'])
    try:
        original_keys = {key for module in projections for key in keys(module)}
        expected_keys = original_keys | {key for module in padding for key in keys(module)}
        require(set(arrays) == expected_keys and set(original_reader.entries) == original_keys,
                'Factor coverage changed')
        for module, shape in ({m: r['shape'] for m, r in projections.items()} | padding).items():
            out_size, in_size = shape
            for name, expected in zip(keys(module), [(RANK, in_size), (out_size, RANK)]):
                array = arrays[name]
                require(array.dtype == np.float32 and array.shape == expected and np.isfinite(array).all(),
                        f'Invalid starter factor: {name}')
                if name in original_keys:
                    require(array.tobytes() == original_reader.raw(name), f'Original factor changed: {name}')
                if module in padding or projections[module]['coefficient'] == 0:
                    require(not np.any(array) and not np.signbit(array).any(), f'Expected positive-zero factor: {name}')
                else:
                    require(np.any(array), f'Unexpected inactive source factor: {name}')
    finally:
        original_reader.close()
    return arrays


def reconstruct(reader, module):
    packed = reader.tensor(module + '.weight_packed')
    scales = reader.tensor(module + '.weight_scale')
    shape = tuple(int(x) for x in reader.array(module + '.weight_shape'))
    require(packed.device.type == scales.device.type == 'cpu', 'Non-CPU quantized input')
    require(torch.isfinite(scales).all().item() and (scales > 0).all().item(), 'Invalid W8 scale')
    signed = torch.empty(shape, dtype=torch.int16)
    for byte in range(4):
        signed[:, byte::4] = ((packed >> (8 * byte)) & 255).to(torch.int16) - 128
    weight = signed.float().reshape(shape[0], shape[1] // GROUP, GROUP)
    weight = (weight * scales.float().unsqueeze(-1)).reshape(shape).contiguous()
    require(torch.isfinite(weight).all().item(), 'Nonfinite reconstructed W8 weight')
    return weight


def factorize(weight, direction, strength):
    """Original full-normalization transformation and rank-3 SVD, on CPU."""
    require(weight.dtype == direction.dtype == torch.float32 and
            weight.device.type == direction.device.type == 'cpu', 'Non-CPU/F32 factorization')
    require(weight.shape[0] == direction.numel(), 'Direction/output dimension mismatch')
    row_norms = LA.vector_norm(weight, dim=1, keepdim=True)
    normalized = F.normalize(weight, p=2, dim=1)
    preliminary_a = (direction @ normalized).view(1, -1)
    preliminary_b = (-strength * direction).view(-1, 1)
    adjusted = normalized + preliminary_b @ preliminary_a
    adjusted = F.normalize(adjusted, p=2, dim=1)
    adjusted = adjusted * row_norms
    delta = adjusted - weight
    # This seeds only the CPU generator, unlike torch.manual_seed's CUDA wrapper.
    torch.random.default_generator.manual_seed(SEED)
    u, singular, v = torch.svd_lowrank(delta, q=2 * RANK + 4, niter=6)
    sqrt_s = torch.sqrt(singular[:RANK])
    b = (u[:, :RANK] @ torch.diag(sqrt_s)).contiguous()
    a = (torch.diag(sqrt_s) @ v[:, :RANK].T).contiguous()
    require(a.shape == (RANK, weight.shape[1]) and b.shape == (weight.shape[0], RANK),
            'Wrong factor orientation')
    require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), 'Nonfinite factors')
    reconstruction = b @ a
    denominator = max(LA.vector_norm(delta).item(), 1e-30)
    relative_error = LA.vector_norm(delta - reconstruction).item() / denominator
    row_error = torch.max(torch.abs(LA.vector_norm(adjusted, dim=1, keepdim=True) - row_norms) /
                          row_norms.clamp_min(1e-30)).item()
    require(relative_error <= 1.0001 and row_error < 2e-5, 'Invalid low-rank/norm-preserving construction')
    return a, b, {'delta_frobenius': denominator, 'rank3_relative_frobenius_error': relative_error,
                  'full_delta_max_relative_row_norm_error': row_error,
                  'rank3_delta_to_weight_frobenius_ratio': LA.vector_norm(reconstruction).item() /
                      max(LA.vector_norm(weight).item(), 1e-30)}


def compile_nodes(nodes, path, namespace):
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                            *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)


def reference_factors(weight, directions, parameters, layer, component):
    """Run the pinned original method, with only CUDA seeding intercepted.

    No Heretic/model/PEFT/bitsandbytes package is imported or instantiated.
    The original method sees ordinary F32 CPU weights, allowing exact comparison
    of the shared projection, interpolation, factor orientation and SVD logic.
    """
    cls = next(n for n in ast.parse(REFERENCE.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'Abliteration')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'modify_model')
    calls = []
    class CpuTorch:
        cuda = SimpleNamespace(manual_seed_all=lambda seed: calls.append(('cuda_seed_intercepted', seed)))
        @staticmethod
        def manual_seed(seed):
            calls.append(('cpu_seed', seed))
            return torch.random.default_generator.manual_seed(seed)
        def __getattr__(self, name):
            return getattr(torch, name)
    namespace = {'torch': CpuTorch(), 'F': F, 'LA': LA, 'math': math,
                 'cast': lambda annotation, value: value, 'Linear': object, 'Tensor': torch.Tensor,
                 'RowNormalization': SimpleNamespace(NONE='none', PRE='pre', FULL='full')}
    compile_nodes([method], REFERENCE, namespace)
    a, b = torch.zeros((RANK, weight.shape[1])), torch.zeros((weight.shape[0], RANK))
    module = SimpleNamespace(weight=weight, base_layer=SimpleNamespace(weight=weight),
                             lora_A={'default': SimpleNamespace(weight=a)},
                             lora_B={'default': SimpleNamespace(weight=b)})
    model = SimpleNamespace(get_layers=lambda: range(30),
                            get_layer_modules=lambda index: {component: [module]} if index == layer else {},
                            peft_config=SimpleNamespace(r=RANK))
    self = SimpleNamespace(residual_directions=directions, settings=SimpleNamespace(row_normalization='full'),
                           heretic_settings=SimpleNamespace(seed=SEED))
    params = SimpleNamespace(direction_index=parameters['direction_index'],
                              weight_distributions={k: SimpleNamespace(**v) for k, v in parameters['weight_distributions'].items()})
    namespace['modify_model'](self, SimpleNamespace(get_model=lambda: model), params)
    require(calls in ([], [('cpu_seed', SEED), ('cuda_seed_intercepted', SEED)]), 'Unexpected reference seeding')
    return a, b


def independent_checks(reader, projections, parameters, directions):
    reports = []
    # Independent float64 matrix expression (not the factor construction):
    # D * row_normalize((I - lambda*v*v^T) * D^-1 * W) - W.
    grid = np.arange(13 * 17, dtype=np.float64).reshape(13, 17)
    synthetic = (np.sin(grid * .17) + .3 * np.cos(grid * .031)).astype(np.float32)
    v = np.linspace(-.9, 1.1, 13).astype(np.float32)
    v = (v / np.linalg.norm(v)).astype(np.float32)
    small_directions = torch.from_numpy(np.tile(v, (31, 1)).copy())
    for layer, component in [(15, 'attn.o_proj'), (23, 'mlp.down_proj'), (0, 'attn.o_proj')]:
        strength = coefficient(layer, parameters['weight_distributions'][component])
        weight = torch.from_numpy(synthetic.copy())
        if strength == 0:
            a, b = reference_factors(weight, small_directions, parameters, layer, component)
            require(torch.count_nonzero(a).item() == torch.count_nonzero(b).item() == 0,
                    'Original inactive layer is not identity')
            reports.append({'fixture': 'inactive_layer', 'layer': layer, 'component': component, 'passed': True})
            continue
        a, b, diag = factorize(weight, small_directions[layer + 1], strength)
        ar, br = reference_factors(weight, small_directions, parameters, layer, component)
        require(torch.equal(a, ar) and torch.equal(b, br), 'Factor construction differs from original method')
        a2, b2, _ = factorize(weight, small_directions[layer + 1], strength)
        require(torch.equal(a, a2) and torch.equal(b, b2), 'CPU factors are not deterministic')
        w64, v64 = synthetic.astype(np.float64), v.astype(np.float64)
        norms = np.linalg.norm(w64, axis=1, keepdims=True)
        normalized = w64 / np.maximum(norms, 1e-12)
        projected = (np.eye(13) - strength * np.outer(v64, v64)) @ normalized
        expected = projected / np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12) * norms - w64
        # Exact SVD gives an independent optimal-rank-3 product for this fixture.
        u, s, vh = np.linalg.svd(expected, full_matrices=False)
        rank3 = (u[:, :RANK] * s[:RANK]) @ vh[:RANK]
        got = b.double().numpy() @ a.double().numpy()
        relative = float(np.linalg.norm(got - rank3) / np.linalg.norm(rank3))
        require(relative < 3e-4, f'Independent rank-3 product mismatch: {relative}')
        x = np.linspace(-1, 1, 17)
        require(np.allclose((ALPHA / RANK) * b.double().numpy() @ (a.double().numpy() @ x),
                            rank3 @ x, rtol=5e-4, atol=3e-5), 'LoRA scaling/orientation mismatch')
        # This identity catches an incorrect projection sign before renormalization.
        require(np.allclose(v64 @ projected, (1 - strength * (v64 @ v64)) * (v64 @ normalized),
                            rtol=1e-11, atol=1e-11), 'Projection sign identity failed')
        reports.append({'fixture': 'independent_double_svd', 'layer': layer, 'component': component,
                        'coefficient': strength, 'original_method_factors_bit_exact': True,
                        'repeat_factors_bit_exact': True, 'independent_rank3_product_relative_error': relative,
                        'alpha_over_rank': ALPHA / RANK, **diag})
    module = f'{LAYER_PREFIX}6.experts.0.down_proj'
    started = time.monotonic()
    weight = reconstruct(reader, module)
    # A separate byte-oriented NumPy decoder verifies every real INT8 code.
    raw = reader.raw(module + '.weight_packed')
    codes = np.frombuffer(raw, dtype=np.uint8).astype(np.int16).reshape(weight.shape) - 128
    scale_bits = reader.array(module + '.weight_scale').astype(np.uint32) << 16
    scales = scale_bits.view(np.float32)
    independent = codes.astype(np.float32).reshape(weight.shape[0], -1, GROUP) * scales[:, :, None]
    require(np.array_equal(weight.numpy(), independent.reshape(weight.shape)), 'Independent W8 decoding mismatch')
    strength = projections[module]['coefficient']
    a, b, diag = factorize(weight, directions[7], strength)
    ar, br = reference_factors(weight, directions, parameters, 6, 'mlp.down_proj')
    require(torch.equal(a, ar) and torch.equal(b, br), 'Real W8 projection differs from original method')
    a2, b2, _ = factorize(weight, directions[7], strength)
    require(torch.equal(a, a2) and torch.equal(b, b2), 'Real W8 factorization not deterministic')
    reports.append({'fixture': 'real_w8_expert_projection', 'module': module,
                    'shape': list(weight.shape), 'coefficient': strength,
                    'direction_row': 7, 'decoded_values_bit_exact': True,
                    'original_method_factors_bit_exact': True, 'repeat_factors_bit_exact': True,
                    'weight_f32_sha256': hashlib.sha256(weight.numpy().tobytes()).hexdigest(),
                    'a_sha256': hashlib.sha256(a.numpy().tobytes()).hexdigest(),
                    'b_sha256': hashlib.sha256(b.numpy().tobytes()).hexdigest(),
                    'three_factorizations_and_decode_seconds': time.monotonic() - started, **diag})
    return reports


def loader_checks(arrays, targets):
    tree = ast.parse(LOADER_CHECKS.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
             node.name in {'compile_nodes', 'keys', 'isolated_loader_checks'}]
    require(len(nodes) == 3, 'Compatibility helper changed')
    namespace = {'ast': ast, 're': re, 'SimpleNamespace': SimpleNamespace, 'require': require,
                 'artifact': artifact, 'BASE_PREFIX': BASE_PREFIX, 'LAYER_PREFIX': LAYER_PREFIX,
                 'VLLM': VLLM, 'PEFT_UTILS': PEFT_UTILS}
    compile_nodes(nodes, LOADER_CHECKS, namespace)
    previous_threads = torch.get_num_threads()
    try:
        return namespace['isolated_loader_checks'](arrays, targets, RANK, ALPHA)
    finally:
        torch.set_num_threads(previous_threads)


def common_receipt(inputs, reader, projections, padding, parameters, directions, args):
    return {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'label': 'NF4-direction / W8-weight rebased strength 1.0', 'model_id': MODEL_ID,
        'builder': artifact(Path(__file__).resolve()), 'output_directory': str(OUTPUT),
        'inputs': inputs, 'baseline_safetensors': reader.files,
        'direction_origin': 'Exact saved original NF4 pilot directions; no new activation extraction.',
        'direction_tensor_sha256': hashlib.sha256(directions.numpy().tobytes()).hexdigest(),
        'direction_indexing': 'Row 0 is the original unused zero embedding direction; decoder layer i uses the unchanged unit direction at row i+1.',
        'selected_trial': 5, 'selected_parameters': parameters, 'external_strength_multiplier': 1.0,
        'weight_reconstruction': 'Decode offset-128 packed INT8; multiply each code by its stored BF16 group32 scale in FP32. No extra BF16 rounding in CPU reconstruction.',
        'factorization': {'dtype': 'float32', 'device': 'cpu', 'row_normalization': 'full',
                          'rank': RANK, 'lora_alpha': ALPHA, 'alpha_over_rank': ALPHA / RANK,
                          'svd_lowrank_q': 2 * RANK + 4, 'svd_lowrank_niter': 6, 'seed_per_projection': SEED,
                          'cpu_threads': args.threads, 'deterministic_algorithms': True},
        'coverage': {'projections': len(projections), 'active_rebuilt': 3111, 'inactive_preserved_zero': 789,
                     'attention_active': 15, 'shared_mlp_active': 24, 'expert_down_active': 3072,
                     'expert_gate_up_zero_padding': len(padding), 'total_factor_tensors': 23160},
        'software': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__},
        'limitations': [
            'NF4-derived directions are retained; this is not fresh W8 activation extraction.',
            'Historical factors were built on GPU. Same seed/q/niter on CPU does not guarantee the same random SVD basis or floating-point rounding.',
            'Rank-3 approximation does not exactly preserve every row norm.',
            'Runtime casts FP32 factors to BF16. Native Marlin arithmetic is not claimed bit-identical to CPU reconstruction.',
            'Unmerged W8+LoRA is not equivalent to a merged/requantized checkpoint.',
            'No model behavior, capability gain or runtime adapter acceptance is established by this CPU receipt.',
        ],
        'runtime_constraints': ['unchanged W8 baseline and profile', 'fully_sharded_loras=false',
                                'expert gate/up factors remain exactly zero', 'new distinct candidate ID'],
        'scope': {'model_loads': 0, 'inference_calls': 0, 'service_actions': 0, 'gpu_initialized': False,
                  'checkpoint_exports': 0, 'source_or_manifest_writes': 0},
        'resource_estimate': {'output_tensor_bytes': 491112960, 'largest_single_f32_matrix_bytes': 88 * 2**20,
                              'suggested_cpu_quota_percent': 400, 'suggested_memory_high_gib': 6,
                              'suggested_memory_max_gib': 8,
                              'note': 'Sequential projection processing; full build also hashes the baseline shards once. Runtime estimate is unmeasured.'},
    }


def assert_unchanged(inputs, baseline_stats):
    for name, expected in inputs.items():
        require(artifact(Path(name)) == expected, f'Immutable input changed during run: {name}')
    for name, expected in baseline_stats.items():
        require(stat_identity(Path(name)) == expected['stat'], f'Baseline shard changed during run: {name}')
    require(not torch.cuda.is_initialized(), 'CUDA initialized during CPU work')
    require(not any(name in sys.modules for name in ('transformers', 'vllm', 'peft', 'bitsandbytes', 'heretic')),
            'A model/runtime framework was unexpectedly imported')


def build(reader, projections, padding, arrays, inputs, receipt, conversion):
    # Whole-file hashes bind the actual tensor bytes to the frozen baseline,
    # beyond the lighter metadata/sample validation performed by preflight.
    verified = []
    for shard in conversion['shards']:
        path = BASE / shard['file']
        require(path.stat().st_size == shard['bytes'] and sha256(path) == shard['sha256'],
                f'Baseline shard hash mismatch: {path}')
        verified.append({'file': shard['file'], 'sha256': shard['sha256'], 'bytes': shard['bytes']})
    OUTPUT.mkdir(exist_ok=False)
    json_new(OUTPUT / 'build-plan.json', receipt | {'status': 'building', 'complete': False})
    direction = torch.load(OLD / 'refusal-directions.pt', map_location='cpu', weights_only=True)
    diagnostics = []
    began = time.monotonic()
    for index, (module, row) in enumerate(projections.items(), 1):
        if row['coefficient'] == 0:
            continue
        weight = reconstruct(reader, module)
        a, b, diag = factorize(weight, direction[row['direction_row']], row['coefficient'])
        for name, factor in zip(keys(module), (a, b)):
            arrays[name] = factor.numpy().copy()
        diagnostics.append({'module': module, 'shape': list(row['shape']),
                            'coefficient': row['coefficient'], 'direction_row': row['direction_row'], **diag})
        del weight, a, b
        if len(diagnostics) % 128 == 0:
            print(json.dumps({'stage': 'rebuilding_factors', 'active_done': len(diagnostics),
                              'active_total': 3111, 'seconds': time.monotonic() - began}), flush=True)
    require(len(diagnostics) == 3111, 'Missing active factors')
    receipt['installed_loader_checks'] = loader_checks(arrays, json_read(STARTER / 'adapter_config.json')['target_modules'])
    # Preserve all zero padding/inactive factors exactly and prove changed scope.
    original_reader = TensorReader([STARTER / 'adapter_model.safetensors'])
    changed = []
    try:
        for module in list(projections) + list(padding):
            active = module in projections and projections[module]['coefficient'] != 0
            for name in keys(module):
                value = arrays[name]
                require(value.dtype == np.float32 and np.isfinite(value).all(), f'Invalid output: {name}')
                before = original_reader.raw(name)
                if not active:
                    require(value.tobytes() == before, f'Inactive/padding factor changed: {name}')
                elif value.tobytes() != before:
                    changed.append(name)
        require(len(changed) == 6222, 'Expected both factors of every active projection to be rebuilt')
    finally:
        original_reader.close()
    from safetensors.numpy import save_file
    temporary = OUTPUT / 'adapter_model.safetensors.incomplete'
    require(not temporary.exists(), 'Unexpected incomplete output')
    save_file(arrays, str(temporary), metadata={'format': 'pt'})
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    saved = TensorReader([temporary])
    try:
        require(set(saved.entries) == set(arrays), 'Serialized tensor names changed')
        for name, expected in arrays.items():
            require(saved.meta(name)['dtype'] == 'F32' and saved.meta(name)['shape'] == list(expected.shape) and
                    saved.raw(name) == expected.tobytes(), f'Serialized tensor changed: {name}')
    finally:
        saved.close()
    # Hard-link publication fails if a final filename already exists.
    os.link(temporary, OUTPUT / 'adapter_model.safetensors')
    temporary.unlink()
    write_new(OUTPUT / 'adapter_config.json', (STARTER / 'adapter_config.json').read_bytes())
    json_new(OUTPUT / 'projection-validation.json', diagnostics)
    assert_unchanged(inputs, reader.files)
    receipt.update({'status': 'completed_cpu_builder_validation', 'complete': True,
                    'baseline_shards_fully_sha256_verified': verified,
                    'changed_active_factor_tensors': len(changed), 'unchanged_factor_tensors': 23160 - len(changed),
                    'zero_padding_proof': 'All 7680 added expert gate/up projections retain finite positive FP32 zero A and B; alpha/r=1, so B@A and B@(A@x) are zero. Installed pack_moe probes cover all 128 experts in all 30 layers.',
                    'original_inputs_and_manifests_unchanged': True,
                    'output_artifacts': [artifact(OUTPUT / name) for name in ('adapter_model.safetensors', 'adapter_config.json', 'projection-validation.json')],
                    'candidate_identity': {'path': str(OUTPUT), 'model_id': MODEL_ID, 'base_model_name_or_path': str(BASE)},
                    'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    'build_factor_seconds': time.monotonic() - began})
    json_new(OUTPUT / 'provenance-validation.json', receipt)
    directory_fd = os.open(OUTPUT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(json.dumps({'status': receipt['status'], 'output': str(OUTPUT),
                      'receipt_sha256': sha256(OUTPUT / 'provenance-validation.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--build', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    require(1 <= args.threads <= 8, 'Use 1 to 8 CPU threads')
    require(not os.path.lexists(OUTPUT), f'Refusing existing candidate directory: {OUTPUT}')
    if args.preflight:
        require(not os.path.lexists(PREFLIGHT), f'Refusing existing preflight receipt: {PREFLIGHT}')
    else:
        require(PREFLIGHT.is_file(), 'Run --preflight before --build')
    cpu_setup(args.threads)
    inputs, config, conversion, parameters, adapter_config, directions = load_inputs()
    index = json_read(BASE / 'model.safetensors.index.json')['weight_map']
    shard_names = {row['file'] for row in conversion['shards']}
    require(set(index.values()) == shard_names and all(Path(name).name == name for name in shard_names),
            'Index/receipt shard coverage mismatch')
    reader = TensorReader([BASE / name for name in shard_names])
    try:
        require(set(reader.entries) == set(index), 'Index/header tensor coverage mismatch')
        require(all(reader.entries[name][3].name == shard for name, shard in index.items()), 'Index shard mapping mismatch')
        projections, padding = inventory(reader, config, parameters, adapter_config)
        arrays = load_starter(projections, padding)
        receipt = common_receipt(inputs, reader, projections, padding, parameters, directions, args)
        if args.build:
            preflight = json_read(PREFLIGHT)
            require(preflight['complete'] and preflight['status'] == 'passed_cpu_preflight', 'Preflight did not pass')
            for key in ('builder', 'inputs', 'baseline_safetensors', 'selected_parameters', 'direction_tensor_sha256', 'factorization', 'software'):
                require(preflight[key] == receipt[key], f'Preflight identity/config changed: {key}')
            receipt['preflight'] = artifact(PREFLIGHT)
            receipt['independent_numeric_checks'] = preflight['independent_numeric_checks']
            build(reader, projections, padding, arrays, inputs, receipt, conversion)
        else:
            receipt['independent_numeric_checks'] = independent_checks(reader, projections, parameters, directions)
            receipt['installed_loader_checks'] = loader_checks(arrays, adapter_config['target_modules'])
            assert_unchanged(inputs, reader.files)
            receipt.update({'status': 'passed_cpu_preflight', 'complete': True,
                            'full_candidate_built': False, 'baseline_full_shard_hash_verification': 'Deferred to --build; preflight binds headers/metadata and independently decodes one real projection.',
                            'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
            json_new(PREFLIGHT, receipt)
            print(json.dumps({'status': receipt['status'], 'receipt': str(PREFLIGHT),
                              'receipt_sha256': sha256(PREFLIGHT), 'max_rss_bytes': receipt['max_rss_bytes']}), flush=True)
    finally:
        reader.close()


if __name__ == '__main__':
    main()
