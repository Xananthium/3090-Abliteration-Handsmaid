#!/usr/bin/env python3
"""Evaluate previously seen benign development cases on private BF16 base/LoRA models.

Two sequential repeats retain full text and the frozen strict correctness and
completion rules. This runner never changes models, adapters, or services.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
import types
import urllib.request


WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
BASE = 'http://127.0.0.1:8002'
CONTEXT_LIMIT = 8192
BUDGETS = [512, 1024, 2048]
REPEATS = [1, 2]
REQUEST_TIMEOUT = 1800
EVIDENCE_STATUS = 'previously seen development diagnostics; not new held-out evidence'
SOURCE_CASES = ROOT / 'broader-validation/capability-cases-v1.json'
HELPER_PATH = ROOT / 'evaluate_markers.py'
FROZEN = {
    HELPER_PATH: 'a0feb6cf923dca4e5fd5330d8a1cc24d10d51ddff0912db4454f23a70108b9bb',
    SOURCE_CASES: '34bdac9e83f9651a2cddbef6e2b8dcbe9eedffa3e758bc2d2402421f97c1f5ba',
}


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_stat(path):
    stat = path.stat()
    return {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'device': stat.st_dev, 'inode': stat.st_ino}


def verify_frozen():
    for path, expected in FROZEN.items():
        if file_sha256(path) != expected:
            raise RuntimeError(f'Frozen input changed: {path}')


def load_inputs(case_ids=None):
    verify_frozen()
    helper = types.ModuleType('_frozen_bf16_integrity_helpers')
    helper.__file__ = str(HELPER_PATH)
    # Definitions only: no classifier, main(), inference, or bytecode write.
    exec(compile(HELPER_PATH.read_bytes(), str(HELPER_PATH), 'exec'), helper.__dict__)
    authored = [case for case in json.loads(SOURCE_CASES.read_text())
                if case['id'].startswith('broader-fresh-')]
    counts = Counter(case['category'] for case in authored)
    if (len(authored) != 24 or len({case['id'] for case in authored}) != 24
            or counts != dict.fromkeys(('reasoning', 'instruction', 'consistency', 'recovery'), 6)):
        raise RuntimeError('Expected the frozen authored24 set with six cases per category')
    if case_ids:
        if len(case_ids) != len(set(case_ids)):
            raise ValueError('--case-id values must be unique')
        unknown = set(case_ids) - {case['id'] for case in authored}
        if unknown:
            raise ValueError(f'Unknown authored case IDs: {sorted(unknown)}')
        selected = [case for case in authored if case['id'] in set(case_ids)]
    else:
        selected = authored
    return helper, authored, selected


def checkpoint_identity(path, helper):
    identity = helper.checkpoint_identity(path)
    if not identity['shards'] or 'config.json' not in identity['files']:
        raise ValueError('Expected checkpoint requires config.json and safetensors shards')
    config = json.loads((Path(identity['path']) / 'config.json').read_text())
    for section in [config, config.get('text_config', {}), config.get('vision_config') or {}]:
        if section.get('quantization_config'):
            raise ValueError('BF16 integrity evaluation requires an unquantized base checkpoint')
        dtype = section.get('dtype', section.get('torch_dtype'))
        if dtype is not None and dtype != 'bfloat16':
            raise ValueError(f'Expected BF16 checkpoint configuration, found {dtype!r}')
    receipt = Path(identity['path']) / 'source-receipt.json'
    if receipt.is_file():
        identity['files'][receipt.name] = {'sha256': file_sha256(receipt)}
    return identity


def adapter_identity(path, checkpoint):
    if path is None:
        return None
    directory = Path(path).resolve(strict=True)
    config_path = directory / 'adapter_config.json'
    weights_path = directory / 'adapter_model.safetensors'
    config = json.loads(config_path.read_text())
    if config.get('peft_type') != 'LORA':
        raise ValueError('--adapter-dir must contain a LoRA adapter')
    base_path = config.get('base_model_name_or_path')
    if not base_path or str(Path(base_path).resolve(strict=True)) != checkpoint['path']:
        raise ValueError('Adapter configuration does not identify the expected base checkpoint')
    return {'path': str(directory), 'base_checkpoint_path': checkpoint['path'],
            'files': {item.name: {'sha256': file_sha256(item), **file_stat(item)}
                      for item in (config_path, weights_path)}}


def verify_adapter_stats(identity):
    if identity is None:
        return
    for name, expected in identity['files'].items():
        current = file_stat(Path(identity['path']) / name)
        if any(current[key] != expected[key] for key in current):
            raise RuntimeError('Adapter files changed during the diagnostic')


def flag_values(argv, flag):
    values = []
    for index, token in enumerate(argv):
        if token.startswith(flag + '='):
            values.append(token[len(flag) + 1:])
        elif token == flag:
            for value in argv[index + 1:]:
                if value.startswith('--'):
                    break
                values.append(value)
    return values


def launch_identity(path, checkpoint, adapter, helper):
    path = Path(path).resolve(strict=True)
    profile = json.loads(path.read_text())
    argv = profile.get('args')
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError('Launch profile must contain its argument list in args')
    try:
        served_path = argv[argv.index('serve') + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError('Launch profile does not identify a vLLM serve checkpoint') from exc
    if str(Path(served_path).resolve(strict=True)) != checkpoint['path']:
        raise ValueError('Launch profile uses a different base checkpoint')
    required = {'--host': ['127.0.0.1'], '--port': ['8002'], '--dtype': ['bfloat16'],
                '--kv-cache-dtype': ['bfloat16'], '--max-model-len': [str(CONTEXT_LIMIT)]}
    if any(flag_values(argv, flag) != values for flag, values in required.items()):
        raise ValueError('Launch profile must describe private port 8002, BF16 weights/KV, and context 8192')
    if (any(item == '--quantization' or item.startswith('--quantization=') for item in argv)
            or any('QUANT' in key for key in profile.get('environment', {}))):
        raise ValueError('Launch profile contains a quantization setting')
    base_ids = flag_values(argv, '--served-model-name')
    if not base_ids:
        raise ValueError('Launch profile must explicitly identify its served base model name')
    if adapter is not None and '--enable-lora' not in argv:
        raise ValueError('An adapter run requires a launch profile with --enable-lora')
    return {'path': str(path), 'sha256': file_sha256(path),
            'canonical_fingerprint': helper.fingerprint(profile), 'base_model_ids': base_ids,
            'attestation_scope': 'Hashes the supplied launch record; endpoint checks validate base/adapter roots and context, not the full running process command.'}


def validate_model_listing(payload, model_id, checkpoint, adapter, launch):
    models = payload.get('data')
    if (not isinstance(models, list) or not all(isinstance(model, dict) for model in models)
            or not all(isinstance(model.get('id'), str) for model in models)):
        raise RuntimeError('Private endpoint returned an invalid model listing')
    by_id = {model['id']: model for model in models}
    if len(by_id) != len(models) or model_id not in by_id:
        raise RuntimeError('Requested model is absent or model IDs are duplicated')
    selected = by_id[model_id]
    if adapter is None:
        base_model = selected
        if selected.get('parent') is not None:
            raise RuntimeError('Base run selected an adapter model')
    else:
        if selected.get('root') != adapter['path']:
            raise RuntimeError('Requested LoRA model root differs from --adapter-dir')
        parent = selected.get('parent')
        if parent not in by_id:
            raise RuntimeError('LoRA model listing does not identify its advertised base model')
        base_model = by_id[parent]
    if (base_model.get('root') != checkpoint['path'] or base_model.get('parent') is not None
            or base_model['id'] not in launch['base_model_ids']):
        raise RuntimeError('Advertised base checkpoint or model ID differs from the launch record')
    if type(base_model.get('max_model_len')) is not int or base_model['max_model_len'] != CONTEXT_LIMIT:
        raise RuntimeError('Base model must advertise the expected 8192-token context')
    # vLLM LoRA cards normally omit max_model_len and inherit the parent limit.
    if selected.get('max_model_len') not in (None, CONTEXT_LIMIT):
        raise RuntimeError('Requested model advertises an incompatible context')
    keys = ('id', 'root', 'parent', 'max_model_len')
    return {'requested': {key: selected.get(key) for key in keys},
            'base': {key: base_model.get(key) for key in keys}, 'context_limit': CONTEXT_LIMIT}


def fetch_models():
    with urllib.request.urlopen(BASE + '/v1/models', timeout=30) as handle:
        return json.load(handle)


def row_identity(case, config, run_fingerprint, helper):
    return {'id': case['id'], 'category': case['category'],
            'configuration_fingerprint': run_fingerprint,
            'protocol_fingerprint': config['protocol_fingerprint'],
            'base_checkpoint_fingerprint': helper.fingerprint(config['checkpoint']),
            'adapter_fingerprint': helper.fingerprint(config['adapter']) if config['adapter'] else None,
            'launch_profile_sha256': config['launch_profile']['sha256'],
            'model_id': config['model_id'],
            'selected_cases_fingerprint': config['protocol']['selected_cases_fingerprint'],
            'case_sha256': helper.fingerprint(case),
            'prompt_sha256': helper.digest_bytes(case['prompt'].encode()),
            'answer_key_sha256': helper.fingerprint({'kind': case['kind'], 'answer': case['answer']})}


def read_rows(path, config, run_fingerprint, cases, helper):
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    schedule = [(case, repeat) for repeat in REPEATS for case in cases]
    if len(rows) > len(schedule):
        raise RuntimeError('Too many cached records')
    for row, (case, repeat) in zip(rows, schedule):
        expected = row_identity(case, config, run_fingerprint, helper) | {'repeat': repeat}
        if any(row.get(key) != value for key, value in expected.items()):
            raise RuntimeError('Cached record has incompatible case/repeat, model, adapter, profile, or protocol identity')
        if row.get('record_sha256') != helper.fingerprint(
                {key: value for key, value in row.items() if key != 'record_sha256'}):
            raise RuntimeError('Cached record hash is invalid')
        answer, raw = row['answer_text'], row['raw_final_text']
        if (not isinstance(answer, str) or not isinstance(raw, str) or raw.strip() != answer
                or row['response_sha256'] != helper.digest_bytes(answer.encode())
                or row['raw_final_text_sha256'] != helper.digest_bytes(raw.encode())
                or row['attempts'][-1]['response_sha256'] != row['response_sha256']
                or row['response_attempts'][-1]['raw_text'] != raw
                or row['planned_budgets'] != BUDGETS):
            raise RuntimeError('Cached text, response hash, or completion budgets are inconsistent')
        if len(row['response_attempts']) != len(row['attempts']):
            raise RuntimeError('Cached full-text attempts do not match completion metadata')
        for response, metadata in zip(row['response_attempts'], row['attempts']):
            if (helper.digest_bytes(response['raw_text'].encode()) != response['raw_text_sha256']
                    or helper.digest_bytes(response['raw_text'].strip().encode()) != metadata['response_sha256']
                    or response['model_id'] != config['model_id']
                    or response['requested_max_tokens'] != metadata['requested_max_tokens']):
                raise RuntimeError('Cached attempt text or model identity is inconsistent')
        score = helper.correct(case, answer)
        if (type(row['generation_complete']) is not bool
                or row['generation_complete'] != (not row['unresolved_reasons'])
                or row['scored_correct_before_completion_gate'] is not score
                or row['correct'] is not (score and row['generation_complete'])):
            raise RuntimeError('Cached score disagrees with the frozen strict grader/completion gate')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--expected-checkpoint', required=True, help='Absolute base BF16 checkpoint path')
    parser.add_argument('--model-id', required=True, help='Advertised base or LoRA ID sent in native requests')
    parser.add_argument('--adapter-dir', help='Optional absolute LoRA directory; omitted for a base-model run')
    parser.add_argument('--launch-profile', required=True, help='JSON launch record for the already running server')
    parser.add_argument('--case-id', action='append', help='Optional exact authored case ID; repeat for a subset')
    args = parser.parse_args()
    # --help exits before inputs, locks, or HTTP requests.
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.label):
        parser.error('--label must be a filename-safe identifier')
    if not args.model_id.strip() or not Path(args.expected_checkpoint).is_absolute():
        parser.error('Provide a nonempty --model-id and absolute --expected-checkpoint')
    if args.adapter_dir and not Path(args.adapter_dir).is_absolute():
        parser.error('--adapter-dir must be absolute')
    helper, authored, cases = load_inputs(args.case_id)
    helper.MODEL = args.model_id
    checkpoint = checkpoint_identity(args.expected_checkpoint, helper)
    adapter = adapter_identity(args.adapter_dir, checkpoint)
    launch = launch_identity(args.launch_profile, checkpoint, adapter, helper)
    lock_path = WORK / 'runtime.lock'
    # The parent controller must hold this same lock while changing/restarting
    # the runtime. Open the existing file read-only and never replace it.
    with lock_path.open('rb') as runtime_lock:
        try:
            fcntl.flock(runtime_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('BF16 runtime is busy; refusing to overlap evaluation or restart') from exc
        lock_stat = os.fstat(runtime_lock.fileno())
        listing = fetch_models()
        served = validate_model_listing(listing, args.model_id, checkpoint, adapter, launch)
        protocol = {
            'version': 1, 'script_sha256': file_sha256(Path(__file__)),
            'frozen_evaluator_version': helper.PROTOCOL_VERSION,
            'frozen_input_sha256': {str(path): digest for path, digest in FROZEN.items()},
            'evidence_status': EVIDENCE_STATUS, 'authored_case_count': len(authored),
            'case_order': [case['id'] for case in cases],
            'selected_cases_fingerprint': helper.fingerprint(cases),
            'repeats': REPEATS, 'output_budgets': BUDGETS, 'max_workers': 1,
            'budget_semantics': 'Retry only max_tokens stops; not three independent budget experiments.',
            'system': helper.CAPABILITY_SYSTEM, 'temperature': 0, 'thinking': False,
            'chat_template_kwargs': {'enable_thinking': False},
            'context_limit': CONTEXT_LIMIT, 'context_margin': helper.CONTEXT_MARGIN,
            'request_timeout_seconds': REQUEST_TIMEOUT,
            'scoring': 'Unchanged frozen correct(case, text) AND generation_complete.',
            'text_retention': 'Full raw text of every attempt plus frozen helper final text after outer-whitespace stripping.',
            'timing_scope': 'Standard helper seconds include identity guards; native_request_seconds measure HTTP calls separately. Neither is a throughput benchmark.',
        }
        config = {'label': args.label, 'endpoint': BASE, 'model_id': args.model_id,
                  'checkpoint': checkpoint, 'adapter': adapter, 'launch_profile': launch,
                  'served_identity': served, 'runtime_lock': str(lock_path),
                  'protocol': protocol, 'protocol_fingerprint': helper.fingerprint(protocol)}
        answers_path = WORK / f'{args.label}-answers.jsonl'
        summary_path = WORK / f'{args.label}-summary.json'
        fingerprint = helper.ensure_manifest(WORK / f'{args.label}-config.json', config,
                                             [answers_path, summary_path])
        rows = read_rows(answers_path, config, fingerprint, cases, helper)
        total = len(cases) * len(REPEATS)
        if summary_path.exists():
            previous = json.loads(summary_path.read_text())
            if (previous.get('configuration_fingerprint') != fingerprint
                    or previous.get('answers_sha256') != file_sha256(answers_path)
                    or previous.get('record_count') != len(rows) or len(rows) != total):
                raise RuntimeError('Cached summary does not match complete compatible results')
            print(json.dumps({'stage': 'already_complete', 'label': args.label,
                              'summary': str(summary_path), 'evidence_status': EVIDENCE_STATUS}), flush=True)
            return

        def guard():
            current_lock = lock_path.stat()
            if (current_lock.st_dev, current_lock.st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
                raise RuntimeError('Runtime lock file was replaced during evaluation')
            if file_sha256(Path(launch['path'])) != launch['sha256']:
                raise RuntimeError('Launch profile changed during evaluation')
            if checkpoint_identity(checkpoint['path'], helper) != checkpoint:
                raise RuntimeError('Base checkpoint identity changed during evaluation')
            verify_adapter_stats(adapter)
            if validate_model_listing(fetch_models(), args.model_id, checkpoint, adapter, launch) != served:
                raise RuntimeError('Requested base/adapter serving identity changed during evaluation')

        original_request = helper.request_json
        captured = []
        native_requests = []

        def guarded_request(base, route, payload, timeout):
            if (base != BASE or route not in ('/v1/messages/count_tokens', '/v1/messages')
                    or payload.get('model') != args.model_id):
                raise RuntimeError('Attempted an unexpected endpoint or model')
            guard()
            began = time.monotonic()
            result = original_request(base, route, payload, timeout)
            seconds = time.monotonic() - began
            native_requests.append({'route': route, 'seconds': seconds})
            guard()
            if route == '/v1/messages':
                if result.get('model') != args.model_id:
                    raise RuntimeError('Response does not identify the requested base/LoRA model')
                raw = ''.join(block.get('text', '') for block in result['content'] if block.get('type') == 'text')
                captured.append({'raw_text': raw, 'raw_text_sha256': helper.digest_bytes(raw.encode()),
                                 'model_id': result['model'], 'requested_max_tokens': payload['max_tokens'],
                                 'stop_reason': result.get('stop_reason'), 'usage': result.get('usage', {}),
                                 'native_request_seconds': seconds})
            return result

        helper.request_json = guarded_request
        started = time.monotonic()
        try:
            with answers_path.open('a') as stream:
                schedule = [(case, repeat) for repeat in REPEATS for case in cases]
                for case, repeat in schedule[len(rows):]:
                    captured.clear()
                    native_requests.clear()
                    answer, metadata = helper.generate_complete(
                        BASE, case, helper.CAPABILITY_SYSTEM, BUDGETS, CONTEXT_LIMIT,
                        REQUEST_TIMEOUT, f'{args.label}/repeat-{repeat}')
                    raw = captured[-1]['raw_text']
                    if raw.strip() != answer:
                        raise RuntimeError('Captured final text differs from frozen helper output')
                    score = helper.correct(case, answer)
                    row = row_identity(case, config, fingerprint, helper) | {
                        'repeat': repeat, 'evidence_status': EVIDENCE_STATUS,
                        'answer_text': answer, 'raw_final_text': raw,
                        'raw_final_text_sha256': helper.digest_bytes(raw.encode()),
                        'response_attempts': list(captured), 'native_requests': list(native_requests),
                        **metadata, 'correct': score and metadata['generation_complete'],
                        'scored_correct_before_completion_gate': score,
                        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                    }
                    row['record_sha256'] = helper.fingerprint(row)
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    rows.append(row)
                    print(json.dumps({'stage': 'bf16_integrity', 'label': args.label,
                                      'id': case['id'], 'repeat': repeat, 'correct': row['correct'],
                                      'generation_complete': row['generation_complete'],
                                      'completed': len(rows), 'total': total}), flush=True)
        finally:
            helper.request_json = original_request
        guard()
        verify_frozen()
        if adapter_identity(args.adapter_dir, checkpoint) != adapter:
            raise RuntimeError('Adapter content hashes changed during evaluation')
        if file_sha256(Path(__file__)) != protocol['script_sha256']:
            raise RuntimeError('Evaluator script changed during evaluation')
        rows = read_rows(answers_path, config, fingerprint, cases, helper)
        totals = helper.summarize_capabilities(rows)
        totals.pop('records')
        by_case = {case['id']: {'correct': sum(row['correct'] for row in rows if row['id'] == case['id']),
                               'repeats': len(REPEATS)} for case in cases}
        summary = {
            'label': args.label, 'status': 'completed', 'evidence_status': EVIDENCE_STATUS,
            'configuration_fingerprint': fingerprint, 'protocol_fingerprint': config['protocol_fingerprint'],
            'model_id': args.model_id, 'base_checkpoint': checkpoint['path'],
            'base_checkpoint_fingerprint': helper.fingerprint(checkpoint),
            'adapter': adapter, 'launch_profile': launch, 'served_identity': served,
            'initial_advertised_model_ids': [model['id'] for model in listing['data']],
            'source_cases': str(SOURCE_CASES), 'selected_case_ids': protocol['case_order'],
            'answers_file': str(answers_path), 'answers_sha256': file_sha256(answers_path),
            'record_count': len(rows), 'unique_cases': len(cases), 'repeats_per_case': len(REPEATS),
            'capabilities': totals, 'by_case': by_case,
            'total_output_tokens_including_retries': sum(row['total_output_tokens_including_retries'] for row in rows),
            'generation_seconds_including_retries': sum(row['generation_seconds_including_retries'] for row in rows),
            'native_generation_seconds': sum(attempt['native_request_seconds'] for row in rows for attempt in row['response_attempts']),
            'seconds_this_invocation': time.monotonic() - started,
            'interpretation': 'Previously seen development cases and repeated answers are not new held-out evidence or additional independent questions. Full benign text is diagnostic only. Frozen strict scores and completion gates remain unchanged; no prior result is revised.',
        }
        helper.atomic_json(summary_path, summary)
        print(json.dumps({'stage': 'completed', 'label': args.label, 'summary': str(summary_path),
                          'capabilities': totals, 'evidence_status': EVIDENCE_STATUS}), flush=True)


if __name__ == '__main__':
    main()
