#!/usr/bin/env python3
"""W8A16: fixed groups of four previously seen benign development diagnostics.

Two fixed repeats use isolated per-task copies of the pinned completion/grading
helpers. Only complete groups contribute scores. The runtime controller must
hold runtime.lock while restarting or changing the server.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import types


WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
V1_PATH = ROOT / 'bf16-iteration/evaluate_bf16_integrity.py'
V1_SHA256 = 'f0a84fc9f8b1ed8f0088df755b42935af75840751a524083b8656b32b30fe288'
BATCH_SIZE = 4
MAX_WORKERS = 4
EVIDENCE_STATUS = 'previously seen development diagnostics; not new held-out evidence'
FORBIDDEN_ACTIVATION_ENV = (
    'VLLM_MARLIN_INPUT_DTYPE',
    'VLLM_HUMMING_INPUT_QUANT_CONFIG',
    'VLLM_HUMMING_ONLINE_QUANT_CONFIG',
)


def now():
    return datetime.now(timezone.utc).isoformat()


def load_support(case_ids=None):
    if hashlib.sha256(V1_PATH.read_bytes()).hexdigest() != V1_SHA256:
        raise RuntimeError('Pinned version 1 support definitions changed')
    support = types.ModuleType('_bf16_integrity_v1_support')
    support.__file__ = str(V1_PATH)
    exec(compile(V1_PATH.read_bytes(), str(V1_PATH), 'exec'), support.__dict__)
    helper, authored, cases = support.load_inputs(case_ids)
    # Reuse only immutable helper/scoring definitions. W8A16 identities are
    # explicit here; the BF16 support file and its artifacts remain untouched.
    original_identity = support.row_identity

    def row_identity(case, config, fingerprint, task_helper):
        return original_identity(case, config, fingerprint, task_helper) | {
            'precision_manifest_sha256': config['precision']['sha256'],
            'weight_bits': 8, 'activation_dtype': 'bfloat16',
            'quantization_backend': config['precision']['manifest']['quantization_backend']}

    support.row_identity = row_identity
    return support, helper, authored, cases


def check_activation_environment(environment):
    if not isinstance(environment, dict):
        raise ValueError('Launch environment must be a JSON object')
    enabled = [name for name in FORBIDDEN_ACTIVATION_ENV
               if environment.get(name) not in (None, '')]
    if enabled:
        raise ValueError(f'W8A16 forbids activation/online quantization environment settings: {enabled}')


def quantization_sections(config):
    result = {}
    for name, section in [('root', config), ('text_config', config.get('text_config')),
                          ('vision_config', config.get('vision_config'))]:
        if section and section.get('quantization_config') is not None:
            result[name] = section['quantization_config']
        if section:
            dtype = section.get('dtype', section.get('torch_dtype'))
            if dtype is not None and dtype != 'bfloat16':
                raise ValueError(f'Expected BF16 activation/unquantized configuration dtype in {name}')
    return result


def validate_weight_only_config(configs, backend):
    if not configs:
        raise ValueError('W8A16 requires an explicit quantization configuration')
    for section, quantization in configs.items():
        if not isinstance(quantization, dict):
            raise ValueError(f'Invalid quantization_config in {section}')
        method = quantization.get('quant_method')
        if method == 'compressed-tensors' and backend == 'compressed-tensors':
            if quantization.get('format') != 'pack-quantized':
                raise ValueError('Expected the compressed-tensors pack-quantized format')
            groups = quantization.get('config_groups')
            if not isinstance(groups, dict) or not groups:
                raise ValueError('Compressed-tensors configuration requires weight groups')
            for group in groups.values():
                weights = group.get('weights') or {}
                if (type(weights.get('num_bits')) is not int or weights['num_bits'] != 8
                        or weights.get('type') != 'int'
                        or weights.get('symmetric') is not True
                        or group.get('input_activations') is not None
                        or group.get('output_activations') is not None):
                    raise ValueError('Expected 8-bit integer weights with unquantized activations')
            if quantization.get('kv_cache_scheme') is not None:
                raise ValueError('Quantized KV cache is outside this W8A16 protocol')
        elif method in ('gptq', 'gptq_marlin') and backend in ('gptq', 'gptq_marlin'):
            if type(quantization.get('bits')) is not int or quantization['bits'] != 8:
                raise ValueError('Expected an 8-bit GPTQ checkpoint')
            if any(quantization.get(key) is not None
                   for key in ('input_activations', 'output_activations', 'activation_scheme')):
                raise ValueError('GPTQ activation quantization is outside this W8A16 protocol')
        else:
            raise ValueError(f'Unsupported or mismatched W8A16 format: {method!r}/{backend!r}')


def checkpoint_identity(path, precision, helper):
    identity = helper.checkpoint_identity(path)
    manifest = precision['manifest']
    if identity['path'] != manifest['expected_base_checkpoint']:
        raise ValueError('Checkpoint path differs from the precision manifest')
    if not identity['shards'] or 'config.json' not in identity['files']:
        raise ValueError('Expected checkpoint requires config.json and safetensors shards')
    if identity['files']['config.json']['sha256'] != manifest['config_sha256']:
        raise ValueError('Checkpoint config hash differs from the precision manifest')
    config = json.loads((Path(identity['path']) / 'config.json').read_text())
    sections = quantization_sections(config)
    if sections != manifest['quantization_configs']:
        raise ValueError('Checkpoint quantization configuration differs from the precision manifest')
    validate_weight_only_config(sections, manifest['quantization_backend'])
    identity['precision_manifest_sha256'] = precision['sha256']
    return identity


def load_precision_manifest(path, expected_checkpoint, launch_profile, support, helper):
    path = Path(path).resolve(strict=True)
    manifest = json.loads(path.read_text())
    required = ('schema_version', 'expected_base_checkpoint', 'config_sha256', 'weight_bits',
                'activation_dtype', 'launch_profile_sha256', 'quantization_backend',
                'quantization_configs')
    if any(key not in manifest for key in required):
        raise ValueError(f'Precision manifest requires {required}')
    if (type(manifest['schema_version']) is not int or manifest['schema_version'] != 1
            or type(manifest['weight_bits']) is not int
            or manifest['weight_bits'] != 8 or manifest['activation_dtype'] != 'bfloat16'):
        raise ValueError('Precision manifest must specify version 1, 8-bit weights, BF16 activations')
    for key in ('config_sha256', 'launch_profile_sha256'):
        if not isinstance(manifest[key], str) or not re.fullmatch('[0-9a-f]{64}', manifest[key]):
            raise ValueError(f'Invalid SHA256 value for {key}')
    expected = str(Path(expected_checkpoint).resolve(strict=True))
    if manifest['expected_base_checkpoint'] != expected:
        raise ValueError('Expected checkpoint differs from the precision manifest')
    if support.file_sha256(Path(launch_profile)) != manifest['launch_profile_sha256']:
        raise ValueError('Launch profile hash differs from the precision manifest')
    if not isinstance(manifest['quantization_configs'], dict):
        raise ValueError('quantization_configs must map config sections to their exact objects')
    check_activation_environment(dict(os.environ))
    identity = {'path': str(path), 'sha256': support.file_sha256(path),
                'canonical_fingerprint': helper.fingerprint(manifest), 'manifest': manifest,
                'quality_reference': 'untouched W8A16 baseline; BF16 source is conversion input only',
                'attestation_scope': 'Binds checkpoint configuration and supplied launch record. Does not independently inspect live kernel arithmetic or all inherited server environment variables.'}
    checkpoint_identity(expected, identity, helper)
    return identity


def launch_identity(path, checkpoint, adapter, precision, support, helper):
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
        raise ValueError('Launch profile uses a different checkpoint')
    required = {'--host': ['127.0.0.1'], '--port': ['8002'], '--dtype': ['bfloat16'],
                '--kv-cache-dtype': ['bfloat16'], '--max-model-len': [str(support.CONTEXT_LIMIT)]}
    if any(support.flag_values(argv, flag) != values for flag, values in required.items()):
        raise ValueError('Launch requires private port 8002, BF16 activations/KV, and context 8192')
    quantization_flags = support.flag_values(argv, '--quantization')
    if quantization_flags not in ([], [precision['manifest']['quantization_backend']]):
        raise ValueError('Launch quantization backend differs from the precision manifest')
    if '--no-enable-prefix-caching' not in argv or '--enable-prefix-caching' in argv:
        raise ValueError('This development protocol requires explicitly disabled prefix caching')
    check_activation_environment(profile.get('environment', {}))
    base_ids = support.flag_values(argv, '--served-model-name')
    if not base_ids:
        raise ValueError('Launch profile must identify its served base model name')
    if adapter is not None and '--enable-lora' not in argv:
        raise ValueError('An adapter run requires --enable-lora')
    digest = support.file_sha256(path)
    if digest != precision['manifest']['launch_profile_sha256']:
        raise ValueError('Launch profile changed after precision-manifest verification')
    return {'path': str(path), 'sha256': digest, 'canonical_fingerprint': helper.fingerprint(profile),
            'base_model_ids': base_ids, 'quantization_backend': precision['manifest']['quantization_backend'],
            'forbidden_activation_environment': list(FORBIDDEN_ACTIVATION_ENV),
            'attestation_scope': 'Supplied launch record is hashed; endpoint checks validate roots and context, not the full running command/environment or kernel precision.'}


def fresh_helper(source, path, model_id):
    helper = types.ModuleType('_isolated_batched_w8a16_helpers')
    helper.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), helper.__dict__)
    helper.MODEL = model_id
    return helper


def group_schedule(cases, repeats):
    groups = []
    for repeat in repeats:
        for offset in range(0, len(cases), BATCH_SIZE):
            groups.append({'index': len(groups), 'repeat': repeat,
                           'case_ids': [case['id'] for case in cases[offset:offset + BATCH_SIZE]]})
    return groups


def atomic_text(path, text):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2) + '\n')


class EventRecorder:
    def __init__(self, invocation_id):
        self.invocation_id = invocation_id
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.events = []

    def record(self, kind, case_id, repeat, **extra):
        with self.lock:
            event = {'sequence': len(self.events), 'invocation_id': self.invocation_id,
                     'kind': kind, 'id': case_id, 'repeat': repeat,
                     'at_utc': now(), 'group_elapsed_seconds': time.monotonic() - self.started,
                     **extra}
            self.events.append(event)
        return event


class DiagnosticTaskError(RuntimeError):
    def __init__(self, error, partial):
        super().__init__(repr(error))
        self.partial = partial


def evaluate_one(case, repeat, group_index, submission_index, config, fingerprint,
                 support, helper_source, guard, recorder):
    # Nothing mutable in this module is shared with another task's transport.
    helper = fresh_helper(helper_source, support.HELPER_PATH, config['model_id'])
    original_request = helper.request_json
    captured, native_requests = [], []
    task_started = recorder.record('task_started', case['id'], repeat,
                                   group_index=group_index, submission_index=submission_index)

    def request(base, route, payload, timeout):
        if (base != support.BASE or route not in ('/v1/messages/count_tokens', '/v1/messages')
                or payload.get('model') != config['model_id']):
            raise RuntimeError('Attempted an unexpected endpoint or model')
        guard()
        start_event = recorder.record('request_started', case['id'], repeat,
                                      group_index=group_index, route=route,
                                      requested_max_tokens=payload.get('max_tokens'))
        started = time.monotonic()
        try:
            result = original_request(base, route, payload, timeout)
        except Exception as exc:
            recorder.record('request_failed', case['id'], repeat, group_index=group_index,
                            route=route, start_sequence=start_event['sequence'], error=repr(exc))
            raise
        seconds = time.monotonic() - started
        end_event = recorder.record('request_finished', case['id'], repeat,
                                    group_index=group_index, route=route,
                                    start_sequence=start_event['sequence'])
        native_requests.append({'route': route, 'seconds': seconds,
                                'started_sequence': start_event['sequence'],
                                'finished_sequence': end_event['sequence']})
        if route == '/v1/messages':
            raw = ''.join(block.get('text', '') for block in result['content'] if block.get('type') == 'text')
            captured.append({'raw_text': raw, 'raw_text_sha256': helper.digest_bytes(raw.encode()),
                             'model_id': result.get('model'), 'requested_max_tokens': payload['max_tokens'],
                             'stop_reason': result.get('stop_reason'), 'usage': result.get('usage', {}),
                             'native_request_seconds': seconds,
                             'started_sequence': start_event['sequence'],
                             'finished_sequence': end_event['sequence']})
            if result.get('model') != config['model_id']:
                raise RuntimeError('Response model differs from the requested base/LoRA ID')
        guard()
        return result

    helper.request_json = request
    try:
        answer, metadata = helper.generate_complete(
            support.BASE, case, helper.CAPABILITY_SYSTEM, support.BUDGETS, support.CONTEXT_LIMIT,
            support.REQUEST_TIMEOUT, f'{config["label"]}/group-{group_index}/repeat-{repeat}')
        raw = captured[-1]['raw_text']
        if raw.strip() != answer:
            raise RuntimeError('Captured answer differs from frozen helper output')
        score = helper.correct(case, answer)
        finished = recorder.record('task_finished', case['id'], repeat,
                                   group_index=group_index, submission_index=submission_index)
        row = support.row_identity(case, config, fingerprint, helper) | {
            'repeat': repeat, 'group_index': group_index, 'submission_index': submission_index,
            'invocation_id': recorder.invocation_id, 'evidence_status': EVIDENCE_STATUS,
            'task_started_sequence': task_started['sequence'],
            'task_finished_sequence': finished['sequence'],
            'answer_text': answer, 'raw_final_text': raw,
            'raw_final_text_sha256': helper.digest_bytes(raw.encode()),
            'response_attempts': captured, 'native_requests': native_requests,
            **metadata, 'correct': score and metadata['generation_complete'],
            'scored_correct_before_completion_gate': score, 'recorded_at_utc': finished['at_utc'],
        }
        row['record_sha256'] = helper.fingerprint(row)
        return row
    except Exception as exc:
        recorder.record('task_failed', case['id'], repeat, group_index=group_index, error=repr(exc))
        raise DiagnosticTaskError(exc, {'id': case['id'], 'repeat': repeat,
                                       'response_attempts': captured,
                                       'native_requests': native_requests,
                                       'scored': False}) from exc


def validate_record(row, case, repeat, group_index, position, config, fingerprint, support, helper):
    expected = support.row_identity(case, config, fingerprint, helper) | {
        'repeat': repeat, 'group_index': group_index, 'submission_index': position}
    if any(row.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Committed record has incompatible identity or group order')
    if row.get('record_sha256') != helper.fingerprint(
            {key: value for key, value in row.items() if key != 'record_sha256'}):
        raise RuntimeError('Committed record hash is invalid')
    answer, raw = row['answer_text'], row['raw_final_text']
    if (raw.strip() != answer or row['response_sha256'] != helper.digest_bytes(answer.encode())
            or row['raw_final_text_sha256'] != helper.digest_bytes(raw.encode())
            or row['response_attempts'][-1]['raw_text'] != raw
            or row['planned_budgets'] != support.BUDGETS
            or len(row['attempts']) != len(row['response_attempts'])):
        raise RuntimeError('Committed final text or retry metadata is inconsistent')
    for response, metadata in zip(row['response_attempts'], row['attempts']):
        if (helper.digest_bytes(response['raw_text'].encode()) != response['raw_text_sha256']
                or helper.digest_bytes(response['raw_text'].strip().encode()) != metadata['response_sha256']
                or response['model_id'] != config['model_id']
                or response['requested_max_tokens'] != metadata['requested_max_tokens']):
            raise RuntimeError('Committed per-attempt capture is inconsistent')
    score = helper.correct(case, answer)
    if (type(row['generation_complete']) is not bool
            or row['generation_complete'] != (not row['unresolved_reasons'])
            or row['scored_correct_before_completion_gate'] is not score
            or row['correct'] is not (score and row['generation_complete'])):
        raise RuntimeError('Committed strict score disagrees with the frozen grader/completion gate')


def load_committed(directory, schedule, config, fingerprint, cases, support, helper):
    by_id = {case['id']: case for case in cases}
    expected_names = {f'group-{group["index"]:04d}.json' for group in schedule}
    if {path.name for path in directory.glob('group-*.json')} - expected_names:
        raise RuntimeError('Unexpected committed group files')
    groups, missing = [], False
    for group in schedule:
        path = directory / f'group-{group["index"]:04d}.json'
        if not path.exists():
            missing = True
            continue
        if missing:
            raise RuntimeError('Committed groups are not a contiguous prefix')
        value = json.loads(path.read_text())
        if (value.get('configuration_fingerprint') != fingerprint or value.get('schedule') != group
                or value.get('status') != 'complete'
                or value.get('group_sha256') != helper.fingerprint(
                    {key: item for key, item in value.items() if key != 'group_sha256'})):
            raise RuntimeError('Committed group identity or hash is invalid')
        if len(value['records']) != len(group['case_ids']):
            raise RuntimeError('A committed group is incomplete')
        for position, (case_id, row) in enumerate(zip(group['case_ids'], value['records'])):
            validate_record(row, by_id[case_id], group['repeat'], group['index'], position,
                            config, fingerprint, support, helper)
        groups.append(value)
    return groups


def sync_answers(path, groups, config, fingerprint, cases, support, helper):
    rows = [row for group in groups for row in group['records']]
    old = support.read_rows(path, config, fingerprint, cases, helper)
    boundaries = {0}
    count = 0
    for group in groups:
        count += len(group['records'])
        boundaries.add(count)
    # A snapshot can lag committed groups if interrupted between atomic writes.
    if len(old) not in boundaries or old != rows[:len(old)]:
        raise RuntimeError('Ordered answer snapshot differs from committed complete groups')
    if not path.exists() or len(old) != len(rows):
        atomic_text(path, ''.join(json.dumps(row) + '\n' for row in rows))
    return rows


def reconcile_pending(directory, groups, schedule, fingerprint, retry_incomplete):
    committed = {group['schedule']['index']: group for group in groups}
    for path in sorted(directory.glob('pending-*.json')):
        value = json.loads(path.read_text())
        index = value.get('schedule', {}).get('index')
        if (value.get('configuration_fingerprint') != fingerprint or type(index) is not int
                or index >= len(schedule) or index < 0 or value['schedule'] != schedule[index]):
            raise RuntimeError('Pending group has incompatible identity')
        if index in committed:
            if value['attempt_id'] != committed[index]['attempt_id']:
                raise RuntimeError('Pending and committed attempt identities conflict')
            path.unlink()  # Crash after group commit and before pending cleanup.
            continue
        if index != len(groups):
            raise RuntimeError('Unexpected pending group beyond the first uncommitted group')
        if not retry_incomplete:
            raise RuntimeError(
                'An interrupted group exists. Use --retry-incomplete-group to archive its unscored '
                'partial outputs and rerun that entire fixed group; no individual rows are reused.')
        archive = directory / f'incomplete-{index:04d}-{value["attempt_id"]}.json'
        if archive.exists():
            raise RuntimeError('Incomplete-attempt archive already exists')
        path.replace(archive)


def execute_group(group, cases_by_id, config, fingerprint, support, helper, helper_source,
                  guard, directory, executor, invocation_id):
    group_started = time.monotonic()
    attempt_id = f'{time.time_ns()}-{os.getpid()}'
    pending_path = directory / f'pending-{group["index"]:04d}.json'
    commit_path = directory / f'group-{group["index"]:04d}.json'
    if pending_path.exists() or commit_path.exists():
        raise RuntimeError('Refusing to overwrite an existing group attempt')
    initial = {'status': 'in_progress', 'configuration_fingerprint': fingerprint,
               'schedule': group, 'attempt_id': attempt_id, 'invocation_id': invocation_id,
               'started_at_utc': now(), 'partial_outputs_scored': False}
    atomic_json(pending_path, initial)
    recorder = EventRecorder(invocation_id)
    futures = {executor.submit(evaluate_one, cases_by_id[case_id], group['repeat'], group['index'],
                               position, config, fingerprint, support, helper_source, guard, recorder): case_id
               for position, case_id in enumerate(group['case_ids'])}
    rows, failures, observed = {}, {}, []
    for future in as_completed(futures):
        case_id = futures[future]
        observed.append(case_id)
        try:
            rows[case_id] = future.result()
        except Exception as exc:
            failures[case_id] = {'error': repr(exc), 'partial': getattr(exc, 'partial', None)}
        # Persist partial successes for audit only, never for reuse/scoring.
        atomic_json(pending_path, initial | {
            'status': 'incomplete' if failures else 'in_progress',
            'records': [rows[key] for key in group['case_ids'] if key in rows],
            'failures': failures, 'observed_future_completion_order': observed,
            'events': list(recorder.events), 'updated_at_utc': now()})
    if failures:
        raise RuntimeError(f'Group {group["index"]} failed; partial outputs remain unscored: {failures}')
    ordered = [rows[key] for key in group['case_ids']]
    for position, row in enumerate(ordered):
        validate_record(row, cases_by_id[row['id']], group['repeat'], group['index'], position,
                        config, fingerprint, support, helper)
    guard()
    value = initial | {
        'status': 'complete', 'partial_outputs_scored': False, 'completed_at_utc': now(),
        'wall_seconds': time.monotonic() - group_started,
        'records': ordered, 'events': list(recorder.events),
        'submission_order': list(group['case_ids']),
        'client_task_start_order': [event['id'] for event in recorder.events if event['kind'] == 'task_started'],
        'client_task_finish_order': [event['id'] for event in recorder.events if event['kind'] == 'task_finished'],
        'observed_future_completion_order': observed,
        'ordering_scope': 'Sequences and elapsed times are local to this group. Client submission, task, and HTTP dispatch/completion order are recorded; GPU scheduler order is not observed.'}
    value['group_sha256'] = helper.fingerprint(value)
    atomic_json(commit_path, value)
    pending_path.unlink()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--expected-checkpoint', required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--adapter-dir')
    parser.add_argument('--launch-profile', required=True)
    parser.add_argument('--precision-manifest', required=True,
                        help='JSON binding W8A16 checkpoint/config, quantization objects, and launch hash')
    parser.add_argument('--case-id', action='append', help='Repeat exact authored case IDs for a fixed subset')
    parser.add_argument('--retry-incomplete-group', action='store_true',
                        help='Archive unscored partial outputs and retry their entire fixed group')
    args = parser.parse_args()
    # No file read, lock, transport, or service activity precedes --help exit.
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.label):
        parser.error('--label must be a filename-safe identifier')
    if not args.model_id.strip() or not Path(args.expected_checkpoint).is_absolute():
        parser.error('Provide a nonempty model ID and absolute base checkpoint path')
    if args.adapter_dir and not Path(args.adapter_dir).is_absolute():
        parser.error('--adapter-dir must be absolute')
    support, helper, authored, cases = load_support(args.case_id)
    helper_source = support.HELPER_PATH.read_bytes()
    precision = load_precision_manifest(args.precision_manifest, args.expected_checkpoint,
                                        args.launch_profile, support, helper)
    checkpoint = checkpoint_identity(args.expected_checkpoint, precision, helper)
    adapter = support.adapter_identity(args.adapter_dir, checkpoint)
    launch = launch_identity(args.launch_profile, checkpoint, adapter, precision, support, helper)
    schedule = group_schedule(cases, support.REPEATS)
    lock_path = WORK / 'runtime.lock'
    with lock_path.open('rb') as runtime_lock:
        try:
            fcntl.flock(runtime_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('W8A16 runtime is busy; refusing concurrent evaluation/restart') from exc
        lock_stat = os.fstat(runtime_lock.fileno())
        listing = support.fetch_models()
        served = support.validate_model_listing(listing, args.model_id, checkpoint, adapter, launch)
        protocol = {
            'version': 1, 'runner': 'w8a16-integrity-fixed-batches',
            'script_sha256': support.file_sha256(Path(__file__)),
            'v1_support_sha256': V1_SHA256, 'frozen_evaluator_version': helper.PROTOCOL_VERSION,
            'precision': {'weight_bits': 8, 'activation_dtype': 'bfloat16',
                          'quality_reference': 'untouched W8A16 baseline'},
            'frozen_input_sha256': {str(path): digest for path, digest in support.FROZEN.items()},
            'evidence_status': EVIDENCE_STATUS, 'authored_case_count': len(authored),
            'case_order': [case['id'] for case in cases], 'selected_cases_fingerprint': helper.fingerprint(cases),
            'repeats': support.REPEATS, 'fixed_group_schedule': schedule,
            'batch_size': BATCH_SIZE, 'max_workers': MAX_WORKERS,
            'batch_policy': 'Submit one fixed group at a time; wait for every task before the next group. Per-task retries remain sequential.',
            'resume_policy': 'Only atomic complete groups contribute scores. Interrupted groups require explicit whole-group retry; partial rows are archived unscored.',
            'output_budgets': support.BUDGETS, 'budget_semantics': 'Retry only max_tokens stops.',
            'system': helper.CAPABILITY_SYSTEM, 'temperature': 0, 'thinking': False,
            'chat_template_kwargs': {'enable_thinking': False},
            'context_limit': support.CONTEXT_LIMIT, 'context_margin': helper.CONTEXT_MARGIN,
            'request_timeout_seconds': support.REQUEST_TIMEOUT,
            'scoring': 'Unchanged frozen correct(case, text) AND generation_complete.',
            'timing_scope': 'Helper elapsed time includes guards and overlapping requests. Native HTTP latency and batch wall time are separate; summed request seconds are not elapsed throughput.',
            'comparison_scope': 'Compare only W8A16 runs under this identical fixed schedule and precision manifest. Previous BF16 runs are separate feasibility data, not the quality reference.'}
        config = {'label': args.label, 'endpoint': support.BASE, 'model_id': args.model_id,
                  'checkpoint': checkpoint, 'adapter': adapter, 'launch_profile': launch,
                  'precision': precision,
                  'served_identity': served, 'runtime_lock': str(lock_path),
                  'protocol': protocol, 'protocol_fingerprint': helper.fingerprint(protocol)}
        directory = WORK / f'{args.label}-groups'
        answers_path = WORK / f'{args.label}-answers.jsonl'
        summary_path = WORK / f'{args.label}-summary.json'
        existing_groups = list(directory.iterdir()) if directory.exists() else []
        fingerprint = helper.ensure_manifest(WORK / f'{args.label}-config.json', config,
                                             [answers_path, summary_path, *existing_groups])
        directory.mkdir(exist_ok=True)
        groups = load_committed(directory, schedule, config, fingerprint, cases, support, helper)
        rows = sync_answers(answers_path, groups, config, fingerprint, cases, support, helper)
        reconcile_pending(directory, groups, schedule, fingerprint, args.retry_incomplete_group)
        if summary_path.exists():
            previous = json.loads(summary_path.read_text())
            cached_totals = helper.summarize_capabilities(rows)
            cached_totals.pop('records')
            if (previous.get('configuration_fingerprint') != fingerprint
                    or previous.get('answers_sha256') != support.file_sha256(answers_path)
                    or previous.get('record_count') != len(rows) or len(groups) != len(schedule)
                    or previous.get('group_sha256') != [group['group_sha256'] for group in groups]
                    or previous.get('capabilities') != cached_totals):
                raise RuntimeError('Cached summary does not match complete committed groups')
            print(json.dumps({'stage': 'already_complete', 'label': args.label,
                              'summary': str(summary_path)}), flush=True)
            return

        def guard():
            current_lock = lock_path.stat()
            if (current_lock.st_dev, current_lock.st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
                raise RuntimeError('Runtime lock file was replaced')
            if support.file_sha256(Path(launch['path'])) != launch['sha256']:
                raise RuntimeError('Launch profile changed')
            if support.file_sha256(Path(precision['path'])) != precision['sha256']:
                raise RuntimeError('Precision manifest changed')
            check_activation_environment(dict(os.environ))
            if checkpoint_identity(checkpoint['path'], precision, helper) != checkpoint:
                raise RuntimeError('Base checkpoint identity changed')
            support.verify_adapter_stats(adapter)
            if support.validate_model_listing(support.fetch_models(), args.model_id, checkpoint, adapter, launch) != served:
                raise RuntimeError('Requested base/adapter serving identity changed')

        started = time.monotonic()
        invocation_id = f'{time.time_ns()}-{os.getpid()}'
        cases_by_id = {case['id']: case for case in cases}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for group in schedule[len(groups):]:
                group_started = time.monotonic()
                value = execute_group(group, cases_by_id, config, fingerprint, support, helper,
                                      helper_source, guard, directory, executor, invocation_id)
                groups.append(value)
                rows = sync_answers(answers_path, groups, config, fingerprint, cases, support, helper)
                print(json.dumps({'stage': 'w8a16_integrity_group', 'label': args.label,
                                  'group_index': group['index'], 'completed_groups': len(groups),
                                  'total_groups': len(schedule), 'completed_rows': len(rows),
                                  'total_rows': len(cases) * len(support.REPEATS),
                                  'group_wall_seconds': time.monotonic() - group_started}), flush=True)
        guard()
        support.verify_frozen()
        if support.file_sha256(V1_PATH) != V1_SHA256 or support.file_sha256(Path(__file__)) != protocol['script_sha256']:
            raise RuntimeError('Pinned runner definitions changed')
        if support.adapter_identity(args.adapter_dir, checkpoint) != adapter:
            raise RuntimeError('Adapter content hashes changed')
        groups = load_committed(directory, schedule, config, fingerprint, cases, support, helper)
        rows = sync_answers(answers_path, groups, config, fingerprint, cases, support, helper)
        totals = helper.summarize_capabilities(rows)
        totals.pop('records')
        summary = {
            'label': args.label, 'status': 'completed', 'protocol_version': 1,
            'evidence_status': EVIDENCE_STATUS, 'configuration_fingerprint': fingerprint,
            'protocol_fingerprint': config['protocol_fingerprint'], 'model_id': args.model_id,
            'base_checkpoint': checkpoint['path'], 'base_checkpoint_fingerprint': helper.fingerprint(checkpoint),
            'adapter': adapter, 'launch_profile': launch, 'served_identity': served,
            'precision': precision,
            'quality_reference': 'untouched W8A16 baseline; BF16 source is conversion input only',
            'initial_advertised_model_ids': [model['id'] for model in listing['data']],
            'selected_case_ids': protocol['case_order'], 'unique_cases': len(cases),
            'repeats_per_case': len(support.REPEATS), 'record_count': len(rows),
            'batch_size': BATCH_SIZE, 'max_workers': MAX_WORKERS, 'complete_groups': len(groups),
            'group_directory': str(directory), 'group_sha256': [group['group_sha256'] for group in groups],
            'answers_file': str(answers_path), 'answers_sha256': support.file_sha256(answers_path),
            'archived_incomplete_attempts': sorted(path.name for path in directory.glob('incomplete-*.json')),
            'capabilities': totals,
            'by_case': {case['id']: {'correct': sum(row['correct'] for row in rows if row['id'] == case['id']),
                                    'repeats': len(support.REPEATS)} for case in cases},
            'total_output_tokens_including_retries': sum(row['total_output_tokens_including_retries'] for row in rows),
            'sum_generation_request_seconds_including_retries': sum(row['generation_seconds_including_retries'] for row in rows),
            'sum_native_generation_seconds': sum(attempt['native_request_seconds'] for row in rows for attempt in row['response_attempts']),
            'sum_committed_group_wall_seconds': sum(group['wall_seconds'] for group in groups),
            'seconds_this_invocation': time.monotonic() - started,
            'refusal_evaluation_performed': False, 'full_quality_retention_gate_evaluated': False,
            'interpretation': 'Only complete fixed batches count. These previously seen benign development cases and repeated answers do not establish held-out quality or refusal rates. Compare candidates with the untouched W8A16 baseline under this same batching protocol; BF16 source weights are conversion input, not the quality reference. Strict grading and completion gates are unchanged.'}
        atomic_json(summary_path, summary)
        print(json.dumps({'stage': 'completed', 'label': args.label, 'summary': str(summary_path),
                          'capabilities': totals, 'evidence_status': EVIDENCE_STATUS}), flush=True)


if __name__ == '__main__':
    main()
