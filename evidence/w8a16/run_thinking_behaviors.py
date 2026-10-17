#!/usr/bin/env python3
"""Wait for committed thinking24 comparisons, then run four sequential benign smokes.

This controller never changes services, models, profiles, or adapters. Each child
wrapper owns runtime.lock and performs live identity checks. Existing outputs are
reused only after offline identity, protocol, integrity, and completion checks.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import types


WORK = Path(__file__).resolve().parent
PYTHON = '/home/isnotreal/vllm-env/bin/python'
ALIAS = 'qwen3.6-35b-hauhau-w4a16-vision'
BRANCHES = ('baseline', 'strength-1.0', 'strength-1.5', 'attention-1.5-mlp-1.0')
PINNED = {
    'check_behaviors_thinking.py': 'cdfd43d3fcf7573e42d42047a3e40b9abebd806a6ab344cdeeb6e84f7738da6b',
    'launch-v1.json': '394fa5c48b6d0206f9e9cf90deba3c3c1701162d9aef8022085cbc08d7ef4f58',
    'precision-manifest-v1.json': 'a8806a05173567d3f86df3781c3feaa8845fe5e1872fc8c7c6a45361ea7f931c',
    'adapter-identities-v1.json': 'b6c8059c4e3c6fefc0c736b223740baf4130ce194a84dd283eefd49ab1687929',
    'run_thinking_batch.py': '08ec7b4892fe6094456353e4ec8df746b2731c19aaabb5de98e2d9cc4a1b3a57',
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def status(stage, **extra):
    value = {'stage': stage, 'updated_utc': now(), **extra}
    write(WORK / 'thinking-behaviors-status-v1.json', value)
    print(json.dumps(value), flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_sources():
    for name, expected in PINNED.items():
        require(sha(WORK / name) == expected, f'Pinned controller input changed: {name}')


def wait_for_batch(timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    result_path = WORK / 'thinking-batch-result-v1.json'
    upstream_status = WORK / 'thinking-batch-status-v1.json'
    while True:
        if upstream_status.exists():
            require(json.loads(upstream_status.read_text()).get('stage') != 'failed',
                    'Thinking batch failed; smoke checks were not started')
        if result_path.exists():
            result = json.loads(result_path.read_text())
            require(result.get('status') == 'completed', 'Thinking batch result is not completed')
            require(result.get('precision_manifest_sha256') == PINNED['precision-manifest-v1.json']
                    and result.get('profile_sha256') == PINNED['launch-v1.json'],
                    'Thinking batch precision/profile differs from this follow-up')
            comparisons = result.get('comparisons', [])
            require([row.get('label') for row in comparisons] == list(BRANCHES),
                    'Thinking batch does not contain all four expected branches in order')
            for row in comparisons:
                path = WORK / f'w8a16-{row["label"]}-thinking24-v1-summary.json'
                require(row.get('path') == str(path) and sha(path) == row.get('sha256'),
                        'Committed thinking summary path/hash differs')
                summary = json.loads(path.read_text())
                require(summary.get('status') == 'completed' and summary.get('record_count') == 48,
                        'Thinking summary is partial or failed')
            return {'path': str(result_path), 'sha256': sha(result_path), 'comparisons': comparisons}
        if time.monotonic() >= deadline:
            raise TimeoutError('Timed out waiting for the committed thinking batch; no smokes started')
        time.sleep(min(5, max(0, deadline - time.monotonic())))


def prepare():
    verify_sources()
    wrapper_path = WORK / 'check_behaviors_thinking.py'
    wrapper = types.ModuleType('_thinking_smoke_controller_helpers')
    wrapper.__file__ = str(wrapper_path)
    exec(compile(wrapper_path.read_bytes(), str(wrapper_path), 'exec'), wrapper.__dict__)
    integrity, support, grading, _ = wrapper.load_helpers()
    profile = WORK / 'launch-v1.json'
    manifest = WORK / 'precision-manifest-v1.json'
    base_path = json.loads(manifest.read_text())['expected_base_checkpoint']
    precision = integrity.load_precision_manifest(str(manifest), base_path, str(profile), support, grading)
    checkpoint = integrity.checkpoint_identity(base_path, precision, grading)
    registered = json.loads((WORK / 'adapter-identities-v1.json').read_text())['adapters']
    branches = []
    for name in BRANCHES:
        spec = registered[name + '-vllm'] if name != 'baseline' else None
        model_id = spec['model_id'] if spec else ALIAS
        adapter = support.adapter_identity(spec['path'] if spec else None, checkpoint)
        if spec:
            require(adapter['files']['adapter_model.safetensors']['sha256'] == spec['adapter_model_sha256']
                    and adapter['files']['adapter_config.json']['sha256'] == spec['adapter_config_sha256'],
                    f'Adapter differs from the registered identity: {name}')
        launch = integrity.launch_identity(str(profile), checkpoint, adapter, precision, support, grading)
        branches.append({'name': name, 'model_id': model_id,
                         'label': f'w8a16-{name}-tools-vision-thinking-v1',
                         'expected': {'checkpoint': checkpoint, 'adapter': adapter,
                                      'precision': precision, 'launch_profile': launch}})
    require(len({branch['model_id'] for branch in branches}) == 4, 'Expected four distinct model IDs')
    return wrapper, support, grading, branches


def validate_output(path, branch, wrapper, support, grading):
    original_sha256 = sha(path)
    result = json.loads(path.read_text())
    require(result.get('result_sha256') == grading.fingerprint({k: v for k, v in result.items() if k != 'result_sha256'}),
            'Behavior result fingerprint differs')
    identity = result['identity']
    require(identity.get('model_id') == branch['model_id'], 'Behavior model identity differs')
    for key, expected in branch['expected'].items():
        require(identity.get(key) == expected, f'Behavior {key} identity differs')
    require(identity.get('runtime_lock') == str(WORK / 'runtime.lock'), 'Behavior runtime lock differs')
    config_path = WORK / f'{branch["label"]}-behaviors-thinking-config.json'
    config = json.loads(config_path.read_text())
    fingerprint = grading.fingerprint(identity)
    require(config == {'configuration': identity, 'configuration_fingerprint': fingerprint}
            and result.get('configuration_fingerprint') == fingerprint, 'Behavior configuration differs')
    protocol = identity['protocol']
    expected_protocol = {
        'version': 1, 'runner': 'w8a16-frozen-behavior-smokes-thinking',
        'helper_sha256': wrapper.HELPER_SHA256, 'integrity_helper_sha256': wrapper.INTEGRITY_SHA256,
        'wrapper_sha256': PINNED['check_behaviors_thinking.py'], 'case_order': wrapper.TESTS,
        'temperature': 0, 'max_tokens': 4096,
        'thinking': {'type': 'enabled', 'budget_tokens': 3072},
        'chat_template_kwargs': {'enable_thinking': True}, 'context_limit': 8192,
        'context_margin': 64, 'request_timeout_seconds': 1800, 'automatic_retries': False,
    }
    require(all(protocol.get(k) == v for k, v in expected_protocol.items())
            and identity.get('protocol_fingerprint') == grading.fingerprint(protocol),
            'Behavior thinking protocol differs')
    integrity, _, _, _ = wrapper.load_helpers()
    require(protocol.get('vllm_protocol_source_sha256') == integrity.PROTOCOL_SOURCE_HASHES
            and protocol.get('chat_template_sha256') == integrity.THINKING_TEMPLATE_SHA256,
            'Behavior inspected protocol/template hashes differ')
    served = identity['served_identity']
    listing = {'data': [served['base']] + ([served['requested']] if branch['expected']['adapter'] else [])}
    require(support.validate_model_listing(listing, branch['model_id'], branch['expected']['checkpoint'],
                branch['expected']['adapter'], branch['expected']['launch_profile']) == served,
            'Stored native base/adapter identity differs')
    records = result.get('records', [])
    require(result.get('completed') is True and result.get('error') is None
            and result.get('total') == result.get('expected_total') == 7
            and [row.get('test') for row in records] == wrapper.TESTS,
            'Behavior run is incomplete, failed, or has missing checks')
    require(all(type(row.get('passed')) is bool for row in records)
            and result.get('passed') == sum(row['passed'] for row in records), 'Behavior pass count differs')
    require(result.get('all_generations_complete') is True and result.get('truncated_generations') == 0,
            'Behavior run contains incomplete/truncated generation')
    native = result.get('native_requests', [])
    generations = [entry for entry in native if entry.get('route') == '/v1/messages']
    expected_count = 5 + int(records[0]['passed']) + int(records[3]['passed'])
    require(result.get('generation_count') == len(generations) == expected_count
            and [entry.get('route') for entry in native] == ['/v1/messages/count_tokens', '/v1/messages'] * expected_count,
            'Behavior native request sequence is partial or inconsistent')
    for index, entry in enumerate(native):
        require(entry.get('request_index') == index
                and entry.get('request_sha256') == wrapper.sha_bytes(json.dumps(entry['request']).encode())
                and entry.get('response_sha256') == grading.fingerprint(entry['response']),
                'Behavior native request/response integrity differs')
    for index, entry in enumerate(generations):
        wrapper.validate_request(entry['request'], branch['model_id'])
        count_entry = native[index * 2]
        count = count_entry['response'].get('input_tokens')
        expected_count_request = {key: value for key, value in entry['request'].items()
                                  if key in ('model', 'messages', 'system', 'tools',
                                             'tool_choice', 'chat_template_kwargs')}
        require(count_entry['request'] == expected_count_request, 'Behavior counted a different prompt/history')
        require(type(count) is int and count > 0 and count + 4096 + 64 <= 8192
                and entry.get('counted_input_tokens') == count, 'Behavior context count differs')
        require(entry.get('generation_complete') is True
                and entry['response'].get('model') == branch['model_id']
                and entry.get('stop_reason') == entry['response'].get('stop_reason')
                and entry.get('usage') == entry['response'].get('usage')
                and entry['response'].get('stop_reason') in ('end_turn', 'tool_use', 'stop_sequence'),
                'Behavior response is partial, truncated, or has wrong model identity')
        parts = wrapper.response_parts(entry['response'])
        require(all(entry.get(k) == v for k, v in parts.items()), 'Behavior response block capture differs')
    responded = [row for row in records if row.get('response') is not None]
    require(len(responded) == expected_count, 'Behavior response records are missing')
    for row, entry in zip(responded, generations):
        require(row['response'] == entry['response'] and row.get('generation_complete') is True
                and row.get('response_sha256') == grading.fingerprint(row['response']),
                'Behavior test record differs from captured complete response')
        require(all(row.get(k) == v for k, v in wrapper.response_parts(row['response']).items()),
                'Behavior test block separation differs')
    require((records[1]['response'] is not None) == records[0]['passed']
            and (records[4]['response'] is not None) == records[3]['passed'],
            'Behavior conditional follow-up sequence differs')
    require(result.get('thinking_block_count') == sum(len(entry['thinking_blocks']) for entry in generations)
            and result.get('total_output_tokens') == sum(entry['usage']['output_tokens'] for entry in generations),
            'Behavior token/block totals differ')
    require(sha(path) == original_sha256, 'Behavior output changed during validation')
    return {'branch': branch['name'], 'label': branch['label'], 'model_id': branch['model_id'],
            'path': str(path), 'sha256': sha(path), 'config_path': str(config_path), 'config_sha256': sha(config_path),
            'configuration_fingerprint': fingerprint, 'protocol_fingerprint': identity['protocol_fingerprint'],
            'completed': True, 'passed': result['passed'], 'total': 7,
            'all_tests_passed': result['passed'] == 7, 'all_generations_complete': True,
            'generation_count': expected_count, 'truncated_generations': 0,
            'completion_by_test': [{'test': row['test'], 'passed': row['passed'],
                'generation_complete': row['generation_complete'],
                'conditional_followup_skipped': row['response'] is None} for row in records],
            'generation_stop_reasons': [entry['response']['stop_reason'] for entry in generations],
            'thinking_block_count': result['thinking_block_count'], 'total_output_tokens': result['total_output_tokens']}


def run(args):
    result_path = WORK / 'thinking-behaviors-result-v1.json'
    comparisons = []
    upstream = None
    active_output = None
    existing = None
    try:
        existing = json.loads(result_path.read_text()) if result_path.exists() else None
        require(existing is None or existing.get('status') == 'completed',
                'Existing failed aggregate is preserved; investigate before choosing new run labels')
        verify_sources()
        status('waiting-for-thinking-batch')
        upstream = wait_for_batch(args.wait_timeout_seconds)
        wrapper, support, grading, branches = prepare()
        for branch in branches:
            verify_sources()
            path = WORK / f'{branch["label"]}-behaviors-thinking.json'
            active_output = path
            reused = path.exists()
            status('validating-existing' if reused else 'thinking-tools-vision',
                   label=branch['label'], model_id=branch['model_id'], completed_branches=len(comparisons))
            if not reused:
                require(existing is None, 'Committed aggregate is missing a referenced branch output')
                command = [PYTHON, '-u', str(WORK / 'check_behaviors_thinking.py'),
                    '--label', branch['label'], '--model-id', branch['model_id'],
                    '--launch-profile', str(WORK / 'launch-v1.json'),
                    '--precision-manifest', str(WORK / 'precision-manifest-v1.json')]
                if branch['expected']['adapter']:
                    command += ['--adapter-dir', branch['expected']['adapter']['path']]
                subprocess.run(command, check=True, timeout=args.branch_timeout_seconds)
            entry = validate_output(path, branch, wrapper, support, grading)
            comparisons.append(entry)
            status('branch-completed', **entry, reused_existing_output=reused)
        verify_sources()
        require(sha(Path(upstream['path'])) == upstream['sha256'], 'Committed thinking batch changed')
        require(len({row['protocol_fingerprint'] for row in comparisons}) == 1,
                'Branches do not share the same thinking smoke protocol')
        result = {'status': 'completed', 'finished_utc': now(), 'comparisons': comparisons,
                  'upstream': upstream, 'input_sha256': PINNED, 'driver_sha256': sha(Path(__file__)),
                  'protocol_fingerprint': comparisons[0]['protocol_fingerprint'],
                  'all_generations_complete': True, 'all_checks_passed': all(row['all_tests_passed'] for row in comparisons),
                  'passed': sum(row['passed'] for row in comparisons), 'total': 28,
                  'scope': 'Seven previously seen benign tool/vision smokes on four W8A16 branches with thinking enabled. Separate from thinking-disabled results; no refusal or broad quality claim.',
                  'quality_reference': 'Untouched W8A16 baseline', 'model_or_service_changes': False}
        result['result_sha256'] = grading.fingerprint(result)
        if existing is not None:
            require(existing.get('result_sha256') == grading.fingerprint({k: v for k, v in existing.items() if k != 'result_sha256'}),
                    'Existing aggregate fingerprint differs')
            require(all(existing.get(k) == result[k] for k in result if k not in ('finished_utc', 'result_sha256')),
                    'Existing aggregate differs from fully validated branch results')
        else:
            write(result_path, result)
        status('completed', result=str(result_path), sha256=sha(result_path),
               all_generations_complete=True, all_checks_passed=result['all_checks_passed'])
    except Exception as exc:
        failure = {'status': 'failed', 'finished_utc': now(), 'error': repr(exc),
                   'comparisons': comparisons, 'upstream': upstream, 'input_sha256': PINNED,
                   'driver_sha256': sha(Path(__file__)),
                   'last_observed_output': {'path': str(active_output), 'sha256': sha(active_output)}
                       if active_output is not None and active_output.exists() else None,
                   'all_generations_complete': False, 'all_checks_passed': False,
                   'note': 'Partial or invalid observations are preserved and never accepted as successful.'}
        failure['result_sha256'] = fingerprint(failure)
        if not result_path.exists():
            write(result_path, failure)
        status('failed', error=repr(exc), completed_branches=len(comparisons), result=str(result_path))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-timeout-seconds', type=float, default=7200)
    parser.add_argument('--branch-timeout-seconds', type=float, default=14400)
    args = parser.parse_args()
    if args.wait_timeout_seconds <= 0 or args.branch_timeout_seconds <= 0:
        parser.error('Timeouts must be positive')
    with (WORK / 'thinking-behaviors.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Thinking behavior controller is already active') from exc
        run(args)


if __name__ == '__main__':
    main()
