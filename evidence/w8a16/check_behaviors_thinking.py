#!/usr/bin/env python3
"""Run seven frozen benign tool/vision checks with native Gemma thinking enabled.

These previously seen development smokes remain separate from thinking-disabled
results. The immutable helper supplies every prompt, tool schema, and pass
predicate. Thinking blocks are retained but excluded from assertion text.
"""

import argparse
from collections import Counter
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import types
import urllib.request


WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
INTEGRITY_PATH = WORK / 'evaluate_integrity_thinking.py'
INTEGRITY_SHA256 = 'b3054a303ebd8371a57bd0951d93dc7cc8ff974a06180f027462d5b621001696'
HELPER_PATH = ROOT / 'check_behaviors.py'
HELPER_SHA256 = 'fe7a1d2a6c7f4b340bb3ee10f65bde28191e4f1008de9b810bf9156e0fdc8590'
TESTS = ['forced_tool_call', 'tool_result_roundtrip', 'automatic_tool_choice',
         'pivot_after_two_errors', 'workflow_completion', 'vision_red', 'vision_blue']
MAX_TOKENS = 4096
THINKING_BUDGET = 3072
CONTEXT_MARGIN = 64
REQUEST_TIMEOUT = 1800


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    return sha_bytes(path.read_bytes())


def load_helpers():
    for path, expected in ((INTEGRITY_PATH, INTEGRITY_SHA256), (HELPER_PATH, HELPER_SHA256)):
        if sha(path) != expected:
            raise RuntimeError(f'Pinned helper changed: {path}')
    integrity = types.ModuleType('_w8a16_thinking_behavior_identity')
    integrity.__file__ = str(INTEGRITY_PATH)
    exec(compile(INTEGRITY_PATH.read_bytes(), str(INTEGRITY_PATH), 'exec'), integrity.__dict__)
    support, grading, _, _ = integrity.load_support()
    behavior = types.ModuleType('_isolated_w8a16_thinking_behavior')
    behavior.__file__ = str(HELPER_PATH)
    exec(compile(HELPER_PATH.read_bytes(), str(HELPER_PATH), 'exec'), behavior.__dict__)
    return integrity, support, grading, behavior


def response_parts(response):
    content = response.get('content')
    if not isinstance(content, list) or not all(isinstance(block, dict) for block in content):
        raise RuntimeError('Native response content must be a list of blocks')
    kinds = Counter(block.get('type') for block in content)
    if set(kinds) - {'text', 'tool_use', 'thinking', 'redacted_thinking'}:
        raise RuntimeError(f'Unexpected native response block types: {dict(kinds)}')
    final_blocks = [block for block in content if block.get('type') == 'text']
    if any(not isinstance(block.get('text'), str) for block in final_blocks):
        raise RuntimeError('Final text blocks must contain strings')
    return {
        'final_text': ''.join(block['text'] for block in final_blocks).strip(),
        'final_blocks': final_blocks,
        'tool_use_blocks': [block for block in content if block.get('type') == 'tool_use'],
        'thinking_blocks': [block for block in content
                            if block.get('type') in ('thinking', 'redacted_thinking')],
        'assertion_blocks': [block for block in content
                             if block.get('type') in ('text', 'tool_use')],
        'content_block_types': dict(kinds),
    }


def final_text(response):
    return response_parts(response)['final_text']


def install_thinking_call(helper):
    original_call = helper.call

    def call(*args, **kwargs):
        kwargs.update(max_tokens=MAX_TOKENS, temperature=0,
                      thinking={'type': 'enabled', 'budget_tokens': THINKING_BUDGET},
                      chat_template_kwargs={'enable_thinking': True})
        return original_call(*args, **kwargs)

    helper.call = call
    # The pass predicates in main() are unchanged. Their text accessor sees
    # only final blocks; complete assistant content stays intact for history.
    helper.text = final_text


def validate_request(payload, model_id):
    if (payload.get('model') != model_id or payload.get('temperature') != 0
            or payload.get('max_tokens') != MAX_TOKENS
            or payload.get('thinking') != {'type': 'enabled', 'budget_tokens': THINKING_BUDGET}
            or payload.get('chat_template_kwargs') != {'enable_thinking': True}):
        raise RuntimeError('Smoke request differs from the pinned thinking protocol')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--expected-checkpoint', help='Optional override; otherwise use the precision manifest')
    parser.add_argument('--adapter-dir')
    parser.add_argument('--launch-profile', required=True)
    parser.add_argument('--precision-manifest', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.label):
        parser.error('--label must be a filename-safe identifier')
    if not args.model_id.strip():
        parser.error('Provide a nonempty model ID')
    if args.adapter_dir and not Path(args.adapter_dir).is_absolute():
        parser.error('--adapter-dir must be absolute')
    output = WORK / f'{args.label}-behaviors-thinking.json'
    if output.exists():
        raise RuntimeError('Existing observations are preserved; use a new label')
    expected_checkpoint = args.expected_checkpoint or json.loads(
        Path(args.precision_manifest).read_text()).get('expected_base_checkpoint')
    if not isinstance(expected_checkpoint, str) or not Path(expected_checkpoint).is_absolute():
        parser.error('The manifest/override must identify an absolute checkpoint path')
    integrity, support, grading, helper = load_helpers()
    precision = integrity.load_precision_manifest(args.precision_manifest, expected_checkpoint,
                                                  args.launch_profile, support, grading)
    checkpoint = integrity.checkpoint_identity(expected_checkpoint, precision, grading)
    adapter = support.adapter_identity(args.adapter_dir, checkpoint)
    launch = integrity.launch_identity(args.launch_profile, checkpoint, adapter, precision, support, grading)
    launch_args = json.loads(Path(launch['path']).read_text())['args']
    if (support.flag_values(launch_args, '--tool-call-parser') != ['gemma4']
            or '--enable-auto-tool-choice' not in launch_args):
        raise RuntimeError('The seven smokes require the inspected Gemma4 tool parser and automatic tool choice')
    helper.MODEL = args.model_id
    install_thinking_call(helper)
    lock_path = WORK / 'runtime.lock'
    with lock_path.open('rb') as runtime_lock:
        try:
            fcntl.flock(runtime_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('W8A16 runtime is busy; refusing concurrent smoke checks/restart') from exc
        # Another invocation may have finished while local identities loaded.
        if output.exists():
            raise RuntimeError('Existing observations are preserved; use a new label')
        lock_stat = os.fstat(runtime_lock.fileno())
        listing = support.fetch_models()
        served = support.validate_model_listing(listing, args.model_id, checkpoint, adapter, launch)
        protocol = {
            'version': 1, 'runner': 'w8a16-frozen-behavior-smokes-thinking',
            'helper_sha256': HELPER_SHA256, 'integrity_helper_sha256': INTEGRITY_SHA256,
            'wrapper_sha256': sha(Path(__file__)), 'case_order': TESTS,
            'temperature': 0, 'max_tokens': MAX_TOKENS,
            'thinking': {'type': 'enabled', 'budget_tokens': THINKING_BUDGET},
            'chat_template_kwargs': {'enable_thinking': True},
            'context_limit': support.CONTEXT_LIMIT, 'context_margin': CONTEXT_MARGIN,
            'request_timeout_seconds': REQUEST_TIMEOUT, 'automatic_retries': False,
            'vllm_protocol_source_sha256': integrity.PROTOCOL_SOURCE_HASHES,
            'chat_template_sha256': integrity.THINKING_TEMPLATE_SHA256,
            'assertion_input': 'Final text blocks and tool_use blocks only; thinking never contributes.',
            'history_policy': 'Echo original assistant content, including thinking/signatures/tool IDs, into tool-result follow-ups.',
            'scoring': 'Unchanged frozen smoke predicates and conditional follow-ups. Completion is reported separately; no semantic regrading.',
        }
        identity = {
            'version': 1, 'model_id': args.model_id, 'checkpoint': checkpoint,
            'adapter': adapter, 'precision': precision, 'launch_profile': launch,
            'served_identity': served, 'runtime_lock': str(lock_path),
            'protocol': protocol, 'protocol_fingerprint': grading.fingerprint(protocol),
            'scope': 'Seven previously seen benign development tool/vision smokes; not a broad benchmark.',
            'comparison_scope': 'Separate thinking-enabled protocol; do not pool with thinking-disabled observations.',
            'quality_reference': 'Untouched W8A16 baseline; BF16 source is conversion input only.',
        }
        fingerprint = grading.ensure_manifest(
            WORK / f'{args.label}-behaviors-thinking-config.json', identity, [output])
        captured = []

        def guard():
            current_lock = lock_path.stat()
            if (current_lock.st_dev, current_lock.st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
                raise RuntimeError('Runtime lock file was replaced')
            if sha(Path(precision['path'])) != precision['sha256'] or sha(Path(launch['path'])) != launch['sha256']:
                raise RuntimeError('Precision manifest or launch profile changed')
            integrity.verify_thinking_protocol()
            integrity.check_activation_environment(dict(os.environ))
            if integrity.checkpoint_identity(checkpoint['path'], precision, grading) != checkpoint:
                raise RuntimeError('Base checkpoint changed')
            support.verify_adapter_stats(adapter)
            current = support.fetch_models()
            if support.validate_model_listing(current, args.model_id, checkpoint, adapter, launch) != served:
                raise RuntimeError('Requested base/adapter serving identity changed')
            return current

        original_urlopen = urllib.request.urlopen

        def request_json(route, payload):
            guard()
            data = json.dumps(payload).encode()
            request = urllib.request.Request(support.BASE + route, data=data,
                headers={'content-type': 'application/json', 'anthropic-version': '2023-06-01'})
            started = time.monotonic()
            with original_urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                value = json.load(response)
            entry = {'request_index': len(captured), 'route': route,
                     'request': payload, 'request_sha256': sha_bytes(data),
                     'seconds': time.monotonic() - started,
                     'response': value, 'response_sha256': grading.fingerprint(value)}
            captured.append(entry)
            if route == '/v1/messages':
                if value.get('model') != args.model_id:
                    raise RuntimeError('Response model differs from the requested base/LoRA ID')
                entry.update(response_parts(value))
                entry['generation_complete'] = value.get('stop_reason') in ('end_turn', 'tool_use', 'stop_sequence')
                entry['usage'] = value.get('usage')
                entry['stop_reason'] = value.get('stop_reason')
            guard()
            return value

        def urlopen(request, *call_args, **kwargs):
            current = guard()
            url = request if isinstance(request, str) else request.full_url
            if url == support.BASE + '/v1/models':
                base_id = served['base']['id']
                ordered = sorted(current['data'], key=lambda model: model['id'] != base_id)
                return io.BytesIO(json.dumps(current | {'data': ordered}).encode())
            if url != support.BASE + '/v1/messages' or not getattr(request, 'data', None):
                raise RuntimeError('Frozen helper attempted an unexpected endpoint')
            payload = json.loads(request.data)
            validate_request(payload, args.model_id)
            # Count the same images/tools/template/history that generation sees.
            count_payload = {key: value for key, value in payload.items()
                             if key in ('model', 'messages', 'system', 'tools',
                                        'tool_choice', 'chat_template_kwargs')}
            count = request_json('/v1/messages/count_tokens', count_payload).get('input_tokens')
            if type(count) is not int or count <= 0:
                raise RuntimeError('Native token count must be a positive integer')
            if count + MAX_TOKENS + CONTEXT_MARGIN > support.CONTEXT_LIMIT:
                raise RuntimeError('Smoke history leaves insufficient context for the fixed thinking/output budget')
            value = request_json('/v1/messages', payload)
            captured[-1]['counted_input_tokens'] = count
            return io.BytesIO(json.dumps(value).encode())

        # Only this isolated helper uses the proxy. Identity requests retain
        # their own transport; no process-global urllib state is changed.
        helper.urllib = types.SimpleNamespace(request=types.SimpleNamespace(
            Request=urllib.request.Request, urlopen=urlopen))
        original_argv = sys.argv
        started = time.monotonic()
        completed, error = False, None
        with tempfile.TemporaryDirectory(prefix=f'.{args.label}-thinking-smokes-', dir=WORK) as staging:
            helper.ROOT = Path(staging)
            staged_output = helper.ROOT / f'{args.label}-behaviors.json'
            try:
                sys.argv = [str(HELPER_PATH), '--label', args.label,
                            '--expected-checkpoint', checkpoint['path']]
                helper.main()
                result = json.loads(staged_output.read_text())
                if [row['test'] for row in result['records']] != TESTS:
                    raise RuntimeError('Frozen helper did not produce the expected seven checks in order')
                guard()
                if support.adapter_identity(args.adapter_dir, checkpoint) != adapter:
                    raise RuntimeError('Adapter content hashes changed')
                if (sha(HELPER_PATH) != HELPER_SHA256 or sha(INTEGRITY_PATH) != INTEGRITY_SHA256
                        or sha(Path(__file__)) != protocol['wrapper_sha256']):
                    raise RuntimeError('Pinned wrapper/helper changed during smoke checks')
                support.verify_frozen()
                completed = True
            except Exception as exc:
                error = repr(exc)
                raise
            finally:
                sys.argv = original_argv
                result = json.loads(staged_output.read_text()) if staged_output.exists() else {
                    'checkpoint': checkpoint['path'], 'records': []}
                for row in result['records']:
                    response = row['response']
                    row['response_sha256'] = grading.fingerprint(response) if response is not None else None
                    if response is not None:
                        row.update(response_parts(response))
                        row['generation_complete'] = response.get('stop_reason') in ('end_turn', 'tool_use', 'stop_sequence')
                    else:
                        row['generation_complete'] = None
                generations = [entry for entry in captured if entry['route'] == '/v1/messages']
                result.update(identity=identity, configuration_fingerprint=fingerprint,
                    completed=completed, error=error, native_requests=captured,
                    seconds=time.monotonic() - started,
                    passed=sum(row['passed'] for row in result['records']),
                    total=len(result['records']), expected_total=len(TESTS),
                    generation_count=len(generations),
                    all_generations_complete=bool(generations) and all(
                        entry.get('generation_complete', False) for entry in generations),
                    truncated_generations=sum(entry.get('stop_reason') == 'max_tokens' for entry in generations),
                    total_output_tokens=sum((entry.get('usage') or {}).get('output_tokens', 0) for entry in generations),
                    thinking_block_count=sum(len(entry.get('thinking_blocks', [])) for entry in generations),
                    refusal_evaluation_performed=False, full_quality_retention_gate_evaluated=False)
                result['result_sha256'] = grading.fingerprint(result)
                integrity.atomic_json(output, result)


if __name__ == '__main__':
    main()
