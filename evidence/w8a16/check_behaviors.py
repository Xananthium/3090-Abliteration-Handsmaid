"""Run seven frozen benign tool/vision smoke checks on manifest-pinned W8A16.

These are previously seen development checks, not a broad capability benchmark.
No model, adapter, service, or scoring definition is changed by this wrapper.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import types
import urllib.request


WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
INTEGRITY_PATH = WORK / 'evaluate_integrity.py'
INTEGRITY_SHA256 = 'dc7129030f78ac34887916e97e75b61c81d143920b0fc94d54a39500c7e73a07'
HELPER_PATH = ROOT / 'check_behaviors.py'
HELPER_SHA256 = 'fe7a1d2a6c7f4b340bb3ee10f65bde28191e4f1008de9b810bf9156e0fdc8590'
TESTS = ['forced_tool_call', 'tool_result_roundtrip', 'automatic_tool_choice',
         'pivot_after_two_errors', 'workflow_completion', 'vision_red', 'vision_blue']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_helpers():
    for path, expected in ((INTEGRITY_PATH, INTEGRITY_SHA256), (HELPER_PATH, HELPER_SHA256)):
        if sha(path) != expected:
            raise RuntimeError(f'Pinned helper changed: {path}')
    integrity = types.ModuleType('_w8a16_integrity_identity_helpers')
    integrity.__file__ = str(INTEGRITY_PATH)
    exec(compile(INTEGRITY_PATH.read_bytes(), str(INTEGRITY_PATH), 'exec'), integrity.__dict__)
    support, grading, _, _ = integrity.load_support()
    behavior = types.ModuleType('_isolated_w8a16_behavior_helpers')
    behavior.__file__ = str(HELPER_PATH)
    exec(compile(HELPER_PATH.read_bytes(), str(HELPER_PATH), 'exec'), behavior.__dict__)
    return integrity, support, grading, behavior


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
    output = WORK / f'{args.label}-behaviors.json'
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
    helper.ROOT = WORK
    helper.MODEL = args.model_id
    lock_path = WORK / 'runtime.lock'
    with lock_path.open('rb') as runtime_lock:
        try:
            fcntl.flock(runtime_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('W8A16 runtime is busy; refusing concurrent smoke checks/restart') from exc
        lock_stat = os.fstat(runtime_lock.fileno())
        listing = support.fetch_models()
        served = support.validate_model_listing(listing, args.model_id, checkpoint, adapter, launch)
        identity = {
            'version': 1, 'runner': 'w8a16-frozen-behavior-smokes', 'model_id': args.model_id,
            'checkpoint': checkpoint, 'adapter': adapter, 'precision': precision,
            'launch_profile': launch, 'served_identity': served, 'runtime_lock': str(lock_path),
            'helper_sha256': HELPER_SHA256, 'integrity_helper_sha256': INTEGRITY_SHA256,
            'wrapper_sha256': sha(Path(__file__)), 'case_order': TESTS,
            'temperature': 0, 'max_tokens': 128, 'thinking': False,
            'chat_template_kwargs': {'enable_thinking': False},
            'scope': 'Seven previously seen benign tool/vision smoke checks; not a broad capability benchmark.',
            'scoring': 'Unchanged frozen smoke predicates, including conditional follow-up checks.',
            'quality_reference': 'Untouched W8A16 baseline; BF16 source is conversion input only.'}
        fingerprint = grading.ensure_manifest(WORK / f'{args.label}-behaviors-config.json', identity, [output])
        captured = []

        def guard():
            current_lock = lock_path.stat()
            if (current_lock.st_dev, current_lock.st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
                raise RuntimeError('Runtime lock file was replaced')
            if sha(Path(precision['path'])) != precision['sha256'] or sha(Path(launch['path'])) != launch['sha256']:
                raise RuntimeError('Precision manifest or launch profile changed')
            integrity.check_activation_environment(dict(os.environ))
            if integrity.checkpoint_identity(checkpoint['path'], precision, grading) != checkpoint:
                raise RuntimeError('Base checkpoint changed')
            support.verify_adapter_stats(adapter)
            current = support.fetch_models()
            if support.validate_model_listing(current, args.model_id, checkpoint, adapter, launch) != served:
                raise RuntimeError('Requested base/adapter serving identity changed')
            return current

        original_urlopen = urllib.request.urlopen

        def urlopen(request, *call_args, **kwargs):
            current = guard()
            url = request if isinstance(request, str) else request.full_url
            if url == support.BASE + '/v1/models':
                # The frozen helper assumes its first advertised model is the
                # base. Normalize only that listing order after validating it.
                base_id = served['base']['id']
                ordered = sorted(current['data'], key=lambda model: model['id'] != base_id)
                return io.BytesIO(json.dumps(current | {'data': ordered}).encode())
            if url != support.BASE + '/v1/messages' or not getattr(request, 'data', None):
                raise RuntimeError('Frozen helper attempted an unexpected endpoint')
            payload = json.loads(request.data)
            if (payload.get('model') != args.model_id or payload.get('temperature') != 0
                    or payload.get('max_tokens') != 128 or payload.get('thinking') != {'type': 'disabled'}
                    or payload.get('chat_template_kwargs') != {'enable_thinking': False}):
                raise RuntimeError('Smoke request differs from the pinned model/generation settings')
            started = time.monotonic()
            with original_urlopen(request, *call_args, **kwargs) as response:
                value = json.load(response)
            captured.append({'request_index': len(captured), 'request_sha256': sha_bytes(request.data),
                             'seconds': time.monotonic() - started,
                             'response': value, 'response_sha256': grading.fingerprint(value)})
            if value.get('model') != args.model_id:
                raise RuntimeError('Response model differs from the requested base/LoRA ID')
            guard()
            return io.BytesIO(json.dumps(value).encode())

        # Isolate the transport proxy to this helper module; no global urllib
        # state changes affect another evaluator or the identity transport.
        helper.urllib = types.SimpleNamespace(request=types.SimpleNamespace(
            Request=urllib.request.Request, urlopen=urlopen))
        original_call = helper.call

        def call(*call_args, **kwargs):
            kwargs['chat_template_kwargs'] = {'enable_thinking': False}
            return original_call(*call_args, **kwargs)

        helper.call = call
        original_argv = sys.argv
        started = time.monotonic()
        completed, error = False, None
        try:
            sys.argv = [str(HELPER_PATH), '--label', args.label,
                        '--expected-checkpoint', checkpoint['path']]
            helper.main()
            result = json.loads(output.read_text())
            if [row['test'] for row in result['records']] != TESTS:
                raise RuntimeError('Frozen helper did not produce the expected seven checks in order')
            guard()
            if support.adapter_identity(args.adapter_dir, checkpoint) != adapter:
                raise RuntimeError('Adapter content hashes changed')
            if (sha(HELPER_PATH) != HELPER_SHA256 or sha(INTEGRITY_PATH) != INTEGRITY_SHA256
                    or sha(Path(__file__)) != identity['wrapper_sha256']):
                raise RuntimeError('Pinned wrapper/helper changed during smoke checks')
            support.verify_frozen()
            completed = True
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            sys.argv = original_argv
            result = json.loads(output.read_text()) if output.exists() else {'checkpoint': checkpoint['path'], 'records': []}
            for row in result['records']:
                row['response_sha256'] = grading.fingerprint(row['response']) if row['response'] is not None else None
            result.update(identity=identity, configuration_fingerprint=fingerprint,
                          completed=completed, error=error, native_requests=captured,
                          seconds=time.monotonic() - started,
                          passed=sum(row['passed'] for row in result['records']),
                          total=len(result['records']), refusal_evaluation_performed=False,
                          full_quality_retention_gate_evaluated=False)
            result['result_sha256'] = grading.fingerprint(result)
            integrity.atomic_json(output, result)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


if __name__ == '__main__':
    main()
