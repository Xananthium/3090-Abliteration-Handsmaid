"""Follow the matched development batch with the same 24 cases in thinking mode."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

WORK = Path(__file__).resolve().parent
PYTHON = '/home/isnotreal/vllm-env/bin/python'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def status(stage, **extra):
    record = {'stage': stage, 'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
    write(WORK / 'thinking-batch-status-v1.json', record)
    print(json.dumps(record), flush=True)


def main():
    with (WORK / 'thinking-batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = WORK / 'integrity-batch-result-v1.json'
        status('waiting-for-integrity-batch')
        for _ in range(720):
            if completed.exists():
                assert json.loads(completed.read_text())['status'] == 'completed'
                break
            first_status = json.loads((WORK / 'integrity-batch-status-v1.json').read_text())
            assert first_status['stage'] != 'failed', first_status
            time.sleep(5)
        else:
            raise TimeoutError('Integrity batch has not completed; thinking checks were not started')
        profile = WORK / 'launch-v1.json'
        manifest_path = WORK / 'precision-manifest-v1.json'
        manifest = json.loads(manifest_path.read_text())
        assert sha(profile) == manifest['launch_profile_sha256']
        identities = json.loads((WORK / 'adapter-identities-v1.json').read_text())['adapters']
        cases = [('baseline', 'qwen3.6-35b-hauhau-w4a16-vision', None)]
        for name in ('strength-1.0-vllm', 'strength-1.5-vllm', 'attention-1.5-mlp-1.0-vllm'):
            branch = identities[name]
            cases.append((name.removesuffix('-vllm'), branch['model_id'], branch['path']))
        results = []
        for name, model_id, adapter in cases:
            label = f'w8a16-{name}-thinking24-v1'
            command = [PYTHON, '-u', str(WORK / 'evaluate_integrity_thinking.py'),
                       '--label', label, '--model-id', model_id, '--launch-profile', str(profile),
                       '--precision-manifest', str(manifest_path)]
            if adapter:
                command += ['--adapter-dir', adapter]
            status('thinking-integrity', label=label, model_id=model_id)
            subprocess.run(command, check=True)
            path = WORK / f'{label}-summary.json'
            summary = json.loads(path.read_text())
            assert summary['status'] == 'completed' and summary['record_count'] == 48
            results.append({'label': name, 'path': str(path), 'sha256': sha(path),
                            'capabilities': summary['capabilities']})
        result = {'status': 'completed', 'finished_utc': datetime.now(timezone.utc).isoformat(),
                  'comparisons': results, 'precision_manifest_sha256': sha(manifest_path),
                  'profile_sha256': sha(profile), 'quality_reference': 'Untouched W8A16 baseline',
                  'scope': '24 previously seen benign reasoning, instruction, consistency and recovery cases, two repeats, thinking explicitly enabled. Separate diagnostic; not broad quality or refusal evidence.',
                  'production_modified': False, 'publication_performed': False}
        write(WORK / 'thinking-batch-result-v1.json', result)
        status('completed', result=str(WORK / 'thinking-batch-result-v1.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=repr(error))
        raise
