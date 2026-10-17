"""Start the private, manifest-pinned W8A16 server without changing production."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

WORK = Path(__file__).resolve().parent
SOURCE_WORK = WORK.parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    profile_path = SOURCE_WORK / 'launch-v1.json'
    profile = json.loads(profile_path.read_text())
    manifest_path = SOURCE_WORK / 'precision-manifest-v1.json'
    manifest = json.loads(manifest_path.read_text())
    base = Path(manifest['expected_base_checkpoint'])
    assert sha(profile_path) == manifest['launch_profile_sha256']
    assert sha(base / 'config.json') == manifest['config_sha256']
    assert sha(base / 'model.safetensors.index.json') == manifest['index_sha256']
    assert sha(base / 'conversion-receipt.json') == manifest['conversion_receipt_sha256']
    assert profile['args'][2] == str(base)
    assert 'VLLM_MARLIN_INPUT_DTYPE' not in profile['environment']
    assert '--fully-sharded-loras' not in profile['args']
    assert profile['args'][profile['args'].index('--host') + 1] == '127.0.0.1'
    unit = 'redcell-authorized-workflow-' + str(time.time_ns())
    receipt = {
        'unit': unit, 'checkpoint': str(base), 'profile_path': str(profile_path),
        'profile_sha256': sha(profile_path), 'precision_manifest_sha256': sha(manifest_path),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'generation_validated': False, 'production_modified': False,
        'source_launcher_sha256': 'c583411fae73972dfc0c6d39d7429bbf313d289ea2e0d0f652bd48677c22d61a',
    }
    with (WORK / 'runtime.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        production = subprocess.run(['systemctl', 'is-active', 'qwen-vllm.service'],
                                    capture_output=True, text=True).stdout.strip()
        assert production == 'inactive', f'Production is {production}'
        with socket.socket() as probe:
            assert probe.connect_ex(('127.0.0.1', 8002)) != 0, 'Port8002 already in use'
        command = [
            'sudo', '-n', 'systemd-run', '--collect', '--unit=' + unit, '--uid=isnotreal',
            '--property=WorkingDirectory=' + str(WORK), '--property=Restart=no',
            '--property=TimeoutStopSec=30',
            '--property=UnsetEnvironment=' + ' '.join(profile['unset_environment']),
            '--property=StandardOutput=append:' + str(WORK / 'runtime.log'),
            '--property=StandardError=append:' + str(WORK / 'runtime.log'),
        ]
        command += [f'--setenv={key}={value}' for key, value in profile['environment'].items()]
        started = time.monotonic()
        try:
            subprocess.run(command + profile['args'], check=True)
            write(WORK / 'runtime-status.json', {**receipt, 'status': 'starting'})
            for _ in range(240):
                state = subprocess.run(['systemctl', 'is-active', unit],
                                       capture_output=True, text=True).stdout.strip()
                if state not in ('active', 'activating'):
                    raise RuntimeError(f'{unit}: {state}; inspect runtime-v1.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8002/v1/models', timeout=3) as response:
                        models = json.load(response)['data']
                    assert models[0]['root'] == str(base), 'Unexpected base checkpoint'
                    assert any(row['id'] == 'redcell-w8a16-strength-1.5' for row in models)
                    receipt.update(status='loaded', advertised_models=models,
                                   load_seconds=time.monotonic() - started)
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(5)
            else:
                raise TimeoutError('Private W8A16 server did not become healthy within20minutes')
        except Exception as error:
            receipt.update(status='failed', error=repr(error), elapsed_seconds=time.monotonic() - started)
            write(WORK / 'runtime-status.json', receipt)
            write(WORK / (unit + '-load.json'), receipt)
            raise
        write(WORK / 'runtime-status.json', receipt)
        write(WORK / (unit + '-load.json'), receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
