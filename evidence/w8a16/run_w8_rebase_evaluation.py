#!/usr/bin/env python3
"""Wait for both thinking controllers, register one pinned LoRA, run four checks.

Running this controller is an explicit runtime action. Import, --help and the
offline test file do not contact the server. No service, launch profile, source
adapter or manifest is changed. Completed observations are immutable; an
interrupted capability group needs the explicit whole-group retry option.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import types
import urllib.request


WORK = Path(__file__).resolve().parent
PYTHON = '/home/isnotreal/vllm-env/bin/python'
BASE = '/home/isnotreal/models/redcell-baseline-w8a16-native'
ALIAS = 'qwen3.6-35b-hauhau-w4a16-vision'
NAME = 'nf4-direction-w8-weight-strength-1.0'
MODEL_ID = 'redcell-w8a16-nf4-direction-w8-weight-strength-1.0-v1'
ADAPTER = WORK / 'adapters/nf4-direction-w8-weight-strength-1.0-vllm-v1'
RECEIPT_SHA = '38aa74f31d44b785704858c545837edcaea1585f2a503f230ace726cb6b1b310'
OLD_NAMES = ('baseline', 'strength-1.0', 'strength-1.5', 'attention-1.5-mlp-1.0')
PREFIX = 'w8-rebase-evaluation'
PINNED = {
    'evaluate_integrity.py': 'dc7129030f78ac34887916e97e75b61c81d143920b0fc94d54a39500c7e73a07',
    'evaluate_integrity_thinking.py': 'b3054a303ebd8371a57bd0951d93dc7cc8ff974a06180f027462d5b621001696',
    'check_behaviors.py': '9928ef1517927b81cd74b9a447989fe5e30e1002dd8cdbffeb8dbeed48fb180b',
    'check_behaviors_thinking.py': 'cdfd43d3fcf7573e42d42047a3e40b9abebd806a6ab344cdeeb6e84f7738da6b',
    'analyze_integrity.py': '8bff2efeca121cb3fe100386d29736edfb2c3abfdc86fa6200658a020ef6f11e',
    'analyze_thinking_integrity.py': 'a2c907c08d1963562516c47d984ae5f67a136689a23373c56d6fded1539d33b6',
    'run_thinking_batch.py': '08ec7b4892fe6094456353e4ec8df746b2731c19aaabb5de98e2d9cc4a1b3a57',
    'run_thinking_behaviors.py': '63350a4126b69d8f5b3278623df7ebb50d97e79c4381ddf47c04758bdf16df4b',
    'launch-v1.json': '394fa5c48b6d0206f9e9cf90deba3c3c1701162d9aef8022085cbc08d7ef4f58',
    'precision-manifest-v1.json': 'a8806a05173567d3f86df3781c3feaa8845fe5e1872fc8c7c6a45361ea7f931c',
    'adapter-identities-v1.json': 'b6c8059c4e3c6fefc0c736b223740baf4130ce194a84dd283eefd49ab1687929',
}
OUTPUT_HASHES = {
    'adapter_model.safetensors': 'd9e1cc5bb190673c80380f38258576b3b1dec0d7d93d36e38c2533ba36a9b7cb',
    'adapter_config.json': '8f404726dab2bbd3ddbe99630a37ed3ded557ab077659c556f03e0ec36804891',
    'projection-validation.json': 'b5e726fe7496c0da58c2cee6cc838d716add24e7f3d7e88fa294d1f255833d46',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stat_key(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def status(stage, **extra):
    value = {'stage': stage, 'updated_utc': now(), **extra}
    atomic(WORK / f'{PREFIX}-status-v1.json', value)
    print(json.dumps(value), flush=True)


def signed(value, field='result_sha256'):
    require(value.get(field) == fingerprint({k: v for k, v in value.items() if k != field}),
            f'Invalid {field}')


def pin(path, expected=None, size=None):
    path = Path(path)
    before = stat_key(path)
    digest = sha(path)
    require(before == stat_key(path), f'File changed while hashing: {path}')
    require(expected is None or digest == expected, f'File hash changed: {path}')
    require(size is None or before[2] == size, f'File size changed: {path}')
    return {'sha256': digest, 'bytes': before[2], 'stat': before}


def module(name):
    path = WORK / name
    require(sha(path) == PINNED[name], f'Pinned Python helper changed: {name}')
    result = types.ModuleType('_rebase_' + name.replace('.', '_'))
    result.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), result.__dict__)
    return result


def wait_upstream(timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while True:
        ready = {}
        for prefix in ('thinking-batch', 'thinking-behaviors'):
            state = WORK / f'{prefix}-status-v1.json'
            if state.exists():
                require(read(state).get('stage') != 'failed', f'Upstream {prefix} failed')
            path = WORK / f'{prefix}-result-v1.json'
            if path.exists():
                value = read(path)
                require(value.get('status') != 'failed', f'Upstream {prefix} failed')
                if value.get('status') == 'completed':
                    ready[prefix] = {'path': str(path), **pin(path)}
        if len(ready) == 2:
            return ready
        require(time.monotonic() < deadline, 'Timed out waiting for both completed upstream controllers')
        time.sleep(min(30, max(0, deadline - time.monotonic())))


def validate_capacity(profile, support):
    argv = profile['args']
    values = support.flag_values(argv, '--max-cpu-loras')
    require(len(values) == 1 and int(values[0]) >= 4,
            'Pinned launch must allow at least four CPU LoRA slots')
    require(support.flag_values(argv, '--max-loras') == ['1']
            and '--enable-lora' in argv and '--no-fully-sharded-loras' in argv
            and profile.get('environment', {}).get('VLLM_ALLOW_RUNTIME_LORA_UPDATING') == '1',
            'Pinned launch lacks the expected reversible LoRA settings')
    return int(values[0])


def prepare():
    inputs = {str(WORK / name): pin(WORK / name, expected) for name, expected in PINNED.items()}
    inputs[str(Path(__file__).resolve())] = pin(Path(__file__).resolve())
    receipt_path = ADAPTER / 'provenance-validation.json'
    inputs[str(receipt_path)] = pin(receipt_path, RECEIPT_SHA)
    receipt = read(receipt_path)
    require(receipt.get('complete') is True and receipt.get('status') == 'completed_cpu_builder_validation',
            'Candidate CPU build is not complete')
    require(receipt['candidate_identity'] == {'base_model_name_or_path': BASE,
                'model_id': MODEL_ID, 'path': str(ADAPTER)} and receipt['model_id'] == MODEL_ID
            and receipt['output_directory'] == str(ADAPTER), 'Candidate receipt identity differs')
    artifacts = receipt['output_artifacts']
    require({row['path'] for row in artifacts} == {str(ADAPTER / n) for n in OUTPUT_HASHES}
            and len(artifacts) == len(OUTPUT_HASHES), 'Candidate artifact coverage differs')
    for row in artifacts:
        path = Path(row['path'])
        require(row['sha256'] == OUTPUT_HASHES[path.name], 'Candidate artifact is not the selected build')
        inputs[str(path)] = pin(path, row['sha256'], row['bytes'])
    for row in [receipt['builder'], receipt['preflight'], *receipt['inputs'].values()]:
        inputs[row['path']] = pin(row['path'], row['sha256'], row['bytes'])
    baseline_stats = {path: row['stat'] for path, row in receipt['baseline_safetensors'].items()}
    require(baseline_stats and all(stat_key(path) == value for path, value in baseline_stats.items()),
            'W8 checkpoint shard identity changed since the fully hashed CPU build')
    off, on = module('evaluate_integrity.py'), module('evaluate_integrity_thinking.py')
    support, grading, _, cases = off.load_support()
    on_support, on_grading, _, on_cases = on.load_support()
    require(cases == on_cases and len(cases) == 24, 'Frozen authored cases differ across modes')
    off.check_activation_environment(dict(os.environ))
    on.verify_thinking_protocol()
    profile = WORK / 'launch-v1.json'
    precision = off.load_precision_manifest(str(WORK / 'precision-manifest-v1.json'), BASE,
                                            str(profile), support, grading)
    checkpoint = off.checkpoint_identity(BASE, precision, grading)
    max_cpu_loras = validate_capacity(read(profile), support)
    registered = read(WORK / 'adapter-identities-v1.json')['adapters']
    require(set(registered) == {name + '-vllm' for name in OLD_NAMES[1:]}, 'Old adapter registry changed')
    specs = {}
    for name in (*OLD_NAMES, NAME):
        registered_row = registered[name + '-vllm'] if name in OLD_NAMES[1:] else None
        path = str(ADAPTER) if name == NAME else (registered_row['path'] if registered_row else None)
        model_id = MODEL_ID if name == NAME else (registered_row['model_id'] if registered_row else ALIAS)
        adapter = support.adapter_identity(path, checkpoint)
        if registered_row:
            for filename, key in [('adapter_config.json', 'adapter_config_sha256'),
                                  ('adapter_model.safetensors', 'adapter_model_sha256')]:
                require(adapter['files'][filename]['sha256'] == registered_row[key], 'Old adapter content changed')
        if adapter:
            for filename, identity in adapter['files'].items():
                inputs[str(Path(path) / filename)] = pin(Path(path) / filename, identity['sha256'])
        launch = off.launch_identity(str(profile), checkpoint, adapter, precision, support, grading)
        specs[name] = {'name': name, 'model_id': model_id, 'expected': {
            'checkpoint': checkpoint, 'adapter': adapter, 'precision': precision, 'launch_profile': launch}}
    require(len({spec['model_id'] for spec in specs.values()}) == 5, 'Expected base plus four distinct adapters')
    for filename, identity in checkpoint['files'].items():
        inputs[str(Path(BASE) / filename)] = pin(Path(BASE) / filename, identity['sha256'])
    common, analysis_on = module('analyze_integrity.py'), module('analyze_thinking_integrity.py')
    audit_inputs = {}
    frozen_cases, grade = common.frozen_inputs(audit_inputs)
    require(cases == frozen_cases, 'Accounting cases differ from frozen runners')
    validator, constants = analysis_on.load_validator(common, audit_inputs)
    for path, identity in audit_inputs.items():
        inputs[path] = pin(path, identity['sha256'])
    wrappers = {False: module('check_behaviors.py'), True: module('check_behaviors_thinking.py')}
    for wrapper in wrappers.values():
        wrapper.load_helpers()  # Read-only imports and source validation; never invoke main.
    return types.SimpleNamespace(inputs=inputs, baseline_stats=baseline_stats, receipt=receipt,
        support=support, grading=grading, cases=cases, grade=grade, off=off, on=on,
        on_support=on_support, on_grading=on_grading, common=common, analysis_on=analysis_on,
        validator=validator, constants=constants, wrappers=wrappers, specs=specs,
        max_cpu_loras=max_cpu_loras, protocols={}, behavior_templates={}, upstream=None)


def verify_local(ctx, full=False):
    ctx.off.check_activation_environment(dict(os.environ))
    for path, expected in ctx.inputs.items():
        require(stat_key(path) == expected['stat'], f'Pinned input identity changed: {path}')
        if full:
            require(sha(path) == expected['sha256'], f'Pinned input content changed: {path}')
    for path, expected in ctx.baseline_stats.items():
        require(stat_key(path) == expected, f'Base checkpoint shard changed: {path}')
    expected = ctx.specs[NAME]['expected']
    require(ctx.off.checkpoint_identity(BASE, expected['precision'], ctx.grading) == expected['checkpoint'],
            'Base checkpoint metadata changed')
    for spec in ctx.specs.values():
        ctx.support.verify_adapter_stats(spec['expected']['adapter'])


def validate_identity(identity, spec, ctx):
    require(identity.get('model_id') == spec['model_id'], 'Output model ID differs')
    for key, expected in spec['expected'].items():
        require(identity.get(key) == expected, f'Output {key} identity differs')
    require(identity.get('runtime_lock') == str(WORK / 'runtime.lock'), 'Output runtime lock differs')
    served = identity['served_identity']
    listing = {'data': [served['base']] + ([served['requested']] if spec['expected']['adapter'] else [])}
    require(ctx.support.validate_model_listing(listing, spec['model_id'], spec['expected']['checkpoint'],
        spec['expected']['adapter'], spec['expected']['launch_profile']) == served,
        'Stored base/adapter routing identity differs')


def stage_spec(kind, thinking, name=NAME):
    suffix = ('thinking24' if thinking else 'authored24') if kind == 'capability' else (
        'tools-vision-thinking' if thinking else 'tools-vision')
    label = f'w8a16-{name}-{suffix}-v1'
    end = 'summary' if kind == 'capability' else ('behaviors-thinking' if thinking else 'behaviors')
    config_end = 'config' if kind == 'capability' else end + '-config'
    runner = ('evaluate_integrity' if kind == 'capability' else 'check_behaviors') + (
        '_thinking' if thinking else '') + '.py'
    return {'kind': kind, 'thinking': thinking, 'name': name, 'label': label, 'runner': runner,
            'path': str(WORK / f'{label}-{end}.json'), 'config_path': str(WORK / f'{label}-{config_end}.json')}


def validate_capability(stage, ctx):
    inputs = {}
    summary = read(stage['path'])
    entry = {'label': stage['name'], 'path': stage['path'], 'sha256': sha(stage['path']),
             'capabilities': summary['capabilities'], 'output_tokens': summary['total_output_tokens_including_retries'],
             'batch_wall_seconds': summary['sum_committed_group_wall_seconds']}
    batch = {'base_checkpoint': BASE, 'profile_sha256': PINNED['launch-v1.json'],
             'precision_manifest_sha256': PINNED['precision-manifest-v1.json']}
    if stage['thinking']:
        result, rows, config = ctx.analysis_on.load_run(entry, batch, ctx.cases, ctx.grade,
            ctx.validator, ctx.constants, ctx.common, inputs)
    else:
        result, rows, config = ctx.common.load_run(entry, batch, ctx.cases, ctx.grade, inputs)
    validate_identity(config, ctx.specs[stage['name']], ctx)
    require(config.get('endpoint') == ctx.support.BASE, 'Capability endpoint differs')
    require(config['protocol']['script_sha256'] == PINNED[stage['runner']], 'Capability runner differs')
    if stage['thinking'] in ctx.protocols:
        require(config['protocol'] == ctx.protocols[stage['thinking']], 'Protocol differs from matched baseline')
    # Run the frozen runner's own record validation in addition to independent accounting.
    runner = ctx.on if stage['thinking'] else ctx.off
    support = ctx.on_support if stage['thinking'] else ctx.support
    grading = ctx.on_grading if stage['thinking'] else ctx.grading
    schedule = runner.group_schedule(ctx.cases, support.REPEATS)
    groups = runner.load_committed(WORK / f'{stage["label"]}-groups', schedule, config,
        fingerprint(config), ctx.cases, support, grading)
    require(len(groups) == 12 and result['pending_group_attempts'] == 0,
            'Completed capability output has missing groups or unresolved pending files')
    require(summary['served_identity'] == config['served_identity'], 'Summary serving identity differs')
    inputs = {path: pin(path, item['sha256'], item['bytes']) for path, item in inputs.items()}
    return {'task_completed': True, **stage, 'sha256': sha(stage['path']),
            'configuration_fingerprint': fingerprint(config), 'protocol_fingerprint': config['protocol_fingerprint'],
            'score': result, 'all_predicates_passed': result['correct'] == 48,
            'all_generations_complete': result['incomplete_outputs'] == 0,
            'truncated_generations': result['final_truncated_outputs'], 'artifacts': inputs}, config, rows


def smoke_predicate(test, response, thinking, wrapper):
    if response is None:
        return False
    content = response['content']
    uses = [b for b in content if b.get('type') == 'tool_use']
    text = wrapper.final_text(response) if thinking else ''.join(b.get('text', '') for b in content).strip()
    tool = {'forced_tool_call': ('add', {'a': 3, 'b': 4}),
            'automatic_tool_choice': ('add', {'a': 19, 'b': 23}),
            'pivot_after_two_errors': ('read_text', {'path': '/reports/quarterly.txt'})}.get(test)
    if tool:
        return len(uses) == 1 and uses[0].get('name') == tool[0] and uses[0].get('input') == tool[1]
    if test in ('tool_result_roundtrip', 'workflow_completion'):
        return ('7' if test == 'tool_result_roundtrip' else '47') in text and not uses
    require(test in ('vision_red', 'vision_blue'), 'Unexpected smoke predicate')
    return test.removeprefix('vision_') in text.lower()


def behavior_template(identity):
    return {k: v for k, v in identity.items() if k not in
            ('model_id', 'checkpoint', 'adapter', 'precision', 'launch_profile', 'served_identity')}


def validate_behavior(stage, ctx):
    path = Path(stage['path'])
    before = pin(path)
    result = read(path)
    signed(result)
    identity, thinking = result['identity'], stage['thinking']
    wrapper = ctx.wrappers[thinking]
    validate_identity(identity, ctx.specs[stage['name']], ctx)
    require(behavior_template(identity) == ctx.behavior_templates[thinking],
            'Behavior protocol differs from matched baseline')
    config = read(stage['config_path'])
    require(config == {'configuration': identity, 'configuration_fingerprint': fingerprint(identity)}
            and result['configuration_fingerprint'] == fingerprint(identity), 'Behavior configuration differs')
    rows = result['records']
    require(result.get('completed') is True and result.get('error') is None
            and result.get('total') == len(rows) == 7 and [r['test'] for r in rows] == wrapper.TESTS,
            'Behavior task is not completely recorded')
    require(result.get('checkpoint') == BASE, 'Behavior checkpoint differs')
    for row in rows:
        require(type(row.get('passed')) is bool
                and row['passed'] is smoke_predicate(row['test'], row['response'], thinking, wrapper),
                'Frozen behavior predicate disagrees with saved response')
        require(row.get('response_sha256') == (fingerprint(row['response']) if row['response'] is not None else None),
                'Behavior response checksum differs')
    require(result.get('passed') == sum(r['passed'] for r in rows), 'Behavior totals differ')
    require((rows[1]['response'] is not None) == rows[0]['passed']
            and (rows[4]['response'] is not None) == rows[3]['passed'], 'Conditional smoke sequence differs')
    count = 5 + int(rows[0]['passed']) + int(rows[3]['passed'])
    native = result['native_requests']
    generations = [v for v in native if v.get('route') == '/v1/messages'] if thinking else native
    require(len(generations) == count, 'Behavior request count differs')
    if thinking:
        require([v.get('route') for v in native] == ['/v1/messages/count_tokens', '/v1/messages'] * count,
                'Native thinking count/generation sequence differs')
    for index, entry in enumerate(native):
        require(entry['request_index'] == index and entry['response_sha256'] == fingerprint(entry['response']),
                'Behavior native request index/response hash differs')
        require(isinstance(entry['request_sha256'], str) and len(entry['request_sha256']) == 64,
                'Behavior request digest missing')
        if thinking:
            require(entry['request_sha256'] == wrapper.sha_bytes(json.dumps(entry['request']).encode()),
                    'Behavior native request hash differs')
    responded = [r for r in rows if r['response'] is not None]
    require(len(responded) == count, 'Behavior response records missing')
    for index, (row, entry) in enumerate(zip(responded, generations)):
        response = entry['response']
        require(row['response'] == response and response.get('model') == identity['model_id'],
                'Behavior response model/record differs')
        if thinking:
            wrapper.validate_request(entry['request'], identity['model_id'])
            counted = native[index * 2]
            count_request = {k: v for k, v in entry['request'].items() if k in
                             ('model', 'messages', 'system', 'tools', 'tool_choice', 'chat_template_kwargs')}
            tokens = counted['response'].get('input_tokens')
            require(counted['request'] == count_request and type(tokens) is int and tokens > 0
                    and tokens + 4096 + 64 <= 8192 and entry['counted_input_tokens'] == tokens,
                    'Behavior native context accounting differs')
            complete = response.get('stop_reason') in ('end_turn', 'tool_use', 'stop_sequence')
            parts = wrapper.response_parts(response)
            require(entry.get('generation_complete') is complete and row.get('generation_complete') is complete
                    and entry.get('stop_reason') == response.get('stop_reason')
                    and entry.get('usage') == response.get('usage')
                    and all(entry.get(k) == v and row.get(k) == v for k, v in parts.items()),
                    'Behavior thinking/final-text/completion capture differs')
    complete = all(e['response'].get('stop_reason') in ('end_turn', 'tool_use', 'stop_sequence') for e in generations)
    truncated = sum(e['response'].get('stop_reason') == 'max_tokens' for e in generations)
    if thinking:
        require(result['expected_total'] == 7 and result['generation_count'] == count
                and result['all_generations_complete'] is complete and result['truncated_generations'] == truncated
                and result['total_output_tokens'] == sum((e['response'].get('usage') or {}).get('output_tokens', 0) for e in generations)
                and result['thinking_block_count'] == sum(len(e['thinking_blocks']) for e in generations),
                'Behavior aggregate completion/usage differs')
        require(all(r.get('generation_complete') is None for r in rows if r['response'] is None),
                'Skipped conditional behavior claims a generation')
    require(pin(path) == before, 'Behavior output changed while validating')
    artifacts = {str(path): before, stage['config_path']: pin(stage['config_path'])}
    return {'task_completed': True, **stage, 'sha256': before['sha256'], 'model_id': identity['model_id'],
            'configuration_fingerprint': fingerprint(identity), 'passed': result['passed'], 'total': 7,
            'all_predicates_passed': result['passed'] == 7, 'all_generations_complete': complete,
            'generation_count': count, 'truncated_generations': truncated,
            'generation_stop_reasons': [e['response'].get('stop_reason') for e in generations],
            'request_audit_limit': None if thinking else 'Frozen off wrapper retains request hashes, not request bodies.',
            'artifacts': artifacts}


def validate_upstream(ctx, upstream):
    batch = read(upstream['thinking-batch']['path'])
    behaviors = read(upstream['thinking-behaviors']['path'])
    require(batch.get('status') == behaviors.get('status') == 'completed', 'Upstream completion changed')
    signed(behaviors)
    require(batch['profile_sha256'] == PINNED['launch-v1.json']
            and batch['precision_manifest_sha256'] == PINNED['precision-manifest-v1.json'], 'Upstream precision differs')
    require([v['label'] for v in batch['comparisons']] == list(OLD_NAMES)
            and [v['branch'] for v in behaviors['comparisons']] == list(OLD_NAMES), 'Upstream branch coverage differs')
    require(behaviors['upstream']['sha256'] == upstream['thinking-batch']['sha256']
            and behaviors['upstream']['path'] == upstream['thinking-batch']['path']
            and behaviors['driver_sha256'] == PINNED['run_thinking_behaviors.py'], 'Upstream controller linkage differs')
    checker = module('run_thinking_behaviors.py')
    require(behaviors['input_sha256'] == checker.PINNED, 'Upstream behavior input pins differ')
    for name, cap_entry, behavior_entry in zip(OLD_NAMES, batch['comparisons'], behaviors['comparisons']):
        stage = stage_spec('capability', True, name)
        require(cap_entry['path'] == stage['path'] and sha(stage['path']) == cap_entry['sha256'],
                'Upstream capability path/hash differs')
        result, config, _ = validate_capability(stage, ctx)
        require(cap_entry['capabilities'] == read(stage['path'])['capabilities'], 'Upstream score differs')
        ctx.protocols[True] = config['protocol']
        ctx.inputs.update(result['artifacts'])
        branch = dict(ctx.specs[name], label=stage_spec('behavior', True, name)['label'])
        path = Path(stage_spec('behavior', True, name)['path'])
        checked = checker.validate_output(path, branch, ctx.wrappers[True], ctx.support, ctx.grading)
        require(checked == behavior_entry, 'Upstream behavior comparison differs')
        ctx.inputs[str(path)] = pin(path, checked['sha256'])
        ctx.inputs[checked['config_path']] = pin(checked['config_path'], checked['config_sha256'])
        if name == 'baseline':
            ctx.behavior_templates[True] = behavior_template(read(path)['identity'])
    require(behaviors['total'] == 28 and behaviors['passed'] == sum(v['passed'] for v in behaviors['comparisons'])
            and behaviors['all_generations_complete'] is True
            and behaviors['all_checks_passed'] is all(v['all_tests_passed'] for v in behaviors['comparisons']),
            'Upstream behavior totals differ')
    baseline_off, config, _ = validate_capability(stage_spec('capability', False, 'baseline'), ctx)
    ctx.protocols[False] = config['protocol']
    ctx.inputs.update(baseline_off['artifacts'])
    off_stage = stage_spec('behavior', False, 'baseline')
    off_identity = read(off_stage['path'])['identity']
    require(off_identity.get('wrapper_sha256') == PINNED['check_behaviors.py']
            and off_identity.get('helper_sha256') == ctx.wrappers[False].HELPER_SHA256
            and off_identity.get('integrity_helper_sha256') == PINNED['evaluate_integrity.py']
            and off_identity.get('thinking') is False and off_identity.get('max_tokens') == 128
            and off_identity.get('temperature') == 0 and off_identity.get('case_order') == ctx.wrappers[False].TESTS
            and off_identity.get('chat_template_kwargs') == {'enable_thinking': False}, 'Off behavior baseline protocol differs')
    ctx.behavior_templates[False] = behavior_template(off_identity)
    ctx.inputs.update(validate_behavior(off_stage, ctx)['artifacts'])
    for reference in upstream.values():
        ctx.inputs[reference['path']] = pin(reference['path'], reference['sha256'])
    ctx.upstream = upstream


def validate_partial(stage, ctx, retry_incomplete):
    """Validate committed prefix without writing, scoring, or salvaging partial rows."""
    config_path = Path(stage['config_path'])
    directory = WORK / f'{stage["label"]}-groups'
    answers = WORK / f'{stage["label"]}-answers.jsonl'
    if not config_path.exists():
        require(not answers.exists() and (not directory.exists() or not list(directory.iterdir())),
                'Capability outputs exist without a pinned configuration')
        return
    saved = read(config_path)
    config = saved['configuration']
    require(saved['configuration_fingerprint'] == fingerprint(config), 'Partial configuration fingerprint differs')
    validate_identity(config, ctx.specs[NAME], ctx)
    require(config['label'] == stage['label'] and config['endpoint'] == ctx.support.BASE
            and config['protocol'] == ctx.protocols[stage['thinking']]
            and config['protocol_fingerprint'] == fingerprint(config['protocol']), 'Partial protocol differs')
    runner = ctx.on if stage['thinking'] else ctx.off
    support = ctx.on_support if stage['thinking'] else ctx.support
    grading = ctx.on_grading if stage['thinking'] else ctx.grading
    schedule = runner.group_schedule(ctx.cases, support.REPEATS)
    groups = runner.load_committed(directory, schedule, config, fingerprint(config), ctx.cases, support, grading)
    rows = [row for group in groups for row in group['records']]
    snapshot = [json.loads(line) for line in answers.read_text().splitlines()] if answers.exists() else []
    require(len(snapshot) in range(0, len(rows) + 1, 4) and snapshot == rows[:len(snapshot)],
            'Partial answer snapshot differs from committed groups')
    for path in directory.glob('pending-*.json'):
        pending = read(path)
        index = pending.get('schedule', {}).get('index')
        require(type(index) is int and 0 <= index < len(schedule)
                and pending['schedule'] == schedule[index]
                and pending.get('configuration_fingerprint') == fingerprint(config), 'Pending identity differs')
        if index < len(groups):
            require(pending.get('attempt_id') == groups[index]['attempt_id'], 'Pending/committed attempt differs')
        else:
            require(index == len(groups) and retry_incomplete,
                    'Interrupted group requires explicit --retry-incomplete-group; partial rows remain unscored')


def validate_stage(stage, ctx):
    return validate_capability(stage, ctx)[0] if stage['kind'] == 'capability' else validate_behavior(stage, ctx)


def validate_listing(listing, ctx, require_candidate):
    ids = [row.get('id') for row in listing.get('data', [])]
    required = {ctx.specs[name]['model_id'] for name in OLD_NAMES}
    allowed = required | {MODEL_ID}
    require(len(ids) == len(set(ids)) and required <= set(ids) <= allowed,
            'Base/three old mappings are missing, duplicated, or an unexpected adapter is registered')
    require(not require_candidate or MODEL_ID in ids, 'Candidate mapping disappeared')
    normalized = {}
    for spec in ctx.specs.values():
        if spec['model_id'] in ids:
            expected = spec['expected']
            normalized[spec['model_id']] = ctx.support.validate_model_listing(listing, spec['model_id'],
                expected['checkpoint'], expected['adapter'], expected['launch_profile'])
            require(normalized[spec['model_id']]['base']['id'] == ALIAS, 'Unexpected base alias')
    return normalized


@contextmanager
def runtime_lock():
    path = WORK / 'runtime.lock'
    with path.open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = os.fstat(stream.fileno())
        def check():
            current = path.stat()
            require((identity.st_dev, identity.st_ino) == (current.st_dev, current.st_ino),
                    'Runtime lock file was replaced')
        check()
        yield check
        check()


def load_candidate(ctx):
    payload = {'lora_name': MODEL_ID, 'lora_path': str(ADAPTER),
               'load_inplace': False, 'is_3d_lora_weight': False}
    request = urllib.request.Request(ctx.support.BASE + '/v1/load_lora_adapter',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    # Never retry an ambiguous POST. A later invocation may reuse only a valid mapping.
    with urllib.request.urlopen(request, timeout=300) as response:
        data = response.read()
        require(response.status == 200, 'Dynamic LoRA registration returned a non-200 response')
    return {'request': payload, 'response_sha256': hashlib.sha256(data).hexdigest(),
            'response_text': data.decode('utf-8', errors='replace')}


def register(ctx):
    with runtime_lock() as check_lock:
        verify_local(ctx)
        require(validate_capacity(read(WORK / 'launch-v1.json'), ctx.support) >= 4, 'LoRA capacity changed')
        before = validate_listing(ctx.support.fetch_models(), ctx, False)
        loaded = MODEL_ID not in before
        response = load_candidate(ctx) if loaded else None
        check_lock()
        after = validate_listing(ctx.support.fetch_models(), ctx, True)
        require(all(after[key] == value for key, value in before.items()), 'Registration changed an existing mapping')
        verify_local(ctx)
        return {'registered_utc': now(), 'candidate_was_newly_loaded': loaded, 'before': before, 'after': after,
                'load': response, 'old_adapters_unloaded': False, 'max_cpu_loras_launch_attestation': ctx.max_cpu_loras,
                'attestation_limit': 'Capacity is pinned launch evidence; /v1/models does not expose engine cache capacity.'}


def guard_runtime(ctx, expected):
    with runtime_lock():
        verify_local(ctx)
        require(validate_listing(ctx.support.fetch_models(), ctx, True) == expected,
                'Runtime base/adapter identity changed')


def command(stage, retry_incomplete=False):
    argv = [PYTHON, '-u', str(WORK / stage['runner']), '--label', stage['label'],
            '--model-id', MODEL_ID, '--adapter-dir', str(ADAPTER), '--expected-checkpoint', BASE,
            '--launch-profile', str(WORK / 'launch-v1.json'),
            '--precision-manifest', str(WORK / 'precision-manifest-v1.json')]
    if retry_incomplete and stage['kind'] == 'capability':
        argv.append('--retry-incomplete-group')
    return argv


def freeze(path, value):
    if path.exists():
        require(read(path) == value, f'Existing immutable record differs: {path}')
    else:
        atomic(path, value)


def aggregate(ctx, entries, plan_sha, registration_sha):
    require(len(entries) == 4 and all(e['task_completed'] for e in entries), 'Four tasks are not complete')
    return {'schema_version': 1, 'status': 'completed', 'task_completed': True,
            'candidate_identity': ctx.receipt['candidate_identity'], 'builder_receipt_sha256': RECEIPT_SHA,
            'plan_sha256': plan_sha, 'registration_sha256': registration_sha,
            'upstream': ctx.upstream, 'comparisons': entries,
            'capability_observations_by_mode': {'thinking_off': 48, 'thinking_on': 48},
            'behavior_checks_by_mode': {'thinking_off': 7, 'thinking_on': 7},
            'all_predicates_passed': all(e['all_predicates_passed'] for e in entries),
            'all_generations_complete': all(e['all_generations_complete'] for e in entries),
            'truncated_generations': sum(e['truncated_generations'] for e in entries),
            'completion_semantics': 'Completed means four fully recorded tasks. Strict misses, failed smoke predicates and exhausted-budget truncations remain separately reported outcomes.',
            'scope': 'NF4-direction/W8-weight candidate; 24 previously seen benign cases twice per mode and seven smokes per mode. No held-out, broad-quality or refusal claim; modes are not pooled.',
            'construction_confound': 'CPU versus original GPU randomized SVD and rounding are construction differences; this is not an isolated causal proof of NF4 weight mismatch.',
            'refusal_evaluation_performed': False, 'service_or_profile_changes': False,
            'existing_adapters_unloaded': False, 'runtime_mutation': 'One distinct reversible LoRA registration only.'}


def run(args):
    completed = []
    final_path = WORK / f'{PREFIX}-result-v1.json'
    try:
        status('waiting-for-upstream')
        upstream = wait_upstream(args.wait_timeout_seconds)
        ctx = prepare()
        validate_upstream(ctx, upstream)
        verify_local(ctx)
        stages = [stage_spec('capability', False), stage_spec('capability', True),
                  stage_spec('behavior', False), stage_spec('behavior', True)]
        # Reject corrupt caches before any runtime mutation. Behavior partials are
        # never retried under the same label; capability prefixes retain runner policy.
        cached = {}
        for stage in stages:
            if Path(stage['path']).exists():
                cached[stage['label']] = validate_stage(stage, ctx)
            elif stage['kind'] == 'capability':
                validate_partial(stage, ctx, args.retry_incomplete_group)
            else:
                require(not Path(stage['config_path']).exists(),
                        'Interrupted behavior configuration is preserved; investigate rather than rerun silently')
        plan = {'schema_version': 1, 'controller_sha256': sha(Path(__file__)), 'input_files': ctx.inputs,
                'baseline_shard_stats_from_fully_hashed_build': ctx.baseline_stats,
                'candidate_identity': ctx.receipt['candidate_identity'], 'upstream': upstream,
                'max_cpu_loras_launch_attestation': ctx.max_cpu_loras,
                'specs': ctx.specs, 'stages': stages, 'commands': [command(stage) for stage in stages],
                'protocol_fingerprints': {str(k): fingerprint(v) for k, v in ctx.protocols.items()},
                'retry_policy': 'No automatic request/partial retries. --retry-incomplete-group delegates whole-group archival/retry to the frozen capability runner only.'}
        plan_path = WORK / f'{PREFIX}-plan-v1.json'
        freeze(plan_path, plan)
        registration_path = WORK / f'{PREFIX}-registration-v1.json'
        if final_path.exists():
            saved = read(final_path)
            signed(saved)
            require(len(cached) == 4 and registration_path.exists(), 'Completed aggregate has missing artifacts')
            signed(read(registration_path), 'registration_sha256')
            expected = aggregate(ctx, [cached[s['label']] for s in stages], sha(plan_path), sha(registration_path))
            require({k: v for k, v in saved.items() if k not in ('finished_utc', 'result_sha256')} == expected,
                    'Completed aggregate differs from independently validated artifacts')
            verify_local(ctx, full=True)
            status('completed', reused_completed_aggregate=True, result=str(final_path), sha256=sha(final_path),
                   all_predicates_passed=saved['all_predicates_passed'], all_generations_complete=saved['all_generations_complete'])
            return saved
        status('registering-candidate', model_id=MODEL_ID, adapter_path=str(ADAPTER))
        registration = register(ctx)
        if registration_path.exists():
            original = read(registration_path)
            signed(original, 'registration_sha256')
            require(original['after'] == registration['after'], 'Registration resume mappings differ')
        else:
            registration['registration_sha256'] = fingerprint(registration)
            atomic(registration_path, registration)
        expected_mapping = registration['after']
        for stage in stages:
            guard_runtime(ctx, expected_mapping)
            reused = stage['label'] in cached
            status('validating-existing' if reused else 'running-stage', label=stage['label'],
                   runner=stage['runner'], completed_tasks=len(completed))
            if not reused:
                # Each frozen child obtains runtime.lock itself. The controller
                # releases its lock before exec and checks every mapping after it.
                subprocess.run(command(stage, args.retry_incomplete_group), check=True,
                               timeout=args.stage_timeout_seconds, cwd=str(WORK))
            entry = validate_stage(stage, ctx)
            guard_runtime(ctx, expected_mapping)
            completed.append(entry)
            status('stage-completed', label=stage['label'], task_completed=True,
                   all_predicates_passed=entry['all_predicates_passed'],
                   all_generations_complete=entry['all_generations_complete'], sha256=entry['sha256'])
        verify_local(ctx, full=True)
        # Re-read all four committed outputs; never trust a stale intermediate summary.
        validated = [validate_stage(stage, ctx) for stage in stages]
        require(validated == completed, 'Stage artifacts changed before aggregate commit')
        guard_runtime(ctx, expected_mapping)
        result = aggregate(ctx, validated, sha(plan_path), sha(registration_path))
        result['finished_utc'] = now()
        result['result_sha256'] = fingerprint(result)
        atomic(final_path, result)
        status('completed', result=str(final_path), sha256=sha(final_path),
               all_predicates_passed=result['all_predicates_passed'], all_generations_complete=result['all_generations_complete'],
               truncated_generations=result['truncated_generations'])
        return result
    except Exception as exc:
        failure = {'status': 'failed', 'task_completed': False, 'finished_utc': now(), 'error': repr(exc),
                   'completed_tasks': completed, 'controller_sha256': sha(Path(__file__)),
                   'note': 'Existing observations, adapters, profiles and final aggregates are preserved.'}
        failure['result_sha256'] = fingerprint(failure)
        failure_path = WORK / f'{PREFIX}-failure-{time.time_ns()}.json'
        atomic(failure_path, failure)
        status('failed', error=repr(exc), failure=str(failure_path), sha256=sha(failure_path))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-timeout-seconds', type=float, default=14400)
    parser.add_argument('--stage-timeout-seconds', type=float, default=14400)
    parser.add_argument('--retry-incomplete-group', action='store_true')
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (args.wait_timeout_seconds, args.stage_timeout_seconds)):
        parser.error('Timeouts must be finite and positive')
    with (WORK / f'{PREFIX}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
