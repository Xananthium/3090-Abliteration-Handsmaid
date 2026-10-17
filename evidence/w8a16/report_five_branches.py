#!/home/isnotreal/model-tools/report-env/bin/python
"""Validate completed W8 evidence and render two separate five-branch reports.

Read-only toward evidence/models/runtime. No API calls, evaluator execution,
services, publication, or deployment. Missing aggregates return not_ready.
Use /home/isnotreal/model-tools/report-env/bin/python for matplotlib.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import types

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
BASE = '/home/isnotreal/models/redcell-baseline-w8a16-native'
ALIAS = 'qwen3.6-35b-hauhau-w4a16-vision'
NEW = 'nf4-direction-w8-weight-strength-1.0'
LABELS = ('baseline', 'strength-1.0', 'strength-1.5', 'attention-1.5-mlp-1.0', NEW)
MODES = ('thinking_off', 'thinking_on')
AGGREGATES = ('integrity-batch', 'thinking-batch', 'thinking-behaviors', 'w8-rebase-evaluation')
CONTROLLER_SHA = 'd4908da0a5a3b9487b28cb7b4fc8fae9a1918faa34654ca640538d07cb5160e3'
COMMON_SHA = '8bff2efeca121cb3fe100386d29736edfb2c3abfdc86fa6200658a020ef6f11e'
THINKING_SHA = 'a2c907c08d1963562516c47d984ae5f67a136689a23373c56d6fded1539d33b6'
WORK_SHA = '7ec041c485f859ef3876a4d6e446713495587ab703dcec55a1d0f51a98cfa738'
SUPPORT_SHA = 'f0a84fc9f8b1ed8f0088df755b42935af75840751a524083b8656b32b30fe288'
BUILD_SHA = '38aa74f31d44b785704858c545837edcaea1585f2a503f230ace726cb6b1b310'
NAMES = {'baseline': 'Untouched baseline', 'strength-1.0': 'Original strength 1.0',
         'strength-1.5': 'Original strength 1.5', 'attention-1.5-mlp-1.0': 'Attention 1.5 · MLP 1.0',
         NEW: 'NF4 direction · W8 weight 1.0'}
SCOPE = ('24 previously seen benign development questions, repeated twice: 48 observations per branch per mode. '
         'Seven separate tool/vision smoke checks per branch per mode. Repeats are not independent questions. '
         'These results do not establish held-out quality, refusal rates/targets, or deployment readiness. '
         'Thinking modes are never pooled; smoke predicates are not added to the 48-answer score.')
TIMING_CAVEAT = ('Observed development timing with concurrency four and reversible unmerged LoRA. '
                 'The four-thread CPU rebase ran 2026-10-15 23:56:38–23:58:47 UTC and overlapped part of '
                 'the original strength-1.0 thinking evaluation. A one-thread factor-comparison diagnostic '
                 'also ran 2026-10-16 00:19:56.735945–00:19:59.259185 UTC. Other host activity is not exhaustively '
                 'measured. These are not controlled idle-host '
                 'or merged-checkpoint deployment benchmarks. Generated-token comparisons are separate '
                 'from latency; native output tokens include thinking and final text, including retries.')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(path, inputs, expected=None, size=None):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns), f'Input changed while hashing: {path}')
    require(expected is None or digest == expected, f'Input hash mismatch: {path}')
    require(size is None or after.st_size == size, f'Input size mismatch: {path}')
    inputs[str(path)] = {'sha256': digest, 'bytes': after.st_size}
    return digest


def module(path, expected, inputs):
    """Definitions-only import of the two stdlib accounting modules; never main."""
    pin(path, inputs, expected)
    result = types.ModuleType('_five_branch_' + Path(path).stem)
    result.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), 'exec'), result.__dict__)
    return result


def isolated(path, expected, functions, constants, inputs, extra=None):
    """Only explicitly named read-only functions/constants are compiled."""
    pin(path, inputs, expected)
    tree = ast.parse(Path(path).read_bytes())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
    require({node.name for node in nodes} == set(functions), f'Missing pure validation functions: {path}')
    namespace = {'Path': Path, 'json': json, 'hashlib': hashlib, 'Counter': Counter,
                 'WORK': WORK, 'BASE': BASE, 'NAME': NEW, **(extra or {})}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants):
            namespace[node.targets[0].id] = ast.literal_eval(node.value)
    require(set(constants) <= namespace.keys(), f'Missing pure validation constants: {path}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


def ready():
    missing = []
    for prefix in AGGREGATES:
        path = WORK / f'{prefix}-result-v1.json'
        if not path.is_file():
            missing.append({'aggregate': prefix, 'state': 'absent'})
        else:
            status = json.loads(path.read_text()).get('status')
            if status != 'completed':
                missing.append({'aggregate': prefix, 'state': status})
    return missing


def load_context(inputs):
    common = module(WORK / 'analyze_integrity.py', COMMON_SHA, inputs)
    thinking = module(WORK / 'analyze_thinking_integrity.py', THINKING_SHA, inputs)
    controller = isolated(WORK / 'run_w8_rebase_evaluation.py', CONTROLLER_SHA,
        ('require', 'fingerprint', 'sha', 'stat_key', 'read', 'pin', 'signed', 'validate_identity',
         'stage_spec', 'smoke_predicate', 'behavior_template', 'validate_behavior'),
        ('PINNED', 'OUTPUT_HASHES'), inputs)
    support = isolated(ROOT / 'bf16-iteration/evaluate_bf16_integrity.py', SUPPORT_SHA,
                       ('validate_model_listing',), (), inputs, {'CONTEXT_LIMIT': 8192})
    off = isolated(WORK / 'check_behaviors.py', controller.PINNED['check_behaviors.py'],
                   (), ('TESTS', 'HELPER_SHA256'), inputs)
    on = isolated(WORK / 'check_behaviors_thinking.py', controller.PINNED['check_behaviors_thinking.py'],
                  ('response_parts', 'final_text', 'validate_request', 'sha_bytes'),
                  ('TESTS', 'MAX_TOKENS', 'THINKING_BUDGET', 'HELPER_SHA256'), inputs)
    work = isolated(WORK / 'analyze_thinking_work.py', WORK_SHA,
                    ('ratio', 'difference', 'total_known', 'observation', 'pair_observations'), (), inputs)
    behavior_driver = isolated(WORK / 'run_thinking_behaviors.py', controller.PINNED['run_thinking_behaviors.py'],
                               (), ('PINNED',), inputs)
    cases, grade = common.frozen_inputs(inputs)
    validator, constants = thinking.load_validator(common, inputs)
    return types.SimpleNamespace(common=common, thinking=thinking, controller=controller,
                                 support=support, wrappers={False: off, True: on}, work=work,
                                 behavior_driver=behavior_driver,
                                 cases=cases, grade=grade, validator=validator, constants=constants,
                                 behavior_templates={})


def validate_links(ctx, inputs):
    common, controller = ctx.common, ctx.controller
    aggregates = {name: common.read(WORK / f'{name}-result-v1.json', inputs) for name in AGGREGATES}
    for name in ('thinking-behaviors', 'w8-rebase-evaluation'):
        common.signed_object(aggregates[name], 'result_sha256')
    off, on = aggregates['integrity-batch'], aggregates['thinking-batch']
    smokes, rebase = aggregates['thinking-behaviors'], aggregates['w8-rebase-evaluation']
    require(all(value['status'] == 'completed' for value in aggregates.values()), 'Aggregate completion changed')
    for value in (off, on):
        require([entry['label'] for entry in value['comparisons']] == list(LABELS[:4]), 'Old branch coverage differs')
        require(value['profile_sha256'] == controller.PINNED['launch-v1.json']
                and value['precision_manifest_sha256'] == controller.PINNED['precision-manifest-v1.json'],
                'Aggregate precision/profile differs')
    require([entry['label'] for entry in off['behaviors']] == list(LABELS[:4])
            and [entry['branch'] for entry in smokes['comparisons']] == list(LABELS[:4]),
            'Old smoke branch coverage differs')
    on_path = str(WORK / 'thinking-batch-result-v1.json')
    smoke_path = str(WORK / 'thinking-behaviors-result-v1.json')
    require(smokes['upstream']['path'] == on_path and smokes['upstream']['sha256'] == inputs[on_path]['sha256']
            and smokes['driver_sha256'] == controller.PINNED['run_thinking_behaviors.py']
            and smokes['input_sha256'] == ctx.behavior_driver.PINNED, 'Thinking smoke upstream/input pins differ')
    require(set(rebase['upstream']) == {'thinking-batch', 'thinking-behaviors'}, 'Rebase upstream coverage differs')
    for key, path in (('thinking-batch', on_path), ('thinking-behaviors', smoke_path)):
        require(rebase['upstream'][key]['path'] == path
                and rebase['upstream'][key]['sha256'] == inputs[path]['sha256'], 'Rebase upstream link differs')
    plan = common.read(WORK / 'w8-rebase-evaluation-plan-v1.json', inputs, rebase['plan_sha256'])
    registration = common.read(WORK / 'w8-rebase-evaluation-registration-v1.json', inputs, rebase['registration_sha256'])
    common.signed_object(registration, 'registration_sha256')
    require(plan['controller_sha256'] == CONTROLLER_SHA and set(plan['specs']) == set(LABELS)
            and plan['upstream'] == rebase['upstream'], 'Rebase plan identity differs')
    for path, identity in plan['input_files'].items():
        pin(path, inputs, identity['sha256'], identity['bytes'])
    for path, expected in plan['baseline_shard_stats_from_fully_hashed_build'].items():
        require(controller.stat_key(path) == expected, f'Base shard changed since its fully hashed build: {path}')
    registry = common.read(WORK / 'adapter-identities-v1.json', inputs, controller.PINNED['adapter-identities-v1.json'])
    receipt_path = WORK / 'adapters/nf4-direction-w8-weight-strength-1.0-vllm-v1/provenance-validation.json'
    build = common.read(receipt_path, inputs, BUILD_SHA)
    require(build['complete'] is True and build['status'] == 'completed_cpu_builder_validation'
            and rebase['builder_receipt_sha256'] == BUILD_SHA
            and plan['candidate_identity'] == rebase['candidate_identity'] == build['candidate_identity'],
            'New candidate construction identity differs')
    for artifact in build['output_artifacts']:
        require(artifact['sha256'] == controller.OUTPUT_HASHES[Path(artifact['path']).name], 'Unexpected candidate artifact')
        pin(artifact['path'], inputs, artifact['sha256'], artifact['bytes'])
    ctx.specs = plan['specs']
    for label, spec in ctx.specs.items():
        expected = spec['expected']
        require(spec['name'] == label and expected['checkpoint']['path'] == BASE, 'Wrong branch/checkpoint')
        require(expected['precision']['sha256'] == controller.PINNED['precision-manifest-v1.json']
                and expected['launch_profile']['sha256'] == controller.PINNED['launch-v1.json'], 'Wrong branch precision/profile')
        if label == 'baseline':
            require(spec['model_id'] == ALIAS and expected['adapter'] is None, 'Baseline is not unedited base')
        else:
            identity = (build['candidate_identity'] if label == NEW else registry['adapters'][label + '-vllm'])
            require(spec['model_id'] == identity['model_id'] and expected['adapter']['path'] == identity['path'],
                    'Adapter label/model path differs from its registry')
            for filename, identity_file in expected['adapter']['files'].items():
                pin(Path(identity['path']) / filename, inputs, identity_file['sha256'])
        after = registration['after'].get(spec['model_id'])
        require(after is not None, 'Registration has no expected branch')
        listing = {'data': [after['base']] + ([after['requested']] if expected['adapter'] else [])}
        require(ctx.support.validate_model_listing(listing, spec['model_id'], expected['checkpoint'],
                expected['adapter'], expected['launch_profile']) == after, 'Registration routing differs')
    require(all(registration['after'].get(key) == value for key, value in registration['before'].items()),
            'Candidate registration changed an earlier mapping')
    require(rebase['task_completed'] is True and len(rebase['comparisons']) == 4
            and all(entry['task_completed'] is True for entry in rebase['comparisons'])
            and rebase['capability_observations_by_mode'] == {'thinking_off': 48, 'thinking_on': 48}
            and rebase['behavior_checks_by_mode'] == {'thinking_off': 7, 'thinking_on': 7}, 'New aggregate task counts differ')
    stages = {(stage['kind'], stage['thinking']): stage for stage in rebase['comparisons']}
    require(set(stages) == {('capability', False), ('capability', True), ('behavior', False), ('behavior', True)},
            'New aggregate mode/task coverage differs')
    for key, entry in stages.items():
        expected_stage = controller.stage_spec(*key, NEW)
        require(all(entry.get(k) == v for k, v in expected_stage.items()), 'New aggregate stage identity differs')
        pin(entry['path'], inputs, entry['sha256'])
        for path, identity in entry['artifacts'].items():
            pin(path, inputs, identity['sha256'], identity['bytes'])
    require(rebase['all_predicates_passed'] is all(entry['all_predicates_passed'] for entry in stages.values())
            and rebase['all_generations_complete'] is all(entry['all_generations_complete'] for entry in stages.values())
            and rebase['truncated_generations'] == sum(entry['truncated_generations'] for entry in stages.values()),
            'New aggregate outcome totals differ from its four stages')
    background = [common.read(WORK / name, inputs) for name in
                  ('w8-weight-rebase-launch-v1.json', 'w8-weight-rebase-completion-v1.json')]
    require(background[1]['receipt_sha256'] == BUILD_SHA, 'CPU completion refers to a different build')
    delta_path = WORK / 'w8-rebase-delta-comparison-v1.json'
    delta = common.read(delta_path, inputs, '3aff4af12403346662adee5bd2826134b725f0ea45feb05c452a0aedf8fd1e3e')
    common.signed_object(delta, 'report_sha256')
    background.append({'path': str(delta_path), 'sha256': inputs[str(delta_path)]['sha256'],
                       'status': delta['status'], 'started_utc': delta['started_utc'],
                       'finished_utc': delta['ended_utc'], 'elapsed_seconds': delta['elapsed_seconds'],
                       'resource_notes': delta['resource_notes'], 'actions': delta['actions'],
                       'effective_edit_comparison': delta['by_component']['all_active'],
                       'interpretation_limits': delta['interpretation_limits']})
    return aggregates, stages, build, registry, background


def smoke_result(ctx, label, enabled, entry, inputs):
    controller, common = ctx.controller, ctx.common
    stage = controller.stage_spec('behavior', enabled, label)
    require(entry['path'] == stage['path'], 'Smoke aggregate points at the wrong branch/mode')
    saved = common.read(stage['path'], inputs, entry['sha256'])
    identity = saved['identity']
    if label == 'baseline':
        protocol = identity['protocol'] if enabled else identity
        require(protocol['wrapper_sha256'] == controller.PINNED[stage['runner']]
                and protocol['helper_sha256'] == ctx.wrappers[enabled].HELPER_SHA256
                and protocol['temperature'] == 0 and protocol['case_order'] == ctx.wrappers[enabled].TESTS
                and protocol['chat_template_kwargs'] == {'enable_thinking': enabled}, 'Baseline smoke protocol differs')
        if enabled:
            require(protocol['max_tokens'] == 4096 and protocol['thinking'] == {'type': 'enabled', 'budget_tokens': 3072}
                    and identity['protocol_fingerprint'] == common.fingerprint(protocol), 'Native thinking smoke settings differ')
        else:
            require(protocol['max_tokens'] == 128 and protocol['thinking'] is False, 'Thinking-off smoke settings differ')
        ctx.behavior_templates[enabled] = controller.behavior_template(identity)
    checked = controller.validate_behavior(stage, ctx)
    inputs.update({path: {k: value[k] for k in ('sha256', 'bytes')} for path, value in checked['artifacts'].items()})
    require(entry['passed'] == checked['passed'] and entry['total'] == 7, 'Smoke aggregate totals differ')
    if label == NEW:
        require(all(entry[k] == checked[k] for k in ('model_id', 'configuration_fingerprint', 'all_predicates_passed',
                    'all_generations_complete', 'generation_count', 'truncated_generations', 'generation_stop_reasons')),
                'New smoke aggregate disagrees with independently validated captures')
    elif enabled:
        require(entry['model_id'] == identity['model_id'] and entry['branch'] == label
                and entry['completed'] is True
                and entry['all_generations_complete'] == checked['all_generations_complete']
                and entry['generation_count'] == checked['generation_count']
                and entry['truncated_generations'] == checked['truncated_generations']
                and entry['config_path'] == stage['config_path']
                and entry['config_sha256'] == inputs[stage['config_path']]['sha256']
                and entry['configuration_fingerprint'] == common.fingerprint(identity), 'Thinking smoke comparison identity differs')
    native = saved['native_requests']
    generations = [row for row in native if row.get('route') == '/v1/messages'] if enabled else native
    usage = [(row['response'].get('usage') or {}).get('output_tokens') for row in generations]
    tokens = ctx.work.total_known(usage)
    rows = [{'label': label, 'mode': MODES[int(enabled)], 'test': row['test'], 'passed': row['passed'],
             'conditional_followup_skipped': row['response'] is None,
             'generation_complete': (row['response'].get('stop_reason') in ('end_turn', 'tool_use', 'stop_sequence'))
                                      if row['response'] else None,
             'stop_reason': row['response'].get('stop_reason') if row['response'] else None} for row in saved['records']]
    return {'passed': checked['passed'], 'total': 7, 'tool_passed': sum(row['passed'] for row in rows[:5]),
            'tool_total': 5, 'vision_passed': sum(row['passed'] for row in rows[5:]), 'vision_total': 2,
            'failed_tests': [row['test'] for row in rows if not row['passed']],
            'all_generations_complete': checked['all_generations_complete'],
            'truncated_generations': checked['truncated_generations'],
            'generated_output_tokens': tokens, 'source_path': stage['path']}, rows


def analyze():
    inputs = {}
    ctx = load_context(inputs)
    common = ctx.common
    aggregates, new_stages, build, registry, background = validate_links(ctx, inputs)
    start = datetime.fromisoformat(background[1]['started_utc'])
    finish = datetime.fromisoformat(background[1]['finished_utc'])
    diagnostic_start = datetime.fromisoformat(background[2]['started_utc'])
    diagnostic_finish = datetime.fromisoformat(background[2]['finished_utc'])
    modes, pairs_csv, smokes_csv, groups_csv = {}, [], [], []
    for enabled, mode in ((False, MODES[0]), (True, MODES[1])):
        old = aggregates['thinking-batch' if enabled else 'integrity-batch']
        entries = {entry['label']: entry for entry in old['comparisons']}
        new_stage = new_stages[('capability', enabled)]
        summary = common.read(new_stage['path'], inputs, new_stage['sha256'])
        entries[NEW] = {'label': NEW, 'path': new_stage['path'], 'sha256': new_stage['sha256'],
                        'capabilities': summary['capabilities'], 'output_tokens': summary['total_output_tokens_including_retries'],
                        'batch_wall_seconds': summary['sum_committed_group_wall_seconds']}
        batch = {'base_checkpoint': BASE, 'profile_sha256': old['profile_sha256'],
                 'precision_manifest_sha256': old['precision_manifest_sha256']}
        results, rows, configs = {}, {}, {}
        smoke_entries = ({entry['branch']: entry for entry in aggregates['thinking-behaviors']['comparisons']}
                         if enabled else {entry['label']: entry for entry in aggregates['integrity-batch']['behaviors']})
        smoke_entries[NEW] = new_stages[('behavior', enabled)]
        for label in LABELS:
            if enabled:
                result, records, config = ctx.thinking.load_run(entries[label], batch, ctx.cases, ctx.grade,
                    ctx.validator, ctx.constants, common, inputs)
            else:
                result, records, config = common.load_run(entries[label], batch, ctx.cases, ctx.grade, inputs)
            ctx.controller.validate_identity(config, ctx.specs[label], ctx)
            require(result['pending_group_attempts'] == 0 and config['endpoint'] == 'http://127.0.0.1:8002'
                    and config['protocol']['script_sha256'] == ctx.controller.PINNED[
                        'evaluate_integrity_thinking.py' if enabled else 'evaluate_integrity.py'], 'Capability identity/pending state differs')
            if label == NEW:
                require(new_stage['score'] == result
                        and new_stage['all_predicates_passed'] is (result['correct'] == 48)
                        and new_stage['all_generations_complete'] is (result['incomplete_outputs'] == 0)
                        and new_stage['truncated_generations'] == result['final_truncated_outputs'],
                        'New capability aggregate differs from fresh accounting')
            if label != 'baseline':
                for key in ('protocol', 'checkpoint', 'launch_profile', 'precision'):
                    require(config[key] == configs['baseline'][key], f'{mode}/{label}: unmatched {key}')
            configs[label], rows[label] = config, records
            result['mode'], result['display_name'] = mode, NAMES[label]
            result['smokes'], tests = smoke_result(ctx, label, enabled, smoke_entries[label], inputs)
            smokes_csv.extend(tests)
            group_dir = WORK / f'{config["label"]}-groups'
            overlap_groups = 0
            for index in range(12):
                group = common.read(group_dir / f'group-{index:04d}.json', inputs)
                began, ended = datetime.fromisoformat(group['started_at_utc']), datetime.fromisoformat(group['completed_at_utc'])
                overlap = max(0.0, (min(ended, finish) - max(began, start)).total_seconds())
                diagnostic_overlap = max(0.0, (min(ended, diagnostic_finish) - max(began, diagnostic_start)).total_seconds())
                overlap_groups += overlap > 0
                groups_csv.append({'mode': mode, 'label': label, 'group_index': index,
                    'repeat': group['schedule']['repeat'], 'committed_wall_seconds': group['wall_seconds'],
                    'generated_tokens_including_retries': ctx.work.total_known([
                        ctx.work.observation(row, common)['generated_tokens_including_retries'] for row in group['records']]),
                    'started_utc': group['started_at_utc'], 'completed_utc': group['completed_at_utc'],
                    'known_cpu_build_interval_overlap_seconds': overlap,
                    'known_cpu_diagnostic_interval_overlap_seconds': diagnostic_overlap,
                    'timing_context': 'observed development timing; possible background CPU work'})
            result['known_cpu_build_overlapping_groups'] = overlap_groups
            result['timing_context'] = 'observed development timing; possible background CPU work; not merged deployment throughput'
            results[label] = result
        for label in LABELS:
            paired = common.paired(rows['baseline'], rows[label])
            common.compare(results[label], results['baseline'], paired)
            if label != 'baseline':
                for base, candidate, score_pair in zip(rows['baseline'], rows[label], paired):
                    cost = ctx.work.pair_observations(label, ctx.work.observation(base, common),
                                                    ctx.work.observation(candidate, common), common)
                    pairs_csv.append({'mode': mode, **cost, 'score_outcome': score_pair['outcome'],
                                      'timing_context': 'observed development timing; possible background CPU work'})
        ranking = sorted(results.values(), key=lambda row: -row['correct'])
        for result in ranking:
            result['development_rank'] = 1 + sum(row['correct'] > result['correct'] for row in ranking)
        modes[mode] = {'thinking_enabled': enabled, 'unique_questions': 24, 'repeats': [1, 2],
                       'observations_per_branch': 48, 'total_observations': 240,
                       'smokes_per_branch': 7, 'ranking': ranking,
                       'ranking_rule': 'Strict correct/48 descending; ties share rank; no speed or smoke tiebreak.'}
    require(aggregates['thinking-behaviors']['passed'] == sum(row['smokes']['passed'] for row in modes['thinking_on']['ranking'] if row['label'] != NEW)
            and aggregates['thinking-behaviors']['total'] == 28
            and aggregates['thinking-behaviors']['all_checks_passed'] is all(
                row['smokes']['passed'] == 7 for row in modes['thinking_on']['ranking'] if row['label'] != NEW)
            and aggregates['thinking-behaviors']['all_generations_complete'] is all(
                row['smokes']['all_generations_complete'] for row in modes['thinking_on']['ranking'] if row['label'] != NEW),
            'Thinking smoke aggregate sum differs')
    for path, identity in inputs.items():
        require(sha(path) == identity['sha256'], f'Input changed during analysis: {path}')
    report = {'schema_version': 1, 'status': 'completed_five_branch_development_report',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': SCOPE,
              'quality_reference': 'Untouched W8A16 baseline separately within each thinking mode; BF16 source is conversion input only.',
              'refusal_evaluation_performed': False, 'full_quality_or_refusal_target_acceptance_evaluated': False,
              'deployment_performed': False, 'publication_performed': False,
              'token_and_timing_scope': TIMING_CAVEAT,
              'throughput_rule': 'Generated tokens including retries divided by committed group wall seconds. Never sum overlapping request latencies as elapsed throughput.',
              'retention_rule': '100 × candidate correct / matched baseline correct, uncapped; null if baseline correct is zero. Losses/gains pair exact case IDs and repeats.',
              'construction': {'old_branch_direction_provenance': registry['direction_provenance'],
                  'new_branch': NEW, 'candidate_identity': build['candidate_identity'], 'receipt_sha256': BUILD_SHA,
                  'direction_origin': build['direction_origin'], 'weight_reconstruction': build['weight_reconstruction'],
                  'factorization': build['factorization'], 'limitations': build['limitations'],
                  'comparison_confound': aggregates['w8-rebase-evaluation']['construction_confound']},
              'known_cpu_background_work': background, 'modes': modes,
              'source_aggregate_sha256': {name: inputs[str(WORK / f'{name}-result-v1.json')]['sha256'] for name in AGGREGATES},
              'evidence_bundle_sha256': common.fingerprint({path: value['sha256'] for path, value in inputs.items()}),
              'generator_sha256': sha(Path(__file__))}
    return report, inputs, pairs_csv, smokes_csv, groups_csv, common


def render(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'svg.fonttype': 'none'})
    colors = {'baseline': '#526475', 'strength-1.0': '#167b82', 'strength-1.5': '#315ca3',
              'attention-1.5-mlp-1.0': '#8865a8', NEW: '#bd6b2b'}
    artifacts = {}
    categories = ('reasoning', 'instruction', 'consistency', 'recovery')
    for mode, data in report['modes'].items():
        rows = data['ranking']
        fig, (bars, heat) = plt.subplots(1, 2, figsize=(15, 8),
            gridspec_kw={'width_ratios': [1.4, 1], 'wspace': .2})
        fig.patch.set_facecolor('#f8fafc')
        mode_name = 'Thinking enabled' if data['thinking_enabled'] else 'Thinking disabled'
        fig.suptitle(f'REDCELL W8A16 | {mode_name}', x=.035, y=.97, ha='left',
                     fontsize=21, fontweight='bold', color='#162534')
        fig.text(.035, .915, 'Five branches · 24 previously seen development questions × 2 fixed repeats', color='#526475', fontsize=12)
        bars.barh(range(5), [row['correct'] for row in rows], height=.6,
                  color=[colors[row['label']] for row in rows], zorder=3)
        labels = [f"#{row['development_rank']}  {NAMES[row['label']]}\nTools {row['smokes']['tool_passed']}/5 · Vision {row['smokes']['vision_passed']}/2" for row in rows]
        bars.set_yticks(range(5), labels)
        bars.invert_yaxis(); bars.set_xlim(0, 48); bars.set_xticks([0, 12, 24, 36, 48])
        bars.set_xlabel('Strictly correct observations / 48', labelpad=10)
        bars.set_title('Final-answer score; smoke results listed separately', loc='left', fontsize=11, pad=15)
        bars.grid(axis='x', color='#dde4eb', zorder=0); bars.set_axisbelow(True)
        bars.tick_params(axis='y', length=0, pad=12)
        for i, row in enumerate(rows):
            inside = row['correct'] >= 15
            bars.text(row['correct'] - .7 if inside else row['correct'] + .7, i,
                      f"{row['correct']}/48 · {row['score_percent']:.1f}%", va='center',
                      ha='right' if inside else 'left', color='white' if inside else '#162534', fontweight='bold')
        values = np.array([[row['by_category'][category]['correct'] for category in categories] for row in rows])
        heat.imshow(values, vmin=0, vmax=12, cmap='YlGnBu', aspect='auto')
        heat.set_yticks([]); heat.set_xticks(range(4), [c.title() for c in categories], rotation=30, ha='right')
        heat.set_title('Category scores / 12', loc='left', fontsize=12, pad=15); heat.tick_params(length=0, pad=8)
        for i in range(5):
            for j in range(4):
                heat.text(j, i, f'{values[i,j]}/12', ha='center', va='center', fontweight='bold',
                          color='white' if values[i,j] >= 8 else '#162534')
        for ax in (bars, heat):
            ax.set_facecolor('#f8fafc')
            for spine in ax.spines.values(): spine.set_visible(False)
        fig.subplots_adjust(left=.29, right=.97, top=.82, bottom=.26)
        fig.text(.035, .12, 'Development diagnostics only. Repeats are not independent questions. No overall quality, refusal-target, or deployment claim.', color='#526475', fontsize=10)
        fig.text(.035, .08, 'Modes remain separate; equal scores share rank. NF4-derived directions are reused; CPU/GPU SVD and rounding differ.', color='#526475', fontsize=10)
        fig.text(.035, .04, 'Token work and timing appear in CSV/JSON. Background CPU activity prevents an idle-host speed comparison.', color='#526475', fontsize=10)
        for suffix in ('png', 'svg'):
            path = output / f'{mode}-ranked.{suffix}'
            fig.savefig(path, format=suffix, dpi=180, facecolor=fig.get_facecolor())
            artifacts[path.name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        plt.close(fig)
    return artifacts


def write_report(report, inputs, pairs, smokes, groups, common, output):
    require(output.parent == WORK, 'Report directory must be directly under the W8 workspace')
    if output.exists():
        existing = json.loads((output / 'report.json').read_text())
        require(existing['evidence_bundle_sha256'] == report['evidence_bundle_sha256']
                and existing['generator_sha256'] == report['generator_sha256'],
                'Existing report is preserved; choose a new --output-dir for different evidence/generator')
        return {'status': 'already_generated', 'output_directory': str(output)}
    temporary = WORK / (output.name + '.partial-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    temporary.mkdir()
    ranking = []
    categories = []
    for mode, data in report['modes'].items():
        for row in data['ranking']:
            ranking.append({**row, 'tool_passed': row['smokes']['tool_passed'], 'tool_total': 5,
                            'vision_passed': row['smokes']['vision_passed'], 'vision_total': 2,
                            'smoke_truncated_generations': row['smokes']['truncated_generations']})
            categories.extend({'mode': mode, 'label': row['label'], 'category': category, **values}
                              for category, values in row['by_category'].items())
    outputs = {'ranking.csv': (ranking, ('mode', 'development_rank', 'label', 'correct', 'total', 'score_percent',
                    'relative_score_retention_percent', 'baseline_losses', 'baseline_gains', 'baseline_correct_retained',
                    'net_correct_change', 'incomplete_outputs', 'final_truncated_outputs', 'output_tokens_including_retries',
                    'committed_group_wall_seconds', 'output_tokens_per_committed_wall_second', 'tool_passed', 'tool_total',
                    'vision_passed', 'vision_total', 'smoke_truncated_generations', 'known_cpu_build_overlapping_groups', 'timing_context')),
               'categories.csv': (categories, ('mode', 'label', 'category', 'correct', 'total', 'baseline_correct',
                    'baseline_losses', 'baseline_gains', 'relative_score_retention_percent', 'unresolved', 'unresolved_truncated')),
               'paired-observations.csv': (pairs, ('mode', 'label', 'case_id', 'repeat', 'category', 'score_outcome',
                    'baseline_correct', 'candidate_correct', 'baseline_complete', 'candidate_complete',
                    'baseline_generated_tokens_including_retries', 'candidate_generated_tokens_including_retries',
                    'extra_generated_tokens', 'generated_token_ratio', 'baseline_retry_count', 'candidate_retry_count',
                    'baseline_native_generation_request_seconds_including_retries',
                    'candidate_native_generation_request_seconds_including_retries',
                    'extra_native_generation_request_seconds', 'native_generation_latency_ratio', 'timing_context')),
               'tool-vision-checks.csv': (smokes, ('mode', 'label', 'test', 'passed', 'conditional_followup_skipped',
                    'generation_complete', 'stop_reason')),
               'committed-groups.csv': (groups, ('mode', 'label', 'group_index', 'repeat', 'generated_tokens_including_retries',
                    'committed_wall_seconds', 'started_utc', 'completed_utc', 'known_cpu_build_interval_overlap_seconds',
                    'known_cpu_diagnostic_interval_overlap_seconds', 'timing_context'))}
    for name, (rows, fields) in outputs.items():
        (temporary / name).write_text(common.csv_text(rows, fields))
    (temporary / 'evidence-manifest.json').write_text(json.dumps(inputs, indent=2) + '\n')
    chart_artifacts = render(report, temporary)
    report['artifacts'] = {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
                           for path in temporary.iterdir() if path.is_file()}
    report['chart_artifacts'] = chart_artifacts
    (temporary / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    temporary.rename(output)
    return {'status': report['status'], 'output_directory': str(output),
            'ranking': {mode: [{'label': row['label'], 'rank': row['development_rank'], 'correct': row['correct'],
                               'total': 48, 'smokes_passed': row['smokes']['passed']} for row in data['ranking']]
                        for mode, data in report['modes'].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=WORK / 'final-five-branch-report-v1')
    args = parser.parse_args()
    missing = ready()
    if missing:
        print(json.dumps({'status': 'not_ready', 'incomplete_evidence': missing, 'files_written': 0}))
        return 2
    report, inputs, pairs, smokes, groups, common = analyze()
    print(json.dumps(write_report(report, inputs, pairs, smokes, groups, common, args.output_dir.resolve())))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
