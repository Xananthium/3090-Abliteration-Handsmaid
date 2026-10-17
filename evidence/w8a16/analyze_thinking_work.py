#!/usr/bin/env python3
"""Snapshot committed thinking-development token work and latency, without inference.

Supports a contiguous prefix of complete groups while the batch is running.
Pending/abandoned groups are excluded. Each invocation creates new timestamped
JSON/CSV artifacts; it never changes evaluation evidence or existing analyses.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import types

WORK = Path(__file__).resolve().parent
THINKING_ANALYZER_SHA256 = 'a2c907c08d1963562516c47d984ae5f67a136689a23373c56d6fded1539d33b6'


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None


def difference(candidate, baseline):
    return candidate - baseline if candidate is not None and baseline is not None else None


def total_known(values):
    return sum(values) if all(value is not None for value in values) else None


def load_helpers(inputs):
    path = WORK / 'analyze_thinking_integrity.py'
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != THINKING_ANALYZER_SHA256:
        raise ValueError('Pinned thinking analysis helpers changed')
    inputs[str(path)] = {'sha256': THINKING_ANALYZER_SHA256, 'bytes': len(source)}
    thinking = types.ModuleType('_saved_thinking_diagnostics')
    thinking.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), thinking.__dict__)
    common = thinking.load_common()
    cases, grade = common.frozen_inputs(inputs)
    validator, constants = thinking.load_validator(common, inputs)
    inputs[str(WORK / 'analyze_integrity.py')] = {
        'sha256': thinking.COMMON_SHA256, 'bytes': (WORK / 'analyze_integrity.py').stat().st_size}
    return thinking, common, cases, grade, validator, constants


def observation(row, common):
    attempts, responses = row['attempts'], row['response_attempts']
    common.require(len(attempts) == len(responses) >= 1, 'Mismatched native generation attempts')
    for attempt, response in zip(attempts, responses):
        tokens = attempt['output_tokens']
        common.require(tokens is None or type(tokens) is int and tokens >= 0, 'Invalid token usage')
        common.require(tokens == response['usage'].get('output_tokens'), 'Native token usage mismatch')
        seconds = response['native_request_seconds']
        common.require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
                       'Invalid native generation request latency')
    tokens = total_known([attempt['output_tokens'] for attempt in attempts])
    common.require(row['total_output_tokens_including_retries'] == sum(a['output_tokens'] or 0 for a in attempts),
                   'Cached generated-token total differs from native attempts')
    return {'case_id': row['id'], 'repeat': row['repeat'], 'category': row['category'],
            'group_index': row['group_index'], 'case_sha256': row['case_sha256'],
            'generated_tokens_including_retries': tokens,
            'known_generated_tokens_including_retries': sum(a['output_tokens'] or 0 for a in attempts),
            'unknown_token_usage_attempts': sum(a['output_tokens'] is None for a in attempts),
            'final_attempt_output_tokens': attempts[-1]['output_tokens'],
            'retry_count': len(attempts) - 1,
            'native_generation_request_seconds_including_retries': sum(r['native_request_seconds'] for r in responses),
            'thinking_characters_including_retries': sum(r['thinking_characters'] for r in responses),
            'final_answer_characters': len(row['answer_text']),
            'correct': row['correct'], 'generation_complete': row['generation_complete'],
            'final_truncated': row['unresolved_truncated']}


def model_snapshot(label, thinking, common, cases, grade, validator, constants, inputs):
    prefix = f'w8a16-{label}-thinking24-v1'
    config_path = WORK / f'{prefix}-config.json'
    if not config_path.exists():
        return {'label': label, 'status': 'not_started', 'committed_observations': 0,
                'expected_observations': 48, 'complete_groups': 0}, None, [], []
    saved = common.read(config_path, inputs)
    config = saved['configuration']
    run_hash = common.fingerprint(config)
    protocol = config['protocol']
    ids = [case['id'] for case in cases]
    schedule = [{'index': i, 'repeat': repeat, 'case_ids': ids[offset:offset + 4]}
                for i, (repeat, offset) in enumerate((r, o) for r in (1, 2) for o in range(0, 24, 4))]
    common.require(saved['configuration_fingerprint'] == run_hash
                   and config['protocol_fingerprint'] == common.fingerprint(protocol)
                   and config['label'] == prefix and (config['adapter'] is None) == (label == 'baseline'),
                   f'{label}: configuration identity mismatch')
    common.require(protocol['runner'] == 'w8a16-thinking-integrity-fixed-batches'
                   and protocol['script_sha256'] == thinking.RUNNER_SHA256
                   and protocol['thinking'] is True and protocol['temperature'] == 0
                   and protocol['chat_template_kwargs'] == {'enable_thinking': True}
                   and protocol['output_budgets'] == [4096, 6144]
                   and protocol['thinking_protocol_sources'] == constants['PROTOCOL_SOURCE_HASHES']
                   and protocol['case_order'] == ids and protocol['repeats'] == [1, 2]
                   and protocol['selected_cases_fingerprint'] == common.fingerprint(cases)
                   and protocol['fixed_group_schedule'] == schedule
                   and protocol['expected_record_count'] == 48,
                   f'{label}: unmatched fixed thinking-development protocol')
    precision = common.read(config['precision']['path'], inputs, config['precision']['sha256'])
    common.read(config['launch_profile']['path'], inputs, config['launch_profile']['sha256'])
    common.require(precision == config['precision']['manifest'] and precision['weight_bits'] == 8
                   and precision['activation_dtype'] == 'bfloat16'
                   and precision['expected_base_checkpoint'] == config['checkpoint']['path'],
                   f'{label}: precision/checkpoint mismatch')
    directory = WORK / f'{prefix}-groups'
    paths = sorted(directory.glob('group-*.json'))  # Capture the immutable committed prefix once.
    common.require(len(paths) <= 12 and [p.name for p in paths] ==
                   [f'group-{i:04d}.json' for i in range(len(paths))], f'{label}: noncontiguous committed groups')
    case_by_id = {case['id']: case for case in cases}
    helper = types.SimpleNamespace(fingerprint=common.fingerprint, correct=grade, digest_bytes=common.digest)
    support = types.SimpleNamespace(row_identity=thinking.row_identity, BUDGETS=[4096, 6144])
    observations, groups = [], []
    for index, path in enumerate(paths):
        saved_group = common.read(path, inputs)
        common.signed_object(saved_group, 'group_sha256')
        common.require(saved_group['configuration_fingerprint'] == run_hash
                       and saved_group['status'] == 'complete' and saved_group['schedule'] == schedule[index]
                       and len(saved_group['records']) == 4, f'{label}: incompatible group commit')
        wall = saved_group['wall_seconds']
        common.require(type(wall) in (int, float) and math.isfinite(wall) and wall > 0,
                       f'{label}: invalid committed group wall time')
        group_observations = []
        for position, (case_id, row) in enumerate(zip(schedule[index]['case_ids'], saved_group['records'])):
            validator(row, case_by_id[case_id], schedule[index]['repeat'], index, position,
                      config, run_hash, support, helper)
            group_observations.append(observation(row, common))
        observations.extend(group_observations)
        tokens = total_known([row['generated_tokens_including_retries'] for row in group_observations])
        groups.append({'group_index': index, 'repeat': schedule[index]['repeat'], 'case_ids': schedule[index]['case_ids'],
                       'generated_tokens_including_retries': tokens, 'committed_wall_seconds': wall,
                       'development_output_tokens_per_committed_wall_second': ratio(tokens, wall)})
    tokens = total_known([row['generated_tokens_including_retries'] for row in observations])
    wall = sum(group['committed_wall_seconds'] for group in groups)
    summary = {'label': label, 'status': 'all_observations_committed' if len(observations) == 48 else 'partial_committed_prefix',
               'committed_observations': len(observations), 'expected_observations': 48, 'complete_groups': len(groups),
               'total_generated_tokens_including_retries': tokens,
               'known_generated_tokens_including_retries': sum(row['known_generated_tokens_including_retries'] for row in observations),
               'unknown_token_usage_attempts': sum(row['unknown_token_usage_attempts'] for row in observations),
               'sum_committed_group_wall_seconds': wall,
               'development_output_tokens_per_committed_wall_second': ratio(tokens, wall),
               'retried_observations': sum(row['retry_count'] > 0 for row in observations),
               'incomplete_outputs': sum(not row['generation_complete'] for row in observations),
               'final_truncated_outputs': sum(row['final_truncated'] for row in observations),
               'pending_files_observed_but_excluded': sorted(p.name for p in directory.glob('pending-*.json')),
               'archived_attempts_observed_but_excluded': sorted(p.name for p in directory.glob('incomplete-*.json'))}
    return summary, config, observations, groups


def pair_observations(label, baseline, candidate, common):
    common.require((baseline['case_id'], baseline['repeat'], baseline['case_sha256']) ==
                   (candidate['case_id'], candidate['repeat'], candidate['case_sha256']), 'Unmatched observation pair')
    b_tokens, c_tokens = baseline['generated_tokens_including_retries'], candidate['generated_tokens_including_retries']
    b_seconds = baseline['native_generation_request_seconds_including_retries']
    c_seconds = candidate['native_generation_request_seconds_including_retries']
    return {'label': label, 'case_id': candidate['case_id'], 'repeat': candidate['repeat'],
            'category': candidate['category'], 'group_index': candidate['group_index'],
            'baseline_generated_tokens_including_retries': b_tokens,
            'candidate_generated_tokens_including_retries': c_tokens,
            'extra_generated_tokens': difference(c_tokens, b_tokens), 'generated_token_ratio': ratio(c_tokens, b_tokens),
            'baseline_native_generation_request_seconds_including_retries': b_seconds,
            'candidate_native_generation_request_seconds_including_retries': c_seconds,
            'extra_native_generation_request_seconds': c_seconds - b_seconds,
            'native_generation_latency_ratio': ratio(c_seconds, b_seconds),
            'baseline_retry_count': baseline['retry_count'], 'candidate_retry_count': candidate['retry_count'],
            'baseline_final_attempt_output_tokens': baseline['final_attempt_output_tokens'],
            'candidate_final_attempt_output_tokens': candidate['final_attempt_output_tokens'],
            'baseline_thinking_characters_including_retries': baseline['thinking_characters_including_retries'],
            'candidate_thinking_characters_including_retries': candidate['thinking_characters_including_retries'],
            'baseline_correct': baseline['correct'], 'candidate_correct': candidate['correct'],
            'baseline_complete': baseline['generation_complete'], 'candidate_complete': candidate['generation_complete'],
            'baseline_final_truncated': baseline['final_truncated'], 'candidate_final_truncated': candidate['final_truncated']}


def analyze():
    inputs = {}
    thinking, common, cases, grade, validator, constants = load_helpers(inputs)
    models, configs, observations, groups = {}, {}, {}, {}
    for label in common.LABELS:
        models[label], configs[label], observations[label], groups[label] = model_snapshot(
            label, thinking, common, cases, grade, validator, constants, inputs)
    pairs, group_pairs, case_totals = [], [], []
    baseline_index = {(row['case_id'], row['repeat']): row for row in observations['baseline']}
    for label in common.LABELS[1:]:
        if configs[label] is None:
            continue
        common.require(configs['baseline'] is not None, 'Candidate records have no baseline configuration')
        for field in ('protocol', 'checkpoint', 'launch_profile', 'precision'):
            common.require(configs[label][field] == configs['baseline'][field], f'{label}: unmatched {field}')
        current = [pair_observations(label, baseline_index[(row['case_id'], row['repeat'])], row, common)
                   for row in observations[label] if (row['case_id'], row['repeat']) in baseline_index]
        pairs.extend(current)
        matched_group_count = min(len(groups[label]), len(groups['baseline']))
        for base, candidate in zip(groups['baseline'][:matched_group_count], groups[label][:matched_group_count]):
            group_pairs.append({'label': label, 'group_index': base['group_index'], 'repeat': base['repeat'],
                                'case_ids': base['case_ids'],
                                'baseline_committed_wall_seconds': base['committed_wall_seconds'],
                                'candidate_committed_wall_seconds': candidate['committed_wall_seconds'],
                                'committed_group_wall_ratio': ratio(candidate['committed_wall_seconds'], base['committed_wall_seconds']),
                                'baseline_generated_tokens_including_retries': base['generated_tokens_including_retries'],
                                'candidate_generated_tokens_including_retries': candidate['generated_tokens_including_retries']})
        b_tokens = total_known([row['baseline_generated_tokens_including_retries'] for row in current])
        c_tokens = total_known([row['candidate_generated_tokens_including_retries'] for row in current])
        b_wall = sum(group['committed_wall_seconds'] for group in groups['baseline'][:matched_group_count])
        c_wall = sum(group['committed_wall_seconds'] for group in groups[label][:matched_group_count])
        models[label]['matched_baseline_comparison'] = {
            'paired_observations': len(current), 'unpaired_candidate_observations': len(observations[label]) - len(current),
            'matched_complete_groups': matched_group_count,
            'baseline_generated_tokens_including_retries': b_tokens, 'candidate_generated_tokens_including_retries': c_tokens,
            'extra_generated_tokens': difference(c_tokens, b_tokens), 'generated_token_ratio': ratio(c_tokens, b_tokens),
            'baseline_committed_group_wall_seconds': b_wall, 'candidate_committed_group_wall_seconds': c_wall,
            'committed_group_wall_ratio': ratio(c_wall, b_wall)}
        for case in cases:
            selected = [row for row in current if row['case_id'] == case['id']]
            if not selected:
                continue
            b = total_known([row['baseline_generated_tokens_including_retries'] for row in selected])
            c = total_known([row['candidate_generated_tokens_including_retries'] for row in selected])
            case_totals.append({'label': label, 'case_id': case['id'], 'category': case['category'],
                                'paired_repeats': len(selected), 'expected_repeats': 2,
                                'baseline_generated_tokens_including_retries': b,
                                'candidate_generated_tokens_including_retries': c,
                                'extra_generated_tokens': difference(c, b), 'generated_token_ratio': ratio(c, b)})
    background_paths = [WORK / 'w8-weight-rebase-launch-v1.json']
    background_paths += sorted(WORK.glob('w8-weight-rebase*completion*.json'))
    background = []
    for path in background_paths:
        if path.is_file():
            record = common.read(path, inputs)
            background.append({'path': str(path), 'sha256': inputs[str(path)]['sha256'], 'record': record})
    timing_context = ('possible background CPU work; see JSON provenance' if background
                      else 'development timing; controlled idle host not established')
    for row in [*models.values(), *pairs, *group_pairs]:
        row['timing_context'] = timing_context
    for path, identity in inputs.items():
        common.require(common.digest(Path(path).read_bytes()) == identity['sha256'], f'Evidence changed while reading: {path}')
    complete = all(model['committed_observations'] == 48 for model in models.values())
    report = {'schema_version': 1, 'status': 'all_observations_committed' if complete else 'partial_committed_development_diagnostic',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Development token work and latency on 24 previously seen cases, two fixed repeats, concurrency four. Not merged-checkpoint deployment throughput, overall quality, or refusal evidence.',
              'token_scope': 'Native generated tokens include thinking and final text, including sequential retries. Separate reasoning/final token counts are unavailable; thinking characters are not token counts.',
              'timing_scope': 'Model elapsed time is the sum of committed group wall seconds. Per-observation native generation request latency includes server queue/prefill/decode/network time and sequential retries; overlapping observations must not be summed into elapsed throughput. Group time is never assigned to an individual case.',
              'background_work_provenance': background,
              'latency_caveat': 'The CPU-only W8 factor rebase launched at 2026-10-15T23:56:38Z with four CPU threads, CPUQuota 400%, an 8 GiB memory limit and PrivateDevices. It overlaps part of the strength-1.0 evaluation. Latencies and committed-wall rates can include background CPU/resource contention and are not controlled idle-host comparisons. Generated-token deltas are reported separately; neither metric establishes merged-checkpoint deployment kernel speed. The attached launch/completion records describe only this known background job, not all host activity.',
              'partial_scope': 'Only hash-verified complete committed groups contribute. Pending and abandoned attempts are excluded. Candidate/baseline cost totals use exactly matched case/repeat observations and group indices.',
              'models': list(models.values()), 'paired_observations': pairs, 'paired_groups': group_pairs,
              'paired_case_totals': case_totals,
              'largest_generated_token_increases': sorted([row for row in pairs if row['extra_generated_tokens'] is not None],
                                                          key=lambda row: row['extra_generated_tokens'], reverse=True)[:12],
              'inputs': inputs, 'analyzer_sha256': common.digest(Path(__file__).read_bytes())}
    return report, common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report, common = analyze()
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    prefix = WORK / f'thinking-work-{tag}'
    files = {'json': Path(str(prefix) + '.json'), 'pairs_csv': Path(str(prefix) + '-pairs.csv'),
             'cases_csv': Path(str(prefix) + '-cases.csv'), 'models_csv': Path(str(prefix) + '-models.csv'),
             'groups_csv': Path(str(prefix) + '-groups.csv')}
    pair_fields = ('label', 'case_id', 'repeat', 'category', 'group_index',
                   'baseline_generated_tokens_including_retries', 'candidate_generated_tokens_including_retries',
                   'extra_generated_tokens', 'generated_token_ratio',
                   'baseline_native_generation_request_seconds_including_retries',
                   'candidate_native_generation_request_seconds_including_retries',
                   'extra_native_generation_request_seconds', 'native_generation_latency_ratio',
                   'baseline_retry_count', 'candidate_retry_count', 'baseline_correct', 'candidate_correct',
                   'baseline_complete', 'candidate_complete', 'baseline_final_truncated', 'candidate_final_truncated',
                   'timing_context')
    common.atomic_text(files['pairs_csv'], common.csv_text(report['paired_observations'], pair_fields))
    common.atomic_text(files['cases_csv'], common.csv_text(report['paired_case_totals'],
                       ('label', 'case_id', 'category', 'paired_repeats', 'expected_repeats',
                        'baseline_generated_tokens_including_retries', 'candidate_generated_tokens_including_retries',
                        'extra_generated_tokens', 'generated_token_ratio')))
    common.atomic_text(files['models_csv'], common.csv_text(report['models'],
                       ('label', 'status', 'committed_observations', 'expected_observations', 'complete_groups',
                        'total_generated_tokens_including_retries', 'sum_committed_group_wall_seconds',
                        'development_output_tokens_per_committed_wall_second', 'retried_observations',
                        'incomplete_outputs', 'final_truncated_outputs', 'unknown_token_usage_attempts', 'timing_context')))
    common.atomic_text(files['groups_csv'], common.csv_text(report['paired_groups'],
                       ('label', 'group_index', 'repeat', 'baseline_committed_wall_seconds',
                        'candidate_committed_wall_seconds', 'committed_group_wall_ratio',
                        'baseline_generated_tokens_including_retries', 'candidate_generated_tokens_including_retries',
                        'timing_context')))
    report['csv_artifacts'] = {name: {'path': str(path), 'sha256': common.digest(path.read_bytes())}
                               for name, path in files.items() if name != 'json'}
    common.atomic_text(files['json'], json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'artifacts': {name: str(path) for name, path in files.items()},
                      'committed_observations': {row['label']: row['committed_observations'] for row in report['models']},
                      'paired_observations': len(report['paired_observations']),
                      'largest_generated_token_increases': [{key: row[key] for key in pair_fields[:9]}
                                                            for row in report['largest_generated_token_increases'][:3]]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
