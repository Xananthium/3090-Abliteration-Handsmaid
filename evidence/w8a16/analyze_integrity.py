#!/usr/bin/env python3
"""Audit and summarize the completed, fixed W8A16 development comparison.

Reads saved artifacts only: no inference, network, model loading, or services.
Run after run_integrity_batch.py commits integrity-batch-result-v1.json.
"""

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re


WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
LABELS = ('baseline', 'strength-1.0', 'strength-1.5', 'attention-1.5-mlp-1.0')
CATEGORIES = ('reasoning', 'instruction', 'consistency', 'recovery')
REPEATS = (1, 2)
CASE_SHA256 = '34bdac9e83f9651a2cddbef6e2b8dcbe9eedffa3e758bc2d2402421f97c1f5ba'
GRADER_SHA256 = 'a0feb6cf923dca4e5fd5330d8a1cc24d10d51ddff0912db4454f23a70108b9bb'
SCOPE = ('Previously seen benign development diagnostics: 24 unique cases, two fixed repeats. '
         'Not held-out evidence, a refusal evaluation, or full quality/refusal acceptance. '
         'Repeated observations are not 48 independent questions.')
SMOKE_TESTS = ('forced_tool_call', 'tool_result_roundtrip', 'automatic_tool_choice',
               'pivot_after_two_errors', 'workflow_completion', 'vision_red', 'vision_blue')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def signed_object(value, field):
    require(value.get(field) == fingerprint({k: v for k, v in value.items() if k != field}),
            f'Invalid {field}')


def read(path, inputs, expected_sha=None, jsonl=False):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    sha = digest(data)
    require(expected_sha is None or sha == expected_sha, f'File hash mismatch: {path}')
    inputs[str(path)] = {'sha256': sha, 'bytes': len(data)}
    if jsonl:
        return [json.loads(line) for line in data.splitlines()]
    return json.loads(data)


def frozen_inputs(inputs):
    cases = read(ROOT / 'broader-validation/capability-cases-v1.json', inputs, CASE_SHA256)
    cases = [case for case in cases if case['id'].startswith('broader-fresh-')]
    require(len(cases) == 24 and len({case['id'] for case in cases}) == 24
            and Counter(case['category'] for case in cases) == dict.fromkeys(CATEGORIES, 6),
            'Expected the fixed authored24 case set')
    path = ROOT / 'evaluate_markers.py'
    source = path.read_bytes()
    require(digest(source) == GRADER_SHA256, 'Frozen grader changed')
    inputs[str(path)] = {'sha256': GRADER_SHA256, 'bytes': len(source)}
    # Compile only the two pure grading functions. No transport code is imported.
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in ('correct', 'json_equivalent')]
    require(len(nodes) == 2, 'Frozen pure grading definitions are missing')
    namespace = {'json': json, 're': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return cases, namespace['correct']


def totals(rows):
    return {'correct': sum(row['correct'] for row in rows), 'total': len(rows),
            'unresolved': sum(not row['generation_complete'] for row in rows),
            'unresolved_truncated': sum(row['unresolved_truncated'] for row in rows),
            'retried': sum(len(row['attempts']) > 1 for row in rows),
            'by_category': {category: {
                'total': sum(row['category'] == category for row in rows),
                'correct': sum(row['correct'] for row in rows if row['category'] == category),
                'unresolved': sum(not row['generation_complete'] for row in rows if row['category'] == category),
                'unresolved_truncated': sum(row['unresolved_truncated'] for row in rows if row['category'] == category),
            } for category in sorted(CATEGORIES)}}


def load_run(entry, batch, cases, grade, inputs):
    label = entry['label']
    run_label = f'w8a16-{label}-authored24-v1'
    summary_path = WORK / f'{run_label}-summary.json'
    require(Path(entry['path']) == summary_path, f'{label}: unexpected summary path')
    summary = read(summary_path, inputs, entry['sha256'])
    require(summary['status'] == 'completed', f'{label}: incomplete runs are not ranked')
    config_record = read(WORK / f'{run_label}-config.json', inputs)
    config = config_record['configuration']
    run_hash = fingerprint(config)
    protocol = config['protocol']
    require(config_record['configuration_fingerprint'] == run_hash
            and summary['configuration_fingerprint'] == run_hash, f'{label}: configuration mismatch')
    require(config['label'] == run_label and summary['label'] == run_label,
            f'{label}: incorrect run label')
    require(config['protocol_fingerprint'] == fingerprint(protocol)
            and summary['protocol_fingerprint'] == fingerprint(protocol), f'{label}: protocol mismatch')
    ids = [case['id'] for case in cases]
    schedule = [{'index': index, 'repeat': repeat, 'case_ids': ids[offset:offset + 4]}
                for index, (repeat, offset) in enumerate((r, o) for r in REPEATS for o in range(0, 24, 4))]
    require(protocol['case_order'] == ids and protocol['repeats'] == list(REPEATS)
            and protocol['fixed_group_schedule'] == schedule
            and protocol['selected_cases_fingerprint'] == fingerprint(cases)
            and protocol['batch_size'] == 4 and protocol['max_workers'] == 4
            and protocol['thinking'] is False and protocol['temperature'] == 0,
            f'{label}: fixed development schedule or decoding changed')
    require(config['checkpoint']['path'] == batch['base_checkpoint']
            and config['launch_profile']['sha256'] == batch['profile_sha256']
            and config['precision']['sha256'] == batch['precision_manifest_sha256']
            and summary['precision'] == config['precision']
            and summary['base_checkpoint_fingerprint'] == fingerprint(config['checkpoint'])
            and summary['adapter'] == config['adapter']
            and summary['model_id'] == config['model_id'], f'{label}: model/precision identity mismatch')
    require((config['adapter'] is None) == (label == 'baseline'), f'{label}: incorrect base/adapter role')
    require(summary['record_count'] == 48 and summary['unique_cases'] == 24
            and summary['repeats_per_case'] == 2 and summary['selected_case_ids'] == ids
            and summary['complete_groups'] == 12, f'{label}: expected exactly 48 observations in 12 groups')
    group_dir = WORK / f'{run_label}-groups'
    require(Path(summary['group_directory']) == group_dir, f'{label}: unexpected groups path')
    expected_names = {f'group-{group["index"]:04d}.json' for group in schedule}
    require({p.name for p in group_dir.glob('group-*.json')} == expected_names,
            f'{label}: missing or unexpected committed groups')
    groups = []
    for expected in schedule:
        group = read(group_dir / f'group-{expected["index"]:04d}.json', inputs)
        signed_object(group, 'group_sha256')
        require(group['status'] == 'complete' and group['schedule'] == expected
                and group['configuration_fingerprint'] == run_hash
                and len(group['records']) == 4, f'{label}: invalid committed group')
        require(type(group['wall_seconds']) in (int, float) and math.isfinite(group['wall_seconds'])
                and group['wall_seconds'] > 0, f'{label}: invalid committed group wall time')
        groups.append(group)
    rows = [row for group in groups for row in group['records']]
    require(summary['group_sha256'] == [group['group_sha256'] for group in groups],
            f'{label}: summary group hashes mismatch')
    expected_pairs = [(case, repeat) for repeat in REPEATS for case in cases]
    for position, (row, (case, repeat)) in enumerate(zip(rows, expected_pairs)):
        signed_object(row, 'record_sha256')
        expected = {'id': case['id'], 'category': case['category'], 'repeat': repeat,
                    'group_index': position // 4, 'submission_index': position % 4,
                    'configuration_fingerprint': run_hash,
                    'protocol_fingerprint': config['protocol_fingerprint'],
                    'base_checkpoint_fingerprint': fingerprint(config['checkpoint']),
                    'adapter_fingerprint': fingerprint(config['adapter']) if config['adapter'] else None,
                    'launch_profile_sha256': batch['profile_sha256'], 'model_id': config['model_id'],
                    'precision_manifest_sha256': batch['precision_manifest_sha256'],
                    'selected_cases_fingerprint': fingerprint(cases), 'case_sha256': fingerprint(case),
                    'prompt_sha256': digest(case['prompt'].encode()),
                    'answer_key_sha256': fingerprint({'kind': case['kind'], 'answer': case['answer']}),
                    'weight_bits': 8, 'activation_dtype': 'bfloat16'}
        require(all(row.get(key) == value for key, value in expected.items()),
                f'{label}: case/repeat/provenance mismatch at observation {position}')
        require(type(row['correct']) is bool and type(row['generation_complete']) is bool
                and type(row['unresolved_truncated']) is bool
                and row['generation_complete'] == (not row['unresolved_reasons'])
                and row['unresolved_truncated'] == ('output_truncated' in row['unresolved_reasons'])
                and row['unresolved_truncated'] == (row['stop_reason'] == 'max_tokens'),
                f'{label}: inconsistent completion/truncation metadata')
        answer = row['answer_text']
        require(row['raw_final_text'].strip() == answer
                and digest(answer.encode()) == row['response_sha256']
                and digest(row['raw_final_text'].encode()) == row['raw_final_text_sha256']
                and row['correct'] is (grade(case, answer) and row['generation_complete']),
                f'{label}: answer hash or frozen strict score mismatch')
        require(row['attempts'] and len(row['attempts']) == len(row['response_attempts']),
                f'{label}: invalid retry count')
        for attempt, response in zip(row['attempts'], row['response_attempts']):
            require(attempt['output_tokens'] is None or
                    (type(attempt['output_tokens']) is int and attempt['output_tokens'] >= 0),
                    f'{label}: invalid token count')
            require(attempt['output_tokens'] == response['usage'].get('output_tokens')
                    and response['model_id'] == config['model_id']
                    and attempt['response_sha256'] == digest(response['raw_text'].strip().encode())
                    and response['raw_text_sha256'] == digest(response['raw_text'].encode()),
                    f'{label}: attempt identity or token accounting mismatch')
        require(row['total_output_tokens_including_retries'] ==
                sum(attempt['output_tokens'] or 0 for attempt in row['attempts']),
                f'{label}: row token accounting mismatch')
    answers_path = WORK / f'{run_label}-answers.jsonl'
    require(Path(summary['answers_file']) == answers_path, f'{label}: unexpected answers path')
    require(read(answers_path, inputs, summary['answers_sha256'], jsonl=True) == rows,
            f'{label}: answer snapshot differs from committed groups')
    calculated = totals(rows)
    require(summary['capabilities'] == calculated and entry['capabilities'] == calculated,
            f'{label}: cached score totals disagree with committed rows')
    tokens = sum(row['total_output_tokens_including_retries'] for row in rows)
    wall = sum(group['wall_seconds'] for group in groups)
    require(summary['total_output_tokens_including_retries'] == tokens == entry['output_tokens']
            and summary['sum_committed_group_wall_seconds'] == wall == entry['batch_wall_seconds'],
            f'{label}: cached token/wall-time totals mismatch')
    archived = sorted(p.name for p in group_dir.glob('incomplete-*.json'))
    pending = sorted(p.name for p in group_dir.glob('pending-*.json'))
    require(summary['archived_incomplete_attempts'] == archived,
            f'{label}: archived incomplete-attempt list changed')
    result = {'label': label, 'role': 'reference' if label == 'baseline' else 'candidate',
              'completed': True, 'model_id': config['model_id'], 'correct': calculated['correct'],
              'total': 48, 'score_percent': 100 * calculated['correct'] / 48,
              'missing_observations': 0, 'incomplete_outputs': calculated['unresolved'],
              'final_truncated_outputs': calculated['unresolved_truncated'],
              'observations_with_any_truncated_attempt': sum(any(a['stop_reason'] == 'max_tokens'
                                                               for a in row['attempts']) for row in rows),
              'retried_observations': calculated['retried'],
              'archived_incomplete_group_attempts': len(archived), 'pending_group_attempts': len(pending),
              'archived_incomplete_files': archived, 'pending_files': pending,
              'by_category': calculated['by_category'],
              'output_tokens_including_retries': tokens, 'committed_group_wall_seconds': wall,
              'output_tokens_per_committed_wall_second': tokens / wall,
              'unknown_attempt_token_counts': sum(a['output_tokens'] is None for row in rows for a in row['attempts']),
              'run_summary': str(summary_path)}
    # Missing usage must not silently turn a lower bound into a measured rate.
    if result['unknown_attempt_token_counts']:
        result['known_output_tokens_per_committed_wall_second'] = result.pop('output_tokens_per_committed_wall_second')
        result['output_tokens_per_committed_wall_second'] = None
    return result, rows, config


def paired(baseline, candidate):
    require(len(baseline) == len(candidate) == 48, 'Pairing requires exactly 48 observations')
    pairs = []
    for base, row in zip(baseline, candidate):
        require((base['id'], base['repeat'], base['case_sha256']) ==
                (row['id'], row['repeat'], row['case_sha256']), 'Pair identity mismatch')
        pairs.append({'case_id': row['id'], 'repeat': row['repeat'], 'category': row['category'],
                      'baseline_correct': base['correct'], 'candidate_correct': row['correct'],
                      'outcome': ('retained' if row['correct'] else 'loss') if base['correct']
                                 else ('gain' if row['correct'] else 'both_incorrect'),
                      'baseline_complete': base['generation_complete'],
                      'candidate_complete': row['generation_complete'],
                      'baseline_truncated': base['unresolved_truncated'],
                      'candidate_truncated': row['unresolved_truncated']})
    return pairs


def compare(result, baseline_result, pairs):
    counts = Counter(pair['outcome'] for pair in pairs)
    numerator, denominator = result['correct'], baseline_result['correct']
    result.update(baseline_correct=denominator, baseline_losses=counts['loss'], baseline_gains=counts['gain'],
                  baseline_correct_retained=counts['retained'], both_incorrect=counts['both_incorrect'],
                  net_correct_change=numerator - denominator,
                  relative_score_retention_percent=100 * numerator / denominator if denominator else None)
    for category, values in result['by_category'].items():
        selected = [pair for pair in pairs if pair['category'] == category]
        base_correct = baseline_result['by_category'][category]['correct']
        values.update(baseline_correct=base_correct,
                      baseline_losses=sum(pair['outcome'] == 'loss' for pair in selected),
                      baseline_gains=sum(pair['outcome'] == 'gain' for pair in selected),
                      relative_score_retention_percent=100 * values['correct'] / base_correct if base_correct else None)
    result['paired_changes'] = [pair for pair in pairs if pair['outcome'] in ('loss', 'gain')]


def analyze(batch_path):
    inputs = {}
    batch = read(batch_path, inputs)
    require(batch['status'] == 'completed', 'Batch is not complete; no development ranking is emitted')
    require([entry['label'] for entry in batch['comparisons']] == list(LABELS),
            'Expected one baseline and the three fixed candidates, in controller order')
    cases, grade = frozen_inputs(inputs)
    results, records, configs = {}, {}, {}
    for entry in batch['comparisons']:
        label = entry['label']
        results[label], records[label], configs[label] = load_run(entry, batch, cases, grade, inputs)
    for label in LABELS[1:]:
        for field in ('protocol', 'checkpoint', 'launch_profile', 'precision'):
            require(configs[label][field] == configs['baseline'][field],
                    f'{label}: unmatched {field} versus baseline')
    all_pairs = []
    for label in LABELS:
        pairs = paired(records['baseline'], records[label])
        compare(results[label], results['baseline'], pairs)
        all_pairs.extend({'label': label, **pair} for pair in pairs)
    # Stable order for equal scores; ties share a rank, with no speed tiebreak.
    ranking = sorted(results.values(), key=lambda row: -row['correct'])
    for row in ranking:
        row['development_rank'] = 1 + sum(other['correct'] > row['correct'] for other in ranking)
    require([entry['label'] for entry in batch['behaviors']] == list(LABELS),
            'Unexpected behavior comparison labels')
    behaviors = []
    for entry in batch['behaviors']:
        saved = read(entry['path'], inputs, entry['sha256'])
        signed_object(saved, 'result_sha256')
        require(saved['completed'] is True and saved['total'] == len(saved['records']) == 7
                and [row['test'] for row in saved['records']] == list(SMOKE_TESTS)
                and saved['passed'] == sum(row['passed'] for row in saved['records']) == entry['passed']
                and saved['identity']['model_id'] == results[entry['label']]['model_id'],
                'Behavior smoke summary mismatch')
        behaviors.append({'label': entry['label'], 'passed': saved['passed'], 'total': 7,
                          'failed_tests': [row['test'] for row in saved['records'] if not row['passed']]})
    # Refuse a mixed snapshot if another process replaced any evidence while reading.
    for path, identity in inputs.items():
        require(digest(Path(path).read_bytes()) == identity['sha256'], f'Input changed during analysis: {path}')
    report = {'schema_version': 1, 'status': 'completed_development_analysis',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': SCOPE,
              'quality_reference': 'Untouched W8A16 baseline; BF16 source is conversion input only.',
              'refusal_evaluation_performed': False, 'full_quality_refusal_acceptance_evaluated': False,
              'pairing': {'unique_cases': 24, 'repeats': list(REPEATS), 'observations_per_model': 48,
                          'models': 4, 'total_observations': 192, 'keys': ['case_id', 'repeat'],
                          'schedule': configs['baseline']['protocol']['fixed_group_schedule']},
              'ranking_rule': 'Completed development strict correct/48 descending; equal scores share rank. Tool/vision smokes and speed are not folded into this score.',
              'retention_rule': '100 * candidate correct / baseline correct; uncapped, null when baseline correct is zero. Paired losses and gains are reported separately.',
              'throughput_rule': 'All output tokens including retries in committed groups / sum of committed group wall seconds. Excludes loading, gaps between groups, abandoned attempts, and tool/vision smokes. Never divides by summed request latency.',
              'ranking': ranking, 'tool_vision_smokes': behaviors, 'paired_observations': all_pairs,
              'inputs': inputs, 'analyzer_sha256': digest(Path(__file__).read_bytes())}
    return report


def atomic_text(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def csv_text(rows, fields):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-result', type=Path, default=WORK / 'integrity-batch-result-v1.json')
    args = parser.parse_args()
    if not args.batch_result.is_file():
        print(json.dumps({'status': 'not_ready', 'reason': 'Completed batch result is not present; no files changed.'}))
        return 2
    report = analyze(args.batch_result)
    paths = {
        'json': WORK / 'integrity-analysis-v1.json',
        'ranking_csv': WORK / 'integrity-ranking-v1.csv',
        'categories_csv': WORK / 'integrity-categories-v1.csv',
        'pairs_csv': WORK / 'integrity-pairs-v1.csv',
    }
    fields = ('development_rank', 'label', 'role', 'correct', 'total', 'score_percent',
              'relative_score_retention_percent', 'baseline_losses', 'baseline_gains',
              'baseline_correct_retained', 'net_correct_change', 'incomplete_outputs',
              'final_truncated_outputs', 'observations_with_any_truncated_attempt',
              'retried_observations', 'missing_observations', 'archived_incomplete_group_attempts',
              'pending_group_attempts', 'output_tokens_including_retries', 'committed_group_wall_seconds',
              'output_tokens_per_committed_wall_second', 'unknown_attempt_token_counts')
    atomic_text(paths['ranking_csv'], csv_text(report['ranking'], fields))
    categories = [{'label': row['label'], 'category': category, **row['by_category'][category]}
                  for row in report['ranking'] for category in CATEGORIES]
    atomic_text(paths['categories_csv'], csv_text(categories,
                ('label', 'category', 'correct', 'total', 'baseline_correct', 'baseline_losses',
                 'baseline_gains', 'relative_score_retention_percent', 'unresolved', 'unresolved_truncated')))
    atomic_text(paths['pairs_csv'], csv_text(report['paired_observations'],
                ('label', 'case_id', 'repeat', 'category', 'baseline_correct', 'candidate_correct',
                 'outcome', 'baseline_complete', 'candidate_complete', 'baseline_truncated', 'candidate_truncated')))
    report['csv_artifacts'] = {name: {'path': str(path), 'sha256': digest(path.read_bytes())}
                               for name, path in paths.items() if name != 'json'}
    atomic_text(paths['json'], json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'artifacts': {k: str(v) for k, v in paths.items()},
                      'ranking': [{key: row[key] for key in fields[:10]} for row in report['ranking']]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
