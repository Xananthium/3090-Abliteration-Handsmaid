#!/usr/bin/env python3
"""Analyze the completed thinking24 batch without inference or service access.

Preserves the separate thinking-off analyzer and artifacts. Scores only final
native text blocks, with the frozen grader and completion gate unchanged.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import types

WORK = Path(__file__).resolve().parent
COMMON_SHA256 = '8bff2efeca121cb3fe100386d29736edfb2c3abfdc86fa6200658a020ef6f11e'
RUNNER_SHA256 = 'b3054a303ebd8371a57bd0951d93dc7cc8ff974a06180f027462d5b621001696'


def load_common():
    import hashlib
    path = WORK / 'analyze_integrity.py'
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != COMMON_SHA256:
        raise ValueError('Pinned thinking-off accounting helpers changed')
    common = types.ModuleType('_read_only_integrity_accounting')
    common.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), common.__dict__)
    return common


def load_validator(common, inputs):
    path = WORK / 'evaluate_integrity_thinking.py'
    source = path.read_bytes()
    common.require(common.digest(source) == RUNNER_SHA256, 'Pinned thinking24 runner changed')
    inputs[str(path)] = {'sha256': RUNNER_SHA256, 'bytes': len(source)}
    tree = ast.parse(source)
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ('PROTOCOL_SOURCE_HASHES', 'THINKING_TEMPLATE_SHA256')}
    validator = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == 'validate_record')
    namespace = {}
    exec(compile(ast.Module(body=[validator], type_ignores=[]), str(path), 'exec'), namespace)
    for name, expected in constants['PROTOCOL_SOURCE_HASHES'].items():
        data = Path(name).read_bytes()
        common.require(common.digest(data) == expected, f'Inspected native thinking implementation changed: {name}')
        inputs[name] = {'sha256': expected, 'bytes': len(data)}
    return namespace['validate_record'], constants


def row_identity(case, config, run_hash, helper):
    return {'id': case['id'], 'category': case['category'], 'configuration_fingerprint': run_hash,
            'protocol_fingerprint': config['protocol_fingerprint'],
            'base_checkpoint_fingerprint': helper.fingerprint(config['checkpoint']),
            'adapter_fingerprint': helper.fingerprint(config['adapter']) if config['adapter'] else None,
            'launch_profile_sha256': config['launch_profile']['sha256'], 'model_id': config['model_id'],
            'selected_cases_fingerprint': config['protocol']['selected_cases_fingerprint'],
            'case_sha256': helper.fingerprint(case), 'prompt_sha256': helper.digest_bytes(case['prompt'].encode()),
            'answer_key_sha256': helper.fingerprint({'kind': case['kind'], 'answer': case['answer']}),
            'precision_manifest_sha256': config['precision']['sha256'], 'weight_bits': 8,
            'activation_dtype': 'bfloat16',
            'quantization_backend': config['precision']['manifest']['quantization_backend']}


def load_run(entry, batch, cases, grade, validate_record, constants, common, inputs):
    require, fingerprint = common.require, common.fingerprint
    label = entry['label']
    run_label = f'w8a16-{label}-thinking24-v1'
    summary_path = WORK / f'{run_label}-summary.json'
    require(Path(entry['path']) == summary_path, f'{label}: unexpected summary path')
    summary = common.read(summary_path, inputs, entry['sha256'])
    require(summary['status'] == 'completed', f'{label}: incomplete runs cannot be ranked')
    saved = common.read(WORK / f'{run_label}-config.json', inputs)
    config = saved['configuration']
    run_hash, protocol = fingerprint(config), config['protocol']
    require(saved['configuration_fingerprint'] == run_hash == summary['configuration_fingerprint']
            and config['protocol_fingerprint'] == fingerprint(protocol) == summary['protocol_fingerprint'],
            f'{label}: configuration/protocol fingerprint mismatch')
    ids = [case['id'] for case in cases]
    schedule = [{'index': i, 'repeat': repeat, 'case_ids': ids[offset:offset + 4]}
                for i, (repeat, offset) in enumerate((r, o) for r in common.REPEATS for o in range(0, 24, 4))]
    require(config['label'] == summary['label'] == run_label
            and protocol['case_order'] == ids and protocol['repeats'] == [1, 2]
            and protocol['fixed_group_schedule'] == schedule
            and protocol['selected_cases_fingerprint'] == fingerprint(cases)
            and protocol['selected_case_count'] == 24 and protocol['expected_record_count'] == 48
            and protocol['selected_category_counts'] == dict.fromkeys(common.CATEGORIES, 6)
            and protocol['batch_size'] == protocol['max_workers'] == 4,
            f'{label}: expected the fixed 24-case, 48-observation schedule')
    require(protocol['runner'] == 'w8a16-thinking-integrity-fixed-batches'
            and protocol['script_sha256'] == RUNNER_SHA256
            and protocol['thinking'] is True and protocol['temperature'] == 0
            and protocol['chat_template_kwargs'] == {'enable_thinking': True}
            and protocol['output_budgets'] == [4096, 6144] and protocol['context_limit'] == 8192
            and protocol['thinking_protocol_sources'] == constants['PROTOCOL_SOURCE_HASHES'],
            f'{label}: native thinking protocol differs from the pinned runner')
    precision = common.read(config['precision']['path'], inputs, batch['precision_manifest_sha256'])
    profile = common.read(config['launch_profile']['path'], inputs, batch['profile_sha256'])
    require(config['precision']['manifest'] == precision
            and config['precision']['sha256'] == batch['precision_manifest_sha256']
            and config['launch_profile']['sha256'] == batch['profile_sha256']
            and precision['expected_base_checkpoint'] == config['checkpoint']['path']
            and precision['weight_bits'] == 8 and precision['activation_dtype'] == 'bfloat16'
            and config['checkpoint']['files']['chat_template.jinja']['sha256'] == constants['THINKING_TEMPLATE_SHA256'],
            f'{label}: precision/checkpoint/template identity mismatch')
    argv = profile['args']
    for flag, expected in {'--reasoning-parser': 'gemma4', '--dtype': 'bfloat16',
                           '--kv-cache-dtype': 'bfloat16', '--host': '127.0.0.1',
                           '--port': '8002', '--max-model-len': '8192'}.items():
        require(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == expected,
                f'{label}: incompatible launch {flag}')
    require(summary['model_id'] == config['model_id'] and summary['adapter'] == config['adapter']
            and (config['adapter'] is None) == (label == 'baseline')
            and summary['base_checkpoint_fingerprint'] == fingerprint(config['checkpoint'])
            and summary['base_checkpoint'] == config['checkpoint']['path']
            and summary['precision'] == config['precision'] and summary['launch_profile'] == config['launch_profile'],
            f'{label}: baseline/adapter serving identity mismatch')
    require(summary['record_count'] == 48 and summary['unique_cases'] == 24
            and summary['repeats_per_case'] == 2 and summary['selected_case_ids'] == ids
            and summary['complete_groups'] == 12 and summary['thinking_requested'] is True
            and summary['selected_category_counts'] == dict.fromkeys(common.CATEGORIES, 6)
            and summary['record_category_counts'] == dict.fromkeys(common.CATEGORIES, 12),
            f'{label}: incomplete or incompatible thinking summary')
    directory = WORK / f'{run_label}-groups'
    require(Path(summary['group_directory']) == directory
            and {p.name for p in directory.glob('group-*.json')} ==
            {f'group-{g["index"]:04d}.json' for g in schedule}, f'{label}: incomplete committed groups')
    groups = []
    helper = types.SimpleNamespace(fingerprint=fingerprint, correct=grade, digest_bytes=common.digest)
    support = types.SimpleNamespace(row_identity=row_identity, BUDGETS=[4096, 6144])
    case_by_id = {case['id']: case for case in cases}
    for expected in schedule:
        group = common.read(directory / f'group-{expected["index"]:04d}.json', inputs)
        common.signed_object(group, 'group_sha256')
        require(group['status'] == 'complete' and group['schedule'] == expected
                and group['configuration_fingerprint'] == run_hash and len(group['records']) == 4,
                f'{label}: committed group mismatch')
        require(type(group['wall_seconds']) in (int, float) and math.isfinite(group['wall_seconds'])
                and group['wall_seconds'] > 0, f'{label}: invalid wall time')
        for position, (case_id, row) in enumerate(zip(expected['case_ids'], group['records'])):
            validate_record(row, case_by_id[case_id], expected['repeat'], expected['index'], position,
                            config, run_hash, support, helper)
            require(row['context_limit'] == 8192 and row['context_margin'] == 64
                    and type(row['counted_input_tokens']) is int and row['counted_input_tokens'] > 0,
                    f'{label}: invalid context accounting')
            available = 8192 - row['counted_input_tokens'] - 64
            budgets = list(dict.fromkeys(min(b, available) for b in [4096, 6144]))
            require(row['available_output_tokens'] == available and row['context_safe_budgets'] == budgets,
                    f'{label}: invalid context-safe output budgets')
            require(1 <= len(row['attempts']) <= len(budgets), f'{label}: invalid retry count')
            for index, (attempt, response) in enumerate(zip(row['attempts'], row['response_attempts'])):
                require(response['requested_max_tokens'] == budgets[index]
                        and response['thinking_budget_tokens'] == budgets[index] - 1024 >= 1024
                        and response['stop_reason'] == attempt['stop_reason']
                        and (index == len(row['attempts']) - 1 or attempt['stop_reason'] == 'max_tokens'),
                        f'{label}: thinking reserve or retry policy mismatch')
                require(attempt['output_tokens'] == response['usage'].get('output_tokens')
                        and (attempt['output_tokens'] is None or
                             (type(attempt['output_tokens']) is int and attempt['output_tokens'] >= 0)),
                        f'{label}: invalid native token usage')
            final = row['response_attempts'][-1]
            require(final['stop_reason'] != 'max_tokens' or len(row['attempts']) == len(budgets),
                    f'{label}: a truncated response stopped before the fixed retries were exhausted')
            reasons = []
            if final['stop_reason'] == 'max_tokens':
                reasons.append('output_truncated')
            elif final['stop_reason'] not in ('end_turn', 'stop_sequence'):
                reasons.append('unexpected_stop_reason')
            if any(block['type'] not in ('text', 'thinking') for block in final['native_content_blocks']):
                reasons.append('unexpected_nontext_content')
            if not row['answer_text']:
                reasons.append('missing_final_text')
            if any(marker in row['answer_text'] for marker in ('<|channel>', '<channel|>', '<|think|>')):
                reasons.append('unparsed_reasoning_markers_in_final_text')
            require(row['unresolved_reasons'] == reasons
                    and type(row['unresolved_truncated']) is bool
                    and row['unresolved_truncated'] == ('output_truncated' in reasons),
                    f'{label}: final-only completion gate mismatch')
            require(row['total_output_tokens_including_retries'] ==
                    sum(attempt['output_tokens'] or 0 for attempt in row['attempts']),
                    f'{label}: output token accounting mismatch')
        groups.append(group)
    rows = [row for group in groups for row in group['records']]
    require(summary['group_sha256'] == [group['group_sha256'] for group in groups],
            f'{label}: summary/group hashes mismatch')
    answers_path = WORK / f'{run_label}-answers.jsonl'
    require(Path(summary['answers_file']) == answers_path
            and common.read(answers_path, inputs, summary['answers_sha256'], jsonl=True) == rows,
            f'{label}: answer snapshot differs from committed groups')
    counts = common.totals(rows)
    require(summary['capabilities'] == entry['capabilities'] == counts, f'{label}: strict totals mismatch')
    tokens = sum(row['total_output_tokens_including_retries'] for row in rows)
    wall = sum(group['wall_seconds'] for group in groups)
    observed = sum(row['thinking_observed'] for row in rows)
    require(summary['total_output_tokens_including_retries'] == tokens
            and summary['sum_committed_group_wall_seconds'] == wall
            and summary['thinking_observed_rows'] == observed
            and summary['final_attempt_thinking_characters'] == sum(row['thinking_characters'] for row in rows),
            f'{label}: cached usage/time/thinking totals mismatch')
    archives = sorted(p.name for p in directory.glob('incomplete-*.json'))
    pending = sorted(p.name for p in directory.glob('pending-*.json'))
    require(summary['archived_incomplete_attempts'] == archives, f'{label}: incomplete-attempt list changed')
    unknown_usage = sum(a['output_tokens'] is None for row in rows for a in row['attempts'])
    result = {'label': label, 'role': 'reference' if label == 'baseline' else 'candidate',
              'completed': True, 'model_id': config['model_id'], 'correct': counts['correct'], 'total': 48,
              'score_percent': 100 * counts['correct'] / 48, 'missing_observations': 0,
              'incomplete_outputs': counts['unresolved'], 'final_truncated_outputs': counts['unresolved_truncated'],
              'retried_observations': counts['retried'],
              'observations_with_any_truncated_attempt': sum(any(a['stop_reason'] == 'max_tokens' for a in row['attempts']) for row in rows),
              'thinking_requested_observations': 48, 'thinking_observed_final_attempts': observed,
              'missing_final_text_outputs': sum('missing_final_text' in row['unresolved_reasons'] for row in rows),
              'unparsed_reasoning_marker_outputs': sum('unparsed_reasoning_markers_in_final_text' in row['unresolved_reasons'] for row in rows),
              'archived_incomplete_group_attempts': len(archives), 'pending_group_attempts': len(pending),
              'archived_incomplete_files': archives, 'pending_files': pending,
              'by_category': counts['by_category'], 'output_tokens_including_retries': tokens,
              'committed_group_wall_seconds': wall, 'unknown_attempt_token_counts': unknown_usage,
              'output_tokens_per_committed_wall_second': tokens / wall if not unknown_usage else None,
              'run_summary': str(summary_path)}
    return result, rows, config


def analyze(batch_path):
    common = load_common()
    inputs = {str(WORK / 'analyze_integrity.py'): {'sha256': COMMON_SHA256,
              'bytes': (WORK / 'analyze_integrity.py').stat().st_size}}
    batch = common.read(batch_path, inputs)
    common.require(batch['status'] == 'completed'
                   and [entry['label'] for entry in batch['comparisons']] == list(common.LABELS),
                   'Only the completed four-model thinking24 batch can be ranked')
    cases, grade = common.frozen_inputs(inputs)
    validator, constants = load_validator(common, inputs)
    results, rows, configs = {}, {}, {}
    for entry in batch['comparisons']:
        label = entry['label']
        results[label], rows[label], configs[label] = load_run(
            entry, batch, cases, grade, validator, constants, common, inputs)
    for label in common.LABELS[1:]:
        for field in ('protocol', 'checkpoint', 'launch_profile', 'precision'):
            common.require(configs[label][field] == configs['baseline'][field],
                           f'{label}: unmatched {field} versus the thinking baseline')
    pairs = []
    for label in common.LABELS:
        paired = common.paired(rows['baseline'], rows[label])
        common.compare(results[label], results['baseline'], paired)
        pairs.extend({'label': label, **row} for row in paired)
    ranking = sorted(results.values(), key=lambda row: -row['correct'])
    for row in ranking:
        row['development_rank'] = 1 + sum(other['correct'] > row['correct'] for other in ranking)
    for path, identity in inputs.items():
        common.require(common.digest(Path(path).read_bytes()) == identity['sha256'], f'Input changed during analysis: {path}')
    return {'schema_version': 1, 'status': 'completed_development_analysis', 'thinking_enabled': True,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': common.SCOPE,
            'quality_reference': 'Untouched W8A16 baseline with native thinking enabled; BF16 source is conversion input only.',
            'refusal_evaluation_performed': False, 'full_quality_refusal_acceptance_evaluated': False,
            'pairing': {'unique_cases': 24, 'repeats': [1, 2], 'observations_per_model': 48,
                        'models': 4, 'total_observations': 192, 'keys': ['case_id', 'repeat'],
                        'schedule': configs['baseline']['protocol']['fixed_group_schedule']},
            'thinking_protocol': {'temperature': 0, 'chat_template_kwargs': {'enable_thinking': True},
                                  'planned_total_token_budgets': [4096, 6144], 'final_token_reserve': 1024,
                                  'minimum_thinking_tokens': 1024, 'native_reasoning_parser': 'gemma4',
                                  'final_only_scoring': True, 'runner_sha256': RUNNER_SHA256},
            'ranking_rule': 'Completed thinking-enabled development strict correct/48 descending; equal scores share rank. Separate from thinking-off scores.',
            'retention_rule': '100 * candidate correct / thinking-enabled baseline correct; uncapped, null when baseline correct is zero. Paired losses and gains are separate.',
            'throughput_rule': 'Native output tokens include reasoning plus final text, including retries, divided by committed group wall time. Separate reasoning/final token counts are unavailable. Excludes loading, between-group gaps and abandoned attempts; summed request latency is not the denominator.',
            'validation_scope': 'Saved configuration, pinned request guard and native response captures are checked. No live inference or independent kernel-precision attestation is performed.',
            'ranking': ranking, 'paired_observations': pairs, 'inputs': inputs,
            'analyzer_sha256': common.digest(Path(__file__).read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-result', type=Path, default=WORK / 'thinking-batch-result-v1.json')
    args = parser.parse_args()
    if not args.batch_result.is_file():
        print(json.dumps({'status': 'not_ready', 'reason': 'Completed thinking batch is absent; no files changed.'}))
        return 2
    report = analyze(args.batch_result)
    common = load_common()
    paths = {'json': WORK / 'thinking-analysis-v1.json', 'ranking_csv': WORK / 'thinking-ranking-v1.csv',
             'categories_csv': WORK / 'thinking-categories-v1.csv', 'pairs_csv': WORK / 'thinking-pairs-v1.csv'}
    fields = ('development_rank', 'label', 'role', 'correct', 'total', 'score_percent',
              'relative_score_retention_percent', 'baseline_losses', 'baseline_gains',
              'baseline_correct_retained', 'net_correct_change', 'incomplete_outputs', 'final_truncated_outputs',
              'missing_final_text_outputs', 'unparsed_reasoning_marker_outputs', 'thinking_observed_final_attempts',
              'retried_observations', 'missing_observations', 'archived_incomplete_group_attempts',
              'pending_group_attempts', 'output_tokens_including_retries', 'committed_group_wall_seconds',
              'output_tokens_per_committed_wall_second', 'unknown_attempt_token_counts')
    common.atomic_text(paths['ranking_csv'], common.csv_text(report['ranking'], fields))
    categories = [{'label': row['label'], 'category': category, **row['by_category'][category]}
                  for row in report['ranking'] for category in common.CATEGORIES]
    common.atomic_text(paths['categories_csv'], common.csv_text(categories,
                       ('label', 'category', 'correct', 'total', 'baseline_correct', 'baseline_losses',
                        'baseline_gains', 'relative_score_retention_percent', 'unresolved', 'unresolved_truncated')))
    common.atomic_text(paths['pairs_csv'], common.csv_text(report['paired_observations'],
                       ('label', 'case_id', 'repeat', 'category', 'baseline_correct', 'candidate_correct',
                        'outcome', 'baseline_complete', 'candidate_complete', 'baseline_truncated', 'candidate_truncated')))
    report['csv_artifacts'] = {name: {'path': str(path), 'sha256': common.digest(path.read_bytes())}
                               for name, path in paths.items() if name != 'json'}
    common.atomic_text(paths['json'], json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'artifacts': {key: str(path) for key, path in paths.items()},
                      'ranking': [{key: row[key] for key in fields[:10]} for row in report['ranking']]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
