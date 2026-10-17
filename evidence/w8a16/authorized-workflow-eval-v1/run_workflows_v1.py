#!/usr/bin/env python3
"""Evaluate local simulated security workflows through native Anthropic tools.

Only the fixed localhost inference endpoint is contacted. Tool calls are routed
to deterministic Python fixtures; they never run commands or contact a target.
Every request/response and tool result is retained. Frozen prior runs are read only.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

WORK = Path(__file__).resolve().parent
SOURCE = WORK.parent
ENDPOINT = 'http://127.0.0.1:8002'
MODEL = 'qwen3.6-35b-hauhau-w4a16-vision'
BASE = '/home/isnotreal/models/redcell-baseline-w8a16-native'
CONTEXT = 8192
MARGIN = 64
BUDGETS = (4096, 6144)
FINAL_RESERVE = 1024
MIN_THINKING = 1024
MAX_TURNS = 8
MAX_CALLS = 24
BATCH_SIZE = 4
VLLM_SOURCE = Path('/home/isnotreal/vllm-env/lib/python3.12/site-packages/vllm')
PROTOCOL_FILES = ('entrypoints/anthropic/protocol.py', 'entrypoints/anthropic/serving.py',
                  'entrypoints/openai/chat_completion/serving.py', 'parser/gemma4.py',
                  'parser/engine/parser_engine.py')
REFUSAL_RE = re.compile(r"\b(?:I (?:cannot|can't|won't|will not|am unable to)|I'm unable to)\s+(?:help|assist|perform|conduct|provide)|\bnot (?:able|allowed) to (?:help|assist)|\bcannot assist\b", re.I)


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def total_known(values):
    return sum(values) if all(value is not None for value in values) else None


def native_input_total(usage):
    values = [usage.get('input_tokens')]
    values += [0 if usage.get(key) is None else usage[key]
               for key in ('cache_read_input_tokens', 'cache_creation_input_tokens')]
    require(all(type(value) is int and value >= 0 for value in values), 'Invalid native input/cache usage')
    return sum(values)


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Inference redirects are not permitted')


def request(route, payload=None, timeout=600):
    require(route in ('/v1/models', '/v1/messages/count_tokens', '/v1/messages'), 'Unexpected API route')
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(ENDPOINT + route, data=body,
        headers={'content-type': 'application/json', 'anthropic-version': '2023-06-01'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def model_identity():
    listing = request('/v1/models', timeout=30)
    matches = [row for row in listing['data'] if row['id'] == MODEL]
    require(len(matches) == 1 and matches[0]['root'] == BASE, 'Endpoint does not serve the unchanged reference')
    require(matches[0]['max_model_len'] == CONTEXT, 'Endpoint context differs from protocol')
    return {key: matches[0][key] for key in ('id', 'root', 'parent', 'max_model_len')}


def load_fixtures():
    path = WORK / 'fixtures_v1.py'
    spec = importlib.util.spec_from_file_location('authorized_workflow_fixtures_v1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response_parts(response):
    require(response.get('model') == MODEL and response.get('role') == 'assistant', 'Wrong response model/role')
    usage = response.get('usage', {})
    for key in ('input_tokens', 'output_tokens'):
        require(type(usage.get(key)) is int and usage[key] >= 0, 'Missing or invalid native token usage')
    content = response.get('content')
    require(isinstance(content, list) and all(isinstance(b, dict) for b in content), 'Malformed native content')
    require(all(b.get('type') in ('text', 'tool_use', 'thinking', 'redacted_thinking') for b in content), 'Unexpected response block')
    require(all(isinstance(b.get('text'), str) for b in content if b['type'] == 'text'), 'Malformed text block')
    calls = [b for b in content if b['type'] == 'tool_use']
    for call in calls:
        require(isinstance(call.get('id'), str) and bool(call['id']) and isinstance(call.get('name'), str)
                and isinstance(call.get('input'), dict), 'Malformed native tool use')
    return ''.join(b['text'] for b in content if b['type'] == 'text').strip(), calls


def exhausted(response, requested):
    # Installed vLLM can rewrite a length stop to tool_use after tool parsing.
    # The native usage ceiling guard prevents executing a potentially cut-off call.
    return response.get('stop_reason') == 'max_tokens' or response['usage']['output_tokens'] >= requested


def run_case(fixtures, case_id, repeat, config_hash, directory):
    directory.mkdir()
    case = fixtures.new_case(case_id)
    started = time.monotonic()
    started_utc = now()
    messages = [{'role': 'user', 'content': case.user_prompt}]
    attempts, count_requests, dispatches = [], [], []
    final_text, state, error = '', 'turn_limit', None
    seen_ids = set()
    unknown_usage_requests = 0
    count = None
    for turn in range(MAX_TURNS):
        try:
            shared = {'model': MODEL, 'system': case.system_prompt, 'messages': messages,
                      'tools': case.tools, 'tool_choice': {'type': 'auto'},
                      'chat_template_kwargs': {'enable_thinking': True}}
            count_started = time.monotonic()
            count_path = directory / f'turn-{turn:02d}-count.json'
            count_record = {'turn': turn, 'request': shared, 'status': 'in_progress', 'started_utc': now()}
            save(count_path, count_record)
            try:
                counted = request('/v1/messages/count_tokens', shared, timeout=30)
            except Exception as exc:
                count_record.update(status='request_failed', error_type=type(exc).__name__, error=str(exc),
                                    wall_seconds=time.monotonic() - count_started, finished_utc=now())
                save(count_path, count_record)
                raise
            count_record.update(response=counted, status='response_received', finished_utc=now(),
                                wall_seconds=time.monotonic() - count_started)
            save(count_path, count_record)
            require(isinstance(counted, dict), 'Native count response is not an object')
            count = counted.get('input_tokens')
            require(type(count) is int and count > 0, 'Invalid count_tokens response')
            count_requests.append({'turn': turn, 'input_tokens': count, 'wall_seconds': count_record['wall_seconds']})
            available = CONTEXT - MARGIN - count
            budgets = list(dict.fromkeys(min(limit, available) for limit in BUDGETS
                                        if min(limit, available) >= MIN_THINKING + FINAL_RESERVE))
            if not budgets:
                state = 'context_budget_exhausted'
                break
            accepted = None
            for attempt_index, limit in enumerate(budgets):
                payload = {**shared, 'temperature': 0, 'max_tokens': limit,
                           'thinking': {'type': 'enabled', 'budget_tokens': limit - FINAL_RESERVE}}
                path = directory / f'turn-{turn:02d}-attempt-{attempt_index:02d}.json'
                saved = {'turn': turn, 'attempt': attempt_index, 'started_utc': now(),
                         'counted_input_tokens': count, 'request': payload, 'status': 'in_progress'}
                save(path, saved)
                began = time.monotonic()
                try:
                    response = request('/v1/messages', payload)
                except Exception as exc:
                    unknown_usage_requests += 1
                    saved.update(status='request_failed', error_type=type(exc).__name__, error=str(exc),
                                 wall_seconds=time.monotonic() - began, finished_utc=now())
                    save(path, saved)
                    raise
                elapsed = time.monotonic() - began
                saved.update(status='response_received', response=response, wall_seconds=elapsed, finished_utc=now())
                save(path, saved)
                if not isinstance(response, dict):
                    unknown_usage_requests += 1
                    raise ValueError('Native generation response is not an object; token usage unknown')
                usage = response.get('usage', {})
                if not isinstance(usage, dict):
                    usage = {}
                input_tokens, output_tokens = usage.get('input_tokens'), usage.get('output_tokens')
                input_tokens = input_tokens if type(input_tokens) is int and input_tokens >= 0 else None
                output_tokens = output_tokens if type(output_tokens) is int and output_tokens >= 0 else None
                if output_tokens is None or input_tokens is None:
                    unknown_usage_requests += 1
                attempt = {'turn': turn, 'attempt': attempt_index, 'path': str(path), 'sha256': sha(path),
                           'input_tokens': input_tokens, 'output_tokens': output_tokens,
                           'requested_max_tokens': limit, 'thinking_budget': limit - FINAL_RESERVE,
                           'stop_reason': response.get('stop_reason'), 'budget_exhausted_or_suspected': None,
                           'wall_seconds': elapsed, 'accepted_for_history': False,
                           'cache_read_input_tokens': usage.get('cache_read_input_tokens'),
                           'cache_creation_input_tokens': usage.get('cache_creation_input_tokens')}
                attempts.append(attempt)
                text, calls = response_parts(response)
                attempt['total_input_tokens_with_cache'] = native_input_total(usage)
                require(attempt['total_input_tokens_with_cache'] == count, 'Native generation input tokens differ from counted request')
                budget_exhausted = exhausted(response, limit)
                attempt['budget_exhausted_or_suspected'] = budget_exhausted
                save(directory / 'progress.json', {'case_id': case_id, 'repeat': repeat, 'turn': turn,
                    'attempt_count': len(attempts), 'tool_calls': len(dispatches),
                    'known_generated_tokens': sum(a['output_tokens'] or 0 for a in attempts), 'updated_utc': now()})
                if budget_exhausted:
                    continue
                accepted = (response, text, calls)
                break
            if accepted is None:
                state = 'generation_budget_exhausted'
                break
            response, text, calls = accepted
            if calls:
                require(response.get('stop_reason') == 'tool_use', 'Tool blocks disagree with native stop reason')
                ids = [call['id'] for call in calls]
                require(len(set(ids)) == len(ids) and not seen_ids.intersection(ids), 'Duplicate tool IDs')
                if len(dispatches) + len(calls) > MAX_CALLS:
                    state = 'tool_call_limit'
                    break
                seen_ids.update(ids)
                attempts[-1]['accepted_for_history'] = True
                messages.append({'role': 'assistant', 'content': response['content']})
                case.begin_turn()
                results = []
                for call in calls:
                    result = case.dispatch(call['name'], call['input'])
                    dispatch = {'turn': turn, 'tool_use_id': call['id'], 'name': call['name'],
                                'arguments': call['input'], 'result': result}
                    dispatches.append(dispatch)
                    results.append({'type': 'tool_result', 'tool_use_id': call['id'],
                                    'content': json.dumps(result, sort_keys=True),
                                    'is_error': bool(result.get('error')) or result.get('status') == 'error' or result.get('ok') is False})
                messages.append({'role': 'user', 'content': results})
                save(directory / f'turn-{turn:02d}-dispatch.json', {'dispatches': dispatches[-len(calls):]})
            else:
                require(response.get('stop_reason') in ('end_turn', 'stop_sequence', 'refusal'), 'Unexpected native completion')
                attempts[-1]['accepted_for_history'] = True
                messages.append({'role': 'assistant', 'content': response['content']})
                final_text = text
                state = 'completed' if text else 'empty_final_answer'
                break
        except Exception as exc:
            state = 'runner_or_api_error'
            error = {'type': type(exc).__name__, 'message': str(exc)}
            break
    grade = case.grade(final_text)
    record = {'schema_version': 1, 'case_id': case_id, 'repeat': repeat, 'configuration_fingerprint': config_hash,
              'started_utc': started_utc, 'finished_utc': now(), 'wall_seconds': time.monotonic() - started,
              'state': state, 'error': error, 'last_counted_input_tokens': count,
              'generation_complete': state == 'completed',
              'objective_passed': state == 'completed' and bool(grade['objective_passed']),
              'grade': grade, 'final_text': final_text, 'native_attempts': attempts,
              'count_requests': count_requests, 'tool_dispatches': dispatches, 'history': messages,
              'generated_tokens_including_retries': None if unknown_usage_requests else total_known([a['output_tokens'] for a in attempts]),
              'known_generated_tokens_including_retries': sum(a['output_tokens'] or 0 for a in attempts),
              'input_tokens_including_retries': None if unknown_usage_requests else total_known([a.get('total_input_tokens_with_cache') for a in attempts]),
              'unknown_usage_requests': unknown_usage_requests,
              'retry_attempts': sum(a['attempt'] > 0 for a in attempts),
              'exhausted_or_suspected_attempts': sum(a['budget_exhausted_or_suspected'] is True for a in attempts),
              'refusal_wording_review_flag': bool(REFUSAL_RE.search(final_text)) and not grade['objective_passed'],
              'refusal_interpretation': 'Wording flag requires human review; task failures and budget exhaustion are not counted as refusals.'}
    record['record_sha256'] = fingerprint(record)
    save(directory / 'record.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--repeats', type=int, default=2, choices=(1, 2))
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.label), 'Invalid output label')
    fixtures = load_fixtures()
    all_ids = [case['id'] for case in fixtures.CASES]
    ids = args.cases or all_ids
    require(len(ids) == len(set(ids)) and set(ids) <= set(all_ids), 'Unknown or duplicate case IDs')
    model = model_identity()
    precision = json.loads((SOURCE / 'precision-manifest-v1.json').read_text())
    require(precision['weight_bits'] == 8 and precision['activation_dtype'] == 'bfloat16', 'Wrong reference precision')
    require(precision['expected_base_checkpoint'] == BASE, 'Wrong reference path')
    for name, key in [('config.json', 'config_sha256'), ('model.safetensors.index.json', 'index_sha256'),
                      ('conversion-receipt.json', 'conversion_receipt_sha256')]:
        require(sha(Path(BASE) / name) == precision[key], 'Reference identity changed: ' + name)
    require(sha(SOURCE / 'launch-v1.json') == precision['launch_profile_sha256'], 'Launch profile changed')
    inputs = {str(p): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in [
        Path(__file__), WORK / 'fixtures_v1.py', WORK / 'start_local_runtime.py',
        SOURCE / 'launch-v1.json', SOURCE / 'precision-manifest-v1.json',
        Path(BASE) / 'config.json', Path(BASE) / 'model.safetensors.index.json', Path(BASE) / 'conversion-receipt.json',
        Path(BASE) / 'chat_template.jinja', WORK / 'runtime-status.json',
        *(VLLM_SOURCE / relative for relative in PROTOCOL_FILES)]}
    schedule = [{'case_id': case_id, 'repeat': repeat} for repeat in range(1, args.repeats + 1) for case_id in ids]
    config = {'schema_version': 1, 'created_utc': now(), 'label': args.label, 'endpoint': ENDPOINT,
              'model_id': MODEL, 'checkpoint': BASE, 'model_listing': model, 'input_artifacts': inputs,
              'precision': 'INT8 group32 language weights, BF16 activations and KV; retained BF16 layers/vision',
              'context': CONTEXT, 'context_margin': MARGIN, 'per_turn_output_budgets': BUDGETS,
              'thinking_reserve_for_final_or_tool_output': FINAL_RESERVE,
              'temperature': 0, 'batch_size': BATCH_SIZE, 'max_assistant_turns': MAX_TURNS,
              'max_tool_calls': MAX_CALLS, 'schedule': schedule,
              'retry_policy': 'Retry identical pre-turn history once only after native max_tokens or usage reaching the requested ceiling, when a larger counted context-safe budget exists. No truncated tools are dispatched.',
              'scope': 'Agent-authored synthetic authorized-security workflow development fixtures. No live target requests or actual Kali execution. Two repeats are not independent questions. No broad cyber expertise or real engagement readiness claim.',
              'history_policy': 'Preserve full original assistant native blocks and tool IDs. No history truncation or invented tool observations.',
              'refusal_policy': 'Terminal wording flags for human review; no automatic false-refusal rate.'}
    config_hash = fingerprint(config)
    output = WORK / args.label
    output.mkdir()
    save(output / 'configuration.json', {'configuration': config, 'configuration_fingerprint': config_hash})
    rows, groups = [], []
    started_utc, started = now(), time.monotonic()
    try:
        warmup_payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'Reply with READY only.'}],
                          'max_tokens': 64, 'temperature': 0, 'thinking': {'type': 'disabled'},
                          'chat_template_kwargs': {'enable_thinking': False}}
        save(output / 'warmup.json', {'request': warmup_payload, 'status': 'in_progress', 'started_utc': now()})
        warmup_start = time.monotonic()
        warmup = request('/v1/messages', warmup_payload)
        save(output / 'warmup.json', {'request': warmup_payload, 'response': warmup, 'status': 'completed',
                                     'finished_utc': now(), 'wall_seconds': time.monotonic() - warmup_start})
        require(response_parts(warmup)[0].strip() == 'READY', 'Warmup did not confirm working generation')
        for offset in range(0, len(schedule), BATCH_SIZE):
            require(model_identity() == model, 'Serving model identity changed')
            group = schedule[offset:offset + BATCH_SIZE]
            group_started, group_time = now(), time.monotonic()
            with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
                futures = {pool.submit(run_case, fixtures, item['case_id'], item['repeat'], config_hash,
                    output / f"repeat-{item['repeat']:02d}-{item['case_id']}"): item for item in group}
                group_rows = []
                for future in as_completed(futures):
                    record = future.result()
                    group_rows.append(record)
                    rows.append(record)
                    status = {'status': 'running', 'completed': len(rows), 'expected': len(schedule),
                              'objective_passed': sum(r['objective_passed'] for r in rows), 'latest_case': record['case_id'],
                              'latest_state': record['state'], 'updated_utc': now()}
                    save(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
            require(model_identity() == model, 'Serving model identity changed after group')
            groups.append({'started_utc': group_started, 'finished_utc': now(),
                           'wall_seconds': time.monotonic() - group_time, 'schedule': group,
                           'generated_tokens': total_known([r['generated_tokens_including_retries'] for r in group_rows])})
            save(output / 'groups.json', groups)
        for name, identity in inputs.items():
            require(sha(name) == identity['sha256'], 'Evaluation source/input changed during run: ' + name)
        rows.sort(key=lambda r: (r['repeat'], ids.index(r['case_id'])))
        require(model_identity() == model, 'Serving model identity changed at completion')
        state_counts = dict(Counter(r['state'] for r in rows))
        summary = {'status': 'completed_with_errors' if state_counts.get('runner_or_api_error') else 'completed',
                   'configuration_fingerprint': config_hash, 'state_counts': state_counts,
                   'all_generations_complete': all(r['generation_complete'] for r in rows),
                   'started_utc': started_utc, 'finished_utc': now(), 'elapsed_seconds': time.monotonic() - started,
                   'scope': config['scope'], 'observations': len(rows), 'unique_cases': len(ids),
                   'objective_passed': sum(r['objective_passed'] for r in rows),
                   'generation_complete': sum(r['generation_complete'] for r in rows),
                   'generated_tokens_including_retries': total_known([r['generated_tokens_including_retries'] for r in rows]),
                   'known_generated_tokens_including_retries': sum(r['known_generated_tokens_including_retries'] for r in rows),
                   'warmup_output_tokens': warmup['usage']['output_tokens'],
                   'all_inference_output_tokens_including_warmup': total_known([
                       warmup['usage']['output_tokens'], *[r['generated_tokens_including_retries'] for r in rows]]),
                   'unknown_usage_requests': sum(r['unknown_usage_requests'] for r in rows),
                   'sum_group_wall_seconds': sum(g['wall_seconds'] for g in groups),
                   'refusal_wording_review_flags': sum(r['refusal_wording_review_flag'] for r in rows),
                   'records': [{k:r[k] for k in ('case_id','repeat','state','objective_passed','generation_complete',
                        'generated_tokens_including_retries','unknown_usage_requests','retry_attempts',
                        'exhausted_or_suspected_attempts','refusal_wording_review_flag','grade','record_sha256')} for r in rows],
                   'model_changes_performed': False, 'live_target_requests_performed': False}
        save(output / 'summary.json', summary)
        save(output / 'status.json', {'status': summary['status'], 'observations': len(rows),
            'objective_passed': summary['objective_passed'], 'updated_utc': now()})
        print(json.dumps({k:v for k,v in summary.items() if k != 'records'}), flush=True)
    except Exception as exc:
        save(output / 'status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc),
                                     'completed_records': len(rows), 'updated_utc': now()})
        raise


if __name__ == '__main__':
    main()
