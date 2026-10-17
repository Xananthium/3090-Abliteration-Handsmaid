#!/usr/bin/env python3
"""Verify and report committed synthetic workflow evidence; never invoke inference."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
WORK = Path(__file__).resolve().parent
STATES = {'completed', 'empty_final_answer', 'turn_limit', 'context_budget_exhausted',
          'generation_budget_exhausted', 'tool_call_limit', 'runner_or_api_error'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def known_sum(values):
    return sum(values) if all(v is not None for v in values) else None


def token(value):
    return value if type(value) is int and value >= 0 else None


def seconds(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid elapsed time')
    return value


def equal(actual, expected, name):
    require(actual == expected, f'Mismatch: {name}')


def close(actual, expected, name):
    require(math.isclose(seconds(actual), seconds(expected), rel_tol=1e-10, abs_tol=1e-8), f'Mismatch: {name}')


class Evidence:
    def __init__(self):
        self.pins = {}

    def pin(self, path):
        path = Path(path).resolve()
        require(path.is_file(), f'Missing evidence: {path}')
        identity = {'sha256': sha(path), 'bytes': path.stat().st_size}
        if str(path) in self.pins:
            equal(identity, self.pins[str(path)], f'evidence changed while reading {path.name}')
        self.pins[str(path)] = identity
        return path

    def read(self, path):
        path = self.pin(path)
        def bad_constant(value):
            raise ValueError(f'Nonfinite JSON constant: {value}')
        return json.loads(path.read_text(), parse_constant=bad_constant)

    def verify_unchanged(self):
        for name, identity in self.pins.items():
            equal({'sha256': sha(name), 'bytes': Path(name).stat().st_size}, identity, f'changed evidence {name}')


def native_total(usage):
    values = [usage.get('input_tokens'), usage.get('cache_read_input_tokens') or 0,
              usage.get('cache_creation_input_tokens') or 0]
    return sum(values) if all(token(v) is not None for v in values) else None


def parts(response, model):
    equal(response.get('model'), model, 'accepted response model')
    equal(response.get('role'), 'assistant', 'accepted response role')
    blocks = response.get('content')
    require(isinstance(blocks, list) and all(isinstance(b, dict) for b in blocks), 'Invalid native content')
    require(all(b.get('type') in ('text', 'tool_use', 'thinking', 'redacted_thinking') for b in blocks), 'Unknown native block')
    texts = [b['text'] for b in blocks if b['type'] == 'text']
    require(all(isinstance(t, str) for t in texts), 'Invalid native text')
    calls = [b for b in blocks if b['type'] == 'tool_use']
    require(all(isinstance(c.get('id'), str) and c['id'] and isinstance(c.get('name'), str)
                and isinstance(c.get('input'), dict) for c in calls), 'Invalid native tool block')
    return ''.join(texts).strip(), calls


def load_fixtures(config, evidence):
    source_pins = config['input_artifacts']
    require(isinstance(source_pins, dict) and source_pins, 'Missing source pins')
    for name, identity in source_pins.items():
        evidence.pin(name)
        equal(evidence.pins[str(Path(name).resolve())], identity, f'source pin {Path(name).name}')
    fixture_path = WORK / 'fixtures_v1.py'
    runner_path = WORK / 'run_workflows_v1.py'
    require(str(fixture_path) in source_pins and str(runner_path) in source_pins, 'Fixture/runner source not pinned')
    spec = importlib.util.spec_from_file_location('verified_workflow_fixtures_report', fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    # Extract only the runner's literal refusal-wording pattern and minimum budget.
    tree = ast.parse(runner_path.read_text())
    constants = {node.targets[0].id: node.value for node in tree.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    expression = constants['REFUSAL_RE']
    require(isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute)
            and isinstance(expression.func.value, ast.Name) and expression.func.value.id == 're'
            and expression.func.attr == 'compile', 'Unexpected refusal wording pattern source')
    flag = expression.args[1]
    require(isinstance(flag, ast.Attribute) and isinstance(flag.value, ast.Name)
            and flag.value.id == 're' and flag.attr == 'I', 'Unexpected wording regex flags')
    wording = re.compile(ast.literal_eval(expression.args[0]), re.I)
    return fixture, wording, ast.literal_eval(constants['MIN_THINKING'])


def validate_record(run, item, config, config_hash, fixtures, wording, minimum_thinking, evidence):
    directory = run / f"repeat-{item['repeat']:02d}-{item['case_id']}"
    record = evidence.read(directory / 'record.json')
    digest = record['record_sha256']
    equal(fingerprint({k:v for k,v in record.items() if k != 'record_sha256'}), digest, 'record fingerprint')
    equal(record['case_id'], item['case_id'], 'case identity')
    equal(record['repeat'], item['repeat'], 'repeat identity')
    equal(record['configuration_fingerprint'], config_hash, 'record configuration')
    require(record['state'] in STATES, 'Unknown terminal state')
    seconds(record['wall_seconds'])
    case = fixtures.new_case(item['case_id'])
    require(callable(getattr(case, 'begin_turn', None)), 'Fixture lacks evidence-visibility boundary')
    history = [{'role':'user','content':case.user_prompt}]
    expected_dispatches, counts, accepted_text = [], [], ''
    seen_ids = set()
    attempts = record['native_attempts']
    attempts_by_key = {(a['turn'], a['attempt']): a for a in attempts}
    require(len(attempts_by_key) == len(attempts), 'Duplicate returned attempt identity')
    all_raw = {}
    failed_generation_requests = 0
    invalid_usage_responses = 0
    terminal_seen = False
    for path in sorted(directory.glob('turn-*-attempt-*.json')):
        raw = evidence.read(path)
        key = (raw['turn'], raw['attempt'])
        require(key not in all_raw, 'Duplicate raw attempt')
        equal(path.name, f'turn-{key[0]:02d}-attempt-{key[1]:02d}.json', 'raw attempt filename')
        seconds(raw['wall_seconds'])
        all_raw[key] = (path, raw)
        require(raw['status'] in ('response_received','request_failed'), 'Uncommitted raw generation')
        if raw['status'] == 'request_failed':
            require(key not in attempts_by_key, 'Failed generation incorrectly listed as returned')
            failed_generation_requests += 1
            continue
        require(key in attempts_by_key, 'Unaccounted generation response')
        saved = attempts_by_key[key]
        equal(Path(saved['path']).resolve(), path.resolve(), 'raw generation path')
        equal(saved['sha256'], sha(path), 'raw generation SHA256')
        response = raw['response']; usage = response.get('usage', {})
        usage = usage if isinstance(usage, dict) else {}
        inputs, outputs = token(usage.get('input_tokens')), token(usage.get('output_tokens'))
        invalid_usage_responses += int(inputs is None or outputs is None)
        equal(saved['input_tokens'], inputs, 'attempt input usage')
        equal(saved['output_tokens'], outputs, 'attempt output usage')
        for name in ('cache_read_input_tokens','cache_creation_input_tokens'):
            equal(saved[name], usage.get(name), f'attempt {name}')
        equal(saved['stop_reason'], response.get('stop_reason'), 'attempt stop reason')
        close(saved['wall_seconds'], raw['wall_seconds'], 'attempt duration')
        equal(saved['requested_max_tokens'], raw['request']['max_tokens'], 'requested max tokens')
        equal(saved['thinking_budget'], raw['request']['thinking']['budget_tokens'], 'thinking budget')
        if 'total_input_tokens_with_cache' in saved:
            equal(saved['total_input_tokens_with_cache'], native_total(usage), 'input plus cache usage')
            equal(saved['total_input_tokens_with_cache'], raw['counted_input_tokens'], 'count versus native input')
        exhausted = saved['budget_exhausted_or_suspected']
        if exhausted is not None:
            require(outputs is not None, 'Exhaustion flag without usage')
            equal(exhausted, response.get('stop_reason') == 'max_tokens' or outputs >= saved['requested_max_tokens'], 'exhaustion guard')
        if saved['accepted_for_history']:
            require(exhausted is False and inputs is not None and outputs is not None, 'Truncated or invalid response accepted')
        elif exhausted is None:
            equal(record['state'], 'runner_or_api_error', 'unvalidated response terminal state')
    equal(set(attempts_by_key), {key for key,(_,raw) in all_raw.items() if raw['status']=='response_received'}, 'returned generation coverage')
    count_paths = sorted(directory.glob('turn-*-count.json'))
    require(len(count_paths) <= config['max_assistant_turns'], 'Turn limit exceeded')
    encountered_attempts = set()
    for turn, path in enumerate(count_paths):
        require(not terminal_seen, 'New turn after terminal response')
        counted = evidence.read(path)
        equal(counted['turn'], turn, 'count turn identity')
        equal(path.name, f'turn-{turn:02d}-count.json', 'count filename')
        seconds(counted['wall_seconds'])
        shared = {'model':config['model_id'],'system':case.system_prompt,'messages':history,
                  'tools':case.tools,'tool_choice':{'type':'auto'},'chat_template_kwargs':{'enable_thinking':True}}
        equal(counted['request'], shared, 'counted request/history')
        raw_turn = sorted((key, value) for key,value in all_raw.items() if key[0] == turn)
        if counted['status'] != 'response_received':
            equal(counted['status'], 'request_failed', 'count request state')
            equal(record['state'], 'runner_or_api_error', 'failed count terminal state')
            require(not raw_turn, 'Generation after failed token count')
            terminal_seen = True
            continue
        count = token(counted['response'].get('input_tokens'))
        if not count:
            equal(record['state'], 'runner_or_api_error', 'invalid count terminal state')
            require(not raw_turn, 'Generation with invalid count')
            terminal_seen = True
            continue
        counts.append({'turn':turn,'input_tokens':count,'wall_seconds':counted['wall_seconds']})
        available = config['context']-config['context_margin']-count
        budgets = list(dict.fromkeys(min(limit,available) for limit in config['per_turn_output_budgets']
                                    if min(limit,available)>=minimum_thinking+config['thinking_reserve_for_final_or_tool_output']))
        require(len(raw_turn)<=len(budgets), 'Too many generation attempts for counted context')
        accepted = None
        for index,(key,(raw_path,raw)) in enumerate(raw_turn):
            equal(key, (turn,index), 'attempt sequence')
            encountered_attempts.add(key)
            payload = {**shared,'temperature':config['temperature'],'max_tokens':budgets[index],
                       'thinking':{'type':'enabled','budget_tokens':budgets[index]-config['thinking_reserve_for_final_or_tool_output']}}
            equal(raw['request'],payload,'generation payload/native thinking/history')
            equal(raw['counted_input_tokens'],count,'attempt counted input')
            if index:
                prior = attempts_by_key[(turn,index-1)]
                require(prior['budget_exhausted_or_suspected'] is True and budgets[index]>budgets[index-1], 'Invalid retry policy')
            if raw['status']=='request_failed':
                equal(record['state'],'runner_or_api_error','failed generation terminal state')
                equal(index,len(raw_turn)-1,'requests after failure')
                terminal_seen=True
                continue
            saved=attempts_by_key[key]
            if saved['accepted_for_history']:
                require(accepted is None and index==len(raw_turn)-1,'Multiple accepted attempts in one turn')
                accepted=raw['response']
        if accepted is None:
            require(turn==len(count_paths)-1,'Unaccepted turn followed by more history')
            terminal_seen=True
            continue
        text,calls=parts(accepted,config['model_id'])
        history.append({'role':'assistant','content':accepted['content']})
        if calls:
            equal(accepted.get('stop_reason'),'tool_use','tool stop reason')
            ids=[c['id'] for c in calls]
            require(len(set(ids))==len(ids) and not seen_ids.intersection(ids),'Duplicate tool IDs')
            seen_ids.update(ids)
            require(len(expected_dispatches)+len(calls)<=config['max_tool_calls'],'Tool call cap exceeded')
            case.begin_turn()
            tool_results=[];batch=[]
            for call in calls:
                result=case.dispatch(call['name'],call['input'])
                dispatch={'turn':turn,'tool_use_id':call['id'],'name':call['name'],'arguments':call['input'],'result':result}
                expected_dispatches.append(dispatch);batch.append(dispatch)
                tool_results.append({'type':'tool_result','tool_use_id':call['id'],'content':json.dumps(result,sort_keys=True),
                    'is_error':bool(result.get('error')) or result.get('status')=='error' or result.get('ok') is False})
            equal(evidence.read(directory/f'turn-{turn:02d}-dispatch.json'),{'dispatches':batch},'saved dispatch batch')
            history.append({'role':'user','content':tool_results})
        else:
            require(accepted.get('stop_reason') in ('end_turn','stop_sequence','refusal'),'Unexpected final stop reason')
            accepted_text=text;terminal_seen=True
    equal(encountered_attempts,set(all_raw),'generation/count coverage')
    equal(counts,record['count_requests'],'count request summaries')
    equal(record['last_counted_input_tokens'],
          evidence.read(count_paths[-1])['response'].get('input_tokens')
          if count_paths and evidence.read(count_paths[-1])['status']=='response_received'
          else counts[-1]['input_tokens'] if counts else None,'last counted input')
    equal(history,record['history'],'preserved native conversation history')
    equal(expected_dispatches,record['tool_dispatches'],'replayed tool observations')
    equal(len(list(directory.glob('turn-*-dispatch.json'))),
          len({d['turn'] for d in expected_dispatches}),'unaccounted dispatch batches')
    equal(accepted_text,record['final_text'],'final-only text')
    replayed_grade=case.grade(record['final_text'])
    equal(replayed_grade,record['grade'],'fresh fixture grade')
    completed=record['state']=='completed'
    require(not completed or bool(accepted_text),'Completed generation lacks accepted final text')
    passed=completed and bool(replayed_grade['objective_passed'])
    equal(record['generation_complete'],completed,'generation completion gate')
    equal(record['objective_passed'],passed,'objective completion gate')
    if record['state']=='runner_or_api_error':
        require(record['error'] is not None and not passed,'Runner/API error cannot pass')
    else:
        equal(record['error'],None,'unexpected error record')
    unknown=failed_generation_requests+invalid_usage_responses
    generated=None if unknown else known_sum([a['output_tokens'] for a in attempts])
    inputs=None if unknown else known_sum([a.get('total_input_tokens_with_cache') for a in attempts])
    known_generated=sum(a['output_tokens'] or 0 for a in attempts)
    retries=sum(a['attempt']>0 for a in attempts)
    exhausted=sum(a['budget_exhausted_or_suspected'] is True for a in attempts)
    flagged=bool(wording.search(record['final_text'])) and not replayed_grade['objective_passed']
    for name,value in {'unknown_usage_requests':unknown,'generated_tokens_including_retries':generated,
        'known_generated_tokens_including_retries':known_generated,'input_tokens_including_retries':inputs,
        'retry_attempts':retries,'exhausted_or_suspected_attempts':exhausted,
        'refusal_wording_review_flag':flagged}.items():
        equal(record[name],value,name)
    grade=record['grade']
    observation={'case_id':record['case_id'],'repeat':record['repeat'],'state':record['state'],
        'objective_passed':passed,'generation_complete':completed,'evidence_complete':grade['evidence_complete'],
        'generated_tokens_including_retries':generated,'known_generated_tokens_including_retries':known_generated,
        'input_tokens_including_retries':inputs,'unknown_usage_requests':unknown,'generation_requests':len(all_raw),
        'returned_retry_attempts':retries,'all_retry_requests':sum(key[1]>0 for key in all_raw),
        'exhausted_or_suspected_attempts':exhausted,'tool_dispatches':len(expected_dispatches),
        'unsupported_findings':grade['unsupported_findings'],'repeated_unproductive_calls':grade['repeated_unproductive_calls'],
        'invalid_tool_calls':grade['invalid_tool_calls'],'scope_violations':grade['scope_violations'],
        'refusal_wording_review_flag':flagged,'observed_wall_seconds':record['wall_seconds'],
        'missing_evidence_facts':'; '.join(grade['details']['missing_evidence_facts']),
        'record_sha256':digest,'record_path':str(directory/'record.json')}
    return record,observation


def validate_run(run,evidence):
    envelope=evidence.read(run/'configuration.json');config=envelope['configuration']
    config_hash=envelope['configuration_fingerprint']
    equal(fingerprint(config),config_hash,'configuration fingerprint')
    equal(config['schema_version'],1,'configuration schema')
    require(config['endpoint']=='http://127.0.0.1:8002','Unexpected configured inference endpoint')
    require(config['temperature']==0 and config['batch_size']>0,'Unexpected development protocol')
    for key,value in [('id',config['model_id']),('root',config['checkpoint']),('max_model_len',config['context'])]:
        equal(config['model_listing'][key],value,f'configured serving identity {key}')
    fixtures,wording,minimum=load_fixtures(config,evidence)
    precision=evidence.read(WORK.parent/'precision-manifest-v1.json')
    equal(precision['expected_base_checkpoint'],config['checkpoint'],'reference checkpoint lineage')
    equal(precision['weight_bits'],8,'reference weight bits')
    equal(precision['activation_dtype'],'bfloat16','reference activation dtype')
    for filename,key in [('config.json','config_sha256'),('model.safetensors.index.json','index_sha256'),
                         ('conversion-receipt.json','conversion_receipt_sha256')]:
        equal(sha(Path(config['checkpoint'])/filename),precision[key],f'precision identity {filename}')
    schedule=config['schedule'];keys=[(s['case_id'],s['repeat']) for s in schedule]
    ids=[c['id'] for c in fixtures.CASES]
    require(schedule and len(keys)==len(set(keys)) and all(k[0] in ids and k[1] in (1,2) for k in keys),'Invalid case/repeat schedule')
    summary=evidence.read(run/'summary.json');status=evidence.read(run/'status.json')
    require(summary['status'] in ('completed','completed_with_errors'),'Run has no completed aggregate')
    equal(status['status'],summary['status'],'terminal run status')
    equal(summary['configuration_fingerprint'],config_hash,'summary configuration')
    records=[];observations=[]
    for item in schedule:
        record,observation=validate_record(run,item,config,config_hash,fixtures,wording,minimum,evidence)
        records.append(record);observations.append(observation)
    equal(len(list(run.glob('repeat-*/record.json'))),len(records),'unaccounted observation records')
    expected_summary=[{k:r[k] for k in ('case_id','repeat','state','objective_passed','generation_complete',
        'generated_tokens_including_retries','unknown_usage_requests','retry_attempts','exhausted_or_suspected_attempts',
        'refusal_wording_review_flag','grade','record_sha256')} for r in records]
    equal(summary['records'],expected_summary,'summary record rows/pairing')
    groups=evidence.read(run/'groups.json');batch=config['batch_size']
    equal(len(groups),(len(schedule)+batch-1)//batch,'committed group count')
    for index,group in enumerate(groups):
        equal(group['schedule'],schedule[index*batch:(index+1)*batch],'committed group pairing')
        group_records=records[index*batch:(index+1)*batch]
        equal(group['generated_tokens'],known_sum([r['generated_tokens_including_retries'] for r in group_records]),'committed group cost')
        seconds(group['wall_seconds'])
    warmup=evidence.read(run/'warmup.json')
    equal(warmup['status'],'completed','warmup completion')
    equal(warmup['request']['thinking'],{'type':'disabled'},'warmup thinking mode')
    equal(warmup['request']['chat_template_kwargs'],{'enable_thinking':False},'warmup template mode')
    equal(parts(warmup['response'],config['model_id'])[0],'READY','warmup response')
    warmup_usage=warmup['response']['usage'];warmup_output=token(warmup_usage.get('output_tokens'))
    require(warmup_output is not None,'Invalid warmup usage')
    generated=known_sum([r['generated_tokens_including_retries'] for r in records])
    state_counts=dict(Counter(r['state'] for r in records))
    calculated={'observations':len(records),'unique_cases':len(set(k[0] for k in keys)),
        'objective_passed':sum(r['objective_passed'] for r in records),
        'generation_complete':sum(r['generation_complete'] for r in records),
        'all_generations_complete':all(r['generation_complete'] for r in records),'state_counts':state_counts,
        'generated_tokens_including_retries':generated,
        'known_generated_tokens_including_retries':sum(r['known_generated_tokens_including_retries'] for r in records),
        'warmup_output_tokens':warmup_output,'all_inference_output_tokens_including_warmup':known_sum([warmup_output,generated]),
        'unknown_usage_requests':sum(r['unknown_usage_requests'] for r in records),
        'refusal_wording_review_flags':sum(r['refusal_wording_review_flag'] for r in records),
        'model_changes_performed':False,'live_target_requests_performed':False}
    for name,value in calculated.items():equal(summary[name],value,f'aggregate {name}')
    equal(summary['status'],'completed_with_errors' if state_counts.get('runner_or_api_error') else 'completed','aggregate completion classification')
    equal(status['observations'],len(records),'status observation total')
    equal(status['objective_passed'],calculated['objective_passed'],'status score')
    group_wall=sum(g['wall_seconds'] for g in groups)
    close(summary['sum_group_wall_seconds'],group_wall,'group wall time total')
    seconds(summary['elapsed_seconds'])
    per_case=[]
    for case_id in ids:
        rows=[r for r in observations if r['case_id']==case_id]
        if not rows:continue
        successes=sum(r['objective_passed'] for r in rows)
        row={'case_id':case_id,'family':next(c['family'] for c in fixtures.CASES if c['id']==case_id),
            'objective_passed':successes,'observations':len(rows),'score_fraction':successes/len(rows),
            'generation_complete':sum(r['generation_complete'] for r in rows),
            'repeat_outcomes_consistent':len({r['objective_passed'] for r in rows})==1,
            'generated_tokens_including_retries':known_sum([r['generated_tokens_including_retries'] for r in rows]),
            'known_generated_tokens_including_retries':sum(r['known_generated_tokens_including_retries'] for r in rows),
            'input_tokens_including_retries':known_sum([r['input_tokens_including_retries'] for r in rows]),
            'observed_sum_request_wall_seconds':sum(r['observed_wall_seconds'] for r in rows)}
        for name in ('unknown_usage_requests','returned_retry_attempts','all_retry_requests','exhausted_or_suspected_attempts',
                     'unsupported_findings','repeated_unproductive_calls','invalid_tool_calls','scope_violations','refusal_wording_review_flag'):
            row[name]=sum(r[name] for r in rows)
        per_case.append(row)
    for row in per_case:
        row['score_rank']=1+sum(other['score_fraction']>row['score_fraction'] for other in per_case)
    per_case.sort(key=lambda r:(-r['score_fraction'],ids.index(r['case_id'])))
    full=set(keys)=={(case_id,repeat) for case_id in ids for repeat in (1,2)}
    family_count=len({row['family'] for row in per_case})
    result={'status':'verified_report_generated','report_generation_completed':True,
        'all_evaluation_objectives_passed':calculated['objective_passed']==len(records),
        'configuration_fingerprint':config_hash,'label':config['label'],'model_id':config['model_id'],
        'checkpoint':config['checkpoint'],'precision':config['precision'],'full_eight_case_two_repeat_suite':full,
        'case_variants':len(per_case),'scenario_families':family_count,
        'repeats_per_variant':{row['case_id']:row['observations'] for row in per_case},
        'scope':f"{len(per_case)} synthetic case variants across {family_count} scenario families, {len(records)} observations; repeats are not independent questions. No actual Kali or live target execution.",
        'configured_scope':config['scope'],'totals':calculated,
        'input_tokens_including_retries':known_sum([r['input_tokens_including_retries'] for r in records]),
        'warmup_input_tokens_with_cache':native_total(warmup_usage),
        'all_inference_input_tokens_including_warmup':known_sum([
            native_total(warmup_usage),*[r['input_tokens_including_retries'] for r in records]]),
        'timing':{'observed_run_wall_seconds':summary['elapsed_seconds'],'committed_group_wall_seconds':group_wall,
            'observed_output_tokens_per_group_wall_second':generated/group_wall if generated is not None and group_wall>0 else None,
            'interpretation':'Observed concurrent development timings, not controlled deployment speed; group throughput never sums individual request latencies.'},
        'refusal_interpretation':'Wording flags require manual adjudication. They are not a refusal rate; failures or exhausted budgets are not automatically refusals.',
        'ranking':'Competition ranks by objective pass fraction only. Equal scores share ranks; token cost does not break ties.',
        'grading_limit':'Deterministic evidence/structured-claim grading; no general semantic judge or real engagement readiness claim.',
        'cases':per_case,'observations':observations,'evidence_validation':{
            'source_pins_verified':True,'configuration_and_record_fingerprints_verified':True,
            'raw_generation_hashes_verified':True,'all_generation_request_costs_reconciled':True,
            'native_history_and_tool_results_replayed':True,'model_visible_evidence_turn_boundaries_replayed':True,
            'final_text_regraded_with_fresh_fixtures':True,'runner_or_api_failures_cannot_pass':True},
        'created_utc':datetime.now(timezone.utc).isoformat()}
    return result


def write_outputs(directory,result):
    for filename,rows in [('observations.csv',result['observations']),('cases-ranked.csv',result['cases'])]:
        with (directory/filename).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR',str(directory/'matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=result['cases'];positions=list(range(len(rows)))
    fig,(scores,costs)=plt.subplots(1,2,figsize=(14,max(5.2,.57*len(rows)+2.1)),sharey=True,
        gridspec_kw={'width_ratios':[1.3,1]})
    colors=['#238b66' if r['score_fraction']==1 else '#d69d30' if r['score_fraction'] else '#bd4851' for r in rows]
    scores.barh(positions,[100*r['score_fraction'] for r in rows],color=colors)
    labels=[f"#{r['score_rank']}  {r['case_id']}" for r in rows]
    scores.set_yticks(positions,labels);scores.invert_yaxis();scores.set_xlim(0,119)
    scores.set_xlabel('Objective success (%)');scores.set_xticks([0,50,100])
    for y,row in enumerate(rows):scores.text(100*row['score_fraction']+2,y,f"{row['objective_passed']}/{row['observations']}",va='center',fontsize=10)
    values=[r['known_generated_tokens_including_retries'] for r in rows]
    bars=costs.barh(positions,values,color='#426ca8')
    for y,(row,bar) in enumerate(zip(rows,bars)):
        unknown=row['generated_tokens_including_retries'] is None
        if unknown:bar.set_hatch('//')
        label=f"{'≥' if unknown else ''}{values[y]:,}"+(' (incomplete)' if unknown else '')
        costs.text(values[y]+max(values+[1])*.025,y,label,va='center',fontsize=9)
    costs.set_xlim(0,max(values+[1])*1.42);costs.set_xlabel('Output tokens, all attempts\nWarmup excluded; retries included')
    for axis in (scores,costs):
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='x',alpha=.15);axis.set_axisbelow(True)
    title=f"{result['label']} — synthetic workflow development"
    fig.suptitle(title,fontsize=15,weight='bold',y=.98)
    repeats=Counter(r['observations'] for r in rows)
    repeat_label=(f"{len(rows)} synthetic variants across {result['scenario_families']} scenario families; "
                  + (f"{next(iter(repeats))} repeats each" if len(repeats)==1 else 'mixed repeat counts'))
    fig.text(.5,.915,f"{repeat_label} ({len(result['observations'])} observations). Equal scores share ranks.",ha='center',fontsize=10)
    fig.text(.5,.025,'No actual Kali/live targets. Observed development timings only. Refusal-wording flags require manual review.',ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=[0,.065,1,.89])
    fig.savefig(directory/'case-rankings.png',dpi=170)
    fig.savefig(directory/'case-rankings.svg')
    plt.close(fig)
    shutil.rmtree(directory/'matplotlib-cache',ignore_errors=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();run=args.run_dir.resolve()
    output=args.output_dir.resolve() if args.output_dir else run/'report-workflows-v1'
    if not (run/'summary.json').is_file():
        status_path=run/'status.json';status=json.loads(status_path.read_text()).get('status') if status_path.is_file() else None
        print(json.dumps({'status':'upstream_failed' if status=='failed' else 'not_ready','run_dir':str(run)}))
        return 3 if status=='failed' else 2
    staging=None
    try:
        require(output.parent in (run,run.parent),'Output must be a new direct run subdirectory or sibling')
        require(not output.exists(),'Output exists; use a new directory to preserve reports')
        evidence=Evidence();result=validate_run(run,evidence)
        staging=Path(tempfile.mkdtemp(prefix='.workflow-report-',dir=output.parent))
        write_outputs(staging,result)
        evidence.verify_unchanged()
        result['input_evidence']=evidence.pins
        result['reporter_sha256']=sha(__file__)
        result['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in staging.iterdir() if p.is_file()}
        (staging/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        staging.rename(output)
        print(json.dumps({'status':'verified_report_generated','output_dir':str(output),
            'objectives_passed':result['totals']['objective_passed'],'observations':result['totals']['observations'],
            'all_objectives_passed':result['all_evaluation_objectives_passed'],'summary_sha256':sha(output/'summary.json')}))
        return 0
    except Exception as error:
        if staging is not None:shutil.rmtree(staging,ignore_errors=True)
        print(json.dumps({'status':'validation_failed','error_type':type(error).__name__,'error':str(error)}))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
