"""Offline checks for native-response accounting and dispatch boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import run_workflows_v1 as runner


def response(tokens=16, stop='tool_use', content=None, input_tokens=128):
    if content is None:
        content = [{'type': 'thinking', 'thinking': 'Inspect the supplied evidence.', 'signature': 'fixture-signature'},
                   {'type': 'tool_use', 'id': 'tool_1', 'name': 'inspect', 'input': {'target': 'app.ctf.test'}}]
    return {'model': runner.MODEL, 'role': 'assistant', 'type': 'message',
            'usage': {'input_tokens': input_tokens, 'output_tokens': tokens},
            'content': content, 'stop_reason': stop}


def final():
    return response(8, 'end_turn', [{'type': 'text', 'text': 'Done with evidence.'}])


class FakeCase:
    system_prompt = 'Use only synthetic test tools.'
    user_prompt = 'Inspect the authorized fixture and report.'
    tools = [{'name': 'inspect', 'description': 'Inspect fixture.',
              'input_schema': {'type': 'object', 'properties': {'target': {'type': 'string'}}}}]

    def __init__(self):
        self.calls = []

    def dispatch(self, name, args):
        self.calls.append((name, args))
        return {'ok': True, 'evidence_id': 'e1'}

    def begin_turn(self):
        pass

    def grade(self, text):
        return {'objective_passed': bool(self.calls) and text == 'Done with evidence.',
                'evidence_complete': bool(self.calls)}


class RunnerTests(unittest.TestCase):
    def execute(self, replies, count=128):
        case = FakeCase()
        fixtures = types.SimpleNamespace(new_case=lambda _: case)
        pending = iter(replies)
        requests = []

        def fake_request(route, payload=None, timeout=600):
            requests.append((route, copy.deepcopy(payload)))
            if route.endswith('/count_tokens'):
                return {'input_tokens': count}
            item = next(pending)
            if isinstance(item, Exception):
                raise item
            return item

        with tempfile.TemporaryDirectory() as temp, patch.object(runner, 'request', fake_request):
            path = Path(temp) / 'case'
            record = runner.run_case(fixtures, 'mock-case', 1, 'fixed-config', path)
            saved = {p.name: json.loads(p.read_text()) for p in path.glob('*.json')}
        return record, case, requests, saved

    def test_tool_stop_at_ceiling_is_retried_before_dispatch(self):
        truncated = response(4096)
        truncated['content'][1]['id'] = 'truncated_call'
        record, case, requests, _ = self.execute([truncated, response(), final()])
        self.assertTrue(record['objective_passed'])
        self.assertEqual(len(case.calls), 1)
        self.assertEqual(record['generated_tokens_including_retries'], 4120)
        self.assertEqual(record['retry_attempts'], 1)
        self.assertFalse(record['native_attempts'][0]['accepted_for_history'])
        generations = [payload for route, payload in requests if route == '/v1/messages']
        self.assertEqual(generations[0]['messages'], generations[1]['messages'])
        self.assertEqual(generations[1]['max_tokens'], 6144)
        self.assertEqual(generations[2]['messages'][1]['content'][0]['signature'], 'fixture-signature')
        self.assertEqual(generations[2]['messages'][2]['content'][0]['tool_use_id'], 'tool_1')
        self.assertNotIn('truncated_call', json.dumps(record['history']))

    def test_exhausted_retry_dispatches_nothing(self):
        record, case, _, _ = self.execute([response(4096), response(6144)])
        self.assertEqual(record['state'], 'generation_budget_exhausted')
        self.assertEqual(case.calls, [])
        self.assertEqual(record['generated_tokens_including_retries'], 10240)
        self.assertFalse(record['refusal_wording_review_flag'])

    def test_context_exhaustion_does_not_generate(self):
        record, case, requests, _ = self.execute([], count=7000)
        self.assertEqual(record['state'], 'context_budget_exhausted')
        self.assertEqual(len(requests), 1)
        self.assertEqual(case.calls, [])

    def test_duplicate_tool_ids_are_rejected_before_dispatch(self):
        duplicate = response()
        duplicate['content'].append(copy.deepcopy(duplicate['content'][1]))
        record, case, _, _ = self.execute([duplicate])
        self.assertEqual(record['state'], 'runner_or_api_error')
        self.assertEqual(case.calls, [])
        self.assertEqual(record['generated_tokens_including_retries'], 16)
        self.assertFalse(record['native_attempts'][0]['accepted_for_history'])

    def test_missing_usage_preserves_known_output_and_marks_total_unknown(self):
        malformed = response(17)
        del malformed['usage']['input_tokens']
        record, case, _, saved = self.execute([malformed])
        self.assertEqual(record['state'], 'runner_or_api_error')
        self.assertEqual(record['known_generated_tokens_including_retries'], 17)
        self.assertIsNone(record['generated_tokens_including_retries'])
        self.assertEqual(record['unknown_usage_requests'], 1)
        self.assertEqual(case.calls, [])
        self.assertEqual(saved['turn-00-attempt-00.json']['response']['usage']['output_tokens'], 17)

    def test_timeout_has_unknown_usage_and_retains_failed_request(self):
        record, case, _, saved = self.execute([TimeoutError('synthetic timeout')])
        self.assertIsNone(record['generated_tokens_including_retries'])
        self.assertEqual(record['unknown_usage_requests'], 1)
        self.assertEqual(case.calls, [])
        self.assertEqual(saved['turn-00-attempt-00.json']['status'], 'request_failed')

    def test_non_object_response_preserves_raw_and_marks_usage_unknown(self):
        record, case, _, saved = self.execute([['malformed native response']])
        self.assertEqual(record['state'], 'runner_or_api_error')
        self.assertIsNone(record['generated_tokens_including_retries'])
        self.assertEqual(record['unknown_usage_requests'], 1)
        self.assertEqual(case.calls, [])
        self.assertEqual(saved['turn-00-attempt-00.json']['response'], ['malformed native response'])

    def test_native_cache_usage_is_included_and_checked(self):
        first = response(input_tokens=100)
        first['usage']['cache_read_input_tokens'] = 28
        record, _, _, _ = self.execute([first, final()])
        self.assertTrue(record['objective_passed'])
        self.assertEqual(record['input_tokens_including_retries'], 256)

    def test_native_count_mismatch_is_not_a_model_refusal(self):
        record, case, _, _ = self.execute([response(input_tokens=127)])
        self.assertEqual(record['state'], 'runner_or_api_error')
        self.assertEqual(case.calls, [])
        self.assertFalse(record['refusal_wording_review_flag'])
        self.assertEqual(record['generated_tokens_including_retries'], 16)

    def test_dynamic_model_listing_fields_do_not_change_identity(self):
        def listing(permission):
            return {'data': [{'id': runner.MODEL, 'root': runner.BASE, 'parent': None,
                              'max_model_len': runner.CONTEXT, 'permission': [{'id': permission}]}]}
        with patch.object(runner, 'request', side_effect=[listing('a'), listing('b')]):
            self.assertEqual(runner.model_identity(), runner.model_identity())


if __name__ == '__main__':
    unittest.main()
