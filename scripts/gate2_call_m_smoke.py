"""Bounded local API proposal capture. Credentials never enter output artifacts."""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.gate2_e0_observe import plan
from synthesis_v2.llm_proposal_contract import build_prompt, parse_response


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--credentials', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lines = [x.strip() for x in args.credentials.read_text().splitlines() if x.strip()]
    endpoint = next(x for x in lines if x.startswith('https://'))
    key = next(x for x in lines if not x.startswith('https://'))
    requests = plan(50, 1)['requests']
    config = {'model': 'qwen3.7-flash-2026-07-15', 'temperature': 0,
              'max_tokens': 512, 'max_calls': 8, 'automatic_retries': 0,
              'endpoint': endpoint, 'cost_currency_amount': None,
              'cost_note': 'provider tariff unavailable; bounded calls/tokens, no monetary claim'}
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    proposals = []
    for index, request in enumerate(requests):
        payload = {k: config[k] for k in ('model','temperature','max_tokens')}
        payload['messages'] = [{'role':'user','content':json.dumps(build_prompt(request))}]
        payload['response_format'] = {'type':'json_object'}
        payload['enable_thinking'] = False
        record = {'request_id':request['request_id'], 'payload':payload,
                  'status':'started', 'api_calls':1}
        path = args.output/f'call_{index:02d}.json'
        path.write_text(json.dumps(record, indent=2))
        started = time.monotonic()
        try:
            req = urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',
                data=json.dumps(payload).encode(), headers={
                'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=60) as response:
                body = json.load(response)
            record['response'] = body
            candidate = parse_response(body['choices'][0]['message']['content'], request['task'])
            record['status'] = 'parsed'
            proposals.append({'request_id':request['request_id'], 'candidate_preset':candidate,
                              'call_index':index, 'usage':body.get('usage'),
                              'response_model':body.get('model')})
        except Exception as exc:
            record['status'] = 'failed'
            record['error_type'] = type(exc).__name__
            record['http_status'] = getattr(exc, 'code', None)
        record['wall_s'] = time.monotonic()-started
        # API responses must not echo credentials into artifacts.
        path.write_text(json.dumps(record, indent=2).replace(key,'[REDACTED]'))
        print(json.dumps({'index':index,'status':record['status']}), flush=True)
        if record['status'] != 'parsed':
            break
    (args.output/'proposals.json').write_text(json.dumps(proposals, indent=2))


if __name__ == '__main__':
    main()
