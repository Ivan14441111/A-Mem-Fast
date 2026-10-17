#!/usr/bin/env python3
"""Export response-usage evidence for failed arms without prompts or inference.

SQLite rows cover unique saved responses, not complete live event histories.
Retries, rejected outputs without diagnostics, or unflushed buffers may be absent.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from urllib.parse import quote
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fast_amem.client import stage_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    if any(c['status'] not in {'complete', 'failed'} for c in suite['cells']):
        parser.error('Wait for every planned live attempt to exit')
    result = {'source_suite': str(args.suite), 'source_suite_sha256': hashlib.sha256(args.suite.read_bytes()).hexdigest(),
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Unique cached response usage plus any uncached final failure diagnostic. A lower bound on observed attempt usage, not complete event/request/retry counts. Prefetched responses may extend beyond the last committed memory. No timing or QA estimate is inferred for incomplete arms.',
              'cells': []}
    for cell in suite['cells']:
        directory = args.suite.parent / cell['name']
        for failure_path in sorted(directory.glob('*-failure.json')):
            failure = json.loads(failure_path.read_text());mode = failure['mode']
            if mode not in {'baseline', 'optimized', 'fused', 'joint'}:
                continue
            cache = directory / (mode + '.sqlite')
            by_stage = defaultdict(lambda: {'saved_unique_responses': 0, 'prompt_tokens': 0, 'completion_tokens': 0})
            keys = set()
            if cache.exists():
                with sqlite3.connect('file:' + quote(str(cache.resolve()), safe='/') + '?mode=ro', uri=True) as conn:
                    for key, payload, response in conn.execute('SELECT key,payload,response FROM responses'):
                        keys.add(key);payload=json.loads(payload);response=json.loads(response)
                        prompt=next(m['content'] for m in payload['request']['messages'] if m['role']=='user')
                        row=by_stage[stage_for(prompt)];row['saved_unique_responses']+=1
                        for field in ('prompt_tokens','completion_tokens'):
                            row[field]+=response.get('usage',{}).get(field,0) or 0
            diagnostics=failure.get('llm_diagnostic',[])
            if isinstance(diagnostics,dict):diagnostics=[diagnostics]
            uncached=[]
            for diagnostic in diagnostics:
                key=diagnostic.get('request_key')
                if key and key not in keys and diagnostic.get('usage'):
                    keys.add(key);uncached.append({'stage':diagnostic.get('stage'),'request_key':key,'usage':diagnostic['usage']})
            totals={field:sum(row[field] for row in by_stage.values()) for field in ('prompt_tokens','completion_tokens')}
            for diagnostic in uncached:
                for field in totals:totals[field]+=diagnostic['usage'].get(field,0) or 0
            result['cells'].append({'cell':cell['name'],'mode':mode,'progress':failure.get('progress'),
                                    'failure_artifact_sha256':hashlib.sha256(failure_path.read_bytes()).hexdigest(),
                                    'saved_unique_response_count':sum(r['saved_unique_responses'] for r in by_stage.values()),
                                    'saved_usage_by_stage':dict(by_stage),'uncached_final_diagnostic_usage':uncached,
                                    'response_usage_lower_bound':totals,
                                    'empty_recording_note':'Empty buffer-backed reference cache cannot recover prior live build usage.' if mode=='baseline' and not by_stage else None})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
