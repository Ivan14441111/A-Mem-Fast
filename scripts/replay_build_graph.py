#!/usr/bin/env python3
"""Compare reference/fast graph construction using one saved exact-response cache.

Run only after the source process exits. This is build-only, zero-HTTP replay;
it cannot recover a lost baseline trace, live timing, or incomplete QA results.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark import canonical_memories, digest
from benchmark_context import ReadOnlySnapshotCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output report')
    directory = args.run_dir
    manifest = json.loads((directory / 'manifest.json').read_text())
    config = manifest['args']
    subset = json.loads((directory / 'subset.json').read_text())
    if digest({k: v for k, v in subset.items() if k != 'fingerprint'}) != subset['fingerprint']:
        raise ValueError('Invalid subset fingerprint')
    expected = manifest['source_sha256']

    def sources():
        actual = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in expected}
        if actual != expected:
            raise RuntimeError('Measured source drift; use the archived version')

    sources()
    cache_path = directory / 'optimized.sqlite'
    snapshot = ReadOnlySnapshotCache(cache_path)
    result = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'scope': 'Build-only robust-versus-fast reconstruction from the SAME exact-arm response cache. No QA is attempted. This is not replay of the lost original live baseline.',
              'source_manifest': str(directory / 'manifest.json'),
              'source_manifest_sha256': hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest(),
              'measured_project_commit': manifest.get('project_commit'),
              'command': [sys.executable, *sys.argv],
              'subset_fingerprint': subset['fingerprint'], 'source_sha256': expected,
              'analysis_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cache_rows': len(snapshot.rows),
              'cache_responses_digest': digest(snapshot.rows), 'runs': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    helper_copy = args.output.with_suffix('.source.py.txt')
    if helper_copy.exists():
        raise FileExistsError('Choose a fresh helper source snapshot')
    helper_copy.write_bytes(Path(__file__).read_bytes())
    result['analysis_source_snapshot'] = str(helper_copy)

    def save():
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')

    save()
    try:
        import torch
        torch.set_num_threads(4)
        from sentence_transformers import SentenceTransformer
        import memory_layer
        from memory_layer_robust import RobustAgenticMemorySystem
        from fast_amem.client import MeasuredClient
        from fast_amem.system import FastAgenticMemorySystem
        memory_layer.SentenceTransformer = lambda name: SentenceTransformer(
            config['embedding_model'] if name == 'all-MiniLM-L6-v2' else name, device='cpu')
        shared = SentenceTransformer(config['embedding_model'], device='cpu')
        for name in ('reference', 'fast'):
            client = MeasuredClient(config['base_url'], config['model'], cache=snapshot, cache_mode='replay',
                                    revision=config['model_revision'], max_tokens=config['max_tokens'], seed=config['seed'],
                                    repetition_penalty=config.get('repetition_penalty'), tag_budget=config.get('tag_budget'),
                                    tag_budget_policy=config.get('tag_budget_policy', 'error'),
                                    structured_policy_outputs=config.get('structured_policy_outputs', False))
            if name == 'reference':
                system = RobustAgenticMemorySystem.__new__(RobustAgenticMemorySystem)
                system.memories = {}
                system.retriever = memory_layer.SimpleEmbeddingRetriever(config['embedding_model'])
                system.llm_controller = SimpleNamespace(llm=client)
                system.evo_cnt = 0
                system.evo_threshold = config['evo_threshold']
            else:
                system = FastAgenticMemorySystem(shared, client, evo_threshold=config['evo_threshold'])
            consolidation = []
            original = system.consolidate_memories
            def consolidate(*a, **kw):
                value = original(*a, **kw)
                consolidation.append(len(system.memories))
                return value
            system.consolidate_memories = consolidate
            trace = []
            def on_note(n):
                if client.summary()['errors']:
                    raise RuntimeError('Strict build replay encountered an error; reference fallback is forbidden')
                trace.append(digest(canonical_memories(system)))
                if n % 64 == 0 or n == len(subset['turns']):
                    print(f'{name}: {n}/{len(subset["turns"])}', flush=True)
            if name == 'reference':
                for n, turn in enumerate(subset['turns'], 1):
                    system.add_note(turn['content'], time=turn['timestamp'])
                    on_note(n)
            else:
                system.ingest(subset['turns'], analysis_workers=config['workers'],
                              prefetch_window=config['prefetch_window'], on_note=on_note)
            summary = client.summary()
            if summary['http_requests'] or summary['errors']:
                raise AssertionError('Build-only replay must have zero HTTP/errors')
            result['runs'][name] = {'memory_trace': trace, 'memory_digest': digest(canonical_memories(system)),
                                    'logical_request_keys': [e['key'] for e in client.events if not e['prefetch']],
                                    'memory_count': len(system.memories), 'evolution_count': system.evo_cnt,
                                    'consolidation_memory_counts': consolidation, 'llm': summary}
            save()
        a, b = result['runs']['reference'], result['runs']['fast']
        checks = {name + '_expected_turn_count': run['memory_count'] == len(subset['turns']) and len(run['memory_trace']) == len(subset['turns']) for name, run in result['runs'].items()}
        checks.update({k + '_equal': a[k] == b[k] for k in ('memory_trace', 'memory_digest', 'logical_request_keys',
                  'memory_count', 'evolution_count', 'consolidation_memory_counts')})
        live = directory / 'optimized.json'
        result['original_live_exact_report_available'] = live.exists()
        if live.exists():
            report = json.loads(live.read_text())
            result['original_live_exact_report_sha256'] = hashlib.sha256(live.read_bytes()).hexdigest()
            if report['subset_fingerprint'] != subset['fingerprint']:
                raise AssertionError('Live exact report subset mismatch')
            for name, run in result['runs'].items():
                checks[name + '_matches_live_exact_evolution_count'] = run['evolution_count'] == report['evolution_count']
                checks[name + '_matches_live_exact_consolidation_count'] = len(run['consolidation_memory_counts']) == report['operation_counts'].get('consolidate_memories', 0)
                checks[name + '_matches_live_exact_trace'] = run['memory_trace'] == report['memory_trace']
                checks[name + '_matches_live_exact_digest'] = run['memory_digest'] == report['memory_digest']
                checks[name + '_matches_live_exact_build_requests'] = run['logical_request_keys'] == [
                    e['key'] for e in report['llm_events'] if not e['prefetch'] and e['stage'] not in {'query_keywords', 'answer'}]
        result['checks'] = checks
        if not all(checks.values()):
            raise AssertionError('Build-only reconstruction mismatch')
        sources()
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', exception_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        actual = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() if (ROOT / name).exists() else None for name in expected}
        result['source_integrity'] = 'passed' if actual == expected else 'failed'
        if actual != expected:
            result['status'] = 'failed'
            result['source_sha256_after'] = actual
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        if actual != expected:
            raise RuntimeError('Measured source drift after build-only verification')


if __name__ == '__main__':
    main()
