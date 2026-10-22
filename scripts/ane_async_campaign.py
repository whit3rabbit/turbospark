#!/usr/bin/env python3
"""Fresh-process paired campaign for the isolated public handoff probe.

This measures synthetic handoffs, not model prefill or measured energy.
Invoke after building the release example, with no concurrent GPU tests.
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary', type=Path)
    p.add_argument('program', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--pairs', type=int, default=6)
    a = p.parse_args()
    if a.pairs < 2:
        p.error('at least two pairs required')
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for pair in range(a.pairs):
        for executor in (['sync', 'async'] if pair % 2 == 0 else ['async', 'sync']):
            result = subprocess.run([str(a.binary.resolve()), str(a.program.resolve()), '1', executor],
                                    capture_output=True, text=True, timeout=120)
            stem = a.output / f'{pair}-{executor}'
            stem.with_suffix('.stderr').write_text(result.stderr)
            stem.with_suffix('.stdout').write_text(result.stdout)
            result.check_returncode()
            samples = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            if len(samples) != 1 or samples[0]['executor'] != executor:
                raise ValueError('probe did not execute the requested arm exactly once')
            rows.append(dict(samples[0], pair=pair))
    deltas = []
    for pair in range(a.pairs):
        arm = {r['executor']: r for r in rows if r['pair'] == pair}
        deltas.append(arm['async']['prediction_ms'] - arm['sync']['prediction_ms'])
    rng = random.Random(0)
    boot = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000))
    summary = {'scope': 'synthetic prediction handoff; not end-to-end prefill',
               'pairs': a.pairs, 'delta_async_minus_sync_ms': statistics.mean(deltas),
               'paired_bootstrap_95_ms': [boot[250], boot[9749]], 'samples': rows}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
