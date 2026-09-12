#!/usr/bin/env python3
"""Summarize observed cache states without accepting contaminated measurements."""

import argparse
import json
from pathlib import Path
import re
import statistics

from z_image_evidence import sha256


def cache_classification(read_bytes, stored_bytes):
    ratio = read_bytes / stored_bytes
    return 'disk-read' if ratio >= .9 else ('cached' if ratio <= .01 else 'mixed')


def eligible(row):
    names = ['load', 'first_execution', 'resident_1', 'resident_2', 'resident_3']
    return (row['benchmark_eligible'] and bool(row['background']) and bool(row['preflight']) and
            [p['name'] for p in row['phases']] == names and
            all(s['cpu']['quiet'] and 'AC Power' in s['power'] for s in row['background']) and
            all(s['quiet'] for s in row['preflight']) and
            all(p.get('exact_reference_match') is True and p.get('finite') is True and
                (row['stage'] != 'denoise' or p.get('actual_forwards') == 9)
                for p in row['phases'][1:]))


def phase_eligible(row, phase):
    if eligible(row):
        return True
    if not row['preflight'] or not all(s['quiet'] for s in row['preflight']):
        return False
    if phase['name'] != 'load' and (phase.get('exact_reference_match') is not True or
            phase.get('finite') is not True or (row['stage'] == 'denoise' and phase.get('actual_forwards') != 9)):
        return False
    if not row['background'] or any('window_start' not in s for s in row['background']):
        return False
    start, end = phase['start']['time'], phase['end']['time']
    samples = [s for s in row['background'] if s['window_end'] >= start and s['window_start'] <= end]
    if not samples or samples[0]['window_start'] > start or samples[-1]['window_end'] < end:
        return False
    if any(a['window_end'] < b['window_start'] for a, b in zip(samples, samples[1:])):
        return False
    return all(s['cpu']['quiet'] and 'AC Power' in s['power'] for s in samples)


def summarize(paths):
    rows = []
    groups = {}
    failures = []
    window_groups = {}
    for directory in paths:
        suite = json.loads((directory / 'suite.json').read_text())
        if not suite['restored']:
            raise ValueError('paused processes not restored')
        failures.extend({'suite': directory.name, **c} for c in suite['commands'] if c['exit_code'])
        for stage in ('encode', 'denoise', 'decode'):
            for arm in ('cold-copy', 'reuse'):
                path = directory / (stage + '-' + arm + '.json')
                if not path.exists():
                    continue
                r = json.loads(path.read_text())
                load = r['phases'][0]
                logfile = path.with_suffix('.log')
                highwater = re.search(r'^\s*(\d+)\s+peak memory footprint\s*$', logfile.read_text(), re.M)
                if highwater is None:
                    raise ValueError('missing kernel high-water footprint')
                row = {'suite': directory.name, 'stage': stage, 'requested_cache_arm': arm,
                       'observed_cache': cache_classification(load['physical_read_bytes'], r['stored_weight_bytes']),
                       'eligible': eligible(r), 'record_sha256': sha256(path), 'log_sha256': sha256(logfile),
                       'qualified_phases': {p['name']: phase_eligible(r, p) for p in r['phases']},
                       'load_s': load['elapsed_s'], 'load_physical_read_bytes': load['physical_read_bytes'],
                       'stored_weight_bytes': r['stored_weight_bytes'],
                       'first_execution_s': r['phases'][1]['elapsed_s'],
                       'resident_execution_s': [v['elapsed_s'] for v in r['phases'][2:]],
                       'kernel_peak_process_footprint_bytes': int(highwater.group(1)),
                       'resident_parameter_bytes': r['resident_parameter_bytes'],
                       'retained_driver_bytes': r['retained_after_delete']['mps_driver_bytes'],
                       'after_empty_cache_driver_bytes': r['after_empty_cache']['mps_driver_bytes'],
                       'swap_start': r['swap_start'], 'swap_end': r['swap_end']}
                rows.append(row)
                for phase in r['phases']:
                    if phase_eligible(r, phase):
                        kind = 'resident' if phase['name'].startswith('resident_') else phase['name']
                        window_groups.setdefault(stage + '/' + kind, []).append({
                            'suite': directory.name, 'arm': arm, 'elapsed_s': phase['elapsed_s'],
                            'sampled_peak_phys_footprint_bytes': phase['sampled_peak_phys_footprint_bytes'],
                            'physical_read_bytes': phase['physical_read_bytes']})
                if row['eligible']:
                    groups.setdefault(stage, []).append(row)
    aggregate = {}
    for stage, members in groups.items():
        resident = [value for row in members for value in row['resident_execution_s']]
        aggregate[stage] = {'qualified_processes': len(members), 'resident_samples': len(resident),
                            'resident_s_min': min(resident), 'resident_s_median': statistics.median(resident),
                            'resident_s_max': max(resident),
                            'max_kernel_process_footprint_bytes': max(r['kernel_peak_process_footprint_bytes'] for r in members)}
    return {'protocol': 'IG0 component load, first execution, then three resident repetitions; lighting seed 42',
            'scope': 'BF16 text/transformer, FP32 VAE; reference execution, not native or packed INT4',
            'rows': rows, 'failed_attempts': failures, 'qualified_aggregate': aggregate,
            'qualified_timed_windows': window_groups,
            'window_policy': 'Same CPU/AC thresholds on every overlapping complete interval; rejects coverage gaps. Whole-process rejection remains recorded.',
            'peak_scope': 'kernel process lifetime peak includes preparation and verification; phase samples are in raw records',
            'cold_scope': 'F_NOCACHE new-inode copy, observed physical reads; no global cache purge',
            'minimum_ram_claim': False, 'scratch_bytes_known': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('suites', type=Path, nargs='+')
    p.add_argument('--out', type=Path, default=Path('docs/verification/z-image-ig0-benchmarks.json'))
    args = p.parse_args()
    result = summarize(args.suites)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
