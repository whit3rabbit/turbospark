"""Admission and process-identity guards for reference benchmark evidence."""

import copy
import signal
import unittest
from unittest.mock import patch, Mock

from z_image_benchmark_summary import cache_classification, eligible, phase_eligible
from z_image_benchmark_suite import signal_same


def evidence():
    return {'benchmark_eligible': True, 'stage': 'denoise',
            'preflight': [{'quiet': True}],
            'background': [{'cpu': {'quiet': True}, 'power': 'AC Power'}],
            'phases': [{'name': 'load'}] + [
                {'name': name, 'exact_reference_match': True, 'finite': True, 'actual_forwards': 9}
                for name in ('first_execution', 'resident_1', 'resident_2', 'resident_3')]}


class BenchmarkTests(unittest.TestCase):
    def test_admission_rejects_dirty_or_incomplete_runs(self):
        base = evidence()
        self.assertTrue(eligible(base))
        edits = [lambda r: r['background'][0]['cpu'].update(quiet=False),
                 lambda r: r['background'][0].update(power='Battery Power'),
                 lambda r: r['phases'][1].update(actual_forwards=8),
                 lambda r: r['phases'][1].pop('exact_reference_match'),
                 lambda r: r['phases'].pop()]
        for edit in edits:
            row = copy.deepcopy(base)
            edit(row)
            self.assertFalse(eligible(row))

    def test_timed_windows_require_quiet_continuous_coverage(self):
        row = evidence()
        row['benchmark_eligible'] = False
        row['background'] = [
            {'window_start': a, 'window_end': a + 1, 'cpu': {'quiet': a != 0}, 'power': 'AC Power'}
            for a in range(3)]
        phase = {**row['phases'][2], 'start': {'time': 1.1}, 'end': {'time': 2.9}}
        self.assertTrue(phase_eligible(row, phase))
        row['background'][1]['window_end'] = 1.9
        self.assertFalse(phase_eligible(row, phase))
        row['background'][1]['window_end'] = 2
        row['background'][2]['cpu']['quiet'] = False
        self.assertFalse(phase_eligible(row, phase))
        row['background'][2]['cpu']['quiet'] = True
        phase['start']['time'] = .9
        self.assertFalse(phase_eligible(row, phase))

    def test_cache_classification_uses_measured_reads(self):
        self.assertEqual(cache_classification(1000, 1000), 'disk-read')
        self.assertEqual(cache_classification(1, 1000), 'cached')
        self.assertEqual(cache_classification(500, 1000), 'mixed')

    def test_recycled_pid_is_not_signaled(self):
        proc = Mock()
        proc.create_time.return_value = 20
        with patch('z_image_benchmark_suite.psutil.Process', return_value=proc):
            signal_same([{'pid': 123, 'created': 10}], signal.SIGCONT)
        proc.send_signal.assert_not_called()

    def test_matching_pid_is_restored(self):
        proc = Mock()
        proc.create_time.return_value = 10
        with patch('z_image_benchmark_suite.psutil.Process', return_value=proc):
            signal_same([{'pid': 123, 'created': 10}], signal.SIGCONT)
        proc.send_signal.assert_called_once_with(signal.SIGCONT)


def mutation_report(destination):
    import hashlib
    import json
    from pathlib import Path
    import re
    import shutil
    import subprocess
    import sys
    import tempfile
    mutations = [
        ('z_image_benchmark_summary.py', "s['window_end'] >= start", "s['window_end'] >= end",
         'test_timed_windows_require_quiet_continuous_coverage'),
        ('z_image_benchmark_summary.py', "all(s['cpu']['quiet'] and 'AC Power' in s['power'] for s in row['background'])", "all('AC Power' in s['power'] for s in row['background'])",
         'test_admission_rejects_dirty_or_incomplete_runs'),
        ('z_image_benchmark_summary.py', 'ratio >= .9', 'ratio >= .1',
         'test_cache_classification_uses_measured_reads'),
        ('z_image_benchmark_suite.py', "if proc.create_time() == row['created']:", 'if True:',
         'test_recycled_pid_is_not_signaled'),
        ('z_image_benchmark_suite.py', 'proc.send_signal(sig)', 'pass',
         'test_matching_pid_is_restored')]
    rows = []
    source = Path(__file__).parent
    for filename, old, new, case in mutations:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('test_z_image_benchmark.py', 'z_image_benchmark_summary.py',
                         'z_image_benchmark_suite.py', 'z_image_evidence.py', 'z_image_probe.py'):
                shutil.copyfile(source / name, root / name)
            path = root / filename
            original = path.read_text()
            assert original.count(old) == 1, (filename, old)
            mutated = original.replace(old, new, 1)
            assert mutated != original
            path.write_text(mutated)
            result = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_z_image_benchmark'],
                                    cwd=root, capture_output=True, text=True)
            failed = re.findall(r'^FAIL: (\w+)', result.stderr, re.M)
            assert result.returncode != 0 and failed == [case], result.stderr
            rows.append({'file': filename, 'case': case, 'mutation': [old, new],
                         'unique_application': True, 'failed_only_intended_case': True,
                         'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
                         'output_sha256': hashlib.sha256(result.stderr.encode()).hexdigest()})
    Path(destination).write_text(json.dumps({'mutations': rows}, indent=2) + '\n')
    print('5 unique benchmark mutations failed only their intended cases')


if __name__ == '__main__':
    import sys
    if '--mutation-report' in sys.argv:
        mutation_report(sys.argv[sys.argv.index('--mutation-report') + 1])
    else:
        unittest.main()
