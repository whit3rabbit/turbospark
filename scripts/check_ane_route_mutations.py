#!/usr/bin/env python3
"""Check that replay regressions are observable, without editing source."""
from pathlib import Path
import subprocess
import tempfile
root = Path(__file__).resolve().parent
source = (root / 'ane_moe_route_replay.py').read_text()
tests = (root / 'test_ane_moe_route_replay.py').read_text()
for before, after, test in [
    ('n >= t', 'n > t', 'test_threshold_boundary_and_conservation'),
    ('"executable": window is None', '"executable": True', 'test_windows_are_hypothetical_and_do_not_invent_io'),
]:
    assert before in source
    changed = source.replace(before, after)
    assert changed != source and before not in changed
    with tempfile.TemporaryDirectory(prefix='ane-replay-mutation-') as tmp:
        Path(tmp, 'ane_moe_route_replay.py').write_text(changed)
        Path(tmp, 'test_ane_moe_route_replay.py').write_text(tests)
        result = subprocess.run(['python3', '-m', 'unittest', f'test_ane_moe_route_replay.ReplayTests.{test}'],
                                cwd=tmp, capture_output=True, text=True)
        if result.returncode != 1 or 'FAIL:' not in result.stderr:
            raise SystemExit(f'mutation not detected: {before}: {result.stderr}')
        print(f'detected: {before} -> {after}')
