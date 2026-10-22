#!/usr/bin/env python3
"""Prove each withheld dependency detects its missing private wait."""
from pathlib import Path
import subprocess
import tempfile

source = (Path(__file__).parent / 'probe.m').read_text()
anchor = 'waitEvents:@[waitReady, waitANE]'
assert source.count(anchor) == 1, 'mutation anchor did not match exactly once'
for replacement, reason in [('waitEvents:@[waitANE]', 'consumer crossed stalled producer'),
                            ('waitEvents:@[waitReady]', 'consumer crossed delayed ANE dependency')]:
    changed = source.replace(anchor, replacement)
    assert changed != source and anchor not in changed
    with tempfile.TemporaryDirectory(prefix='ane-event-mutation-') as tmp:
        src, exe = Path(tmp) / 'mutated.m', Path(tmp) / 'probe'
        src.write_text(changed)
        subprocess.run(['xcrun', 'clang', '-fobjc-arc', '-fblocks', '-O2',
                        '-framework', 'Foundation', '-framework', 'Metal', '-framework', 'IOSurface',
                        str(src), '-o', str(exe)], check=True)
        result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
        if result.returncode != 3 or reason not in result.stderr:
            raise SystemExit(f'mutation not detected as expected: {replacement}: {result.returncode}: {result.stderr}')
        print(f'detected: {replacement}: {reason}')
