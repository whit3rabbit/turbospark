#!/usr/bin/env python3
"""Compile standalone research, then run with a process-level watchdog."""
from pathlib import Path
import subprocess
import hashlib
import json
import sys
import tempfile

root = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='ane-events-') as temp:
    binary = str(Path(temp) / 'probe')
    subprocess.run(['xcrun', 'clang', '-fobjc-arc', '-fblocks', '-O2',
                    '-framework', 'Foundation', '-framework', 'Metal', '-framework', 'IOSurface',
                    str(root / 'probe.m'), '-o', binary], check=True)
    print(json.dumps({'source_sha256': hashlib.sha256((root / 'probe.m').read_bytes()).hexdigest(),
                      'binary_sha256': hashlib.sha256(Path(binary).read_bytes()).hexdigest()}), file=sys.stderr)
    try:
        result = subprocess.run([binary], timeout=60)
    except subprocess.TimeoutExpired:
        raise SystemExit('watchdog killed stalled probe; no device event was fabricated')
    raise SystemExit(result.returncode)
