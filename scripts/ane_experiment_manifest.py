#!/usr/bin/env python3
"""Write experiment provenance. Hash requested artifacts, never credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def command(*args):
    r = subprocess.run(args, capture_output=True, text=True, check=False)
    return {"exit": r.returncode, "output": r.stdout.strip(), "error": r.stderr.strip()}


def digest(path):
    path = Path(path).resolve(strict=True)
    if path.is_file():
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}
    files = [digest(f) for f in sorted(path.rglob('*')) if f.is_file()]
    h = hashlib.sha256()
    for f in files:
        h.update((str(Path(f['path']).relative_to(path)) + '\0' + f['sha256'] + '\n').encode())
    return {"path": str(path), "tree_sha256": h.hexdigest(), "files": files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', action='append', default=[])
    p.add_argument('--cache-state', choices=['warm', 'cold', 'unknown'], required=True)
    p.add_argument('--expert-cache-slots', type=int, help='omit when no expert cache exists')
    p.add_argument('--setting', action='append', default=[], help='explicit resolved key=value; no secrets')
    a = p.parse_args()
    if a.expert_cache_slots is not None and a.expert_cache_slots < 1:
        p.error('expert-cache-slots must be positive')
    settings = dict(s.split('=', 1) for s in a.setting)
    diff = subprocess.run(['git', 'diff', '--binary'], capture_output=True, check=True).stdout
    result = {"version": 1, "source": command('git', 'rev-parse', 'HEAD'),
              "branch": command('git', 'branch', '--show-current'),
              "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
              "dirty": command('git', 'status', '--porcelain'),
              "rust": command('rustc', '-Vv'), "cargo": command('cargo', '-V'),
              "os": command('sw_vers'), "hardware": command('sysctl', '-n', 'machdep.cpu.brand_string'),
              "memory": command('sysctl', '-n', 'hw.memsize'), "xcode": command('xcodebuild', '-version'),
              "metal": command('system_profiler', 'SPDisplaysDataType'),
              "power": command('pmset', '-g', 'batt'), "cache_state": a.cache_state,
              "expert_cache_slots": a.expert_cache_slots, "resolved_settings": settings,
              "artifacts": [digest(f) for f in a.artifact]}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
