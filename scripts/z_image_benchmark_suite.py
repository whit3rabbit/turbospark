#!/usr/bin/env python3
"""Run quiet AC component pairs, temporarily pausing identified UI/photo work."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

from z_image_evidence import command, sha256


def candidates():
    selected = []
    for proc in psutil.process_iter(['pid', 'name', 'uids', 'cmdline', 'create_time', 'status']):
        i = proc.info
        if not i['uids'] or i['uids'].real != os.getuid() or i['status'] == psutil.STATUS_STOPPED:
            continue
        if '.hermes/' in ((i['cmdline'] or [''])[0]) or i['name'].startswith('Brave Browser') or i['name'] in ('WhatsApp', 'contactsd', 'mediaanalysisd', 'photolibraryd', 'photoanalysisd', 'corespotlightd', 'appstoreagent', 'triald', 'TrialArchivingService', 'cloudd', 'Codex (Renderer)') or (
                i['name'] == 'Codex (Service)' and '--type=gpu-process' in (i['cmdline'] or [])):
            selected.append({'pid': proc.pid, 'created': i['create_time'], 'name': i['name']})
    for row in list(selected):
        try:
            proc = psutil.Process(row['pid'])
            if '.hermes/' not in (proc.cmdline() or [''])[0]:
                continue
            for child in proc.children(recursive=True):
                if child.uids().real == os.getuid() and child.status() != psutil.STATUS_STOPPED:
                    selected.append({'pid': child.pid, 'created': child.create_time(), 'name': child.name()})
        except psutil.Error:
            pass
    return list({r['pid']: r for r in selected}.values())


def signal_same(rows, sig):
    for row in rows:
        try:
            proc = psutil.Process(row['pid'])
            if proc.create_time() == row['created']:
                proc.send_signal(sig)
        except psutil.NoSuchProcess:
            pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pause-display-and-photo-work', action='store_true', required=True)
    p.add_argument('--stages', nargs='+', choices=['encode', 'denoise', 'decode'], default=['encode', 'denoise', 'decode'])
    args = p.parse_args()
    if args.out.exists():
        p.error('use a new suite directory')
    if 'AC Power' not in command(['pmset', '-g', 'batt']):
        raise RuntimeError('AC power required')
    args.out.mkdir(parents=True)
    rows = candidates()
    receipt = {'paused': rows, 'commands': [], 'restored': False,
               'watchdog_timeout_s': 1800, 'script_sha256': sha256(__file__)}
    receipt_path = args.out / 'suite.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    parent = os.getpid()
    watchdog = os.fork()
    if watchdog == 0:
        # A detached restore path survives a killed supervisor or failed child.
        try:
            for _ in range(1800):
                time.sleep(1)
                if not psutil.pid_exists(parent):
                    break
            signal_same(rows, signal.SIGCONT)
        finally:
            os._exit(0)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    process = None
    try:
        signal_same(rows, signal.SIGSTOP)
        time.sleep(3)
        for stage in args.stages:
            copy = args.out / (stage + '-model.noindex')
            for cache in ('cold-copy', 'reuse'):
                label = stage + '-' + cache
                argv = ['/usr/bin/time', '-l', sys.executable, 'scripts/z_image_benchmark.py', stage,
                        '--cache', cache, '--copy', str(copy), '--out', str(args.out / (label + '.json'))]
                print('START', label, flush=True)
                with (args.out / (label + '.log')).open('w') as log:
                    process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    position = 0
                    while True:
                        try:
                            code = process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            code = None
                        with (args.out / (label + '.log')).open() as reader:
                            reader.seek(position)
                            chunk = reader.read()
                            position = reader.tell()
                        if chunk:
                            print(chunk, end='', flush=True)
                        if code is not None:
                            break
                receipt['commands'].append({'argv': argv, 'exit_code': code,
                                            'log_sha256': sha256(args.out / (label + '.log'))})
                receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
                if code:
                    raise RuntimeError(f'{label} failed, see its log')
                print('COMPLETE', label, flush=True)
            # Only delete copies created in this new suite directory.
            shutil.rmtree(copy)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        signal_same(rows, signal.SIGCONT)
        receipt['restored'] = True
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        os.kill(watchdog, signal.SIGTERM)
        os.waitpid(watchdog, 0)


if __name__ == '__main__':
    main()
