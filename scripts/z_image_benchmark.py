#!/usr/bin/env python3
"""Sequential component benchmark, with explicit inputs and observed cache state."""

import argparse
import ctypes
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import numpy as np
import torch
from diffusers import AutoencoderKL, ZImageTransformer2DModel
from transformers import AutoModel, AutoTokenizer

from z_image_capture import PROMPTS, pipeline, verify_weights
from z_image_evidence import (background_delta, background_snapshot, check_environment,
                              command, preflight, sha256, validate_capture)


class Resources:
    def __init__(self):
        self.rows = []
        self.stop = threading.Event()
        self.lib = ctypes.CDLL('/usr/lib/libproc.dylib')

    def point(self):
        buf = ctypes.create_string_buffer(1024)
        if self.lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(buf)) != 0:
            raise RuntimeError('process resource counter unavailable')
        return {'time': time.monotonic(),
                'phys_footprint_bytes': ctypes.c_uint64.from_buffer(buf, 72).value,
                'physical_read_bytes': ctypes.c_uint64.from_buffer(buf, 144).value,
                'mps_live_bytes': torch.mps.current_allocated_memory(),
                'mps_driver_bytes': torch.mps.driver_allocated_memory()}

    def loop(self):
        try:
            while not self.stop.wait(.1):
                self.rows.append(self.point())
        except Exception as exc:
            self.error = str(exc)

    def measure(self, name, fn):
        torch.mps.synchronize()
        start = self.point()
        t = time.perf_counter()
        value = fn()
        torch.mps.synchronize()
        elapsed = time.perf_counter() - t
        end = self.point()
        rows = [start, *[r for r in self.rows if start['time'] <= r['time'] <= end['time']], end]
        return value, {'name': name, 'elapsed_s': elapsed, 'start': start, 'end': end,
                       'sampled_peak_phys_footprint_bytes': max(r['phys_footprint_bytes'] for r in rows),
                       'sampled_peak_mps_live_bytes': max(r['mps_live_bytes'] for r in rows),
                       'sampled_peak_mps_driver_bytes': max(r['mps_driver_bytes'] for r in rows),
                       'physical_read_bytes': end['physical_read_bytes'] - start['physical_read_bytes']}

    def __enter__(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        if hasattr(self, 'error'):
            raise RuntimeError(self.error)


class Background:
    def __init__(self):
        self.rows = []
        self.stop = threading.Event()

    def sample(self):
        after = background_snapshot()
        self.rows.append({'time': after[0], 'window_start': self.previous[0], 'window_end': after[0],
                          'cpu': background_delta(self.previous, after),
                          'power': command(['pmset', '-g', 'batt'])})
        self.previous = after

    def loop(self):
        try:
            while not self.stop.wait(1):
                self.sample()
        except Exception as exc:
            self.error = str(exc)

    def __enter__(self):
        self.previous = background_snapshot()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        if hasattr(self, 'error'):
            raise RuntimeError(self.error)
        # Cover the final timed interval without a noisy sub-second CPU sample.
        time.sleep(max(0, 1 - (time.monotonic() - self.previous[0])))
        self.sample()


def prepare_copy(source, dest, component, inventory):
    if dest.exists():
        raise ValueError('cold copy must use a new directory')
    dest.mkdir(parents=True)
    copied = {}
    for path in source.rglob('*'):
        relative = path.relative_to(source)
        target = dest / relative
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif relative.as_posix() in inventory['weights'] and relative.parts[0] == component:
            # F_NOCACHE writes establish a new uncached inode, without evicting
            # unrelated applications' cache. Actual reads decide cache classification.
            digest = hashlib.sha256()
            with path.open('rb') as src, target.open('xb') as out:
                fcntl.fcntl(out.fileno(), 48, 1)
                while block := src.read(8 << 20):
                    digest.update(block)
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            key = relative.as_posix()
            if digest.hexdigest() != inventory['weights'][key]['sha256']:
                raise ValueError('canonical copy checksum mismatch')
            copied[key] = digest.hexdigest()
        else:
            target.symlink_to(path.resolve())
    return copied


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['encode', 'denoise', 'decode'])
    p.add_argument('--model', type=Path, default=Path('target/ig0/model'))
    p.add_argument('--reference', type=Path, default=Path('target/ig0/runs/lighting'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cache', choices=['cold-copy', 'reuse'], required=True)
    p.add_argument('--copy', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if args.out.exists() or args.repeats != 3:
        p.error('use a new output path and exactly three measured resident repetitions')
    packages = check_environment()
    ref = validate_capture(args.reference / (args.stage + '.json'))
    invpath = Path('docs/verification/z-image-ig0-inputs.json')
    inv = json.loads(invpath.read_text())
    component = {'encode': 'text_encoder', 'denoise': 'transformer', 'decode': 'vae'}[args.stage]
    if args.cache == 'cold-copy':
        verified = prepare_copy(args.model, args.copy, component, inv)
    else:
        # Verification follows measurement: hashing here would warm the load arm.
        verified = {k: v['sha256'] for k, v in inv['weights'].items() if k.startswith(component + '/')}
    inputs = {k: np.load(args.reference / (k + '.npy')) for k in
              ({'encode': [], 'denoise': ['conditioning', 'initial_noise'], 'decode': ['final_latents']}[args.stage])}
    preflight_attempts = []
    for attempt in range(10):
        try:
            before = preflight()
            break
        except RuntimeError as exc:
            preflight_attempts.append(str(exc))
            if 'quiet background load' not in str(exc) or attempt == 9:
                raise
            print('Waiting for quiet background load', flush=True)
    swap_start = command(['sysctl', 'vm.swapusage'])
    thermal_start = command(['pmset', '-g', 'therm'])
    phases = []
    def load():
        if args.stage == 'encode':
            model = AutoModel.from_pretrained(args.copy / 'text_encoder', dtype=torch.bfloat16,
                                             local_files_only=True).eval().to('mps')
            model.config.use_cache = False
            pipe = pipeline(args.copy)
            pipe.text_encoder = model
            pipe.tokenizer = AutoTokenizer.from_pretrained(args.copy / 'tokenizer', local_files_only=True)
        elif args.stage == 'denoise':
            model = ZImageTransformer2DModel.from_pretrained(args.copy, subfolder='transformer',
                          torch_dtype=torch.bfloat16, local_files_only=True).eval().to('mps')
            pipe = pipeline(args.copy)
            pipe.transformer = model
        else:
            model = AutoencoderKL.from_pretrained(args.copy, subfolder='vae', torch_dtype=torch.float32,
                                                  local_files_only=True).eval().to('mps')
            pipe = None
        if pipe is not None:
            pipe.set_progress_bar_config(disable=True)
        return model, pipe
    with Background() as background, Resources() as resources, torch.inference_mode():
        (model, pipe), row = resources.measure('load', load)
        phases.append(row)
        print('load', row['elapsed_s'], flush=True)
        weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        calls = []
        if args.stage == 'denoise':
            model.register_forward_hook(lambda *unused: calls.append(1))
        def execute():
            if args.stage == 'encode':
                return pipe.encode_prompt([PROMPTS['lighting']], device='mps', do_classifier_free_guidance=False)[0][0]
            if args.stage == 'denoise':
                return pipe(prompt_embeds=[torch.from_numpy(inputs['conditioning']).to('mps', torch.bfloat16)],
                            latents=torch.from_numpy(inputs['initial_noise']).to('mps'), height=1024, width=1024,
                            num_inference_steps=9, guidance_scale=0, output_type='latent').images
            return model.decode(torch.from_numpy(inputs['final_latents']).to('mps') / .3611 + .1159,
                                return_dict=False)[0]
        expected_name = {'encode': 'conditioning', 'denoise': 'final_latents', 'decode': 'decoded_pixels'}[args.stage]
        expected = np.load(args.reference / (expected_name + '.npy'))
        for i in range(args.repeats + 1):
            count_before = len(calls)
            output, row = resources.measure('first_execution' if i == 0 else f'resident_{i}', execute)
            # Verification and CPU copies are outside each execution timer.
            value = output.float().cpu().numpy()
            row['exact_reference_match'] = bool(np.array_equal(value, expected))
            row['finite'] = bool(np.isfinite(value).all())
            row['output_bytes'] = value.nbytes
            if args.stage == 'denoise':
                row['actual_forwards'] = len(calls) - count_before
            if not row['exact_reference_match'] or not row['finite'] or (args.stage == 'denoise' and row['actual_forwards'] != 9):
                raise ValueError('benchmark numerical validation failed')
            phases.append(row)
            print(row['name'], row['elapsed_s'], 'exact', row['exact_reference_match'], flush=True)
            del output, value
        del model, pipe
        gc.collect()
        torch.mps.synchronize()
        retained = resources.point()
        torch.mps.empty_cache()
        torch.mps.synchronize()
        released = resources.point()
    swap_end = command(['sysctl', 'vm.swapusage'])
    # Verify every consumed component byte after measurement as well.
    actual = verify_weights(args.copy, invpath, component)
    if actual != verified:
        raise ValueError('benchmark payload provenance mismatch')
    report = {'stage': args.stage, 'cache_arm': args.cache, 'component': component,
              'source_capture_sha256': sha256(args.reference / (args.stage + '.json')),
              'script_sha256': sha256(__file__), 'packages': packages, 'host': {'chip': command(['sysctl', '-n', 'machdep.cpu.brand_string']),
              'ram_bytes': int(command(['sysctl', '-n', 'hw.memsize'])), 'macos': command(['sw_vers'])}, 'verified_weights': actual,
              'stored_weight_bytes': sum((args.copy / k).stat().st_size for k in verified),
              'resident_parameter_bytes': weight_bytes, 'input_tensor_bytes': {k: v.nbytes for k, v in inputs.items()},
              'preflight': before, 'preflight_attempts': preflight_attempts, 'background': background.rows, 'phases': phases,
              'retained_after_delete': retained, 'after_empty_cache': released,
              'swap_start': swap_start, 'swap_end': swap_end, 'thermal_start': thermal_start,
              'thermal_end': command(['pmset', '-g', 'therm']),
              'resources': resources.rows, 'sampling_interval_s': .1,
              'benchmark_eligible': bool(background.rows) and all(r['cpu']['quiet'] and 'AC Power' in r['power'] for r in background.rows),
              'scratch_bytes': None, 'scratch_note': 'MPS does not expose exact per-operator scratch; driver-minus-live is not scratch.',
              'cold_method': 'new inode written with macOS F_NOCACHE; classify by measured load physical reads',
              'settings': {'case': 'lighting', 'width': 1024, 'height': 1024, 'forwards': 9, 'guidance': 0,
                           'text_transformer_dtype': 'bfloat16', 'vae_dtype': 'float32'}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(args.out, report['benchmark_eligible'], flush=True)


if __name__ == '__main__':
    main()
