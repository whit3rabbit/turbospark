#!/usr/bin/env python3
"""Build isolated Core ML crate copies to verify two async lifetime guards."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parent.parent
source = (root / 'crates/coreml/src/sys.rs').read_text()
mutations = [
    ('if !same {', 'if false && !same {',
     'sys::shared_plane::async_tests::callback_refuses_a_different_or_missing_output_backing'),
    ('while result.is_none() {', 'while false && result.is_none() {',
     'sys::shared_plane::async_tests::missing_completion_is_not_treated_as_cancellation'),
    ('let _compile = COMPILE_AND_PUBLISH\n        .lock()\n        .unwrap_or_else(|p| p.into_inner());',
     'let _compile = ();', 'sys::compile_serialization_tests::cold_publication_excludes_other_compilers_until_cache_is_ready'),
]
for anchor, replacement, test in mutations:
    assert anchor in source
    # Only mutate the first backing check, the async callback validator.
    changed = source.replace(anchor, replacement, 1)
    assert changed != source and replacement in changed
    with tempfile.TemporaryDirectory(prefix='ane-async-mutation-') as tmp:
        crate = Path(tmp) / 'coreml'
        shutil.copytree(root / 'crates/coreml', crate)
        manifest = (crate / 'Cargo.toml').read_text()
        values = {'version': '"0.4.0"', 'edition': '"2021"', 'rust-version': '"1.82"',
                  'license': '"MIT"', 'repository': '"https://github.com/whit3rabbit/turbospark"',
                  'publish': 'false'}
        for key, value in values.items():
            manifest = manifest.replace(f'\n{key}.workspace = true\n', f'\n{key} = {value}\n')
        (crate / 'Cargo.toml').write_text(manifest)
        (crate / 'src/sys.rs').write_text(changed)
        env = dict(os.environ, CARGO_TARGET_DIR=str(root / 'target/ane-mutations'))
        result = subprocess.run(['cargo', 'test', '--offline', '--manifest-path', str(crate / 'Cargo.toml'),
                                 '--features', 'ane-prefill', '--lib', test, '--', '--exact'],
                                capture_output=True, text=True, env=env, timeout=120)
        output = result.stdout + result.stderr
        if result.returncode == 0 or 'test result: FAILED' not in output:
            raise SystemExit(f'mutation did not fail its regression: {anchor}\n{output}')
        print(f'detected: {anchor} -> {replacement}: {test}')
