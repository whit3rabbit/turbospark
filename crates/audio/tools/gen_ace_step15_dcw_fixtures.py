#!/usr/bin/env python3
"""Regenerate the ACE-Step 1.5 DCW (wavelet differential correction) fixtures.

Normative reference: upstream ACE-Step/ACE-Step-1.5 at
ca1e85fe9430179831e6bc6be790c332190a3866,
acestep/models/common/dcw_correction.py (`DCWCorrector.apply`) over
`dcw_primitives` with pytorch-wavelets 1.3.0 `DWT1DForward(J=1,
mode="zero", wave="haar")` / `DWT1DInverse`, CPU float32. Upstream copies
its acestep/models/turbo modeling file over the checkpoint's at load, and
that file applies DCW after every turbo step by default (mode "double",
scaler 0.05, high_scaler 0.02, haar).

    ../.venv-acestep/bin/python -I crates/audio/tools/gen_ace_step15_dcw_fixtures.py ../ACE-Step-1.5

Writes crates/audio/testdata/ace_step15_dcw/: per case `{name}_x.npy`,
`{name}_y.npy`, `{name}_out.npy` ([T, C] time-major) and cases.json.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parents[1] / "testdata" / "ace_step15_dcw"
UPSTREAM_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"


def load_dcw(upstream: Path):
    loguru = types.ModuleType("loguru")

    class _Logger:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    loguru.logger = _Logger()
    sys.modules["loguru"] = loguru
    for name, path in (
        ("acestep", upstream / "acestep"),
        ("acestep.models", upstream / "acestep" / "models"),
        ("acestep.models.common", upstream / "acestep" / "models" / "common"),
    ):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg
    common = upstream / "acestep" / "models" / "common"
    for mod in ("dcw_loader", "dcw_primitives", "dcw_correction"):
        full = f"acestep.models.common.{mod}"
        spec = importlib.util.spec_from_file_location(full, common / f"{mod}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
    return sys.modules["acestep.models.common.dcw_correction"]


# name, mode, scaler, high_scaler, t, frames, channels
CASES = [
    ("double_even", "double", 0.05, 0.02, 0.9, 16, 4),
    ("double_odd", "double", 0.05, 0.02, 0.5, 13, 4),
    ("double_t1", "double", 0.05, 0.02, 1.0, 10, 3),
    ("double_last", "double", 0.05, 0.02, 0.3, 9, 5),
    ("low_odd", "low", 0.1, 0.0, 0.75, 11, 4),
    ("high_even", "high", 0.1, 0.0, 0.25, 12, 4),
    ("pix", "pix", 0.07, 0.0, 0.6, 7, 3),
    ("double_strong", "double", 0.5, 0.3, 0.4, 15, 6),
]


def main(upstream: Path) -> None:
    dcw = load_dcw(upstream)
    OUT.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(29)
    meta = []
    files = []
    for name, mode, scaler, high, t, frames, ch in CASES:
        x = torch.randn(1, frames, ch, generator=gen)
        y = torch.randn(1, frames, ch, generator=gen)
        corrector = dcw.DCWCorrector(enabled=True, mode=mode, scaler=scaler, high_scaler=high, wavelet="haar")
        assert corrector.is_active
        out = corrector.apply(x, y, t)
        # A no-op means pytorch_wavelets failed to load: refuse.
        if mode != "pix":
            assert not torch.equal(out, x), f"{name}: DCW was a no-op (wavelets missing?)"
        for tag, arr in (("x", x), ("y", y), ("out", out)):
            fname = f"{name}_{tag}.npy"
            np.save(OUT / fname, np.ascontiguousarray(arr[0].numpy(), dtype=np.float32))
            files.append(fname)
        meta.append(dict(name=name, mode=mode, scaler=scaler, high_scaler=high, t=t, frames=frames, channels=ch))
    import pytorch_wavelets
    import pywt

    doc = {
        "generator": "tools/gen_ace_step15_dcw_fixtures.py",
        "upstream_commit": UPSTREAM_COMMIT,
        "pytorch_wavelets": getattr(pytorch_wavelets, "__version__", "1.3.0"),
        "pywt": pywt.__version__,
        "torch": torch.__version__,
        "cases": meta,
        "files": {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in files},
    }
    (OUT / "cases.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {len(meta)} cases to {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: gen_ace_step15_dcw_fixtures.py <ACE-Step-1.5 checkout>")
    main(Path(sys.argv[1]).resolve())
