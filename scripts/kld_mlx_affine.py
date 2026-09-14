#!/usr/bin/env python3
"""Cross-engine KL for the MLX-AFFINE families: this port against MLX on the
identical bytes (ROADMAP's 1-bit entry step 5, and its ternary entry).

It said SUB-4-BIT while its only entries were 1- and 2-bit, which was a true
statement about two checkpoints rather than a limit of the driver: nothing
below is narrower than "MLX affine at some width", and `ornith-35b-4bit` is
4- and 8-bit mixed. See `CHECKPOINTS`.

    TURBOSPARK_QWEN35_INSTALL_DIR=~/models/bonsai27b.gturbo \
    TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/bonsai-warm \
      cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
    /tmp/prism-venv/bin/python scripts/kld_mlx_affine.py /tmp/kld/bonsai-warm bonsai-1bit

    TURBOSPARK_TERNARY_INSTALL_DIR=~/models/ternary27b.gturbo \
    TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/ternary-warm \
      cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
    uv run --python 3.12 --with 'mlx-lm==0.31.2' --with numpy \
      scripts/kld_mlx_affine.py /tmp/kld/ternary-warm ternary-2bit

    TURBOSPARK_ORNITH35B_INSTALL_DIR=~/models/ornith35b.gturbo \
    TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/ornith35b-warm \
      cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
    uv run --python 3.12 --with mlx-lm --with numpy \
      scripts/kld_mlx_affine.py /tmp/kld/ornith35b-warm ornith-35b-4bit

    # NEVER RUN: needs `~/models/qwen36.gturbo` re-streamed first (cleared
    # 2026-08-15). Everything else about the row is pinned; see `CHECKPOINTS`.
    TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
    TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/qwen36-warm \
      cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
    uv run --python 3.12 --with mlx-lm --with numpy \
      scripts/kld_mlx_affine.py /tmp/kld/qwen36-warm qwen36

**THE CHECKPOINT NAME IS REQUIRED, NOT DEFAULTED, and that is the whole
reason this file is parameterized rather than copied.** A model-specific
constant in a measurement script is a wrong ANSWER waiting for its second
caller (AGENTS.md Gotcha 38), and this one has a worse failure than most:
point the ternary dump at the 1-bit reference and every check below still
passes -- the reference really is 1-bit throughout, the shapes agree, and the
only symptom is a big divergence that reads as a kernel bug. That is the same
shape as `kld_llamacpp.py`'s cache-name bug, which silently compared one
model's logits against another's because every arm is the same size.

**THE ENVIRONMENT DIFFERS PER WIDTH AND IT IS MEASURED, NOT ASSUMED.**
Upstream `mlx` refuses `bits=1` at the API LEVEL -- not merely on Metal -- so
`mx.quantize` reports "The supported bits are 2, 3, 4, 5, 6 and 8" and the
1-bit reference has to be `github.com/PrismML-Eng/mlx@prism` built from
source. TWO bits is in that list, so the ternary arm runs under an ordinary
`uv run --with mlx-lm` ephemeral env like `kld.py`'s. `require_the_mlx_build`
probes for the width it is about to use rather than checking a version
string, so either environment is accepted when it can actually do the work.

**THE REFERENCE HAD A NORM BUG ON THIS EXACT FAMILY UNTIL 2026-08-18, AND IT
IS THE `+1` AGAIN.** mlx-lm's `qwen3_5.py::sanitize` shifted every RMSNorm
weight by 1.0 when the checkpoint carried `mtp.*` weights OR had an
unsanitized conv1d. A conversion that KEEPS its MTP head while already
storing conv1d in MLX layout therefore got the shift applied a SECOND time,
to norms the converter had already shifted. Fixed in `4eeaf20` ("Fix Qwen3.6
converted norm sanitization", #1623) by dropping the `mtp.*` clause and
keeping the raw-conv1d layout as the only signal; the upstream test is named
`..._norm_not_shift_twice`. That fix is in git main (0.32.0) and NOT in
0.31.3, which is what `uv run --with mlx-lm` still resolves.

EVERY ROW BELOW IS UNAFFECTED, and it is worth saying why rather than
leaving it to luck: the clause keys on `mtp.*` being PRESENT, and all three
mlx conversions here drop it (`qwen36` 0 of 2,090 tensors and
`ornith-35b-4bit` 0, both read off the published index). So the two mlx-lm
versions do the same thing to these checkpoints. A reference that RETAINS a
head -- `mlx-community/Qwen3.8-27B-MTP-4bit`,
`scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp` -- is the case that needs 0.32.0,
and those are precisely the checkpoints an MTP drafter comparison would
reach for. Pin the build before believing such a number, because a
double-shifted norm is a plausible-looking reference, not a crashing one.

Note this port hit the MIRROR of that bug from the other side (AGENTS.md
Gotcha 50's second instance): it read the MTP head's CENTERED q/k norms
plainly and put the true token at median rank 248,308 of 248,320. Same
convention, same family, opposite direction -- which is the reason to treat
`+1` on this family as a place where both engines have been wrong.

**THIS IS A SEPARATE DRIVER FROM `kld.py` AND NOT A WIDENING OF IT**, which
is a decision rather than duplication. `kld.py` runs mlx-lm out of a `uv run
--with mlx-lm` ephemeral environment, and that environment cannot serve this
model at all: upstream `mlx` refuses `bits=1` at the API level -- not merely
on Metal -- so `mx.quantize` reports "The supported bits are 2, 3, 4, 5, 6
and 8" and a forward pass is impossible on any device. The reference here is
`github.com/PrismML-Eng/mlx@prism`, which the checkpoint's own README names,
BUILT FROM SOURCE into a venv. A script that had to be handed an interpreter
is a different script from one that provisions its own.

Everything that is not the environment is IMPORTED from `kld.py` rather than
restated -- `divergences`, `perplexity` -- exactly as `kld_llamacpp.py` does.
The three things this file owns are the per-artifact pins, the environment
note above, and the reading below.

TWO FLOORS, NOT ONE, and this family can only have one of them (bench crate
Gotcha 8, AGENTS.md Gotcha 34). The SHAPE floor is mlx-lm against itself,
cached against batched, which holds weights and kernels and engine fixed and
varies only the forward shape; it is measured here. The BACKEND floor -- the
same engine's CPU against its Metal -- is NOT: this model is DENSE, so every
one of its 24.8B backbone weights is read per token, where the families that
have a cheap CPU arm (`qwen3moe`, `gpt-oss`) activate 3B. Its absence is
reported rather than skipped silently, because a missing floor is a missing
scale and the number above it cannot be read as small without one.

AND THE PORT NARROWS SOMETHING MLX DOES NOT (AGENTS.md Gotcha 45): every
unquantized tensor in BOTH checkpoints is F16 and this port stores them BF16,
which loses 19.5% of the norm values at up to 2^-8 relative. That is a real
difference between the two engines' weights, it is on the port's side of this
comparison, and it is the first thing to suspect if the headline lands high.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kld import divergences, perplexity  # noqa: E402

# Pinned in the matching `crates/repack/tests/*_checkpoint_network.rs`: the
# exact repo and commit each install was streamed from. Same quantized bytes
# on both sides is the whole point.
#
# `widths` is the checkpoint's quantization COMPOSITION: how many modules sit
# at each `(bits, group_size)`, counted from the header's `.scales` companions
# and cross-checked against the loaded reference by `assert_reference_matches`.
#
# IT IS A MAP RATHER THAN A SCALAR BECAUSE A CHECKPOINT CAN CARRY TWO WIDTHS.
# It was `bits` + `group_size` + `modules` while every entry here was uniform,
# which is a true statement about two checkpoints and not a property of the
# format: `ornith-35b-4bit` lifts its ROUTER and its shared-expert gate to 8
# bits on all 40 layers, exactly as Gemma and Qwen 3.6 do, so a uniform-width
# guard refuses a perfectly good reference. A map is also strictly STRONGER
# than what it replaces -- it pins the composition, where the old pair pinned
# a total plus a claim that every module agreed with it.
CHECKPOINTS = {
    "bonsai-1bit": {
        "repo": "prism-ml/Bonsai-27B-mlx-1bit",
        "revision": "ef22f239c670078e1507f9769bcaa66657332b96",
        "widths": {(1, 128): 498},
        "backend_floor_note": (
            "not measured (dense 27B; a CPU arm reads all 24.8B backbone "
            "weights per token)"
        ),
    },
    "ternary-2bit": {
        "repo": "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
        "revision": "70f75f3ad081ab840a42f3304c02c27e7f89bfb7",
        "widths": {(2, 128): 498},
        "backend_floor_note": (
            "not measured (dense 27B; a CPU arm reads all 24.8B backbone "
            "weights per token)"
        ),
    },
    # Qwen 3.6 35B-A3B: the OTHER checkpoint of `qwen_gdn_moe_35b_a3b()`, and
    # the row `docs/BENCHMARKS.md` has been describing as blocked since Phase
    # Q -- "`logit_dump.rs` accepts `TURBOSPARK_QWEN36_INSTALL_DIR` and would
    # produce one, but `kld.py`'s reference is pinned to the Gemma repo".
    # That sentence names the wrong obstacle twice over. `kld.py`'s pin is
    # gone, and this checkpoint would not have belonged there anyway: it is an
    # MoE and that driver has no reference guard, which is the one thing an
    # MoE reference needs (see `assert_reference_matches`).
    #
    # NEVER RUN. `~/models/qwen36.gturbo` was cleared 2026-08-15, so nothing
    # here has been through a forward pass. Everything that could be settled
    # without the install was, off the published config and index:
    #   - `model_type: qwen3_5_moe`, which resolves to
    #     `mlx_lm/models/qwen3_5_moe.py` -- a thin `sanitize`-only subclass of
    #     `qwen3_5.py`, i.e. the SAME reference module `ornith-35b-4bit` below
    #     already runs through. Read, not guessed: mlx-lm has no module named
    #     for this family's own string.
    #   - 512 `.scales`, 80 of them at 8 bits (`mlp.gate` and
    #     `mlp.shared_expert_gate` on all 40 layers), so 432 at 4 bits.
    #   - its 333 VISION tensors carry ZERO `.scales` and `sanitize` strips
    #     them, so they do not enter the count. That check is the reason this
    #     map can be trusted before the run: Ornith's conversion has no vision
    #     tower at all, so copying its numbers across would have been a guess
    #     that happened to be right.
    "qwen36": {
        "repo": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "revision": "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        "widths": {(4, 64): 432, (8, 64): 80},
        # NOT copied from the row below, though the number would be. That one
        # is a MEASUREMENT of mlx's CPU path on a different checkpoint; this
        # is an expectation, and saying so is the difference between the two.
        "backend_floor_note": (
            "not measured (no install on disk; expect it to be unaffordable "
            "for `ornith-35b-4bit`'s measured reason -- same architecture, "
            "and mlx's CPU backend runs that one at 15.4 s/position)"
        ),
    },
    # Ornith-1.5-35B-A3B, the MoE half, and the first MIXED-width entry.
    # 432 modules at 4 bits plus the 80 eight-bit ones named above; 512 in
    # total, which is the header's `.scales` count.
    "ornith-35b-4bit": {
        "repo": "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
        "revision": "19504d912fa8fc7622bf6b1de3db5d5d890b1f02",
        "widths": {(4, 64): 432, (8, 64): 80},
        # MEASURED before being written down, and the expectation was wrong:
        # only 3B of 35B are active per token, so by the reasoning that makes
        # `qwen3moe`'s llama.cpp CPU arm cost ~40 s this one should have been
        # cheap. mlx's CPU backend runs it at 15.4 s/position, i.e. ~2.5 h for
        # the corpus. Affordability of a backend floor is a property of the
        # REFERENCE ENGINE's CPU path, not of the model's active-parameter
        # count.
        "backend_floor_note": (
            "not measured (mlx's CPU backend runs this at 15.4 s/position, "
            "~2.5 h for the corpus, despite only 3B of 35B active)"
        ),
    },
}


def snapshot_dir(spec) -> pathlib.Path:
    repo, revision = spec["repo"], spec["revision"]
    path = (
        pathlib.Path.home()
        / ".cache/huggingface/hub"
        / f"models--{repo.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    if not (path / "config.json").exists():
        sys.exit(
            f"reference checkpoint not found at {path}\n"
            f"  hf download {repo} --revision {revision}"
        )
    return path


def require_the_mlx_build(spec) -> str:
    """Refuse to run under an mlx that cannot represent this checkpoint.

    At ONE bit upstream mlx does not merely lack a Metal kernel, it rejects
    the argument outright, so the failure without this check is an exception
    thrown somewhere inside model loading -- far from the cause. At TWO bits
    upstream is fine and this passes on any recent build. Checked by ASKING
    mlx to do the thing rather than by comparing a version string: the fork's
    version is `0.31.2.dev...`, which is neither newer nor distinguishable
    from upstream's by ordering.
    """
    import mlx.core as mx

    # EVERY width the checkpoint carries, not just its dominant one: a mixed
    # reference is only representable if mlx can do all of them, and finding
    # that out here beats finding it out inside model loading.
    for bits, group in sorted(spec["widths"]):
        try:
            w = mx.random.normal((256, 256))
            q, s, b = mx.quantize(w, group_size=group, bits=bits)
            mx.eval(mx.quantized_matmul(mx.random.normal((1, 256)), q, s, b,
                                        transpose=True, group_size=group, bits=bits))
        except Exception as exc:  # noqa: BLE001 -- the message IS the diagnosis
            sys.exit(
                f"this mlx cannot do bits={bits} ({type(exc).__name__}: {exc})\n"
                "  at one bit that needs github.com/PrismML-Eng/mlx@prism, built from source:\n"
                "    git clone -b prism https://github.com/PrismML-Eng/mlx.git\n"
                "    uv venv /tmp/prism-venv --python 3.12\n"
                "    uv pip install --python /tmp/prism-venv/bin/python cmake ninja setuptools nanobind\n"
                "    uv pip install --python /tmp/prism-venv/bin/python -e mlx/ --no-build-isolation\n"
                "    uv pip install --python /tmp/prism-venv/bin/python 'mlx-lm==0.31.2' transformers numpy"
            )
    return f"{mx.__version__} ({mx.__file__})"


def assert_reference_matches(model, spec) -> dict:
    """The reference must be running the PACKED weights, not a widened copy,
    and it must be running the WIDTH this dump was taken at.

    Without this the comparison could silently become "this port's 1-bit
    kernels against MLX's fp16 kernels on dequantized weights", which is a
    different question with the same shape of answer -- and the tell would be
    a suspiciously SMALL divergence, i.e. the direction nobody investigates.

    Counted rather than spot-checked, and the COMPOSITION is the cross-check:
    the checkpoint's safetensors header says how many tensors carry a
    `.scales` companion at each width, so the same map here is the whole
    model and one fewer at any width is a layer that quietly did not
    quantize.

    Comparing the whole map is also what catches a dump paired with the WRONG
    reference -- the two 27B checkpoints of that architecture have the same
    module count and the same shapes, and differ only in their width.

    The map is keyed on `(bits, group_size)` and NOT on the module type. A
    header cannot tell a QuantizedLinear from a QuantizedEmbedding either, so
    keying on it would be asserting something the evidence does not contain;
    the type still reaches the report, which is where it is useful.

    **MODULES ARE FOUND BY DUCK TYPING, NOT BY A CLASS LIST, AND THAT IS THE
    FIX FOR A REAL MISCOUNT.** This used to test
    `isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding))`, which is
    complete for a DENSE checkpoint and silently misses an MoE one: mlx packs
    each layer's routed experts into a `QuantizedSwitchLinear`, a THIRD type
    that lives in `mlx_lm.models.switch_layers` rather than `mlx.nn`. On
    `ornith-35b-4bit` that is 40 layers x 3 roles = 120 modules, so the guard
    saw 312 of 432 and refused a perfectly good reference. Naming the classes
    would have had to be revised again for the next one; asking each module
    whether it carries a `(bits, group_size)` pair asks the question the
    header's `.scales` count actually answers. Over-counting is not a hazard
    here because the comparison is an EQUALITY against that count, so a
    spurious module fails just as loudly as a missing one.
    """
    want: dict[tuple, int] = dict(spec["widths"])
    seen: dict[tuple, int] = {}
    detail: dict[tuple, int] = {}
    for _, module in model.named_modules():
        bits = getattr(module, "bits", None)
        group = getattr(module, "group_size", None)
        if bits is None or group is None:
            continue
        seen[(bits, group)] = seen.get((bits, group), 0) + 1
        full = (type(module).__name__, bits, group)
        detail[full] = detail.get(full, 0) + 1
    if seen != want:
        sys.exit(
            "reference model's quantization composition does not match the "
            f"checkpoint header.\n  expected {dict(sorted(want.items()))}\n"
            f"  observed {dict(sorted(seen.items())) or 'no quantized modules'}\n"
            f"  by module type: {dict(sorted(detail.items(), key=str))}\n"
            "  a SHORTFALL at one width is usually a module type this walk did "
            "not recognise, not a reference that failed to quantize."
        )
    return {f"{name}(bits={b},group={g})": n for (name, b, g), n in sorted(detail.items(), key=str)}


def assert_dump_matches_checkpoint(meta: dict, spec: dict) -> None:
    """Bind the dump's install identity to the selected MLX checkpoint."""
    install = pathlib.Path(meta["install"])
    manifest_path = install / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"cannot read dump install identity from {manifest_path}: {exc}")

    installed_repo = manifest.get("modelID")
    if installed_repo != spec["repo"]:
        sys.exit(
            "dump install does not match the selected reference checkpoint.\n"
            f"  dump install: {installed_repo or 'missing modelID'}\n"
            f"  selected reference: {spec['repo']}"
        )

    receipt_path = install / "verified-install.json"
    if not receipt_path.exists():
        return
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"cannot read dump install provenance from {receipt_path}: {exc}")
    source_repo = receipt.get("sourceRepoId")
    source_revision = receipt.get("sourceRevision")
    if source_repo is not None and source_repo != spec["repo"]:
        sys.exit(
            f"dump receipt source repo {source_repo} does not match {spec['repo']}"
        )
    if source_revision is not None and source_revision != spec["revision"]:
        sys.exit(
            "dump receipt source revision does not match the selected reference.\n"
            f"  dump revision: {source_revision}\n"
            f"  selected revision: {spec['revision']}"
        )


def mlx_logits(token_ids: list[int], cached: bool, spec) -> tuple[np.ndarray, dict]:
    """mlx-lm's next-token logits per position, as float32 [rows, vocab].

    Row i is the logits after consuming `token_ids[i]`, matching what
    `logit_dump.rs` writes, so the last position is dropped. `cached` picks
    the forward SHAPE: True steps one token at a time through a prompt cache
    (this port's shape, and the headline), False runs one batched pass.
    """
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, _ = load(str(snapshot_dir(spec)))
    quantization = assert_reference_matches(model, spec)
    if not cached:
        out = model(mx.array([token_ids]))
        mx.eval(out)
        return np.array(out[0, :-1, :].astype(mx.float32)), quantization

    cache = make_prompt_cache(model)
    rows = []
    for token in token_ids[:-1]:
        out = model(mx.array([[token]]), cache=cache)
        mx.eval(out)
        rows.append(np.array(out[0, -1, :].astype(mx.float32)))
    return np.stack(rows), quantization


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[2] not in CHECKPOINTS:
        sys.exit(
            f"usage: {sys.argv[0]} <dump-dir> <{'|'.join(CHECKPOINTS)}>\n"
            "  the checkpoint name is REQUIRED; see the module doc for why"
        )
    dump = pathlib.Path(sys.argv[1])
    name = sys.argv[2]
    spec = CHECKPOINTS[name]
    meta = json.loads((dump / "meta.json").read_text())
    rows, vocab, ids = meta["rows"], meta["vocab_size"], meta["token_ids"]
    assert_dump_matches_checkpoint(meta, spec)

    mlx_build = require_the_mlx_build(spec)

    port = np.fromfile(dump / "logits.f16", dtype=np.float16)
    if port.size != rows * vocab:
        sys.exit(f"{dump}/logits.f16 holds {port.size} values, expected {rows * vocab}")
    port = port.reshape(rows, vocab).astype(np.float32)

    cached, quantization = mlx_logits(ids, cached=True, spec=spec)
    batched, _ = mlx_logits(ids, cached=False, spec=spec)
    for name, arr in (("cached", cached), ("batched", batched)):
        if arr.shape != port.shape:
            sys.exit(f"mlx {name} returned {arr.shape}, this port dumped {port.shape}")

    # REPORTED, never checked against an invented bound. This family declares
    # no logit softcap, so there is no transform for the two heads to
    # disagree about -- AGENTS.md Gotcha 38's lesson from the first
    # non-Gemma caller of `kld_llamacpp.py`, which aborted a correct run on a
    # bound that was Gemma's `final_logit_softcapping` all along.
    maxima = {
        "turbospark": float(np.abs(port).max()),
        "mlx_cached": float(np.abs(cached).max()),
    }

    first = meta["first_scored_position"]
    report = {
        "rows": rows,
        "mlx_build": mlx_build,
        "reference_quantization": quantization,
        # This port against MLX in the SAME forward shape. The headline.
        "kl_vs_mlx": divergences(cached, port),
        # MLX against itself across its two shapes: the SHAPE floor.
        "kl_mlx_self": divergences(batched, cached),
        # Named rather than omitted: see the module doc.
        "backend_floor": spec["backend_floor_note"],
        "perplexity": {
            "turbospark": perplexity(port, ids, first),
            "mlx_cached": perplexity(cached, ids, first),
            "mlx_batched": perplexity(batched, ids, first),
        },
        "max_abs_logit": maxima,
        "reference": f"{spec['repo']}@{spec['revision'][:8]}",
        "install": meta["install"],
        "cache_state": meta["cache_state"],
        "prompt_len": meta["prompt_len"],
    }
    print(json.dumps(report, indent=2))
    # Named per artifact, for `kld_llamacpp.py`'s reason: every arm of every
    # model is the same shape, so a shared filename lets one run's report be
    # read as another's.
    (dump / f"kld_mlx_affine-{name}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
