#!/usr/bin/env python3
"""Cross-engine KL for a TEXT+IMAGE prompt: this port against mlx-vlm on the
identical checkpoint (ROADMAP M-V5, stage 2).

Every other `kld_*.py` here compares a text prompt. This one is the first that
puts a picture in front of the model, so it is the only instrument that can see
the three things M-V5 adds: which rows land at which positions, which rope
angle each position gets, and whether the mRoPE selector agrees with the
reference's.

    # 1. The reference's run, which also OWNS the id sequence (see below).
    uv run --python 3.12 --with mlx --with mlx-vlm --with numpy --with pillow \
      --with transformers -- \
      python scripts/kld_mlx_vlm.py prepare /tmp/vision-kld \
        --image ~/models/vision-probe-qwen38/imgs/page.png

    # 2. This port's run, replaying the reference's ids AND its patch rows.
    TURBOSPARK_QWEN38_VISION_INSTALL_DIR=~/models/qwen38-27b-vision.gturbo \
    TURBOSPARK_VISION_KLD_DIR=/tmp/vision-kld \
      cargo test -p turbospark-bench --test vision_logit_dump --release -- \
      --ignored --nocapture

    # 3. The comparison.
    uv run --python 3.12 --with numpy -- \
      python scripts/kld_mlx_vlm.py compare /tmp/vision-kld

# THE ORDER IS INVERTED FROM EVERY OTHER SCRIPT HERE, AND THAT IS TEMPORARY

`logit_dump.rs` writes `meta.json` and the Python side replays its ids. Here
the reference goes FIRST, because M-V6 does not exist: this port cannot yet
build a text+image id sequence at all -- there is no splice, no `--image` flag
and no server arm. So the processor is the authority on the ids until M-V6
lands, at which point this should flip to match its siblings and the two
splices should be compared instead of one being taken on trust.

Both engines still consume the SAME id list, which is the rule that matters
(`logit_dump.rs`'s own header): a tokenizer or template difference would
surface as a divergence and be misread as a numerics gap.

# IT REPLAYS THE PATCH ROWS TOO, WHICH IS WHAT MAKES A GAP ATTRIBUTABLE

`vision_tower_parity.rs` established this discipline one milestone ago and the
reason is the same: preprocessing is held to the reference by
`crates/vision-io`'s golden fixtures, and the tower by that parity gate, so
feeding this port its own preprocessing here would let a pixel difference
present as an injection bug. The dump carries the exact `pixel_values` the
reference fed its own tower and this port replays them.

What is left in the gap, therefore, is exactly what M-V5 added: the blit, the
position table, and the mRoPE dispatch.

# TWO FLOORS, NOT ONE

AGENTS.md Gotcha 34: a cross-engine number needs a shape floor AND a backend
floor, and the second is invisible unless measured. Both engines here run MLX
on Metal on the same bytes, so the backend floor is zero by construction --
which is a property of THIS pairing and does not transfer to the llama.cpp
driver next door. The shape floor is the batched-versus-cached difference,
reported by `--shape-floor`.

# THE CHECKPOINT NAME IS REQUIRED AND NOT DEFAULTED

AGENTS.md Gotcha 38's rule, and the sibling `kld_mlx_affine.py` states the
failure it prevents: point a dump at the wrong reference and every shape check
still passes, the vocabulary is the same size, and the only symptom is a
divergence that reads as a kernel bug.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

CHECKPOINTS = {
    "qwen38-27b-vision": {
        "repo": "mlx-community/Qwen3.8-27B-4bit",
        # LOAD-BEARING. `~/models/qwen38-27b-vision.gturbo` was streamed from
        # this revision, and the published towers have identical shapes -- so
        # `main` would compare this port running one set of weights against the
        # reference running another and fail nothing loudly.
        "revision": "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
        "model_type": "qwen3_5",
    },
}

DEFAULT_QUESTION = "Transcribe the text in this image."


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


def log_softmax(row: np.ndarray) -> np.ndarray:
    """Max-subtracted and accumulated in float64.

    248,320 exponentials per row overflow float32 and lose the tail in it;
    `quality_common::negative_log_prob` applies the same discipline for the
    same reason.
    """
    row = row.astype(np.float64)
    row -= row.max()
    return row - np.log(np.exp(row).sum())


def divergences(p_logits: np.ndarray, q_logits: np.ndarray) -> dict:
    """Per-position KL between two logit matrices of the same shape.

    A restatement of `kld.py`'s function rather than an import, because that
    module executes an mlx-lm dependency chain at import time and this driver
    runs its `compare` step under numpy alone.
    """
    forward, reverse, agree = [], [], 0
    for i in range(p_logits.shape[0]):
        lp = log_softmax(p_logits[i])
        lq = log_softmax(q_logits[i])
        forward.append(float((np.exp(lp) * (lp - lq)).sum()))
        reverse.append(float((np.exp(lq) * (lq - lp)).sum()))
        agree += int(lp.argmax() == lq.argmax())

    def stats(values):
        a = np.asarray(values)
        return {
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p99": float(np.percentile(a, 99)),
            "max": float(a.max()),
        }

    return {
        "forward": stats(forward),
        "reverse": stats(reverse),
        "top1_agreement": agree / p_logits.shape[0],
    }


def softcap_of(install: str) -> float:
    """The install's declared `final_logit_softcapping`, or 0.0 for none.

    Read the property, never recall it (AGENTS.md Gotcha 38). This family
    declares none, so there is no transform to mismatch and both maxima are
    REPORTED rather than checked against an invented tolerance.
    """
    try:
        manifest = json.loads((pathlib.Path(install) / "manifest.json").read_text())
    except OSError:
        return float("nan")
    return float(manifest.get("arch", {}).get("finalLogitSoftcap", 0.0))


# ---------------------------------------------------------------------------
# prepare: the reference's run
# ---------------------------------------------------------------------------


def prepare(args) -> None:
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template

    spec = CHECKPOINTS[args.checkpoint]
    path = snapshot_dir(spec)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    declared = json.loads((path / "config.json").read_text()).get("model_type")
    if declared != spec["model_type"]:
        sys.exit(
            f"{spec['repo']} declares model_type {declared!r}, not "
            f"{spec['model_type']!r}: this driver is pointed at the wrong "
            "checkpoint"
        )

    model, processor = load(str(path))
    config = model.config

    # The processor owns BOTH the resize and the splice. Everything below
    # replays what it produced rather than recomputing any of it.
    from PIL import Image

    image = Image.open(args.image).convert("RGB")
    prompt = apply_chat_template(
        processor, config, args.question, num_images=1
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="np")

    input_ids = np.asarray(inputs["input_ids"]).reshape(-1).astype(np.int64)
    pixel_values = np.asarray(inputs["pixel_values"], dtype=np.float32)
    grid_thw = np.asarray(inputs["image_grid_thw"]).reshape(-1).astype(np.int64)

    image_token_id = int(config.image_token_id)
    placeholders = int((input_ids == image_token_id).sum())
    if placeholders == 0:
        sys.exit(
            "the processor emitted no image placeholder; the prompt template "
            "did not receive the image"
        )

    forward = model(
        mx.array(input_ids)[None],
        pixel_values=mx.array(pixel_values),
        image_grid_thw=mx.array(grid_thw)[None],
    )
    # `LanguageModel.__call__` returns a `LanguageModelOutput` wrapper rather
    # than a bare array; unwrap rather than assume either shape.
    logits = getattr(forward, "logits", forward)
    # CAST IN MLX, not in numpy. These come back bfloat16, which numpy cannot
    # read through the buffer protocol at all ("not a valid PEP 3118 buffer
    # format string") -- so this is a hard error rather than a silent
    # precision question. Worth knowing which side is coarser either way:
    # bfloat16's 8 mantissa bits are FEWER than the float16 this port dumps,
    # so the reference is the lower-precision arm and the round trip through
    # the port's dump width costs nothing (`crates/bench` Gotcha 8 measured
    # the same thing for the text path).
    logits = logits.astype(mx.float32)
    mx.eval(logits)
    reference = np.asarray(logits, dtype=np.float32)[0]

    # Row i holds the next-token logits after consuming ids[i], so the last id
    # is fed to nobody. Dropping that row here rather than in the comparison
    # keeps the two dumps the same shape.
    reference = reference[:-1]

    # PERMUTED TO THIS PORT'S ROW ORDER BEFORE IT IS WRITTEN, and this is the
    # one line in the file that a reader must not skip.
    #
    # The processor emits each patch row as `(C, T, P_h, P_w)`; this port emits
    # `(T, P_h, P_w, C)` on purpose, so the repack can copy
    # `patch_embed.proj.weight` verbatim and the GEMM reads both operands
    # row-major (`crates/vision-io` Gotcha 1). Writing the raw rows here does
    # not fail: the GEMM keeps its shape, the tower runs, and the model reads a
    # DIFFERENT IMAGE. Measured on the first run of this script -- 18.87 mean
    # nats and 1.5% top-1 on the image positions against a near-exact 0.00026
    # median on the text ones, which is what that decomposition is for.
    #
    # `scripts/vision_tower_probe.py`'s dump mode does the identical transpose
    # for the identical reason; this is not a second convention.
    patches = pixel_values.shape[0]
    channels = int(config.vision_config.in_channels)
    temporal = int(config.vision_config.temporal_patch_size)
    patch = int(config.vision_config.patch_size)
    rows = (
        pixel_values.reshape(patches, channels, temporal, patch, patch)
        .transpose(0, 2, 3, 4, 1)
        .reshape(patches, -1)
    )

    (out / "reference.f32").write_bytes(reference.astype("<f4").tobytes())
    # PERMUTED, for the Rust side to replay through this port's tower.
    (out / "pixel_values.f32").write_bytes(
        np.ascontiguousarray(rows).astype("<f4").tobytes()
    )
    # RAW, in the processor's own order, for `shape-floor` to feed back to the
    # reference. Two files rather than one permutation applied twice: the
    # order is the exact thing that went wrong on this script's first run, so
    # each consumer reads the layout it wants and neither has to transpose.
    (out / "pixel_values_raw.f32").write_bytes(
        np.ascontiguousarray(pixel_values).astype("<f4").tobytes()
    )

    # THE REFERENCE'S OWN MERGER ROWS, for the Rust side to inject INSTEAD of
    # running this port's tower. That is the stage-localizing arm: with the
    # port's tower the comparison measures the tower AND the injection
    # together, and this file's four-stage ancestor (`vision_tower_parity.rs`)
    # exists because a composite gap localizes only when the stages can be
    # substituted one at a time.
    #
    # They are read out of `get_input_embeddings`' own output at the
    # placeholder positions, which is where it scattered them -- not
    # recomputed, so a difference in how the merge is addressed cannot hide
    # here either.
    features = model.get_input_embeddings(
        mx.array(input_ids)[None],
        pixel_values=mx.array(pixel_values),
        image_grid_thw=mx.array(grid_thw)[None],
    )
    merged = np.asarray(features.inputs_embeds.astype(mx.float32))[0]
    mask = input_ids == image_token_id
    (out / "image_features.f32").write_bytes(
        np.ascontiguousarray(merged[mask]).astype("<f4").tobytes()
    )
    header = {
        "checkpoint": args.checkpoint,
        "repo": spec["repo"],
        "revision": spec["revision"],
        "image": str(pathlib.Path(args.image).resolve()),
        "question": args.question,
        "input_ids": input_ids.tolist(),
        "grid_thw": grid_thw.tolist(),
        "pixel_values_shape": list(pixel_values.shape),
        "image_token_id": image_token_id,
        "vision_start_token_id": int(config.vision_start_token_id),
        "rows": int(reference.shape[0]),
        "vocab": int(reference.shape[1]),
        "placeholders": placeholders,
        "reference_dtype": "float32",
        "image_features_shape": [int(mask.sum()), int(merged.shape[1])],
    }
    (out / "header.json").write_text(json.dumps(header, indent=2))

    print(f"prepare: {len(input_ids)} ids, {placeholders} placeholders")
    print(f"prepare: grid_thw {grid_thw.tolist()}, pixels {pixel_values.shape}")
    print(f"prepare: {reference.shape[0]} rows x {reference.shape[1]} -> {out}")


# ---------------------------------------------------------------------------
# shape-floor: the reference against ITSELF, batched versus cached
# ---------------------------------------------------------------------------


def shape_floor(args) -> None:
    """The reference token-by-token through a cache, for `compare` to divide by.

    **WITHOUT THIS THE HEADLINE NUMBER HAS NO SCALE** (`crates/bench` Gotcha
    8). This port walks a prompt one token at a time through a KV cache;
    `prepare` runs ONE batched pass over every position. That is a different
    reduce shape on identical weights, and on the MoE families it alone costs
    0.0352 mean nats and 4% of the argmaxes -- more than this port's whole
    gap against mlx-lm on the text path.

    **AND AN IMAGE PROMPT IS NOT A TEXT PROMPT ON THIS AXIS.** The dense text
    floor for this architecture is 0.0000024 nats, which is what makes it
    tempting to skip this step here. It does not transfer: a batched pass over
    1,280 image positions attends over a span nothing in the text corpus
    reaches, so the floor has to be measured on THIS prompt rather than quoted
    from that one.

    Slow by construction -- one forward per position, no batching -- which is
    why it is a separate mode and not part of `prepare`.
    """
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_lm.models.cache import make_prompt_cache

    out = pathlib.Path(args.out)
    header = json.loads((out / "header.json").read_text())
    spec = CHECKPOINTS[header["checkpoint"]]
    model, _ = load(str(snapshot_dir(spec)))

    input_ids = mx.array(np.asarray(header["input_ids"], dtype=np.int64))[None]
    grid = mx.array(np.asarray(header["grid_thw"], dtype=np.int64))[None]
    pixels = np.frombuffer((out / "pixel_values_raw.f32").read_bytes(), dtype="<f4")
    pixels = mx.array(pixels.reshape(header["pixel_values_shape"]))

    # MERGE ONCE, THEN STEP. `get_input_embeddings` scatters the tower's rows
    # into the placeholder positions for a WHOLE prompt; it cannot do it for a
    # one-token step, which is what a naive cached loop tries and why the first
    # version of this died on "tokens: 0, features 1280". Computing the merged
    # matrix once and feeding it a row at a time is the same inputs in the
    # cached SHAPE, which is exactly the axis being measured.
    features = model.get_input_embeddings(
        input_ids, pixel_values=pixels, image_grid_thw=grid
    )
    embeds = features.inputs_embeds
    position_ids = features.position_ids
    mx.eval(embeds, position_ids)

    cache = make_prompt_cache(model.language_model)
    rows = []
    n = input_ids.shape[1] - 1
    for i in range(n):
        forward = model.language_model(
            input_ids[:, i : i + 1],
            inputs_embeds=embeds[:, i : i + 1],
            position_ids=position_ids[..., i : i + 1],
            cache=cache,
        )
        logits = getattr(forward, "logits", forward).astype(mx.float32)
        mx.eval(logits)
        rows.append(np.asarray(logits)[0, -1, :])
        if i % 200 == 0:
            print(f"  shape-floor: {i}/{n}", flush=True)

    cached = np.stack(rows)
    (out / "reference_cached.f32").write_bytes(cached.astype("<f4").tobytes())
    print(f"shape-floor: wrote {cached.shape[0]} cached rows")


# ---------------------------------------------------------------------------
# generate: the end-to-end arm
# ---------------------------------------------------------------------------


def generate(args) -> None:
    """The reference's own greedy continuation, and this port's beside it.

    The divergence numbers are the instrument; this is the thing a reader
    wants to know. It is also the only arm that exercises the DECODE side of
    the position rule -- past the prompt `rope_position` resolves to
    `position + rope_delta`, which no prompt position reaches.

    Both continuations are decoded with the REFERENCE's tokenizer, because the
    vision install carries no sidecars. That is fine here and stops being fine
    at M-V6, when this port starts building the ids itself.
    """
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_lm.models.cache import make_prompt_cache

    out = pathlib.Path(args.out)
    header = json.loads((out / "header.json").read_text())
    spec = CHECKPOINTS[header["checkpoint"]]
    model, processor = load(str(snapshot_dir(spec)))
    tokenizer = getattr(processor, "tokenizer", processor)

    input_ids = mx.array(np.asarray(header["input_ids"], dtype=np.int64))[None]
    grid = mx.array(np.asarray(header["grid_thw"], dtype=np.int64))[None]
    pixels = np.frombuffer((out / "pixel_values_raw.f32").read_bytes(), dtype="<f4")
    pixels = mx.array(pixels.reshape(header["pixel_values_shape"]))

    features = model.get_input_embeddings(
        input_ids, pixel_values=pixels, image_grid_thw=grid
    )
    embeds, position_ids = features.inputs_embeds, features.position_ids
    delta = int(np.asarray(features.rope_deltas).reshape(-1)[0])
    mx.eval(embeds, position_ids)

    cache = make_prompt_cache(model.language_model)
    forward = model.language_model(
        input_ids, inputs_embeds=embeds, position_ids=position_ids, cache=cache
    )
    logits = getattr(forward, "logits", forward).astype(mx.float32)
    mx.eval(logits)
    token = int(np.asarray(logits)[0, -1].argmax())

    produced = []
    n = len(header["input_ids"])
    for step in range(args.tokens):
        produced.append(token)
        p = n + step + delta
        step_ids = mx.array([[token]])
        forward = model.language_model(
            step_ids,
            inputs_embeds=model.language_model.model.embed_tokens(step_ids),
            position_ids=mx.array([[[p]], [[p]], [[p]]]),
            cache=cache,
        )
        logits = getattr(forward, "logits", forward).astype(mx.float32)
        mx.eval(logits)
        token = int(np.asarray(logits)[0, -1].argmax())

    reference_text = tokenizer.decode(produced)
    result = {"reference_ids": produced, "reference_text": reference_text}

    port_meta = out / "port_meta.json"
    if port_meta.exists():
        port_ids = json.loads(port_meta.read_text()).get("generated_ids")
        if port_ids:
            result["port_ids"] = port_ids
            result["port_text"] = tokenizer.decode(port_ids)
            # SHIFT-TOLERANT, and the first version was not.
            #
            # Measured on the real page: the two engines produce the SAME
            # transcription -- same table rows, same figures, same word
            # sequence -- and this port emits two leading newlines the
            # reference does not. A positional comparison called that 10.42%
            # agreement diverging at token 0, which is the
            # metric-fails-a-correct-implementation trap this repo has now hit
            # three times (`docs/VISION.md`'s cosine note, `rope_mrope_parity`'s
            # FP16 bar).
            #
            # A greedy continuation is free to start with whitespace and the
            # question is whether the CONTENT matches, so the best alignment in
            # a small window is what gets reported -- with the offset printed
            # beside it, because an offset that is not small is a real
            # divergence wearing the same clothes.
            best = (0, -1)
            for shift in range(-8, 9):
                a = produced[max(0, shift) :]
                b = port_ids[max(0, -shift) :]
                n_cmp = min(len(a), len(b))
                if n_cmp < 8:
                    continue
                agree = sum(int(x == y) for x, y in zip(a[:n_cmp], b[:n_cmp]))
                if agree / n_cmp > best[1]:
                    best = (shift, agree / n_cmp)
            result["greedy_alignment_shift"] = best[0]
            result["greedy_agreement"] = best[1]
            a = produced[max(0, best[0]) :]
            b = port_ids[max(0, -best[0]) :]
            result["first_divergence"] = next(
                (i for i, (x, y) in enumerate(zip(a, b)) if x != y), None
            )

    (out / "generation.json").write_text(json.dumps(result, indent=2))
    print("reference:", json.dumps(reference_text))
    if "port_text" in result:
        print("port     :", json.dumps(result["port_text"]))
        print(
            f"greedy agreement: {result['greedy_agreement']:.2%} at alignment "
            f"shift {result['greedy_alignment_shift']}, first divergence at "
            f"{result['first_divergence']}"
        )


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def compare(args) -> None:
    out = pathlib.Path(args.out)
    header = json.loads((out / "header.json").read_text())
    rows, vocab = header["rows"], header["vocab"]

    reference = np.frombuffer((out / "reference.f32").read_bytes(), dtype="<f4")
    reference = reference.reshape(rows, vocab).astype(np.float32)

    port_path = out / "port.f16"
    if not port_path.exists():
        sys.exit(
            f"{port_path} is missing; run the Rust dump step (see this file's "
            "docstring) before comparing"
        )
    port = np.frombuffer(port_path.read_bytes(), dtype="<f2")
    if port.size != rows * vocab:
        sys.exit(
            f"port dump is {port.size} values against the reference's "
            f"{rows * vocab}: the two engines did not walk the same prompt"
        )
    port = port.reshape(rows, vocab).astype(np.float32)

    port_meta = json.loads((out / "port_meta.json").read_text())
    if port_meta["input_ids"] != header["input_ids"]:
        sys.exit(
            "the port replayed a different id sequence than the reference "
            "produced; the comparison would be meaningless"
        )

    # Finiteness BEFORE any statistic. NaN fails every comparison, so an
    # argmax over a NaN row reads as agreement and a KL over one reads NaN --
    # the first is a perfect score on garbage (AGENTS.md Gotcha 59).
    for name, a in (("reference", reference), ("port", port)):
        if not np.isfinite(a).all():
            sys.exit(f"{name} logits carry a non-finite value")

    softcap = softcap_of(port_meta["install"])
    heads = {
        "declared_softcap": softcap,
        "max_abs_logit_mlx_vlm": float(np.abs(reference).max()),
        "max_abs_logit_port": float(np.abs(port).max()),
    }
    if softcap > 0.0 and heads["max_abs_logit_mlx_vlm"] > softcap * 1.001:
        sys.exit(
            f"mlx-vlm's max |logit| is {heads['max_abs_logit_mlx_vlm']:.4f}, "
            f"over the {softcap} softcap this install declares: the two heads "
            "are not the same function"
        )

    result = {"heads": heads, "all_positions": divergences(reference, port)}

    # THE FLOOR, if it has been measured. A headline divergence without one
    # is a number with no scale (`crates/bench` Gotcha 8), and this prompt's
    # floor cannot be quoted from the family's text row -- a batched pass over
    # 1,280 image positions attends over a span no text prompt reaches.
    floor_path = out / "reference_cached.f32"
    if floor_path.exists():
        cached = np.frombuffer(floor_path.read_bytes(), dtype="<f4")
        cached = cached.reshape(rows, vocab).astype(np.float32)
        result["shape_floor"] = divergences(reference, cached)
        result["port_vs_cached"] = divergences(cached, port)
    else:
        result["shape_floor"] = (
            "NOT MEASURED -- run the `shape-floor` mode; the headline below "
            "has no scale without it"
        )

    # The IMAGE SPAN alone, which is the only region M-V5 can have broken in a
    # way the whole-prompt mean would dilute. A 1,280-token image inside a
    # 1,300-token prompt would hide almost anything in the average; a
    # 12-token image would hide everything.
    ids = np.asarray(header["input_ids"])
    span = np.flatnonzero(ids[:rows] == header["image_token_id"])
    if span.size:
        result["image_positions"] = divergences(reference[span], port[span])
        text = np.setdiff1d(np.arange(rows), span)
        if text.size:
            result["text_positions"] = divergences(reference[text], port[text])

    print(json.dumps(result, indent=2))

    # Reported, never asserted. This script measures; the pass/fail bar lives
    # in whatever gate quotes it, so a threshold invented here would be a
    # second and staler copy of one.
    top1 = result["all_positions"]["top1_agreement"]
    mean = result["all_positions"]["forward"]["mean"]
    print(f"\nsummary: {mean:.6f} mean nats, {top1:.4%} top-1 agreement")
    if isinstance(result["shape_floor"], dict):
        f = result["shape_floor"]
        pc = result["port_vs_cached"]
        print(
            f"         shape floor: {f['forward']['mean']:.6f} nats, "
            f"{f['top1_agreement']:.4%} top-1 (the reference against ITSELF)"
        )
        print(
            f"         port vs the CACHED arm: {pc['forward']['mean']:.6f} nats, "
            f"{pc['top1_agreement']:.4%} top-1  <- the comparison that matches shapes"
        )
    if "image_positions" in result:
        i = result["image_positions"]
        print(
            f"         image span: {i['forward']['mean']:.6f} nats, "
            f"{i['top1_agreement']:.4%} top-1, {span.size} positions"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("prepare", help="run the reference and own the ids")
    p.add_argument("out")
    p.add_argument("--image", required=True)
    p.add_argument("--question", default=DEFAULT_QUESTION)
    p.add_argument("--checkpoint", default="qwen38-27b-vision", choices=CHECKPOINTS)
    p.set_defaults(func=prepare)

    f = sub.add_parser("shape-floor", help="the reference against itself, cached")
    f.add_argument("out")
    f.set_defaults(func=shape_floor)

    g = sub.add_parser("generate", help="both engines' greedy continuations")
    g.add_argument("out")
    g.add_argument("--tokens", type=int, default=48)
    g.set_defaults(func=generate)

    c = sub.add_parser("compare", help="compare the two dumps")
    c.add_argument("out")
    c.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
