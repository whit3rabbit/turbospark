"""Export the inworld Voice Isolator realtime checkpoint as a stateful Core ML
streaming step, verify it, and measure it.

One call of the program is one block of F = 2 new STFT frames (20 ms at
48 kHz, hop 480) of the plain stream (`streaming.Stream`, window_frames
None, cache = the 1200-frame attention window) at the default look-ahead
position (2 frames). The host does the analysis, the read-ahead, the
correction and the overlap-add; the program does everything in between.
Contract (the Rust host is written against it; see
crates/audio/src/sts/voice_isolator/COREML.md):

    inputs   spectrum  f32 [1, 2, 1025, 2]  rfft(hann_periodic(2048) * frame),
                                            (re, im), the 2 new frames
             rope_cos  f32 [2, 32]          cos(t * 10000^(-d/32)), t absolute
             rope_sin  f32 [2, 32]
             mask      f32 [2, 1201]        additive (0 or -1e4) over the 1199
                                            cached key slots (oldest first) and
                                            the 2 new frames
    states   k{b}, v{b} f16 [46, 8, 1199, 64]  b in 0..4: rotated keys, values
             conv{m}   f16 [46, 1024, 3]       m in 0..3: time Mamba conv tail
             ssm{m}    f16 [46, 1024, 16]      time Mamba scan state
    output   raw       f32 [2, 1025, 4]     decoder output of the 2 new frames

Each call shifts every k/v state left by 2 and appends the new rotated keys
and values. Zero states plus a mask hiding the empty slots is a fresh stream.

Stages (all in one run):
  1. Build the torch StreamStep from the reference modules and weights and
     check it against the reference `streaming.Stream` (block_frames 2,
     pushes of 960 samples) on real_clip.npy, in float32, through a Python
     replica of the host synthesis.
  2. Trace and convert with coremltools (ct.StateType, macOS 15) at each
     requested compute precision; save .mlpackage and the compiled .mlmodelc.
  3. Run each Core ML program over the clip on each compute unit setting
     (quality vs the float32 torch stream) and time 300 calls after warmup.

Run with a pinned, isolated interpreter:

    uv run --python 3.12 --with coremltools==9.0 --with numpy==2.2.6 \
        --with torch==2.7.0 --with safetensors==0.5.3 \
        python -I crates/audio/tools/export_voice_isolator_coreml.py \
        --reference /path/to/voice-isolator-c73b3fab81a870410cbac50d2f05f25edc4d10aa \
        --checkpoint ~/models/voice-isolator/denoiser_realtime.safetensors \
        --clip ~/models/voice-isolator/real_clip.npy \
        --out ~/models/voice-isolator-coreml

The fp16 program is written as VoiceIsolatorStep.mlpackage, the fp32 one
(if requested) as VoiceIsolatorStep-fp32.mlpackage. Results go to
<out>/report.json.
"""

import argparse
import importlib.util
import json
import math
import pathlib
import shutil
import statistics
import sys
import time
import types

import numpy as np
import torch

F = 2  # frames per call
LOOKAHEAD = 2  # trained look-ahead position baked into the program
MASKED = -1e4
# The encoder LayerNorm sees raw spectrum values (up to sum(hann) = 1024 for
# full-scale input), whose squares overflow fp16. LN is scale invariant, so
# the program normalizes x / 16 with eps / 256: exact in fp32, overflow-free
# in fp16 for any input within [-1, 1].
ENC_SCALE = 16.0


def import_reference(root: pathlib.Path):
    sys.path.insert(0, str(root / "src"))
    # The package __init__ imports file loaders the network never touches.
    for missing in ("torchaudio", "soundfile"):
        if importlib.util.find_spec(missing) is None:
            sys.modules[missing] = types.ModuleType(missing)
    from inworld.voice_isolation import isolator  # pylint: disable=import-outside-toplevel
    from inworld.voice_isolation import model as modeling  # pylint: disable=import-outside-toplevel
    from inworld.voice_isolation import streaming  # pylint: disable=import-outside-toplevel
    return isolator, modeling, streaming


def mamba_parallel(mb, x: torch.Tensor) -> torch.Tensor:
    """`Mamba.forward` without a carried state over x [batch, T, width], the
    scan as a Hillis-Steele associative scan (log2 T steps) instead of T
    sequential steps. Every factor is a product of decays in (0, 1], so it
    cannot overflow in fp16."""
    frames = x.shape[1]
    xz = mb.in_proj(x)
    u, z = xz.chunk(2, dim=-1)
    pad = mb.conv1d.kernel_size[0] - 1
    convolved = torch.nn.functional.conv1d(
        torch.nn.functional.pad(u.transpose(1, 2), (pad, 0)),
        mb.conv1d.weight, mb.conv1d.bias, groups=mb.conv1d.groups)
    u = torch.nn.functional.silu(convolved).transpose(1, 2)
    dt, b, c = torch.split(mb.x_proj(u), [mb.dt_rank, mb.d_state, mb.d_state],
                           dim=-1)
    delta = torch.nn.functional.softplus(mb.dt_proj(dt))
    a_rate = -torch.exp(mb.A_log)
    decay = torch.exp(delta[..., None] * a_rate)  # [B, T, C, N]
    drive = (delta * u)[..., None] * b[:, :, None, :]
    step = 1
    while step < frames:
        ones = torch.ones_like(decay[:, :step])
        zeros = torch.zeros_like(drive[:, :step])
        prev_decay = torch.cat((ones, decay[:, :-step]), dim=1)
        prev_drive = torch.cat((zeros, drive[:, :-step]), dim=1)
        drive = decay * prev_drive + drive
        decay = decay * prev_decay
        step *= 2
    y = (drive * c[:, :, None, :]).sum(-1) + mb.D * u
    y = y * torch.nn.functional.silu(z)
    return mb.out_proj(y)


def bidirectional_freq(bm, x: torch.Tensor) -> torch.Tensor:
    """`BidirectionalMamba.forward` (two-sided) over the bands."""
    forward = mamba_parallel(bm.forward_blocks, x)
    backward = mamba_parallel(bm.backward_blocks, x.flip(1)).flip(1)
    return bm.norm(
        bm.output_proj(torch.cat((forward + x, backward + x), dim=-1)))


class StreamStep(torch.nn.Module):
    """One call of the plain stream: 2 new frames in, their decoder output
    out, the attention caches and time Mamba states carried as buffers."""

    def __init__(self, net, modeling, state_dtype=torch.float16):
        super().__init__()
        cfg = net.config
        if not cfg.causal or cfg.mode_film or cfg.dual_mode:
            raise ValueError("expects the causal realtime checkpoint")
        if cfg.attention_window_frames < 2:
            raise ValueError("expects a bounded attention window")
        self.modeling = modeling
        self.net = net
        self.width = cfg.width
        self.heads = cfg.num_heads
        self.head_dim = cfg.width // cfg.num_heads
        self.bins = cfg.n_fft // 2 + 1
        self.cache = cfg.attention_window_frames - 1
        edges = net.edges
        self.bands = len(edges)
        widths = [e - s for s, e in edges]
        inputs, outputs = 2 * max(widths), 4 * max(widths)
        self.outputs = outputs
        bins = self.bins

        gather = torch.full((self.bands, inputs), 2 * bins, dtype=torch.long)
        valid = torch.zeros(self.bands, inputs)
        keep = []
        for band, (start, end) in enumerate(edges):
            gather[band, :2 * (end - start)] = torch.arange(2 * start, 2 * end)
            valid[band, :2 * (end - start)] = 1.0
            keep.append(band * outputs + torch.arange(4 * (end - start)))
        self.register_buffer("gather", gather.flatten(), persistent=False)
        self.register_buffer("valid", valid[:, None], persistent=False)
        self.register_buffer("counts",
                             torch.tensor([2.0 * w for w in widths])[:, None,
                                                                     None],
                             persistent=False)
        self.register_buffer("keep", torch.cat(keep), persistent=False)

        def stack(tensors, size):
            out = torch.zeros(self.bands, size, *tensors[0].shape[1:])
            for band, tensor in enumerate(tensors):
                out[band, :tensor.shape[0]] = tensor.detach()
            return out

        enc = [(e[0], e[1]) for e in net.encoders]
        self.enc_eps = enc[0][0].eps / (ENC_SCALE * ENC_SCALE)
        self.register_buffer("enc_ln_w",
                             stack([n.weight for n, _ in enc], inputs)[:, None],
                             persistent=False)
        self.register_buffer("enc_ln_b",
                             stack([n.bias for n, _ in enc], inputs)[:, None],
                             persistent=False)
        self.register_buffer("enc_w", stack([l.weight.t() for _, l in enc],
                                            inputs),
                             persistent=False)
        embed = net.band_embedding.detach()
        choices = list(cfg.lookahead_frames_choices)
        if choices:
            embed = embed + net.lookahead_embedding.weight[choices.index(
                LOOKAHEAD)].detach()[None]
        self.register_buffer(
            "enc_b",
            (torch.stack([l.bias.detach() for _, l in enc]) + embed)[:, None],
            persistent=False)

        dec = net.denoise_decoder
        heads = dec.heads
        scale, shift = dec.condition(dec.controls).detach().chunk(2, dim=-1)
        self.register_buffer("dec_scale", 1 + scale, persistent=False)
        self.register_buffer("dec_shift", shift, persistent=False)
        self.head_eps = heads[0][0].eps
        self.register_buffer("head_ln_w",
                             torch.stack([h[0].weight.detach() for h in heads
                                         ])[:, None],
                             persistent=False)
        self.register_buffer("head_ln_b",
                             torch.stack([h[0].bias.detach() for h in heads
                                         ])[:, None],
                             persistent=False)
        self.register_buffer("head_w1",
                             torch.stack([h[1].weight.detach().t()
                                          for h in heads]),
                             persistent=False)
        self.register_buffer("head_b1",
                             torch.stack([h[1].bias.detach() for h in heads
                                         ])[:, None],
                             persistent=False)
        self.register_buffer("head_w2",
                             torch.stack([
                                 torch.nn.functional.pad(
                                     h[3].weight.detach().t(),
                                     (0, outputs - h[3].weight.shape[0]))
                                 for h in heads
                             ]),
                             persistent=False)
        self.register_buffer("head_b2",
                             torch.stack([
                                 torch.nn.functional.pad(
                                     h[3].bias.detach(),
                                     (0, outputs - h[3].bias.shape[0]))
                                 for h in heads
                             ])[:, None],
                             persistent=False)

        self.attention_blocks = []
        self.mamba_blocks = []
        for block in net.blocks:
            if isinstance(block, modeling.DualPathAttention):
                self.attention_blocks.append(block)
            else:
                self.mamba_blocks.append(block)
        inner = cfg.mamba_expand * cfg.width
        conv = cfg.mamba_conv - 1
        for b in range(len(self.attention_blocks)):
            shape = (self.bands, self.heads, self.cache, self.head_dim)
            self.register_buffer(f"k{b}", torch.zeros(shape, dtype=state_dtype))
            self.register_buffer(f"v{b}", torch.zeros(shape, dtype=state_dtype))
        for m in range(len(self.mamba_blocks)):
            self.register_buffer(
                f"conv{m}",
                torch.zeros(self.bands, inner, conv, dtype=state_dtype))
            self.register_buffer(
                f"ssm{m}",
                torch.zeros(self.bands, inner, cfg.mamba_state,
                            dtype=state_dtype))
        self.state_shapes = {
            name: tuple(buf.shape)
            for name, buf in self.named_buffers()
            if name[:1] in "kv" and name[1:].isdigit() or
            name.startswith(("conv", "ssm"))
        }

    def reset(self):
        for name in self.state_shapes:
            getattr(self, name).zero_()

    def encode(self, spectrum: torch.Tensor) -> torch.Tensor:
        flat = spectrum.reshape(F, 2 * self.bins) * (1.0 / ENC_SCALE)
        flat = torch.cat((flat, torch.zeros_like(flat[:, :1])), dim=1)
        x = torch.index_select(flat, 1, self.gather).view(
            F, self.bands, -1).transpose(0, 1)  # [bands, F, inputs]
        mean = x.sum(-1, keepdim=True) / self.counts
        centered = (x - mean) * self.valid
        variance = (centered * centered).sum(-1, keepdim=True) / self.counts
        x = centered * torch.rsqrt(variance + self.enc_eps)
        x = x * self.enc_ln_w + self.enc_ln_b
        return torch.bmm(x, self.enc_w) + self.enc_b  # [bands, F, width]

    def attend(self, block, index: int, x: torch.Tensor, cos, sin, mask):
        modeling = self.modeling
        tb = block.time_transformer
        attn = tb.attn
        qkv = attn.qkv(tb.norm1(x)).view(self.bands, F, 3, self.heads,
                                         self.head_dim)
        query = qkv[:, :, 0].transpose(1, 2)
        key = qkv[:, :, 1].transpose(1, 2)
        value = qkv[:, :, 2].transpose(1, 2)  # [bands, heads, F, dim]
        query = modeling.apply_rope(query, cos, sin)
        key = modeling.apply_rope(key, cos, sin)
        k_state = getattr(self, f"k{index}")
        v_state = getattr(self, f"v{index}")
        past_k = k_state.to(key.dtype)
        past_v = v_state.to(value.dtype)
        keys = torch.cat((past_k, key), dim=2)
        values = torch.cat((past_v, value), dim=2)
        scores = torch.matmul(query, keys.transpose(-1, -2)) * (
            1.0 / math.sqrt(self.head_dim)) + mask
        output = torch.matmul(torch.softmax(scores, dim=-1), values)
        if attn.exclusive:
            output = modeling.exclusive(output, value)
        k_state[:, :, :, :] = torch.cat((past_k[:, :, F:], key),
                                        dim=2).to(k_state.dtype)
        v_state[:, :, :, :] = torch.cat((past_v[:, :, F:], value),
                                        dim=2).to(v_state.dtype)
        x = x + attn.proj(
            output.transpose(1, 2).reshape(self.bands, F, self.width))
        x = x + tb.ffn(tb.norm2(x))
        banded = block.band_transformer(x.transpose(0, 1))  # [F, bands, w]
        return banded.transpose(0, 1)

    def scan(self, block, index: int, x: torch.Tensor) -> torch.Tensor:
        tm = block.time_mamba
        mb = tm.forward_blocks
        conv_state = getattr(self, f"conv{index}")
        ssm_state = getattr(self, f"ssm{index}")
        xz = mb.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        history = torch.cat((conv_state.to(u.dtype), u.transpose(1, 2)),
                            dim=-1)  # [bands, inner, 3 + F]
        convolved = torch.nn.functional.conv1d(history,
                                               mb.conv1d.weight,
                                               mb.conv1d.bias,
                                               groups=mb.conv1d.groups)
        conv_state[:, :, :] = history[:, :, F:].to(conv_state.dtype)
        u = torch.nn.functional.silu(convolved).transpose(1, 2)
        dt, b, c = torch.split(mb.x_proj(u),
                               [mb.dt_rank, mb.d_state, mb.d_state],
                               dim=-1)
        delta = torch.nn.functional.softplus(mb.dt_proj(dt))
        a_rate = -torch.exp(mb.A_log)
        h = ssm_state.to(u.dtype)
        ys = []
        for t in range(F):
            h = (torch.exp(delta[:, t, :, None] * a_rate) * h +
                 (delta[:, t] * u[:, t])[..., None] * b[:, t, None, :])
            ys.append((h * c[:, t, None, :]).sum(-1) + mb.D * u[:, t])
        ssm_state[:, :, :] = h.to(ssm_state.dtype)
        y = torch.stack(ys, dim=1) * torch.nn.functional.silu(z)
        forward = mb.out_proj(y)
        time_out = x + tm.norm(tm.output_proj(forward + x))
        freq = time_out.transpose(0, 1)  # [F, bands, width]
        freq = freq + bidirectional_freq(block.freq_mamba, freq)
        return freq.transpose(0, 1)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.dec_scale + self.dec_shift
        x = torch.nn.functional.layer_norm(x, (self.width,), None, None,
                                           self.head_eps)
        x = x * self.head_ln_w + self.head_ln_b
        x = torch.nn.functional.gelu(torch.bmm(x, self.head_w1) + self.head_b1)
        x = torch.bmm(x, self.head_w2) + self.head_b2  # [bands, F, outputs]
        x = x.transpose(0, 1).reshape(F, self.bands * self.outputs)
        return torch.index_select(x, 1, self.keep).view(F, self.bins, 4)

    def forward(self, spectrum, rope_cos, rope_sin, mask):
        x = self.encode(spectrum)
        a = m = 0
        for block in self.net.blocks:
            if isinstance(block, self.modeling.DualPathAttention):
                x = self.attend(block, a, x, rope_cos, rope_sin, mask)
                a += 1
            else:
                x = self.scan(block, m, x)
                m += 1
        return self.decode(x)


# --- host replica -----------------------------------------------------------


class Host:
    """What the Rust host does around the program: analysis, rotary tables,
    the mask, the read-ahead, the correction and the overlap-add."""

    def __init__(self, net, cache: int):
        cfg = net.config
        self.net = net
        self.n_fft, self.hop = cfg.n_fft, cfg.hop_length
        self.cache = cache
        self.half = (cfg.width // cfg.num_heads) // 2
        self.window = torch.hann_window(self.n_fft).float()

    def inputs(self, audio: np.ndarray, first: int):
        """Program inputs for frames first, first + 1 (zeros where a frame's
        window has no audio, as streaming._analyze pads)."""
        spectra = []
        for t in range(first, first + F):
            start = t * self.hop - self.n_fft // 2
            seg = np.zeros(self.n_fft, dtype=np.float32)
            lo, hi = max(start, 0), min(start + self.n_fft, len(audio))
            if hi > lo:
                seg[lo - start:hi - start] = audio[lo:hi]
            spectra.append(
                torch.fft.rfft(torch.from_numpy(seg) * self.window))
        spec = torch.stack(spectra)  # [F, bins] complex64
        d = np.arange(self.half, dtype=np.float64)
        inv = 10000.0**(-d / self.half)
        angles = np.outer(np.arange(first, first + F, dtype=np.float64), inv)
        mask = np.full((F, self.cache + F), MASKED, dtype=np.float32)
        for i in range(F):
            query = first + i
            for j in range(self.cache):
                frame = first - self.cache + j
                if frame >= 0 and frame > query - (self.cache + 1):
                    mask[i, j] = 0.0
            mask[i, self.cache:self.cache + i + 1] = 0.0
        return {
            "spectrum":
                torch.view_as_real(spec)[None].numpy().astype(np.float32),
            "rope_cos": np.cos(angles).astype(np.float32),
            "rope_sin": np.sin(angles).astype(np.float32),
            "mask": mask,
        }, spec

    def run(self, audio: np.ndarray, step) -> tuple[np.ndarray, list]:
        """The stream over a whole clip: `step(inputs) -> raw [F, bins, 4]`.
        Returns the output audio and every call's inputs (for timing)."""
        total = len(audio) // self.hop + 1
        raws, spectra, calls = [], [], []
        for first in range(0, total, F):
            feed, spec = self.inputs(audio, first)
            raw = np.asarray(step(feed), dtype=np.float32)
            raws.append(raw)
            spectra.append(spec)
            calls.append(feed)
        raw = torch.from_numpy(np.concatenate(raws)[:total])
        spec = torch.cat(spectra)[:total]
        index = torch.clamp(torch.arange(total) + LOOKAHEAD, max=total - 1)
        delta = self.net.correction(raw[index], spec)
        frames = torch.fft.irfft(delta, n=self.n_fft) * self.window
        length = (total - 1) * self.hop + self.n_fft
        ola = torch.zeros(length)
        env = torch.zeros(length)
        sq = self.window**2
        for k in range(total):
            ola[k * self.hop:k * self.hop + self.n_fft] += frames[k]
            env[k * self.hop:k * self.hop + self.n_fft] += sq
        off = self.n_fft // 2
        n = len(audio)
        out = audio + (ola[off:off + n] /
                       env[off:off + n].clamp(min=1e-11)).numpy()
        return out.astype(np.float32), calls


def reference_stream(streaming, net, audio: np.ndarray) -> np.ndarray:
    session = streaming.Stream(net, block_frames=F)
    pieces = []
    for at in range(0, len(audio), 960):
        pieces.append(session.push(audio[at:at + 960]))
    pieces.append(session.flush(as_numpy=True))
    return np.concatenate(pieces).astype(np.float32)


def snr_db(ref: np.ndarray, x: np.ndarray) -> float:
    err = float(np.sum((ref.astype(np.float64) - x)**2))
    sig = float(np.sum(ref.astype(np.float64)**2))
    return float("inf") if err == 0 else 10 * math.log10(sig / err)


def compare(ref, x, audio):
    return {
        "max_abs": float(np.max(np.abs(ref - x))),
        "snr_db": snr_db(ref, x),
        # The output is input plus a correction; the correction alone is the
        # stricter view of what the program computes.
        "correction_snr_db": snr_db(ref - audio, x - audio),
    }


def convert(step: StreamStep, example: dict, precision: str, ct):
    names = ("spectrum", "rope_cos", "rope_sin", "mask")
    args = tuple(torch.from_numpy(example[n]) for n in names)
    step.reset()
    with torch.no_grad():
        traced = torch.jit.trace(step.eval(), args)
    step.reset()
    states = [
        ct.StateType(wrapped_type=ct.TensorType(shape=shape, dtype=np.float16),
                     name=name) for name, shape in step.state_shapes.items()
    ]
    return ct.convert(
        traced,
        inputs=[
            ct.TensorType(name=n, shape=example[n].shape, dtype=np.float32)
            for n in names
        ],
        outputs=[ct.TensorType(name="raw", dtype=np.float32)],
        states=states,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=(ct.precision.FLOAT16
                           if precision == "fp16" else ct.precision.FLOAT32),
        convert_to="mlprogram",
        skip_model_load=True,
    )


def coreml_runner(mlmodel):
    state = mlmodel.make_state()

    def step(feed):
        return mlmodel.predict(feed, state)["raw"]

    return step


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", required=True, type=pathlib.Path)
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--clip", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--precisions", default="fp16,fp32")
    ap.add_argument("--units", default="CPU_AND_NE,CPU_AND_GPU,ALL")
    ap.add_argument("--calls", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--reuse-torch",
                    action="store_true",
                    help="load <out>/torch_stream.npy and the torch results"
                    " of <out>/report.json instead of recomputing them")
    args = ap.parse_args()

    import coremltools as ct  # pylint: disable=import-outside-toplevel

    isolator, modeling, streaming = import_reference(args.reference.resolve())
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    net = isolator.load_model(str(args.checkpoint.expanduser()), "cpu").eval()
    net.set_lookahead(LOOKAHEAD)
    audio = np.load(args.clip.expanduser()).astype(np.float32)
    host = Host(net, net.config.attention_window_frames - 1)
    report = {
        "versions": {
            "torch": torch.__version__,
            "coremltools": ct.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "clip_samples": int(len(audio)),
    }

    # 1. torch StreamStep (float32 states) vs the reference stream.
    torch_step = StreamStep(net, modeling, state_dtype=torch.float32).eval()

    def torch_runner(module):
        module.reset()

        def step(feed):
            return module(*(torch.from_numpy(feed[n]) for n in (
                "spectrum", "rope_cos", "rope_sin", "mask"))).numpy()

        return step

    half_step = StreamStep(net, modeling, state_dtype=torch.float16).eval()
    previous = out / "report.json"
    if args.reuse_torch:
        torch_out = np.load(out / "torch_stream.npy")
        total = len(audio) // host.hop + 1
        calls = [host.inputs(audio, first)[0] for first in range(0, total, F)]
        if previous.exists():
            old = json.loads(previous.read_text())
            for name in ("torch_vs_reference", "torch_fp16_states_vs_torch"):
                if name in old:
                    report[name] = old[name]
    else:
        t0 = time.perf_counter()
        torch_out, calls = host.run(audio, torch_runner(torch_step))
        print(f"torch StreamStep stream: {time.perf_counter() - t0:.1f} s",
              flush=True)
        np.save(out / "torch_stream.npy", torch_out)
    if not args.skip_verify and not args.reuse_torch:
        t0 = time.perf_counter()
        ref_out = reference_stream(streaming, net, audio)
        print(f"reference stream: {time.perf_counter() - t0:.1f} s")
        report["torch_vs_reference"] = compare(ref_out, torch_out, audio)
        print("torch StreamStep vs reference:", report["torch_vs_reference"],
              flush=True)
    if not args.skip_verify and not args.reuse_torch:
        half_out, _ = host.run(audio, torch_runner(half_step))
        report["torch_fp16_states_vs_torch"] = compare(torch_out, half_out,
                                                       audio)
        print("torch fp16 states vs torch fp32:",
              report["torch_fp16_states_vs_torch"], flush=True)

    # 2. Convert.
    report["coreml"] = {}
    for precision in args.precisions.split(","):
        name = ("VoiceIsolatorStep" if precision == "fp16" else
                f"VoiceIsolatorStep-{precision}")
        t0 = time.perf_counter()
        mlmodel = convert(half_step, calls[0], precision, ct)
        package = out / f"{name}.mlpackage"
        if package.exists():
            shutil.rmtree(package)
        mlmodel.save(str(package))
        compiled = out / f"{name}.mlmodelc"
        if compiled.exists():
            shutil.rmtree(compiled)
        ct.models.utils.compile_model(str(package), str(compiled))
        print(f"converted {name} in {time.perf_counter() - t0:.1f} s", flush=True)
        del mlmodel

        # 3. Quality and latency per compute unit setting.
        for unit in args.units.split(","):
            key = f"{precision}/{unit}"
            t0 = time.perf_counter()
            try:
                loaded = ct.models.CompiledMLModel(str(compiled),
                                                   compute_units=getattr(
                                                       ct.ComputeUnit, unit))
                load_s = time.perf_counter() - t0
                cm_out, _ = host.run(audio, coreml_runner(loaded))
            except Exception as error:  # pylint: disable=broad-except
                # A unit that cannot build the program (an ANE compile
                # failure) is a result, not a crash of the whole run.
                report["coreml"][key] = {"error": str(error)[:500]}
                print(key, "failed:", error, flush=True)
                continue
            entry = {"load_s": load_s}
            entry.update(compare(torch_out, cm_out, audio))
            entry["finite"] = bool(np.isfinite(cm_out).all())
            run = coreml_runner(loaded)
            for i in range(args.warmup):
                run(calls[i % len(calls)])
            times = []
            for i in range(args.calls):
                t0 = time.perf_counter()
                run(calls[i % len(calls)])
                times.append((time.perf_counter() - t0) * 1e3)
            times.sort()
            entry["median_ms"] = statistics.median(times)
            entry["p95_ms"] = times[int(math.ceil(0.95 * len(times))) - 1]
            entry["realtime"] = entry["p95_ms"] < 20.0
            report["coreml"][key] = entry
            print(key, json.dumps(entry), flush=True)
            del loaded

    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
