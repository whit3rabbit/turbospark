#!/usr/bin/env python3
"""Convert the NVIDIA Multitalker Parakeet .nemo to an F32 safetensors snapshot.

Output directory (`multitalker_asr` family):
  config.json               model_type "multitalker_asr", NeMo-derived graph config
  model.safetensors         F32, MLX tensor layouts (the layouts the TurboSpark
                            loaders read): Conv2d [O,kH,kW,I], Conv1d [O,k,I],
                            LSTM `Wx`/`Wh` plus one summed `bias` per layer
  tokenizer.model           SentencePiece model from the .nemo (reference only)
  conversion_manifest.json  input, member, script and output sha256 values

Weights stay F32, the dtype Frozen E ran in. Speaker kernels are kept under
their NeMo names (`spk_kernels.0.{0,3}`, `bg_spk_kernels.0.{0,3}`). The only
arithmetic change is the summed LSTM bias (b_ih + b_hh), which the Rust LSTM
adds once; that differs from torch's two adds by at most one rounding step.

`check-diar` compares the Nemotron-3 diarizer .nemo against the HF
`model.safetensors` by tensor content (sha256 of each tensor's bytes, allowing
transposes) and reports anything that does not match.

Usage (pinned reference venv):
  python convert_multitalker_nemo.py convert --nemo FILE.nemo --out DIR
  python convert_multitalker_nemo.py check-diar --nemo DIAR.nemo --safetensors model.safetensors
"""

import argparse
import hashlib
import io
import json
import platform
import re
import sys
import tarfile
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.numpy import save_file
from safetensors import safe_open

MULTITALKER_NEMO_SHA256 = "afaefe89829e201a0ee22c67a715eb5a467bd06c0704cc1579fbfb370fb5be73"
DIAR_NEMO_SHA256 = "867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d"
DIAR_SAFETENSORS_SHA256 = "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6"
KNOWN_PREFIXES = ("encoder.", "decoder.", "joint.", "spk_kernels.", "bg_spk_kernels.")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_nemo(path: Path):
    """Returns (config dict, state dict, {member: bytes} for small members, member sha256s)."""
    members = {}
    digests = {}
    state = None
    with tarfile.open(path, "r:*") as tar:
        for info in tar:
            if not info.isfile():
                continue
            data = tar.extractfile(info).read()
            name = info.name.lstrip("./")
            digests[name] = hashlib.sha256(data).hexdigest()
            if name.endswith("model_weights.ckpt"):
                state = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            else:
                members[name] = data
    if state is None:
        raise SystemExit(f"{path}: no model_weights.ckpt member")
    cfg_name = next(n for n in members if n.endswith("model_config.yaml"))
    cfg = yaml.safe_load(members[cfg_name].decode("utf-8"))
    return cfg, state, members, digests


def supported_punctuation(vocab) -> list:
    """NeMo's `extract_punctuation_from_vocab` (tokenizer_utils.py at 218d0b4).

    RNN-T decoding deletes one whitespace character before any of these marks
    (`decode_tokens_to_str_with_strip_punctuation`), so the Rust detokenizer
    needs the exact set.
    """
    special = [r"^\[.*\]$", r"^<.*>$", r"^##", "^▁", r"^\s*$"]
    patterns = [re.compile(p) for p in special]
    return sorted(
        {
            char
            for token in vocab
            for char in token
            if unicodedata.category(char).startswith("P") and not any(p.match(token) for p in patterns)
        }
    )


def build_config(cfg: dict) -> dict:
    pre, enc, dec, joint = cfg["preprocessor"], cfg["encoder"], cfg["decoder"], cfg["joint"]
    contexts = [list(c) for c in enc["att_context_size"]]
    if [70, 13] not in contexts:
        raise SystemExit(f"att_context_size {contexts} lacks the Frozen E [70, 13]")
    return {
        "model_type": "multitalker_asr",
        "target": cfg.get("target", ""),
        "nemo_version": cfg.get("nemo_version", ""),
        "preprocessor": {
            "sample_rate": pre["sample_rate"],
            "features": pre["features"],
            "n_fft": pre["n_fft"],
            "window_size": pre["window_size"],
            "window_stride": pre["window_stride"],
            "window": pre["window"],
            # NeMo's AudioToMelSpectrogramPreprocessor default when the key is absent.
            "preemph": 0.97 if pre.get("preemph") is None else pre["preemph"],
            "dither": pre.get("dither", 1.0e-05),
            "normalize": str(pre.get("normalize", "NA")),
            "log_zero_guard_value": float(pre.get("log_zero_guard_value", 2.0**-24)),
            "pad_to": pre.get("pad_to", 0),
            "pad_value": pre.get("pad_value", 0.0),
            "stft_pad_mode": "constant",
        },
        "encoder": {
            "feat_in": enc["feat_in"],
            "n_layers": enc["n_layers"],
            "d_model": enc["d_model"],
            "n_heads": enc["n_heads"],
            "ff_expansion_factor": enc["ff_expansion_factor"],
            "subsampling": enc["subsampling"],
            "subsampling_factor": enc["subsampling_factor"],
            "subsampling_conv_channels": enc["subsampling_conv_channels"],
            "conv_kernel_size": enc["conv_kernel_size"],
            "causal_downsampling": enc["causal_downsampling"],
            "conv_context_size": enc["conv_context_size"],
            "conv_norm_type": enc["conv_norm_type"],
            "self_attention_model": enc["self_attention_model"],
            "att_context_style": enc["att_context_style"],
            "att_context_size": contexts,
            "pos_emb_max_len": enc["pos_emb_max_len"],
            "untie_biases": enc["untie_biases"],
            "use_bias": enc["use_bias"],
            "xscaling": enc["xscaling"],
        },
        "speaker_kernels": {
            "type": cfg["spk_kernel_type"],
            "layers": list(cfg["spk_kernel_layers"]),
            "background": bool(cfg["add_bg_spk_kernel"]),
            "num_speakers": cfg["model_defaults"]["num_speakers"],
            "hidden": cfg["model_defaults"]["enc_hidden"],
        },
        "decoder": {
            "pred_hidden": dec["prednet"]["pred_hidden"],
            "pred_rnn_layers": dec["prednet"]["pred_rnn_layers"],
            "vocab_size": dec["vocab_size"],
            "blank_as_pad": dec.get("blank_as_pad", True),
        },
        "joint": {
            "joint_hidden": joint["jointnet"]["joint_hidden"],
            "activation": joint["jointnet"]["activation"],
            "encoder_hidden": joint["jointnet"]["encoder_hidden"],
            "pred_hidden": joint["jointnet"]["pred_hidden"],
            "num_classes": joint["num_classes"],
        },
        "vocabulary": list(joint["vocabulary"]),
        "supported_punctuation": supported_punctuation(joint["vocabulary"]),
        "default_att_context_size": [70, 13],
        "max_symbols": cfg["decoding"]["greedy"]["max_symbols"],
    }


def featurizer_buffer(key: str, tensor) -> tuple:
    """NeMo's stored STFT window and mel filterbank, under the names the Rust loaders read."""
    arr = tensor.detach().to("cpu").to(torch.float32).numpy()
    if key == "preprocessor.featurizer.window":
        return "preprocessor.window", arr
    if key == "preprocessor.featurizer.fb":
        return "preprocessor.fb", np.ascontiguousarray(arr.reshape(arr.shape[-2], arr.shape[-1]))
    raise SystemExit(f"unexpected preprocessor tensor {key}")


def convert_weights(state: dict) -> dict:
    out = {}
    lstm_bias = {}
    unknown = Counter()
    for key, tensor in state.items():
        if key.startswith("preprocessor."):
            name, arr = featurizer_buffer(key, tensor)
            out[name] = arr
            continue
        if not key.startswith(KNOWN_PREFIXES):
            unknown[key.split(".")[0]] += 1
            continue
        arr = tensor.detach().to("cpu").to(torch.float32).numpy()
        if ".dec_rnn.lstm." in key:
            base, suffix = key.split(".dec_rnn.lstm.")
            stem = f"{base}.dec_rnn.lstm"
            if suffix.startswith("weight_ih_l"):
                out[f"{stem}.{suffix[len('weight_ih_l'):]}.Wx"] = arr
            elif suffix.startswith("weight_hh_l"):
                out[f"{stem}.{suffix[len('weight_hh_l'):]}.Wh"] = arr
            elif suffix.startswith(("bias_ih_l", "bias_hh_l")):
                lstm_bias.setdefault(f"{stem}.{suffix.split('_l')[-1]}.bias", []).append(arr)
            else:
                raise SystemExit(f"unexpected LSTM tensor {key}")
            continue
        if arr.ndim == 4:  # Conv2d (O, I, kH, kW) -> (O, kH, kW, I)
            arr = arr.transpose(0, 2, 3, 1)
        elif arr.ndim == 3:  # Conv1d (O, I, k) -> (O, k, I)
            arr = arr.transpose(0, 2, 1)
        out[key] = np.ascontiguousarray(arr)
    for key, parts in lstm_bias.items():
        if len(parts) != 2:
            raise SystemExit(f"{key}: expected bias_ih and bias_hh, found {len(parts)}")
        out[key] = (parts[0] + parts[1]).astype(np.float32)
    if unknown:
        raise SystemExit(f"unrecognized checkpoint prefixes (refusing to drop silently): {dict(unknown)}")
    return out


def cmd_convert(a) -> None:
    nemo = a.nemo.expanduser().resolve()
    got = sha256_file(nemo)
    if got != MULTITALKER_NEMO_SHA256:
        raise SystemExit(f"{nemo}: sha256 {got} != pinned {MULTITALKER_NEMO_SHA256}")
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cfg, state, members, digests = read_nemo(nemo)
    config = build_config(cfg)
    weights = convert_weights(state)
    save_file(dict(sorted(weights.items())), str(out / "model.safetensors"), metadata={"format": "mlx"})
    (out / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False) + "\n")
    tok = next(n for n in members if n.endswith("tokenizer.model"))
    (out / "tokenizer.model").write_bytes(members[tok])
    outputs = {n: sha256_file(out / n) for n in ("config.json", "model.safetensors", "tokenizer.model")}
    manifest = {
        "source": "nvidia/multitalker-parakeet-streaming-0.6b-v1@8749fc71fd6e2d88ef230159bbf2aea69b524ee1",
        "nemo_sha256": got,
        "nemo_members_sha256": digests,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "tensors": len(weights),
        "outputs_sha256": outputs,
    }
    (out / "conversion_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"tensors": len(weights), **outputs}, indent=1))


def convert_diar_weights(state: dict) -> dict:
    """NeMo Sortformer/Nemotron-3 names to the names `NemotronDiarization::load` reads.

    Values are upcast to f32 (the .nemo stores bfloat16; Frozen E ran them in
    f32 after `diar_model.to(torch.float32)`). The subpixel conv goes to the
    MLX `[out, kernel, in]` layout the loader permutes back.
    """
    out = {}
    for key, tensor in state.items():
        if key.startswith("preprocessor."):
            name, arr = featurizer_buffer(key, tensor)
            out[name] = arr
            continue
        arr = tensor.detach().to("cpu").to(torch.float32).numpy()
        name = key
        if ".ffn.net.0." in key:
            name = key.replace(".ffn.net.0.", ".ffn.linear1.")
        elif ".ffn.net.3." in key:
            name = key.replace(".ffn.net.3.", ".ffn.linear2.")
        elif key.startswith("sortformer_modules.activity_head."):
            name = key.replace("sortformer_modules.activity_head.", "sortformer_modules.activity_head.layers.")
        elif key == "sortformer_modules.subpixel_upsample.weight":
            arr = arr.transpose(0, 2, 1)  # (O, I, k) -> (O, k, I)
        elif ".ffn." in key:
            raise SystemExit(f"unexpected FFN tensor {key}")
        out[name] = np.ascontiguousarray(arr)
    return out


def diar_config(hf_config: dict) -> dict:
    """The loader's config layout, filled from NVIDIA's config.json at the pinned revision."""
    a, h, s = hf_config["audio_config"], hf_config["head_config"], hf_config["streaming_config"]
    if a.get("partial_rotary_factor", 1.0) != 1.0 or a["num_key_value_heads"] != a["num_attention_heads"]:
        raise SystemExit("diarizer config outside the graph the Rust loader implements")
    return {
        "model_type": "nemotron_diarization",
        "num_speakers": h["num_speakers"],
        "dtype": "float32",
        "output_subsampling_factor": 1,
        "encoder_config": {
            "feat_in": a["num_mel_bins"],
            "d_model": a["hidden_size"],
            "n_layers": a["num_hidden_layers"],
            "n_heads": a["num_attention_heads"],
            "subsampling_factor": a["subsampling_factor"],
            "ff_expansion": a["intermediate_size"] / a["hidden_size"],
            "qkv_bias": False,
            "qk_norm": False,
            "pre_block_norm": True,
            "xscaling": False,
            "rope_base": a["rope_parameters"]["rope_theta"],
            "rotary_fraction": a["partial_rotary_factor"],
        },
        "modules_config": {
            "model_type": "sortformer_modules",
            "num_speakers": h["num_speakers"],
            "fc_d_model": h["audio_hidden_size"],
            "tf_d_model": h["hidden_size"],
            "subsampling_factor": h["subsampling_factor"],
            # Offline defaults from config.json; streaming callers override them.
            "chunk_len": hf_config["chunk_length"],
            "fifo_len": hf_config["fifo_length"],
            "spkcache_len": s["speaker_cache_length"],
            "spkcache_update_period": hf_config["speaker_cache_update_period"],
            "chunk_left_context": 0,
            "chunk_right_context": hf_config["chunk_right_context"],
            "spkcache_sil_frames_per_spk": s["speaker_cache_silence_frames_per_speaker"],
            "pred_score_threshold": s["prediction_score_threshold"],
            "max_index": 99999,
            "scores_boost_latest": s["latest_frames_score_boost"],
            "sil_threshold": 0.2,
            "strong_boost_rate": s["strong_boost_rate"],
            "weak_boost_rate": s["weak_boost_rate"],
            "min_pos_scores_rate": s["min_positive_scores_rate"],
            "use_aosc": True,
            "use_learnable_sil_emb": True,
            "use_activity_head": True,
        },
        "processor_config": {
            "feature_size": 128,
            "sampling_rate": 16000,
            "hop_length": 160,
            "n_fft": 512,
            "win_length": 400,
            "preemphasis": 0.97,
            "padding_value": 0.0,
            "pad_to": 16,
        },
        "source_repo": "nvidia/Nemotron-3-Diarization",
        "source_revision": "f667ed73aee57d40cc39428eb768b4fd87a0a29e",
    }


def cmd_convert_diar(a) -> None:
    nemo, hf_cfg = a.nemo.expanduser().resolve(), a.hf_config.expanduser().resolve()
    got = sha256_file(nemo)
    if got != DIAR_NEMO_SHA256:
        raise SystemExit(f"{nemo}: sha256 {got} != pinned {DIAR_NEMO_SHA256}")
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    _, state, _, digests = read_nemo(nemo)
    weights = convert_diar_weights(state)
    save_file(dict(sorted(weights.items())), str(out / "model.safetensors"), metadata={"format": "mlx"})
    (out / "config.json").write_text(json.dumps(diar_config(json.loads(hf_cfg.read_text())), indent=1) + "\n")
    outputs = {n: sha256_file(out / n) for n in ("config.json", "model.safetensors")}
    manifest = {
        "source": "nvidia/Nemotron-3-Diarization@f667ed73aee57d40cc39428eb768b4fd87a0a29e",
        "nemo_sha256": got,
        "hf_config_sha256": sha256_file(hf_cfg),
        "nemo_members_sha256": digests,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "tensors": len(weights),
        "outputs_sha256": outputs,
        "note": "bfloat16 .nemo values upcast to float32 exactly",
    }
    (out / "conversion_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"tensors": len(weights), **outputs}, indent=1))


def tensor_digests(arrays):
    """Content digests that ignore layout: sha256 over each tensor's sorted f32 values plus its sorted shape."""
    out = Counter()
    for arr in arrays:
        flat = np.sort(np.asarray(arr, dtype=np.float32).ravel())
        out[(tuple(sorted(arr.shape)), hashlib.sha256(flat.tobytes()).hexdigest())] += 1
    return out


def cmd_check_diar(a) -> None:
    nemo, st = a.nemo.expanduser().resolve(), a.safetensors.expanduser().resolve()
    for path, want in ((nemo, DIAR_NEMO_SHA256), (st, DIAR_SAFETENSORS_SHA256)):
        got = sha256_file(path)
        if got != want:
            raise SystemExit(f"{path}: sha256 {got} != pinned {want}")
    _, state, _, _ = read_nemo(nemo)
    nemo_named = {
        k: t.detach().to(torch.float32).numpy() for k, t in state.items() if not k.startswith("preprocessor.")
    }
    # NeMo fuses q/k/v into one [3*d, d] matrix per layer; HF stores three [d, d].
    pieces = dict(nemo_named)
    for k, arr in nemo_named.items():
        if arr.ndim == 2 and arr.shape[0] == 3 * arr.shape[1]:
            for i, part in enumerate(np.split(arr, 3, axis=0)):
                pieces[f"{k}[{i}/3]"] = part
    with safe_open(str(st), framework="numpy") as f:
        hf_named = {k: f.get_tensor(k) for k in f.keys()}
    source = tensor_digests(pieces.values())
    missing = []
    for k, arr in sorted(hf_named.items()):
        d = next(iter(tensor_digests([arr])))
        if source[d] == 0:
            missing.append((k, arr.shape))
    hf_digests = tensor_digests(hf_named.values())
    unused = [
        (k, arr.shape)
        for k, arr in sorted(nemo_named.items())
        if next(iter(tensor_digests([arr]))) not in hf_digests
        and not (arr.ndim == 2 and arr.shape[0] == 3 * arr.shape[1])
    ]
    print(f"HF tensors {len(hf_named)}: {len(hf_named) - len(missing)} have an exact .nemo source")
    for k, shape in missing:
        print("HF tensor with no .nemo source:", k, shape)
    for k, shape in unused:
        print(".nemo tensor not in HF (unused by the HF graph):", k, shape)
    sys.exit(1 if missing else 0)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--nemo", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("check-diar")
    d.add_argument("--nemo", type=Path, required=True)
    d.add_argument("--safetensors", type=Path, required=True)
    cd = sub.add_parser("convert-diar")
    cd.add_argument("--nemo", type=Path, required=True)
    cd.add_argument("--hf-config", type=Path, required=True, help="NVIDIA config.json at the pinned revision")
    cd.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    {"convert": cmd_convert, "check-diar": cmd_check_diar, "convert-diar": cmd_convert_diar}[a.cmd](a)


if __name__ == "__main__":
    main()
