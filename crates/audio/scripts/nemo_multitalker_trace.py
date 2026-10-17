#!/usr/bin/env python3
"""Trace NeMo's Frozen E pipeline step by step, for the Rust parity tests.

Runs the unmodified `launch_parallel_streaming` from the pinned NeMo driver
(218d0b4, sha256 checked) on one clip with the Frozen E settings, on CPU, and
records every stage the Rust port must match:

  features.npy         [128, T] whole-file log-mel (the streaming buffer)
  trace.jsonl          one line per step: geometry, diarizer chunk preds and
                       state lengths, gate window maxima, active ids, the exact
                       target/background tensors passed to set_speaker_targets,
                       per-speaker token ids and frames, and stage digests
  steps/step_NNNN.npz  full tensors for the step: ASR chunk, diarizer input,
                       pre-encode output, layer-0 input after the speaker
                       kernels, encoder output, cache lengths, and (for
                       --cache-steps) cache slices for layers 0 and last
  seglst.json          the driver's SegLST output for the clip

The wrappers only observe. They never change arguments or results.

Usage (pinned reference venv):
  python nemo_multitalker_trace.py --driver DRIVER.py --asr ASR.nemo --diar DIAR.nemo \
    --wav CLIP.wav --out DIR [--cache-steps 0,1,2,50] [--threads 1]
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

DRIVER_SHA256 = "7c34077ce6f4286e5ad3f390994683ef957d965bb93e05dea1a51fb2c8b0b56e"
FROZEN_E = {
    "max_num_of_spks": 4,
    "single_speaker_mode": False,
    "masked_asr": False,
    "parallel_speaker_strategy": True,
    "cache_gating": True,
    "binary_diar_preds": True,
    "att_context_size": [70, 13],
    "fifo_len": 264,
    "spkcache_update_period": 222,
    "spk_supervision": "diar",
    "batch_size": 1,
    "precision": "32",
    "use_amp": False,
    "generate_realtime_scripts": False,
    "diar_collar": 0.0,
    "diar_ignore_overlap": False,
    "print_time": False,
    "device": "cpu",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def np32(t) -> np.ndarray:
    # Copy: `.numpy()` shares memory with a CPU f32 tensor, and NeMo mutates some
    # outputs in place afterwards (the driver zeroes speakers >= max_num_of_spks).
    return t.detach().to("cpu").to(torch.float32).numpy().copy()


def summary(arr: np.ndarray) -> dict:
    flat = np.asarray(arr, dtype=np.float32).ravel()
    return {
        "shape": list(np.shape(arr)),
        "l2": float(np.sqrt(np.sum(flat.astype(np.float64) ** 2))),
        "first8": [float(x) for x in flat[:8]],
        "sha256": hashlib.sha256(np.ascontiguousarray(flat).tobytes()).hexdigest(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--driver", type=Path, required=True)
    p.add_argument("--asr", type=Path, required=True)
    p.add_argument("--diar", type=Path, required=True)
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-steps", default="0,1,2")
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    driver_path = a.driver.expanduser().resolve()
    if sha256(driver_path) != DRIVER_SHA256:
        raise SystemExit(f"{driver_path}: not the pinned NeMo 218d0b4 driver")
    out = a.out.expanduser()
    (out / "steps").mkdir(parents=True, exist_ok=True)
    cache_steps = {int(x) for x in a.cache_steps.split(",") if x}

    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)

    spec = importlib.util.spec_from_file_location("frozen_e_driver", driver_path)
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    from omegaconf import OmegaConf

    from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking
    from nemo.collections.asr.parts.utils import multispk_transcribe_utils as mtu
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    import nemo.collections.asr as nemo_asr

    wav = a.wav.expanduser().resolve()
    manifest = out / "manifest.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": str(wav), "duration": None, "session_id": wav.stem}) + "\n")
    cfg = OmegaConf.structured(
        drv.MultitalkerTranscriptionConfig(
            asr_model=str(a.asr.expanduser().resolve()),
            diar_model=str(a.diar.expanduser().resolve()),
            manifest_file=str(manifest),
            output_path=str(out / "seglst.json"),
            **{k: v for k, v in FROZEN_E.items() if k != "att_context_size"},
        )
    )
    cfg.att_context_size = FROZEN_E["att_context_size"]

    # Same setup sequence as the driver's main(), CPU and FP32.
    diar_model = drv.load_diar_model(cfg.diar_model, torch.device("cpu")).to(dtype=torch.float32).eval()
    asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=cfg.asr_model, map_location="cpu")
    asr_model.encoder.set_default_att_context_size(att_context_size=cfg.att_context_size)
    drv.configure_asr_for_multitalker_streaming(cfg, asr_model)
    asr_model = asr_model.to("cpu").eval()
    drv.autocast = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    mtu.validate_feature_frame_strides(asr_model=asr_model, diar_model=diar_model)
    scfg = asr_model.encoder.streaming_cfg
    diar_chunk_len = scfg.valid_out_len + scfg.cache_drop_size
    mtu.configure_diar_streaming(
        diar_model=diar_model,
        cfg=cfg,
        output_subsampling_factor=asr_model.encoder.subsampling_factor,
        diar_chunk_len=diar_chunk_len,
    )
    if isinstance(diar_model.encoder.pre_encode, FeatureStacking) and not cfg.pad_and_drop_preencoded:
        cfg.pad_and_drop_preencoded = True

    samples, _ = mtu.get_multi_talker_samples_from_manifest(
        cfg, manifest_file=str(manifest), feat_per_sec=0.08, max_spks=cfg.max_num_of_spks
    )
    buf = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=cfg.online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )
    buf.append_audio_file(samples[0]["audio_filepath"], stream_id=-1)
    np.save(out / "features.npy", np32(buf.buffer[0]))

    meta = {
        "wav": str(wav),
        "wav_sha256": sha256(wav),
        "driver_sha256": DRIVER_SHA256,
        "torch": torch.__version__,
        "threads": a.threads,
        "streams_length": int(buf.streams_length[0]),
        "pad_and_drop_preencoded": bool(cfg.pad_and_drop_preencoded),
        "streaming_cfg": {k: (v if not hasattr(v, "tolist") else v.tolist()) for k, v in vars(scfg).items()},
        "diar_modules": {
            k: getattr(diar_model.sortformer_modules, k)
            for k in (
                "chunk_len",
                "chunk_left_context",
                "chunk_right_context",
                "fifo_len",
                "spkcache_len",
                "spkcache_update_period",
            )
        },
        "diar_output_subsampling_factor": diar_model.output_subsampling_factor,
        "diar_async_streaming": bool(getattr(diar_model, "async_streaming", False)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))

    # ----------------------------------------------------------- observers
    rec = {}
    tensors = {}
    trace_f = (out / "trace.jsonl").open("w")
    STA = mtu.SpeakerTaggedASR

    orig_perform = STA.perform_parallel_streaming_stt_spk

    def perform(self, step_num, chunk_audio, chunk_lengths, is_buffer_empty, drop_extra_pre_encoded, **kw):
        rec.clear()
        tensors.clear()
        rec.update(
            step=int(step_num),
            asr_chunk_frames=int(chunk_audio.shape[-1]),
            asr_chunk_len=int(chunk_lengths[0]),
            is_buffer_empty=bool(is_buffer_empty),
            drop_extra_pre_encoded=int(drop_extra_pre_encoded),
        )
        tensors["asr_chunk"] = np32(chunk_audio[0])
        # NeMo's @measure_eta wrapper reads kwargs["step_num"], so pass keywords.
        result = orig_perform(
            self,
            step_num=step_num,
            chunk_audio=chunk_audio,
            chunk_lengths=chunk_lengths,
            is_buffer_empty=is_buffer_empty,
            drop_extra_pre_encoded=drop_extra_pre_encoded,
            **kw,
        )
        if rec.get("skipped") is None and "active" not in rec:
            rec["skipped"] = True
        trace_f.write(json.dumps(rec) + "\n")
        trace_f.flush()
        np.savez(out / "steps" / f"step_{int(step_num):04d}.npz", **tensors)
        return result

    STA.perform_parallel_streaming_stt_spk = perform

    orig_diar_step = diar_model.forward_streaming_step

    def diar_step(*args, **kw):
        sig, sig_len = kw["processed_signal"], kw["processed_signal_length"]
        tensors["diar_input"] = np32(sig[0])
        rec["diar_input_frames"] = int(sig.shape[1])
        rec["diar_input_len"] = int(sig_len[0])
        rec["diar_drop_extra"] = int(kw.get("drop_extra_pre_encoded", 0))
        state, total = orig_diar_step(*args, **kw)
        rec["diar_total_frames"] = int(total.shape[1])
        rec["spkcache_len"] = int(state.spkcache.shape[1])
        rec["fifo_len"] = int(state.fifo.shape[1])
        tensors["diar_total_tail"] = np32(total[0, -14:])
        return state, total

    diar_model.forward_streaming_step = diar_step

    orig_find = STA._find_active_speakers

    def find_active(self, diar_preds, n_active_speakers_per_stream):
        rec["gate_window_frames"] = int(diar_preds.shape[1])
        rec["gate_window_max"] = [float(x) for x in torch.max(diar_preds, dim=1).values[0]]
        ids = orig_find(self, diar_preds, n_active_speakers_per_stream)
        rec["active"] = ids[0]
        return ids

    STA._find_active_speakers = find_active

    orig_update_diar = mtu.MultiTalkerInstanceManager.update_diar_state

    def update_diar(self, diar_pred_out_stream, previous_chunk_preds, diar_streaming_state):
        tensors["chunk_preds"] = np32(previous_chunk_preds[0])
        rec["chunk_preds"] = tensors["chunk_preds"].tolist()
        return orig_update_diar(self, diar_pred_out_stream, previous_chunk_preds, diar_streaming_state)

    mtu.MultiTalkerInstanceManager.update_diar_state = update_diar

    orig_pre = STA.forward_pre_encoded

    def pre_encoded(self, audio_signal, length, drop_extra_pre_encoded=0):
        y, ylen = orig_pre(self, audio_signal, length, drop_extra_pre_encoded)
        tensors["pre_encoded"] = np32(y[0])
        rec["pre_encoded_len"] = int(ylen[0])
        rec["pre_encoded"] = summary(tensors["pre_encoded"])
        return y, ylen

    STA.forward_pre_encoded = pre_encoded

    orig_targets = asr_model.set_speaker_targets

    def set_targets(spk, bg=None):
        tensors["spk_targets"] = np32(spk)
        tensors["bg_targets"] = np32(bg)
        rec["spk_targets"] = tensors["spk_targets"].tolist()
        rec["bg_targets"] = tensors["bg_targets"].tolist()
        return orig_targets(spk, bg)

    asr_model.set_speaker_targets = set_targets

    # Registered after NeMo's speaker-kernel pre-hook, so it sees the kernel output.
    def layer0_hook(module, args, kwargs):
        x = kwargs["x"] if "x" in kwargs else args[0]
        tensors["layer0_input"] = np32(x)
        rec["layer0_input"] = summary(tensors["layer0_input"])
        return None

    asr_model.encoder.layers[0].register_forward_pre_hook(layer0_hook, with_kwargs=True)

    orig_enc_step = asr_model.encoder.cache_aware_stream_step

    def enc_step(*args, **kw):
        res = orig_enc_step(*args, **kw)
        encoded, enc_len, ch, tm, ch_len = res
        tensors["encoded"] = np32(encoded)
        tensors["encoded_len"] = enc_len.detach().cpu().numpy()
        tensors["cache_last_channel_len"] = ch_len.detach().cpu().numpy()
        rec["encoded"] = summary(tensors["encoded"])
        rec["cache_last_channel_len"] = [int(x) for x in ch_len]
        if rec["step"] in cache_steps:
            tensors["cache_last_channel_first_last"] = np.stack([np32(ch[0]), np32(ch[-1])])
            tensors["cache_last_time_first_last"] = np.stack([np32(tm[0]), np32(tm[-1])])
        return res

    asr_model.encoder.cache_aware_stream_step = enc_step

    orig_stream = asr_model.conformer_stream_step

    def stream_step(*args, **kw):
        res = orig_stream(*args, **kw)
        hyps = res[5]
        rec["tokens"] = [[int(t) for t in h.y_sequence] for h in hyps]
        rec["token_frames"] = [[int(t) for t in h.timestamp] for h in hyps]
        rec["texts"] = [h.text for h in hyps]
        rec["decoded_length"] = [
            int(h.dec_state.decoded_length) if getattr(h.dec_state, "decoded_length", None) is not None else None
            for h in hyps
        ]
        return res

    asr_model.conformer_stream_step = stream_step

    # ------------------------------------------------------------ run
    streamer = drv.launch_parallel_streaming(
        cfg=cfg,
        asr_model=asr_model,
        diar_model=diar_model,
        streaming_buffer=buf,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )
    trace_f.close()
    seglst = streamer.generate_seglst_dicts_from_parallel_streaming(samples=samples)
    mtu.write_seglst_file(seglst_dict_list=seglst, output_path=str(out / "seglst.json"))
    np.save(out / "diar_preds.npy", np32(streamer.instance_manager.diar_states.diar_pred_out_stream[0]))
    print(f"traced {sum(1 for _ in (out / 'trace.jsonl').open())} steps into {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
