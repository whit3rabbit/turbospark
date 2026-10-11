#!/usr/bin/env python3
"""Convert MegaTTS3 PyTorch checkpoints to the safetensors layout this crate loads.

Why this exists: every public MegaTTS3 upload ships the aligner, duration LM,
DiT and WavVAE as PyTorch pickle `.ckpt` files. The Rust runtime reads
safetensors only, so conversion happens once, offline, here.

Safety: the sources are untrusted pickles. This script never calls
`torch.load` and never imports torch. It parses the torch zip container with
an allowlisted unpickler (tensor rebuild helpers, storage type markers,
OrderedDict) and refuses any other global. Every source file must match a
pinned sha256 before it is parsed. Run it in a scratch directory, not the repo
root. Dependencies: Python 3 stdlib and numpy.

Output layout (MLX convention, matching `codec/wnconv.rs`):
  * 3-D conv weights swap their last two axes: Conv1d `[out, in, K]` ->
    `[out, K, in]`, ConvTranspose1d `[in, out, K]` -> `[in, K, out]`. Both are
    the same swap, so no per-module classification is needed.
  * `weight_g` / `weight_v` pairs stay unfolded (`weight_g` is `[d0, 1, 1]`);
    the Rust loader folds them, like `dacvae` does.
  * Everything else is copied unchanged. Floats are widened to f32 (lossless
    for f16/bf16 sources). Other dtypes are an error, so a surprise surfaces
    in `--inventory` instead of becoming a silent mis-load.

Usage:
  convert_megatts3.py --src SRC --inventory
  convert_megatts3.py --src SRC --out OUT --dict-json dict.json --license LICENSE \
      [--wavvae community|official]

SRC is a Hugging Face snapshot directory (`aligner_lm/`, `diffusion_transformer/`,
`duration_lm/`, `wavvae/`, `g2p/`). `dict.json` is
`tts/utils/text_utils/dict.json` and LICENSE is the Apache-2.0 text, both from
the MegaTTS3 code repo. The output also gets a root `config.json` (the runtime
dispatches on its `model_type`), `NOTICE.md` (sources and provenance) and a
model card `README.md`, so the folder can be published as is. Every output is
deterministic: the same inputs give the same bytes, which catalog pins rely on.
"""

import argparse
import collections
import hashlib
import json
import os
import pickle
import re
import shutil
import struct
import sys
import zipfile

import numpy as np

# Upstream revisions the pins below were taken from (HF LFS sha256).
SOURCES = {
    "ByteDance/MegaTTS3": "409a7002b006d80f0730fca6f80441b08c10e738",
    "mrfakename/MegaTTS3-VoiceCloning": "b3a2c297eea319931ee985751f117f866085d0c5",
}

# relpath -> (sha256, size). The four non-WavVAE components are byte-identical
# in both repos; only the WavVAE differs.
PINNED = {
    "aligner_lm/model_only_last.ckpt": (
        "a00f18ec36f8c1328ddab7a405c8e388790a1c14fdbdd07c546fcacaf5d19296",
        218434266,
    ),
    "diffusion_transformer/model_only_last.ckpt": (
        "12233b95be177504551034390cf71aa748f0c66cbe2fd0ce433b9f9686122da9",
        1836341777,
    ),
    "duration_lm/model_only_last.ckpt": (
        "0f21f4205c5d3ec4bef69716a85ca3d37f25c35b429bac500477a2085039b43f",
        267955084,
    ),
    "g2p/model.safetensors": (
        "0f9d70d454ee35d023a9a54552716a8ccf2411c967abc6a857160527046f62a2",
        1018490136,
    ),
    "g2p/tokenizer.json": (
        "18bf578a236efa19e36ee7be04c327ba4abc23aed0213a31d3199a55ea7d2411",
        14796960,
    ),
    # Official: decoder only (no encoder, cannot clone from a raw wav).
    "wavvae/decoder.ckpt": (
        "f309a09fa0c629551d86b0bc68f5d0c4a24448c5fea6bbed421fb3772977a363",
        904541298,
    ),
    # Community: encoder + decoder. ByteDance withheld the encoder on purpose;
    # this file is a third-party re-upload whose origin we could not verify.
    "wavvae/model_only_last.ckpt": (
        "5f432d0740f5238c573be49097a69ac5271e24d0d310e6ea63023f038e5e408a",
        1022420843,
    ),
}

# component dir -> (ckpt file, key inside checkpoint["state_dict"]); the keys
# are the names `MegaTTS3DiTInfer.build_model` passes to `load_ckpt`.
COMPONENTS = {
    "aligner_lm": ("model_only_last.ckpt", "model"),
    "diffusion_transformer": ("model_only_last.ckpt", "dit"),
    "duration_lm": ("model_only_last.ckpt", "dur_model"),
}
WAVVAE_FILE = {
    "official": "decoder.ckpt",
    "community": "model_only_last.ckpt",
}
G2P_COPY = [
    "config.json",
    "generation_config.json",
    "added_tokens.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "model.safetensors",
]

# Storage type name -> numpy dtype; bf16 is handled separately.
_STORAGE_DTYPES = {
    "FloatStorage": "<f4",
    "HalfStorage": "<f2",
    "DoubleStorage": "<f8",
    "LongStorage": "<i8",
    "IntStorage": "<i4",
    "ShortStorage": "<i2",
    "CharStorage": "i1",
    "ByteStorage": "u1",
    "BoolStorage": "?",
    "BFloat16Storage": "bf16",
}


class ConvertError(Exception):
    pass


# ---------------------------------------------------------------- unpickler


class _StorageType:
    def __init__(self, name):
        self.name = name


class _Storage:
    def __init__(self, type_name, key, numel, view=None):
        self.type_name = type_name
        self.key = key
        self.numel = numel  # elements in the root storage blob
        self.view = view  # legacy format only: (element offset, element count)


class _LazyTensor:
    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = offset
        self.size = tuple(size)
        self.stride = tuple(stride)


def _rebuild_tensor_v2(
    storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None
):
    return _LazyTensor(storage, storage_offset, size, stride)


def _rebuild_parameter(data, requires_grad=False, backward_hooks=None, *rest):
    return data


_ALLOWED_GLOBALS = {
    ("torch._utils", "_rebuild_tensor"): _rebuild_tensor_v2,  # legacy format, same leading args
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
    ("collections", "OrderedDict"): collections.OrderedDict,
}
for _name in _STORAGE_DTYPES:
    _ALLOWED_GLOBALS[("torch", _name)] = _StorageType(_name)


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, fh):
        super().__init__(fh)
        self.storages = {}  # root key -> _Storage, in first-seen order

    def find_class(self, module, name):
        try:
            return _ALLOWED_GLOBALS[(module, name)]
        except KeyError:
            raise ConvertError(f"refusing pickle global {module}.{name}") from None

    def persistent_load(self, pid):
        # zip format: ("storage", type, key, location, numel)
        # legacy:     ("storage", type, key, location, numel, view_metadata)
        if not (isinstance(pid, tuple) and len(pid) in (5, 6) and pid[0] == "storage"):
            raise ConvertError(f"unexpected persistent id {pid!r}")
        storage_type, key, numel = pid[1], pid[2], pid[4]
        if not isinstance(storage_type, _StorageType):
            raise ConvertError("storage type is not an allowlisted marker")
        if not isinstance(numel, int) or numel < 0:
            raise ConvertError(f"bad storage numel {numel!r}")
        view = None
        if len(pid) == 6 and pid[5] is not None:
            meta = pid[5]
            if not (isinstance(meta, tuple) and len(meta) == 3 and all(isinstance(x, (int, str)) for x in meta)):
                raise ConvertError(f"bad view metadata {meta!r}")
            _view_key, off, size = meta
            if not (isinstance(off, int) and isinstance(size, int) and 0 <= off and 0 <= size and off + size <= numel):
                raise ConvertError(f"view {meta!r} outside storage of {numel} elements")
            view = (off, size)
        key = str(key)
        prev = self.storages.get(key)
        if prev is not None and (prev.type_name, prev.numel) != (storage_type.name, numel):
            raise ConvertError(f"storage {key} declared twice with different shapes")
        self.storages.setdefault(key, _Storage(storage_type.name, key, numel))
        return _Storage(storage_type.name, key, numel, view)


def _bf16_to_f32(raw):
    # bf16 is the top half of an f32, so a 16-bit left shift is exact.
    return (raw.astype("<u4") << 16).view("<f4")


def _materialize(tensor, read_raw):
    st = tensor.storage
    raw = read_raw(st.key)
    dt = _STORAGE_DTYPES[st.type_name]
    if dt == "bf16":
        flat = _bf16_to_f32(np.frombuffer(raw, dtype="<u2"))
    else:
        flat = np.frombuffer(raw, dtype=dt)
    if flat.size != st.numel:
        raise ConvertError(f"storage {st.key}: {flat.size} elements, header says {st.numel}")
    if st.view is not None:
        flat = flat[st.view[0] : st.view[0] + st.view[1]]
    if any(d < 0 for d in tensor.size) or any(s < 0 for s in tensor.stride):
        raise ConvertError("negative size or stride")
    if len(tensor.size) != len(tensor.stride):
        raise ConvertError("size/stride rank mismatch")
    count = int(np.prod(tensor.size, dtype=object)) if tensor.size else 1
    if count:
        last = tensor.offset + sum((d - 1) * s for d, s in zip(tensor.size, tensor.stride))
        if tensor.offset < 0 or last >= flat.size:
            raise ConvertError("tensor view runs past its storage")
    item = flat.itemsize
    view = np.lib.stride_tricks.as_strided(
        flat[tensor.offset :] if count else flat[:0],
        shape=tensor.size,
        strides=tuple(s * item for s in tensor.stride),
        writeable=False,
    )
    return np.array(view)  # contiguous copy, detached from the file buffer


_LEGACY_MAGIC = 0x1950A86A20F9469CFC6C
_LEGACY_PROTOCOL = 1001


def _extract_state_dict(obj, path, component_key, read_raw):
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise ConvertError(f"{path}: no top-level state_dict")
    sd_all = obj["state_dict"]
    if component_key not in sd_all:
        raise ConvertError(f"{path}: state_dict has {sorted(sd_all)}, wanted {component_key!r}")
    out = {}
    for name, t in sd_all[component_key].items():
        if not isinstance(t, _LazyTensor):
            raise ConvertError(f"{path}: {component_key}.{name} is not a tensor")
        # Same key cleanup as the reference `load_ckpt` (DDP / torch.compile).
        clean = name.replace("module.", "").replace("_orig_mod.", "")
        out[clean] = _materialize(t, read_raw)
    return out


def _load_zip(path, component_key):
    with zipfile.ZipFile(path) as zf:
        pkls = [n for n in zf.namelist() if n.endswith("/data.pkl")]
        if len(pkls) != 1:
            raise ConvertError(f"{path}: expected one data.pkl, found {len(pkls)}")
        root = pkls[0][: -len("/data.pkl")]
        if f"{root}/byteorder" in zf.namelist():
            if zf.read(f"{root}/byteorder").strip() != b"little":
                raise ConvertError(f"{path}: big-endian checkpoint")
        with zf.open(pkls[0]) as fh:
            obj = _RestrictedUnpickler(fh).load()
        cache = {}

        def read_raw(key):
            if key not in cache:
                cache[key] = zf.read(f"{root}/data/{key}")
            return cache[key]

        return _extract_state_dict(obj, path, component_key, read_raw)


def _load_legacy(path, component_key):
    """Pre-1.6 torch format: four pickles, then every storage blob in key order.

    Each blob is an int64 element count followed by the raw elements, which is
    what `torch.serialization._legacy_load` consumes.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        if _RestrictedUnpickler(fh).load() != _LEGACY_MAGIC:
            raise ConvertError(f"{path}: not a torch checkpoint (bad magic)")
        if _RestrictedUnpickler(fh).load() != _LEGACY_PROTOCOL:
            raise ConvertError(f"{path}: unsupported legacy protocol version")
        sys_info = _RestrictedUnpickler(fh).load()
        if not (isinstance(sys_info, dict) and sys_info.get("little_endian") is True):
            raise ConvertError(f"{path}: big-endian or unreadable sys_info")
        main = _RestrictedUnpickler(fh)
        obj = main.load()
        keys = _RestrictedUnpickler(fh).load()
        if not (isinstance(keys, list) and all(isinstance(k, str) for k in keys)):
            raise ConvertError(f"{path}: storage key list is not a list of strings")
        if sorted(keys) != sorted(main.storages) or len(set(keys)) != len(keys):
            raise ConvertError(f"{path}: storage key list does not match the tensors")
        blobs = {}
        for key in keys:
            st = main.storages[key]
            dt = _STORAGE_DTYPES[st.type_name]
            item = 2 if dt == "bf16" else np.dtype(dt).itemsize
            head = fh.read(8)
            if len(head) != 8 or struct.unpack("<q", head)[0] != st.numel:
                raise ConvertError(f"{path}: storage {key} element count mismatch")
            nbytes = st.numel * item
            if fh.tell() + nbytes > size:
                raise ConvertError(f"{path}: storage {key} runs past end of file")
            blobs[key] = fh.read(nbytes)
            if len(blobs[key]) != nbytes:
                raise ConvertError(f"{path}: storage {key} truncated")
        return _extract_state_dict(obj, path, component_key, blobs.__getitem__)


def load_ckpt_state_dict(path, component_key):
    """Read `checkpoint["state_dict"][component_key]` as {name: ndarray}."""
    if zipfile.is_zipfile(path):
        return _load_zip(path, component_key)
    return _load_legacy(path, component_key)


# ------------------------------------------------------------ transformation


def convert_tensor(name, arr):
    """Return (new_name, new_array), or None for keys that are dropped."""
    name = name.replace("parametrizations.weight.original0", "weight_g")
    name = name.replace("parametrizations.weight.original1", "weight_v")
    if name.endswith("num_batches_tracked"):
        return None
    if arr.dtype.kind != "f":
        raise ConvertError(f"{name}: non-float dtype {arr.dtype}; decide explicitly before keeping it")
    arr = arr.astype("<f4", copy=False)
    if name.endswith(".weight_g"):
        if arr.ndim != 3 or arr.shape[1:] != (1, 1):
            raise ConvertError(f"{name}: weight_g shape {arr.shape}, expected [d0,1,1]")
        return name, np.ascontiguousarray(arr)
    if name.endswith((".weight_v", ".weight")):
        if arr.ndim == 4:
            raise ConvertError(f"{name}: 4-D weight; no Conv2d is expected in MegaTTS3")
        if arr.ndim == 3:
            return name, np.ascontiguousarray(arr.transpose(0, 2, 1))
    return name, np.ascontiguousarray(arr)


def convert_state_dict(sd):
    out = {}
    for name, arr in sd.items():
        res = convert_tensor(name, arr)
        if res is None:
            continue
        new_name, new_arr = res
        if new_name in out:
            raise ConvertError(f"duplicate tensor after renaming: {new_name}")
        out[new_name] = new_arr
    for name in list(out):
        if name.endswith(".weight_v"):
            g = name[: -len("weight_v")] + "weight_g"
            if g not in out:
                raise ConvertError(f"{name} has no matching weight_g")
            if out[g].shape[0] != out[name].shape[0]:
                raise ConvertError(f"{name}: weight_g/weight_v disagree on dim 0")
        elif name.endswith(".weight_g"):
            if name[: -len("weight_g")] + "weight_v" not in out:
                raise ConvertError(f"{name} has no matching weight_v")
    return out


# --------------------------------------------------------------------- output


def write_safetensors(path, tensors, metadata):
    header = {"__metadata__": {k: str(v) for k, v in metadata.items()}}
    offset = 0
    for name in sorted(tensors):
        arr = tensors[name]
        assert arr.dtype == np.dtype("<f4")
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + arr.nbytes],
        }
        offset += arr.nbytes
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)  # keep the data section 8-byte aligned
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for name in sorted(tensors):
            fh.write(tensors[name].tobytes(order="C"))
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pinned(src, rel):
    path = os.path.join(src, rel)
    want_sha, want_size = PINNED[rel]
    if not os.path.isfile(path):
        raise ConvertError(f"missing source file {rel}")
    if os.path.getsize(path) != want_size:
        raise ConvertError(f"{rel}: size {os.path.getsize(path)}, pinned {want_size}")
    got = sha256_file(path)
    if got != want_sha:
        raise ConvertError(f"{rel}: sha256 {got} does not match pin {want_sha}")
    return got


_FLAT_SCALAR = re.compile(r"^([A-Za-z0-9_]+): ([^\s#\[{][^#]*?)\s*$")


def flat_yaml_scalars(path):
    """Top-level `key: scalar` lines of a training config (no PyYAML needed)."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = _FLAT_SCALAR.match(line.rstrip("\n"))
            if not m:
                continue
            key, val = m.groups()
            low = val.lower()
            if low in ("true", "false"):
                out[key] = low == "true"
            else:
                try:
                    out[key] = int(val)
                except ValueError:
                    try:
                        out[key] = float(val)
                    except ValueError:
                        out[key] = val.strip("'\"")
    return out


def load_dict_json(path):
    with open(path, encoding="utf-8-sig") as fh:
        data = json.load(fh)
    for key in ("phone", "tone"):
        if not (isinstance(data.get(key), list) and all(isinstance(x, str) for x in data[key])):
            raise ConvertError(f"{path}: '{key}' must be a list of strings")
    return data


# --------------------------------------------------------------- root files

# Bumped when the converted layout changes in a way a loader must notice.
FORMAT_VERSION = 1
ENCODER_PROVENANCE = (
    "ByteDance withheld the WavVAE encoder from its release. The encoder in this "
    "folder is a third-party re-upload by mrfakename (mrfakename/MegaTTS3-VoiceCloning), "
    "credited to ModelScope ACoderPassBy/MegaTTS-SFT. Its origin and license were not "
    "independently verified; the re-upload's repository declares Apache-2.0."
)


def root_config(wavvae_variant):
    """The root `config.json`: runtimes dispatch on `model_type`, and
    `voice_clone` must agree with whether the WavVAE carries its encoder."""
    return {
        "format_version": FORMAT_VERSION,
        "model_type": "megatts3",
        "sample_rate": 24000,
        "voice_clone": wavvae_variant == "community",
        "wavvae_variant": wavvae_variant,
    }


def notice_text(wavvae_variant):
    """NOTICE.md: where every source came from. Deterministic (no timestamps),
    so the same inputs always give the same bytes and the same pins."""
    used = [f"aligner_lm/{COMPONENTS['aligner_lm'][0]}",
            f"diffusion_transformer/{COMPONENTS['diffusion_transformer'][0]}",
            f"duration_lm/{COMPONENTS['duration_lm'][0]}",
            "g2p/model.safetensors", "g2p/tokenizer.json",
            f"wavvae/{WAVVAE_FILE[wavvae_variant]}"]
    lines = [
        "# Notice",
        "",
        "MegaTTS3 (ByteDance) weights converted to safetensors for TurboSpark by",
        "`crates/audio/tools/convert_megatts3.py`. Values are unchanged: f16 and bf16",
        "are widened to f32 and conv weights reorder their axes; nothing was retrained.",
        "",
        "## Sources",
        "",
        f"- `ByteDance/MegaTTS3` at `{SOURCES['ByteDance/MegaTTS3']}`: aligner, DiT,",
        "  duration LM and G2P. Declared license: Apache-2.0.",
    ]
    if wavvae_variant == "community":
        lines += [
            f"- `mrfakename/MegaTTS3-VoiceCloning` at `{SOURCES['mrfakename/MegaTTS3-VoiceCloning']}`:",
            "  the WavVAE with its encoder.",
        ]
    else:
        lines += ["- The WavVAE is ByteDance's decoder-only release (no voice cloning)."]
    lines += ["", "| Source file | sha256 |", "|---|---|"]
    lines += [f"| `{rel}` | `{PINNED[rel][0]}` |" for rel in used]
    if wavvae_variant == "community":
        lines += ["", "## WavVAE encoder provenance", "", ENCODER_PROVENANCE]
    lines += [
        "",
        "## Use",
        "",
        "Clone only voices you have the right to use. TurboSpark asks for explicit",
        "consent on every cloning request.",
        "",
    ]
    return "\n".join(lines)


TURBOSPARK_URL = "https://github.com/whit3rabbit/turbospark"
PORT_README = TURBOSPARK_URL + "/blob/main/crates/audio/src/tts/megatts3/README.md"
CONVERTER = "crates/audio/tools/convert_megatts3.py"

# path -> what it is, for the model card's file table.
CARD_ROWS = [
    ("config.json", "Root config. TurboSpark dispatches on `model_type: megatts3`, and `voice_clone` says whether the WavVAE encoder is present."),
    ("aligner_lm/", "Whisper-style aligner. Finds the phones and their timing in the reference recording."),
    ("duration_lm/", "Duration model. Predicts how long each phone of the new text lasts."),
    ("diffusion_transformer/", "24-layer latent diffusion transformer. Generates the speech latents in the reference voice."),
    ("wavvae/", "WavVAE codec. The encoder turns the reference into a latent, and the decoder turns latents into 24 kHz audio."),
    ("g2p/", "Qwen2-0.5B grapheme-to-phoneme model and tokenizer, copied unchanged (bf16)."),
    ("frontend/dict.json", "Upstream phone and tone dictionary. Kept for reference. TurboSpark does not read it."),
    ("manifest.json", "sha256, size, and tensor count of every file, plus the upstream sources."),
    ("NOTICE.md", "Exact upstream sources, their sha256, and where the encoder came from."),
    ("LICENSE", "Apache-2.0, from the upstream MegaTTS3 code."),
]


def _human_size(n):
    for unit, scale in (("GB", 10**9), ("MB", 10**6), ("KB", 10**3)):
        if n >= scale:
            return f"{n / scale:.3g} {unit}"
    return f"{n} B"


def _row_numbers(path, files):
    """Parameter count and size of a table row, from the manifest."""
    if path.endswith("/"):
        weights = files.get(path + "model.safetensors", {})
        size = sum(v["size"] for k, v in files.items() if k.startswith(path))
        params = weights.get("params")
        return (f"{params / 1e6:.1f}M" if params else ""), _human_size(size)
    entry = files.get(path)
    return "", (_human_size(entry["size"]) if entry else "")


# Measured in the port README (Verification, Steps and quality) on an Apple M4
# Max. Change both together: these are results, not generated values.
CARD_BENCHMARKS = [
    "## Benchmarks",
    "",
    "Measured with TurboSpark on one Apple M4 Max (36 GB). At the default settings the Metal",
    "path takes about 5 s per second of speech and peaks near 9 GiB for a short request. The",
    "timing ran on battery with other work on the machine, so read it as ratios between rows,",
    f"not absolute numbers. Method and raw results are in the [port's README]({PORT_README}).",
    "",
    "### Speed",
    "",
    "One 3.04 s sentence cloned from a 2.8 s reference. Each row is the median of three",
    "interleaved runs, each a fresh process after a discarded warmup. RTF is seconds of",
    "compute per second of speech.",
    "",
    "| Device | Steps | Time | RTF |",
    "|---|---|---|---|",
    "| Metal | 8 (Fast) | 12.7 s | 4.2 |",
    "| Metal | 16 (Balanced, default) | 15.5 s | 5.1 |",
    "| Metal | 32 (Best) | 24.6 s | 8.1 |",
    "| CPU | 32 | 128.5 s | 42.3 |",
    "",
    "### Memory",
    "",
    "Peak process footprint at 32 steps, the figure macOS charges the app, one run per request:",
    "",
    "| Request | Metal | CPU |",
    "|---|---|---|",
    "| 2.8 s reference, 3.0 s of speech | 8.6 GiB | 6.6 GiB |",
    "| 7.0 s reference, 7.4 s of speech | 9.8 GiB | 8.4 GiB |",
    "| 27.5 s reference, 13.3 s of speech | 14.3 GiB | 11.4 GiB |",
    "",
    "Metal holds 1.4 to 2.9 GiB more because the uploaded weights keep a CPU copy. TurboSpark",
    "checks memory before every request and refuses one that would not fit. By that check, a",
    "16 GB Mac would run the short request and refuse the 7 s one on either device. That is",
    "arithmetic from these numbers, not a run on 16 GB hardware.",
    "",
    "### Accuracy",
    "",
    "- **Against PyTorch.** With every random draw replayed, TurboSpark's CPU path matches the",
    "  upstream PyTorch pipeline on these weights: max sample difference 5.4e-5 and 1.1e-3 on",
    "  two cases, correlation 1.000000.",
    "- **Metal against CPU.** Same seed, 32 steps, 8 LibriSpeech test-clean speakers and two",
    "  multi-chunk passages: identical lengths, RMS difference at most 2.3e-5, and a largest",
    "  single-sample difference of 1.8e-3 (on a 13 s passage).",
    "- **Steps.** 8 LibriSpeech speakers, two sentences each, on Metal. WER is distil-whisper",
    "  large-v3. Similarity is the cosine of WavLM-base-plus-sv embeddings against the",
    "  reference: a second real recording of the same speaker scores 0.963, and two different",
    "  speakers 0.774.",
    "",
    "| Steps | WER | Similarity |",
    "|---|---|---|",
    "| 4 | 8.5% | 0.968 |",
    "| 8 (Fast) | 6.4% | 0.971 |",
    "| 16 (Balanced) | 6.8% | 0.976 |",
    "| 32 (Best) | 6.4% | 0.974 |",
    "| 32, another seed | 8.2% | 0.973 |",
    "",
    "From 8 steps up, no difference is larger than changing the seed. Most of the WER is rare",
    "names and archaic words in the LibriSpeech texts, which every setting misreads alike.",
    "None of this is a listening test.",
    "",
]


def model_card(wavvae_variant, files, hub_repo=None):
    """README.md for the Hugging Face repo. Deterministic: sizes and parameter
    counts come from the manifest, so a rerun writes the same bytes."""
    clone = wavvae_variant == "community"
    folder = hub_repo.split("/")[-1] if hub_repo else "MegaTTS3-TurboSpark"
    table = ["| Path | What it is | Params | Size |", "|---|---|---|---|"]
    for path, what in CARD_ROWS:
        if path == "wavvae/" and not clone:
            what = "WavVAE decoder only (ByteDance's release). It turns latents into 24 kHz audio but cannot encode a reference."
        params, size = _row_numbers(path, files)
        table.append(f"| `{path}` | {what} | {params} | {size} |")
    tags = ["- megatts3", "- text-to-speech", "- safetensors", "- turbospark"]
    if clone:
        tags.insert(1, "- voice-cloning")
    head = [
        "---",
        "license: apache-2.0",
        "base_model: ByteDance/MegaTTS3",
        "pipeline_tag: text-to-speech",
        "language:",
        "- en",
        "tags:",
        *tags,
        "---",
        "",
        "# MegaTTS3 for TurboSpark (MLX-layout safetensors)",
        "",
    ]
    if clone:
        intro = [
            "ByteDance's MegaTTS3 zero-shot voice cloning model as MLX-layout safetensors, converted",
            f"for [TurboSpark]({TURBOSPARK_URL}), which runs it natively on Apple Silicon (on Metal by",
            "default). Give it 1 to 30 seconds of someone's voice and some English text, and it",
            "speaks the text in that voice at 24 kHz. The weights are the upstream values in a new",
            "file format, not a retrain or a fine-tune.",
        ]
    else:
        intro = [
            "ByteDance's MegaTTS3 model as MLX-layout safetensors, converted for",
            f"[TurboSpark]({TURBOSPARK_URL}). It carries ByteDance's decoder-only WavVAE, so it cannot",
            "clone a voice from a recording, and TurboSpark's runtime refuses it for that reason. The",
            "weights are the upstream values in a new file format, not a retrain or a fine-tune.",
        ]
    body = [
        *intro,
        "",
        "MLX layout means the converted tensors follow MLX's conventions: f32, with conv weights",
        "stored as `[out, kernel, in]`. It is not an MLX model: mlx-audio has no MegaTTS3 port,",
        "so these files will not load there, and the upstream PyTorch code cannot load them",
        "without reversing the conversion. TurboSpark's Rust runtime loads them directly. If you",
        "are not using TurboSpark, you probably want the",
        "[upstream release](https://huggingface.co/ByteDance/MegaTTS3) instead.",
        "",
        "## Why this exists",
        "",
        "TurboSpark is a hobby project: a native macOS app and Rust workspace for running local",
        "models on Apple Silicon. Its runtime reads safetensors only, and its model catalog",
        "installs only files pinned to a commit and a sha256. MegaTTS3 ships as PyTorch pickle",
        "checkpoints, which TurboSpark will not load, because opening a pickle can run code.",
        "",
        "Converting once and publishing the result lets the app install MegaTTS3 like any other",
        "model and check every byte before it runs. It also means you can rerun the converter",
        "and compare hashes rather than trusting this upload.",
        "",
        "## What is in the repo",
        "",
        *table,
        "",
        "Each component folder also has a `config.json` with the upstream hyperparameters and the",
        "sha256 of the checkpoint it came from.",
        "",
        "## How it was converted",
        "",
        f"Every file here comes from `{CONVERTER}` in the TurboSpark repo.",
        "",
        "- It checks each upstream file against a pinned sha256 before reading it.",
        "- It parses the checkpoints with an allowlisted unpickler and never imports torch.",
        "- It widens f16 and bf16 tensors to f32, which loses nothing.",
        "- It stores conv weights as `[out, kernel, in]`, the layout MLX uses, and keeps weight",
        "  norm as separate `weight_g` and `weight_v` tensors.",
        "- It writes the same bytes for the same inputs, so `manifest.json` doubles as a",
        "  reproducibility check.",
        "",
        "Sources, also listed with their hashes in `NOTICE.md`:",
        "",
        f"- [ByteDance/MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3/tree/{SOURCES['ByteDance/MegaTTS3']})"
        f" at `{SOURCES['ByteDance/MegaTTS3'][:8]}`: the aligner, duration model, diffusion transformer, and G2P.",
    ]
    if clone:
        body.append(
            "- [mrfakename/MegaTTS3-VoiceCloning]"
            f"(https://huggingface.co/mrfakename/MegaTTS3-VoiceCloning/tree/{SOURCES['mrfakename/MegaTTS3-VoiceCloning']})"
            f" at `{SOURCES['mrfakename/MegaTTS3-VoiceCloning'][:8]}`: the WavVAE with its encoder."
        )
    else:
        body.append("- The WavVAE is ByteDance's decoder-only release from the same repo.")
    if clone:
        download = (
            [f"hf download {hub_repo} --local-dir {folder}"] if hub_repo else []
        )
        body += [
            "",
            "## Using it with TurboSpark",
            "",
            "Voice cloning is on TurboSpark's `main` branch and not in a release yet. Every entry",
            "point asks you to confirm that you have the right to use the voice.",
            "",
            "- **App.** In Audio Studio, open Voice Clone, pick MegaTTS3, accept the download notice,",
            "  choose or record a reference, and pick Fast, Balanced or Best.",
            "- **Command line.** `turbospark-clone` takes this folder or the installed alias. It runs",
            "  on Metal on macOS, `--quality fast|balanced|best` picks 8, 16 or 32 steps, and `--cpu`",
            "  runs the slower CPU reference path:",
            "",
            "```sh",
            *download,
            f"turbospark-clone speak --model {folder} --reference me.wav \\",
            "  --text \"Hello there.\" --consent --output hello.wav",
            "```",
            "",
            "- **HTTP.** `turbospark-server --tts-model` serves it on the OpenAI-style speech route,",
            "  with the reference as a base64 WAV:",
            "",
            "```sh",
            f"turbospark-server --tts-model {folder}",
            "curl http://127.0.0.1:8080/v1/audio/speech -H 'content-type: application/json' \\",
            f"  -d \"{{\\\"model\\\":\\\"{folder}\\\",\\\"input\\\":\\\"Hello there.\\\",\\\"consent\\\":true,"
            "\\\"reference_audio\\\":\\\"$(base64 -i me.wav)\\\"}\" \\",
            "  -o hello.wav",
            "```",
            "",
            "Optional settings are `steps` (default 16, with 8 as Fast and 32 as Best), `p_w` (intelligibility,",
            "default 1.4), `t_w` (similarity to the reference, default 3.0), `seed`, and `speed`.",
            "",
            *CARD_BENCHMARKS,
            "## Consent and the encoder",
            "",
            "ByteDance withheld the WavVAE encoder, the part that turns a recording into a voice the",
            "model can imitate, over misuse concerns. The encoder here is a third-party re-upload by",
            "mrfakename, credited to ModelScope ACoderPassBy/MegaTTS-SFT. Its origin and license were",
            "not independently verified, and the re-upload's repository declares Apache-2.0.",
            "",
            "TurboSpark asks you to accept that before it downloads these files, and asks on every",
            "request whether you have the right to use the voice. Please only clone your own voice or",
            "the voice of someone who agreed to it.",
        ]
    body += [
        "",
        "## Limits",
        "",
        "- English only. TurboSpark's text normalizer covers a documented subset of upstream's, and",
        "  non-Latin text is refused.",
        "- The reference can be 1 to 30 seconds, and only the first 28 are used. Shorter",
        "  references synthesize faster.",
        "- Output is mono 24 kHz.",
        *([
            "- It is not real time: about 5 s of compute per second of speech at the default",
            "  settings on Metal (see [Benchmarks](#benchmarks)).",
            "- No listening test or MOS has been run.",
        ] if clone else [
            "- TurboSpark's port tracks the PyTorch reference on these weights to within about",
            "  1e-3 per sample. No listening test or MOS has been run.",
        ]),
        "",
        f"The measurements behind these numbers are in the [port's README]({PORT_README}).",
        "",
        "## Credits",
        "",
        "MegaTTS3 is ByteDance's work: [paper](https://arxiv.org/abs/2502.18924),",
        "[code](https://github.com/bytedance/MegaTTS3), and",
        "[weights](https://huggingface.co/ByteDance/MegaTTS3). Thanks to them for releasing it",
        "under Apache-2.0" + (", and to mrfakename for the encoder re-upload." if clone else "."),
        "",
    ]
    return "\n".join(head + body)


def write_root_files(out, wavvae_variant, license_path, manifest, hub_repo=None):
    with open(license_path, encoding="utf-8") as fh:
        if "Apache License" not in fh.read(4096):
            raise ConvertError(f"{license_path} does not look like the Apache-2.0 LICENSE")
    shutil.copyfile(license_path, os.path.join(out, "LICENSE"))
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(root_config(wavvae_variant), fh, indent=2, sort_keys=True)
    for name, text in (("NOTICE.md", notice_text(wavvae_variant)),
                       ("README.md", model_card(wavvae_variant, manifest["files"], hub_repo))):
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    for name in ("config.json", "LICENSE", "NOTICE.md", "README.md"):
        path = os.path.join(out, name)
        manifest["files"][name] = {"sha256": sha256_file(path), "size": os.path.getsize(path)}


# ------------------------------------------------------------------------ main


def component_jobs(wavvae_variant):
    jobs = [(c, f"{c}/{f}", key) for c, (f, key) in COMPONENTS.items()]
    jobs.append(("wavvae", f"wavvae/{WAVVAE_FILE[wavvae_variant]}", "model_gen"))
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out")
    ap.add_argument("--dict-json")
    ap.add_argument("--wavvae", choices=sorted(WAVVAE_FILE), default="community")
    ap.add_argument(
        "--license",
        help="upstream LICENSE text (Apache-2.0, from the MegaTTS3 code repo), copied to LICENSE",
    )
    ap.add_argument(
        "--hub-repo",
        help="Hugging Face repo the output will be published to; the model card's download example names it",
    )
    ap.add_argument("--inventory", action="store_true", help="print tensors, write nothing")
    ap.add_argument(
        "--skip-g2p",
        action="store_true",
        help="development only: do not copy the 1 GB G2P model (recorded in the manifest)",
    )
    args = ap.parse_args(argv)
    if not args.inventory and not (args.out and args.dict_json and args.license):
        ap.error("--out, --dict-json and --license are required unless --inventory")
    if args.out and os.path.exists(args.out) and os.listdir(args.out):
        ap.error(f"--out {args.out} exists and is not empty")

    manifest = {"sources": SOURCES, "wavvae_variant": args.wavvae, "files": {}}
    if args.wavvae == "community":
        manifest["wavvae_provenance"] = (
            "unofficial encoder re-uploaded by mrfakename, credited to ModelScope "
            "ACoderPassBy/MegaTTS-SFT; origin and license not independently verified"
        )

    for comp, rel, key in component_jobs(args.wavvae):
        src_sha = verify_pinned(args.src, rel)
        sd = load_ckpt_state_dict(os.path.join(args.src, rel), key)
        converted = convert_state_dict(sd)
        n_params = sum(int(a.size) for a in converted.values())
        print(f"== {comp}: {len(sd)} source tensors -> {len(converted)} kept, {n_params:,} params")
        if args.inventory:
            for name in sorted(sd):
                res = convert_tensor(name, sd[name])
                tail = "(dropped)" if res is None else f"-> {res[0]} {list(res[1].shape)}"
                print(f"   {name} {list(sd[name].shape)} {sd[name].dtype} {tail}")
            continue
        out_dir = os.path.join(args.out, comp)
        os.makedirs(out_dir, exist_ok=True)
        meta = {"model": "megatts3", "component": comp, "source_sha256": src_sha, "layout": "mlx"}
        st_path = os.path.join(out_dir, "model.safetensors")
        write_safetensors(st_path, converted, meta)
        cfg_src = os.path.join(args.src, comp, "config.yaml")
        cfg = {
            "model_type": f"megatts3_{comp}",
            "source_file": rel,
            "source_sha256": src_sha,
            "hparams": flat_yaml_scalars(cfg_src) if os.path.isfile(cfg_src) else {},
        }
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, sort_keys=True)
        manifest["files"][f"{comp}/model.safetensors"] = {
            "params": n_params,
            "sha256": sha256_file(st_path),
            "size": os.path.getsize(st_path),
            "tensors": len(converted),
        }

    if args.inventory:
        return 0

    # G2P is already safetensors + HF tokenizer files; copy after pin checks.
    if args.skip_g2p:
        manifest["g2p"] = "skipped (--skip-g2p)"
    else:
        for rel in ("g2p/model.safetensors", "g2p/tokenizer.json"):
            verify_pinned(args.src, rel)
        g2p_out = os.path.join(args.out, "g2p")
        os.makedirs(g2p_out, exist_ok=True)
        for fname in G2P_COPY:
            src_path = os.path.join(args.src, "g2p", fname)
            if not os.path.isfile(src_path):
                raise ConvertError(f"missing g2p/{fname}")
            shutil.copyfile(src_path, os.path.join(g2p_out, fname))
            manifest["files"][f"g2p/{fname}"] = {
                "sha256": sha256_file(os.path.join(g2p_out, fname)),
                "size": os.path.getsize(os.path.join(g2p_out, fname)),
            }

    load_dict_json(args.dict_json)
    os.makedirs(os.path.join(args.out, "frontend"), exist_ok=True)
    shutil.copyfile(args.dict_json, os.path.join(args.out, "frontend", "dict.json"))
    manifest["files"]["frontend/dict.json"] = {
        "sha256": sha256_file(os.path.join(args.out, "frontend", "dict.json")),
        "size": os.path.getsize(os.path.join(args.out, "frontend", "dict.json")),
    }

    write_root_files(args.out, args.wavvae, args.license, manifest, args.hub_repo)
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConvertError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
