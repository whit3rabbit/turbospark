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
  convert_megatts3.py --src SRC --out OUT --dict-json dict.json [--wavvae community|official]

SRC is a Hugging Face snapshot directory (`aligner_lm/`, `diffusion_transformer/`,
`duration_lm/`, `wavvae/`, `g2p/`). `dict.json` is
`tts/utils/text_utils/dict.json` from the MegaTTS3 code repo.
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
    ap.add_argument("--inventory", action="store_true", help="print tensors, write nothing")
    ap.add_argument(
        "--skip-g2p",
        action="store_true",
        help="development only: do not copy the 1 GB G2P model (recorded in the manifest)",
    )
    args = ap.parse_args(argv)
    if not args.inventory and not (args.out and args.dict_json):
        ap.error("--out and --dict-json are required unless --inventory")
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
