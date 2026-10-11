"""Tests for convert_megatts3.py. Run: python3 -I crates/audio/tools/test_convert_megatts3.py

The fake checkpoints are written without torch: a stub `torch` module is
registered only while pickling, so the bytes carry the same global references a
real torch zip checkpoint does. The converter itself never imports torch.
"""

import collections
import json
import os
import pickle
import struct
import sys
import tempfile
import types
import unittest
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_megatts3 as cv  # noqa: E402


class _Fake:
    """Registers a stub `torch` for the duration of a `with` block."""

    def __enter__(self):
        self.saved = {k: sys.modules.get(k) for k in ("torch", "torch._utils")}
        torch = types.ModuleType("torch")
        utils = types.ModuleType("torch._utils")

        def _rebuild_tensor_v2(*a):
            raise RuntimeError("stub")

        _rebuild_tensor_v2.__module__ = "torch._utils"
        _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"  # pickle rejects "<locals>"
        utils._rebuild_tensor_v2 = _rebuild_tensor_v2

        def _rebuild_tensor(*a):
            raise RuntimeError("stub")

        _rebuild_tensor.__module__ = "torch._utils"
        _rebuild_tensor.__qualname__ = "_rebuild_tensor"
        utils._rebuild_tensor = _rebuild_tensor
        for name in ("FloatStorage", "LongStorage", "BFloat16Storage"):
            cls = type(name, (), {})
            cls.__module__ = "torch"
            setattr(torch, name, cls)
        torch._utils = utils
        sys.modules["torch"] = torch
        sys.modules["torch._utils"] = utils
        self.torch = torch
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class _StorageRef:
    def __init__(self, storage_cls, key, numel):
        self.cls, self.key, self.numel = storage_cls, key, numel
        self.view = None


class _FakeTensor:
    def __init__(self, fake, storage, offset, size, stride):
        self.fake, self.storage = fake, storage
        self.args = (storage, offset, tuple(size), tuple(stride), False, collections.OrderedDict())

    def __reduce__(self):
        return (self.fake.torch._utils._rebuild_tensor_v2, self.args)


class _Pickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, _StorageRef):
            return ("storage", obj.cls, obj.key, "cpu", obj.numel)
        return None


def write_ckpt(path, state_dict_all, extra=None):
    """state_dict_all: {component: {name: (kind, ndarray, size, stride)}}."""
    import io

    blobs = {}
    with _Fake() as fake:
        sd = {}
        for comp, tensors in state_dict_all.items():
            od = collections.OrderedDict()
            for name, (kind, flat, size, stride) in tensors.items():
                key = str(len(blobs))
                cls = getattr(fake.torch, kind)
                blobs[key] = flat.tobytes()
                od[name] = _FakeTensor(fake, _StorageRef(cls, key, flat.size), 0, size, stride)
            sd[comp] = od
        buf = io.BytesIO()
        _Pickler(buf, protocol=2).dump({"state_dict": sd, "global_step": 7, **(extra or {})})
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("arch/data.pkl", buf.getvalue())
        zf.writestr("arch/byteorder", "little")
        for key, raw in blobs.items():
            zf.writestr(f"arch/data/{key}", raw)


class _LegacyTensor:
    def __init__(self, fake, storage, offset, size, stride):
        self.fake, self.args = fake, (storage, offset, tuple(size), tuple(stride))

    def __reduce__(self):
        return (self.fake.torch._utils._rebuild_tensor, self.args)


class _LegacyPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, _StorageRef):
            return ("storage", obj.cls, obj.key, "cpu", obj.numel, obj.view)
        return None


def write_legacy_ckpt(path, comp, tensors, magic=cv._LEGACY_MAGIC):
    """tensors: {name: (kind, root_flat, size, stride, view_or_None)}; one root storage each."""
    with open(path, "wb") as fh, _Fake() as fake:
        pickle.dump(magic, fh, protocol=2)
        pickle.dump(1001, fh, protocol=2)
        pickle.dump(
            {"protocol_version": 1001, "little_endian": True, "type_sizes": {"short": 2, "int": 4, "long": 4}},
            fh,
            protocol=2,
        )
        od, keys, blobs = collections.OrderedDict(), [], []
        for i, (name, (kind, flat, size, stride, view)) in enumerate(tensors.items()):
            key = str(1000 + i)
            ref = _StorageRef(getattr(fake.torch, kind), key, flat.size)
            ref.view = None if view is None else (key + "v", view[0], view[1])
            od[name] = _LegacyTensor(fake, ref, 0, size, stride)
            keys.append(key)
            blobs.append(struct.pack("<q", flat.size) + flat.tobytes())
        _LegacyPickler(fh, protocol=2).dump({"state_dict": {comp: od}, "global_step": 3})
        pickle.dump(keys, fh, protocol=2)
        for b in blobs:
            fh.write(b)


def contiguous(arr, kind="FloatStorage"):
    return (kind, arr.ravel(), arr.shape, tuple(s // arr.itemsize for s in arr.strides))


def read_safetensors(path):
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
        base = 8 + n
        data = fh.read()
    assert base % 8 == 0
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        assert info["dtype"] == "F32"
        lo, hi = info["data_offsets"]
        out[name] = np.frombuffer(data[lo:hi], dtype="<f4").reshape(info["shape"])
    return out, header["__metadata__"]


def wn_fold(v, g):
    # weight = g * v / ||v||, norm over every axis except 0 (torch weight_norm dim=0).
    axes = tuple(range(1, v.ndim))
    return g * v / np.sqrt((v**2).sum(axis=axes, keepdims=True))


class ConvertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rng = np.random.default_rng(0)

    def p(self, name):
        return os.path.join(self.tmp.name, name)

    def f32(self, *shape):
        return self.rng.standard_normal(shape).astype("<f4")

    def test_roundtrip_layout_and_weight_norm(self):
        conv_v, conv_g = self.f32(4, 3, 5), np.abs(self.f32(4, 1, 1)) + 0.5
        ct_v, ct_g = self.f32(3, 4, 5), np.abs(self.f32(3, 1, 1)) + 0.5
        lin = self.f32(6, 2)
        base = self.f32(3, 2)  # stored transposed: size (2,3), stride (1,2)
        write_ckpt(
            self.p("m.ckpt"),
            {
                "model": {
                    "module.net.conv.weight_v": contiguous(conv_v),
                    "module.net.conv.weight_g": contiguous(conv_g),
                    "net.convt.weight_v": contiguous(ct_v),
                    "net.convt.weight_g": contiguous(ct_g),
                    "net.lin.weight": contiguous(lin),
                    "net.view": ("FloatStorage", base.ravel(), (2, 3), (1, 2)),
                    "net.bn.num_batches_tracked": ("LongStorage", np.array([5], "<i8"), (), ()),
                }
            },
        )
        sd = cv.load_ckpt_state_dict(self.p("m.ckpt"), "model")
        self.assertIn("net.conv.weight_v", sd)  # `module.` stripped like load_ckpt
        np.testing.assert_array_equal(sd["net.view"], base.T)
        out = cv.convert_state_dict(sd)
        self.assertNotIn("net.bn.num_batches_tracked", out)
        self.assertEqual(out["net.conv.weight_v"].shape, (4, 5, 3))  # [out, K, in]
        self.assertEqual(out["net.convt.weight_v"].shape, (3, 5, 4))  # [in, K, out]
        self.assertEqual(out["net.conv.weight_g"].shape, (4, 1, 1))
        np.testing.assert_array_equal(out["net.lin.weight"], lin)
        # The Rust loader folds in the stored layout; it must equal the torch fold.
        for v, g, key in ((conv_v, conv_g, "conv"), (ct_v, ct_g, "convt")):
            want = wn_fold(v, g).transpose(0, 2, 1)
            got = wn_fold(out[f"net.{key}.weight_v"], out[f"net.{key}.weight_g"])
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

        st = self.p("out.safetensors")
        cv.write_safetensors(st, out, {"component": "t"})
        back, meta = read_safetensors(st)
        self.assertEqual(meta["component"], "t")
        self.assertEqual(sorted(back), sorted(out))
        for k in out:
            np.testing.assert_array_equal(back[k], out[k])

    def test_legacy_format_with_storage_view(self):
        w = self.f32(2, 3, 4)
        root = self.f32(12).ravel()
        write_legacy_ckpt(
            self.p("l.ckpt"),
            "model",
            {
                "a.weight": ("FloatStorage", w.ravel(), (2, 3, 4), (12, 4, 1), None),
                "b": ("FloatStorage", root, (2, 3), (3, 1), (4, 6)),  # elements 4..10 of root
                "c": ("LongStorage", np.array([1, 2], "<i8"), (2,), (1,), None),
            },
        )
        self.assertFalse(zipfile.is_zipfile(self.p("l.ckpt")))
        sd = cv.load_ckpt_state_dict(self.p("l.ckpt"), "model")
        np.testing.assert_array_equal(sd["a.weight"], w)
        np.testing.assert_array_equal(sd["b"], root[4:10].reshape(2, 3))
        self.assertEqual(sd["c"].dtype.kind, "i")

    def test_legacy_rejects_bad_magic_and_truncation(self):
        flat = self.f32(6).ravel()
        spec = {"x": ("FloatStorage", flat, (6,), (1,), None)}
        write_legacy_ckpt(self.p("m.ckpt"), "model", spec, magic=123)
        with self.assertRaises(cv.ConvertError):
            cv.load_ckpt_state_dict(self.p("m.ckpt"), "model")
        write_legacy_ckpt(self.p("t.ckpt"), "model", spec)
        with open(self.p("t.ckpt"), "rb") as fh:
            data = fh.read()
        with open(self.p("t.ckpt"), "wb") as fh:
            fh.write(data[:-8])  # drop the tail of the last blob
        with self.assertRaises(cv.ConvertError):
            cv.load_ckpt_state_dict(self.p("t.ckpt"), "model")

    def test_legacy_refuses_unlisted_global(self):
        marker = self.p("pwned2")

        class Evil:
            def __reduce__(self):
                return (open, (marker, "w"))

        with open(self.p("e.ckpt"), "wb") as fh:
            pickle.dump(cv._LEGACY_MAGIC, fh, protocol=2)
            pickle.dump(1001, fh, protocol=2)
            pickle.dump({"little_endian": True}, fh, protocol=2)
            pickle.dump({"state_dict": {"model": Evil()}}, fh, protocol=2)
        with self.assertRaises(cv.ConvertError):
            cv.load_ckpt_state_dict(self.p("e.ckpt"), "model")
        self.assertFalse(os.path.exists(marker))

    def test_bf16_and_f16_widen_exactly(self):
        vals = np.array([1.0, -2.5, 0.15625, 3.0e4], "<f4")
        bf = (vals.view("<u4") >> 16).astype("<u2")
        write_ckpt(self.p("b.ckpt"), {"m": {"x": ("BFloat16Storage", bf, (4,), (1,))}})
        sd = cv.load_ckpt_state_dict(self.p("b.ckpt"), "m")
        trunc = (bf.astype("<u4") << 16).view("<f4")
        np.testing.assert_array_equal(sd["x"], trunc)

    def test_refuses_unlisted_pickle_global(self):
        marker = self.p("pwned")

        class Evil:
            def __reduce__(self):
                return (open, (marker, "w"))

        with open(self.p("evil.pkl"), "wb") as fh:
            pickle.dump({"state_dict": {"m": Evil()}}, fh, protocol=2)
        with zipfile.ZipFile(self.p("evil.ckpt"), "w") as zf:
            zf.write(self.p("evil.pkl"), "arch/data.pkl")
        with self.assertRaises(cv.ConvertError):
            cv.load_ckpt_state_dict(self.p("evil.ckpt"), "m")
        self.assertFalse(os.path.exists(marker), "unlisted global must not execute")

    def test_rejects_view_past_storage(self):
        flat = self.f32(4).ravel()
        write_ckpt(self.p("o.ckpt"), {"m": {"x": ("FloatStorage", flat, (3,), (2,))}})
        with self.assertRaises(cv.ConvertError):
            cv.load_ckpt_state_dict(self.p("o.ckpt"), "m")

    def test_structural_errors(self):
        with self.assertRaises(cv.ConvertError):  # orphan weight_g
            cv.convert_state_dict({"a.weight_g": self.f32(2, 1, 1)})
        with self.assertRaises(cv.ConvertError):  # orphan weight_v
            cv.convert_state_dict({"a.weight_v": self.f32(2, 3, 4)})
        with self.assertRaises(cv.ConvertError):  # g not [d0,1,1]
            cv.convert_state_dict({"a.weight_g": self.f32(2, 3, 1), "a.weight_v": self.f32(2, 3, 4)})
        with self.assertRaises(cv.ConvertError):  # conv2d
            cv.convert_state_dict({"a.weight": self.f32(2, 3, 4, 5)})
        with self.assertRaises(cv.ConvertError):  # int tensor that is not dropped
            cv.convert_state_dict({"a.idx": np.zeros(3, "<i8")})

    def test_new_style_parametrization_names(self):
        out = cv.convert_state_dict(
            {
                "c.parametrizations.weight.original0": self.f32(2, 1, 1),
                "c.parametrizations.weight.original1": self.f32(2, 3, 4),
            }
        )
        self.assertEqual(sorted(out), ["c.weight_g", "c.weight_v"])

    def test_pin_mismatch_is_refused(self):
        rel = "g2p/tokenizer.json"
        os.makedirs(self.p("g2p"))
        with open(self.p(rel), "wb") as fh:
            fh.write(b"not the real file")
        with self.assertRaises(cv.ConvertError):
            cv.verify_pinned(self.tmp.name, rel)
        saved = cv.PINNED[rel]
        try:
            import hashlib

            cv.PINNED[rel] = (hashlib.sha256(b"not the real file").hexdigest(), 17)
            self.assertEqual(cv.verify_pinned(self.tmp.name, rel), cv.PINNED[rel][0])
        finally:
            cv.PINNED[rel] = saved

    def test_flat_yaml_scalars(self):
        with open(self.p("c.yaml"), "w") as fh:
            fh.write("amp: true\nhop_size: 240\nlr: 0.5\nname: abc\nlist:\n- 1\nnested:\n  x: 1\nf0: 6e-2\n")
        got = cv.flat_yaml_scalars(self.p("c.yaml"))
        self.assertEqual(got["amp"], True)
        self.assertEqual(got["hop_size"], 240)
        self.assertEqual(got["lr"], 0.5)
        self.assertEqual(got["name"], "abc")
        self.assertNotIn("x", got)
        self.assertNotIn("list", got)

    def test_root_files_are_deterministic_and_declare_cloning(self):
        lic = self.p("LICENSE.src")
        with open(lic, "w", encoding="utf-8") as fh:
            fh.write("                                 Apache License\n Version 2.0\n")
        outs = []
        for run in ("a", "b"):
            out = self.p(run)
            os.makedirs(out)
            manifest = {"files": {}}
            cv.write_root_files(out, "community", lic, manifest)
            self.assertEqual(sorted(manifest["files"]), ["LICENSE", "NOTICE.md", "README.md", "config.json"])
            outs.append(manifest["files"])
        self.assertEqual(outs[0], outs[1], "same inputs must give the same bytes")
        with open(self.p("a/config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        self.assertEqual((cfg["model_type"], cfg["voice_clone"], cfg["sample_rate"]), ("megatts3", True, 24000))
        self.assertFalse(cv.root_config("official")["voice_clone"])
        with open(self.p("a/NOTICE.md"), encoding="utf-8") as fh:
            notice = fh.read()
        community_sha = cv.PINNED["wavvae/model_only_last.ckpt"][0]
        self.assertIn(community_sha, notice)
        self.assertIn(cv.ENCODER_PROVENANCE, notice)
        self.assertIn("not independently verified", cv.ENCODER_PROVENANCE)
        official = cv.notice_text("official")
        self.assertNotIn(community_sha, official)
        self.assertNotIn(cv.ENCODER_PROVENANCE, official)
        # Only the Apache text is accepted as LICENSE.
        with open(self.p("bad"), "w", encoding="utf-8") as fh:
            fh.write("MIT License\n")
        os.makedirs(self.p("c"))
        with self.assertRaises(cv.ConvertError):
            cv.write_root_files(self.p("c"), "community", self.p("bad"), {"files": {}})

    def test_model_card_uses_manifest_numbers_and_names_the_repo(self):
        files = {
            "diffusion_transformer/model.safetensors": {"params": 459037376, "size": 1836204936},
            "diffusion_transformer/config.json": {"size": 2995},
            "LICENSE": {"size": 11341},
        }
        card = cv.model_card("community", files, "someone/MegaTTS3-TurboSpark")
        self.assertTrue(card.startswith("---\nlicense: apache-2.0\n"))
        self.assertIn("| 459.0M | 1.84 GB |", card)
        self.assertIn("hf download someone/MegaTTS3-TurboSpark --local-dir MegaTTS3-TurboSpark", card)
        self.assertIn("not an MLX model", card)
        self.assertIn(cv.SOURCES["mrfakename/MegaTTS3-VoiceCloning"], card)
        self.assertNotIn("<", card.split("```")[1], "no unfilled placeholder in the commands")
        self.assertEqual(card, cv.model_card("community", files, "someone/MegaTTS3-TurboSpark"))
        # Without a repo there is no download line to get wrong.
        self.assertNotIn("hf download", cv.model_card("community", files))
        # The decoder-only variant says it cannot clone and drops the clone usage.
        official = cv.model_card("official", files)
        self.assertIn("cannot", official)
        self.assertNotIn("turbospark-clone", official)
        self.assertNotIn("mrfakename", official)

    def test_dict_json_validation(self):
        with open(self.p("d.json"), "w", encoding="utf-8-sig") as fh:
            json.dump({"phone": ["a", "b"], "tone": ["x"]}, fh)
        self.assertEqual(cv.load_dict_json(self.p("d.json"))["phone"], ["a", "b"])
        with open(self.p("bad.json"), "w") as fh:
            json.dump({"phone": [1], "tone": []}, fh)
        with self.assertRaises(cv.ConvertError):
            cv.load_dict_json(self.p("bad.json"))


if __name__ == "__main__":
    unittest.main()
