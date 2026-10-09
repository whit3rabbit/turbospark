#!/usr/bin/env python3
"""Regenerate Whistle operator goldens with the pinned release oracle.

Research only: private C++ ABI calls are confined to this script. The Rust
crate has no Needle runtime dependency. No assets are downloaded here.
"""

import argparse
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import subprocess
import tempfile

MODEL_SHA = "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb"
LIB_SHA = "98da47c15e1065b4cdc7ddc55e825be78414d4586832db3becaeded39a373df4"
MODEL_REV = "b358ddadd89b7a713b5aa131f23032d3cca1b251"
RUNTIME_REV = "2ae11323dc000f5e70c49f7403efa6af12ba9e67"


def checked(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"{path}: digest differs from pinned oracle")
    return data


def reduced_archive(data):
    # Keep eight Q rows, all thirteen Hadamard MLP tensors, permutations,
    # the mel bank, and tokenizer. Numeric tests need no complete model.
    ids = [325, *range(336, 349), 226, 227, 679, 680]
    records, payloads = [], []
    for index in ids:
        record = list(struct.unpack_from("<BBH4IQQII", data, 308 + index * 44))
        offset, size = record[7:9]
        payload = data[offset:offset + size]
        if index == 325:
            rows, cols, bits = record[3], record[4], record[10]
            row_bytes = ((cols + 127) // 128 * 128) * bits // 8
            norms_per_row = ((cols + 127) // 128) * 2
            payload = payload[:8 * row_bytes] + payload[rows * row_bytes:rows * row_bytes + 8 * norms_per_row]
            record[3] = 8
        records.append(record)
        payloads.append(payload)
    header = bytearray(data[:308])
    struct.pack_into("<I", header, 4, len(records))
    output = header + bytearray(44 * len(records))
    for i, (record, payload) in enumerate(zip(records, payloads)):
        output.extend(bytes((-len(output)) % 64))
        record[7:9] = len(output), len(payload)
        struct.pack_into("<BBH4IQQII", output, 308 + 44 * i, *record)
        output.extend(payload)
    return bytes(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("library", type=Path, help="pinned macos-arm64 libneedle.a")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("the reference ABI is pinned to macOS arm64")
    data = checked(args.model, MODEL_SHA)
    checked(args.library, LIB_SHA)
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="whistle-oracle-") as tmp:
        dylib = Path(tmp) / "libneedle.dylib"
        sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        subprocess.run(["xcrun", "clang++", "-isysroot", sdk, "-dynamiclib",
                        "-Wl,-all_load", str(args.library.resolve()), "-o", str(dylib)], check=True)
        lib = C.CDLL(str(dylib))
        symbols = {}
        for line in subprocess.check_output(["llvm-nm", "--defined-only", str(dylib)], text=True).splitlines():
            fields = line.split()
            if len(fields) == 3 and fields[1] in ("T", "t"):
                symbols[fields[2]] = int(fields[0], 16)
        base = C.cast(lib.needle_load, C.c_void_p).value - symbols["_needle_load"]

        def function(name, result, arguments):
            return C.CFUNCTYPE(result, C.c_void_p, *arguments)(base + symbols["_" + name])

        storage = C.create_string_buffer(16400)
        engine = (C.addressof(storage) + 15) & ~15
        function("_ZN7whistle6EngineC2Ev", None, [])(engine)
        try:
            load = function("_ZN7whistle6Engine4loadEPKhm", C.c_bool, [C.c_char_p, C.c_size_t])
            if not load(engine, data, len(data)):
                raise RuntimeError("reference rejected the pinned checkpoint")

            def vector(offset):
                start, end, _ = struct.unpack("<QQQ", C.string_at(engine + offset, 24))
                return C.string_at(start, end - start)

            def save(name, values):
                if not isinstance(values, bytes):
                    values = struct.pack("<" + "f" * len(values), *values)
                (args.output / name).write_bytes(values)

            save("ops.cact", reduced_archive(data))
            pcm = (C.c_float * 16000)(*[0.2 * math.sin(math.tau * 440 * i / 16000) for i in range(16000)])
            n = function("_ZN7whistle6Engine14audio_featuresEPKfi", C.c_int,
                         [C.POINTER(C.c_float), C.c_int])(engine, pcm, len(pcm))
            if n != 100:
                raise RuntimeError("reference frontend frame count changed")
            save("tone_features.f32", vector(0xb18))
            encoder_weights = struct.unpack("<Q", C.string_at(engine + 0x990, 8))[0]
            tensor = struct.unpack("<Q", C.string_at(encoder_weights + 0x30, 8))[0]
            row = (C.c_float * 512)()
            function("_ZNK7whistle6Engine13cq_decode_rowEPKN6needle6TensorEiPf", None,
                     [C.c_void_p, C.c_int, C.POINTER(C.c_float)])(engine, tensor, 0, row)
            save("q_row.f32", row)
            x = (C.c_float * 512)(*[math.sin(i * 0.1) * 0.3 for i in range(512)])
            out = (C.c_float * 384)()
            function("_ZNK7whistle6Engine15cq_matmul_batchEPKfPKN6needle6TensorEPfi", None,
                     [C.POINTER(C.c_float), C.c_void_p, C.POINTER(C.c_float), C.c_int])(engine, x, tensor, out, 1)
            save("q_project.f32", list(out)[:8])
            mlp = (C.c_float * 512)()
            function("_ZNK7whistle6Engine12hadamard_mlpEPKfRKNS_15HadamardWeightsEPf", None,
                     [C.POINTER(C.c_float), C.c_void_p, C.POINTER(C.c_float)])(engine, x, encoder_weights + 0x98, mlp)
            save("hada.f32", mlp)
        finally:
            function("_ZN7whistle6EngineD2Ev", None, [])(engine)
    manifest = {
        "model_revision": MODEL_REV, "model_sha256": MODEL_SHA,
        "runtime_revision": RUNTIME_REV, "runtime_library_sha256": LIB_SHA,
        "reference": "released CPU oracle; private speech frontend absent from public JAX source",
        "fixture_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(args.output.iterdir()) if p.suffix in (".cact", ".f32")},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
