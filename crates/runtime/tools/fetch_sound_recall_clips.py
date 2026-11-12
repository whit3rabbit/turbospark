#!/usr/bin/env python3
"""Fetch and convert the sound-lane recall clips.

Reads crates/runtime/tests/fixtures/sound_recall/clips.json, downloads each clip
from its pinned URL, refuses any file whose SHA-256 differs from the pin,
and converts it with ffmpeg to 16 kHz mono 16-bit WAV, the format the app
hands EmbeddingGemma 2 (AudioMediaIO.readMono16k). Writes converted.json
next to the WAVs; the gate (crates/runtime/tests/coreml_sound_recall.rs)
reads it.

    python3 -I crates/runtime/tools/fetch_sound_recall_clips.py [OUT_DIR]

OUT_DIR defaults to $TURBOSPARK_SOUND_RECALL_CLIPS, then
~/models/sound-recall-clips. Standard library plus an ffmpeg binary.
Downloaded files are only hashed and passed to ffmpeg, never executed.
"""

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "crates/runtime/tests/fixtures/sound_recall/clips.json"
# Wikimedia asks automated clients to identify themselves.
USER_AGENT = "turbospark-sound-recall-fixture/1.0 (https://github.com/whit3rabbit/turbospark)"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest, max_bytes):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    # upload.wikimedia.org answers bursts with 429; back off and retry.
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
                total = 0
                while block := resp.read(1 << 16):
                    total += len(block)
                    if total > max_bytes:
                        raise SystemExit(f"{url}: larger than the pinned {max_bytes} bytes")
                    out.write(block)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))
    tmp.replace(dest)
    time.sleep(2)


def main():
    out_dir = pathlib.Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("TURBOSPARK_SOUND_RECALL_CLIPS")
        or pathlib.Path.home() / "models/sound-recall-clips"
    ).expanduser()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH")
    manifest = json.loads(MANIFEST.read_text())
    sources = out_dir / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    converted = {}
    for clip in manifest["clips"]:
        name = clip["name"]
        ext = pathlib.PurePosixPath(clip["url"]).suffix.lower()
        src = sources / f"{name}{ext}"
        if not src.is_file() or sha256(src) != clip["sha256"]:
            print(f"downloading {name}: {clip['url']}", file=sys.stderr)
            download(clip["url"], src, clip["bytes"])
        got = sha256(src)
        if got != clip["sha256"]:
            src.unlink()
            raise SystemExit(f"{name}: SHA-256 {got} does not match the pin {clip['sha256']}")
        wav = f"{name}.wav"
        cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        if clip.get("start_seconds"):
            cmd += ["-ss", str(clip["start_seconds"])]
        cmd += ["-i", str(src)]
        if clip.get("seconds"):
            cmd += ["-t", str(clip["seconds"])]
        cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-map_metadata", "-1"]
        cmd += [str(out_dir / wav)]
        subprocess.run(cmd, check=True)
        converted[name] = {
            "wav": wav,
            "source_sha256": got,
            "wav_sha256": sha256(out_dir / wav),
        }
        print(f"converted {name}", file=sys.stderr)
    version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, check=True)
    report = {
        "manifest_sha256": sha256(MANIFEST),
        "ffmpeg": version.stdout.splitlines()[0],
        "clips": converted,
    }
    (out_dir / "converted.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {len(converted)} clips to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
