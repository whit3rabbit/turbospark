"""Run with the pinned mlx-audio environment to regenerate the offline golden."""
import json
import subprocess
from pathlib import Path
import mlx.core as mx
from mlx_audio.tts.models.qwen3_tts.qwen3_tts import mel_spectrogram

reference = Path(__import__("mlx_audio").__file__).resolve().parent.parent
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=reference, text=True).strip()
assert commit == "e1b19b9054bf163f5d812221a54fcc346f1890e9", commit
samples = [((i * 73) % 2001 - 1000) / 2000 for i in range(4096)]
mel = mel_spectrogram(mx.array(samples))[0]
mx.eval(mel)
fixture = {"reference": commit, "samples": samples, "shape": list(mel.shape), "mel": mel.reshape(-1).tolist()}
output = Path(__file__).resolve().parent.parent / "Tests/Qwen3TTSTests/Fixtures/speaker-mel.json"
output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
