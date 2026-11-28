#!/usr/bin/env python3
"""Regenerate the ACE-Step 1.5 prompt-assembly fixtures.

Normative reference: upstream ACE-Step/ACE-Step-1.5 at
ca1e85fe9430179831e6bc6be790c332190a3866, the pure helpers the handler
uses to build DiT text inputs:

- acestep/constants.py: SFT_GEN_PROMPT, DEFAULT_DIT_INSTRUCTION,
  TASK_INSTRUCTIONS
- core/generation/handler/metadata_utils.py: _build_metadata_dict,
  _dict_to_meta_string (via _parse_metas)
- core/generation/handler/prompt_utils.py: _format_instruction,
  _format_lyrics, extract_caption_from_sft_format

The modules are loaded by file path (the handler package __init__
imports every mixin); `loguru` is stubbed. Token ids come from the
checkpoint's Qwen3-Embedding tokenizer through transformers 4.57.6
`AutoTokenizer` with upstream's call (`padding="longest",
truncation=True`, max_length 256 for captions and 2048 for lyrics).

    ../.venv-acestep/bin/python -I crates/audio/tools/gen_ace_step15_prompt_fixtures.py \
        ../ACE-Step-1.5 <checkpoint>/Qwen3-Embedding-0.6B

Writes crates/audio/testdata/ace_step15_prompt/cases.json. Strings are
checked by a default test; ids by an ignored test that needs the
tokenizer file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "testdata" / "ace_step15_prompt"
UPSTREAM_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"


def stub_modules(upstream: Path) -> None:
    loguru = types.ModuleType("loguru")

    class _Logger:
        def __getattr__(self, _name):
            return lambda *a, **k: None

    loguru.logger = _Logger()
    sys.modules["loguru"] = loguru
    pkg = types.ModuleType("acestep")
    pkg.__path__ = [str(upstream / "acestep")]
    sys.modules["acestep"] = pkg
    load("acestep.constants", upstream / "acestep" / "constants.py")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CASES = [
    # name, task, instruction override, caption, lyrics, language, bpm, key, timesig, duration
    ("defaults", "text2music", None, "upbeat synthpop, female vocals", "[verse]\nhello world", "en", None, "", "", None),
    ("full_metas", "text2music", None, "orchestral strings, solo cello, slow and warm, classical",
     "[Instrumental]", "en", 72, "D minor", "4", 30.0),
    ("fractional_duration", "text2music", None, "lofi hip hop", "[Instrumental]", "en", 85, "A minor", "4", 47.9),
    ("cover", "cover", None, "orchestral, classical, no vocals", "[Instrumental]", "en", None, "", "", 31.5),
    ("repaint", "repaint", None, "rock, electric guitar", "[chorus]\nla la la", "en", 120, "E major", "3", 12.0),
    ("instruction_no_colon", "custom", "Fill the audio semantic mask based on the given conditions",
     "jazz trio", "", "en", None, "", "", None),
    ("unicode_lyrics", "text2music", None, "mandopop ballad, piano",
     "[verse]\n\u6211\u7231\u4f60\uff0c\u5c31\u50cf\u8001\u9f20\u7231\u5927\u7c73\n[chorus]\n\u3042\u308a\u304c\u3068\u3046 caf\u00e9 \u00fcber", "zh", 68, "C major", "4", 20.0),
    ("whitespace_meta", "text2music", None, "  ambient drone  ", "[Instrumental]", "en", 0, "   ", "N/A", 10.0),
    ("sft_caption", "text2music", None,
     "# Instruction\nWhatever:\n\n# Caption\nextracted caption text\n\n# Metas\n- bpm: 90\n",
     "[Instrumental]", "en", None, "", "", None),
    ("long_lyrics", "text2music", None, "punk", "\n".join(["[verse] shout it loud again and again"] * 400),
     "en", 180, "", "", 60.0),
]


def main(upstream: Path, tok_dir: Path) -> None:
    stub_modules(upstream)
    constants = sys.modules["acestep.constants"]
    handler = upstream / "acestep" / "core" / "generation" / "handler"
    meta = load("ace_metadata_utils", handler / "metadata_utils.py")
    prompt = load("ace_prompt_utils", handler / "prompt_utils.py")

    class H(meta.MetadataMixin, prompt.PromptMixin):
        pass

    h = H()
    from transformers import AutoTokenizer
    import transformers

    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    cases = []
    for name, task, instr, caption, lyrics, lang, bpm, key, ts, dur in CASES:
        instruction = instr or constants.TASK_INSTRUCTIONS[task]
        meta_dict = h._build_metadata_dict(bpm, key, ts, dur)
        metas = h._parse_metas([meta_dict])[0]
        pure = h.extract_caption_from_sft_format(caption)
        text_prompt = constants.SFT_GEN_PROMPT.format(h._format_instruction(instruction), pure, metas)
        lyric_prompt = h._format_lyrics(lyrics, lang)
        cap = tok(text_prompt, padding="longest", truncation=True, max_length=256)
        lyr = tok(lyric_prompt, padding="longest", truncation=True, max_length=2048)
        full_lyr = tok(lyric_prompt)
        cases.append(
            {
                "name": name,
                "task": task,
                "instruction": instruction,
                "caption": caption,
                "lyrics": lyrics,
                "language": lang,
                "bpm": bpm,
                "key_scale": key,
                "time_signature": ts,
                "duration": dur,
                "metas": metas,
                "caption_prompt": text_prompt,
                "lyric_prompt": lyric_prompt,
                "caption_ids": cap["input_ids"],
                "lyric_ids": lyr["input_ids"],
                "lyric_untruncated_len": len(full_lyr["input_ids"]),
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {
        "generator": "tools/gen_ace_step15_prompt_fixtures.py",
        "upstream_commit": UPSTREAM_COMMIT,
        "transformers_version": transformers.__version__,
        "default_dit_instruction": constants.DEFAULT_DIT_INSTRUCTION,
        "task_instructions": {k: constants.TASK_INSTRUCTIONS[k] for k in ("text2music", "cover", "repaint")},
        "cases": cases,
    }
    (OUT / "cases.json").write_text(json.dumps(doc, indent=1, ensure_ascii=True) + "\n")
    print(f"wrote {len(cases)} cases to {OUT / 'cases.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: gen_ace_step15_prompt_fixtures.py <ACE-Step-1.5 checkout> <Qwen3-Embedding dir>")
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
