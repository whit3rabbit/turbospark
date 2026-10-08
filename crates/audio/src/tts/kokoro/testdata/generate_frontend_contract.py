#!/usr/bin/env python3
"""Regenerate offline parity fixtures with the pinned independent Python frontend.

This helper is never invoked by Rust tests or product synthesis. It requires
Misaki fba1236595f2d2bf21d414ba6e57d25256afada3 and the documented spaCy model
already installed, so regeneration cannot silently select or download a model.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import string

from misaki import en

SOURCE_REVISION = "fba1236595f2d2bf21d414ba6e57d25256afada3"
SOURCE_SHA256 = "a9c36c90a3fa6c6084e9042fd8e19f6049add4f1435d8fe5ca1b77f3529a98f4"
DICTIONARIES = {
    "us_gold.json": "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f",
    "us_silver.json": "de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_fixture = Path(__file__).with_name("frontend-contract.json")
    parser.add_argument("--output", type=Path, default=default_fixture)
    args = parser.parse_args()
    source = Path(inspect.getfile(en))
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise SystemExit("Misaki frontend source differs from the pinned revision")
    for name, expected in DICTIONARIES.items():
        if hashlib.sha256((source.parent / "data" / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Misaki dictionary {name} differs from its pin")
    for package, version in {"spacy": "3.8.7", "en-core-web-sm": "3.8.0"}.items():
        if importlib.metadata.version(package) != version:
            raise SystemExit(f"Regeneration requires installed {package} {version}")

    base = en.G2P(trf=False, british=False, fallback=lambda _: (None, None))

    def spell(token):
        if not token.text or any(c not in string.ascii_letters for c in token.text):
            return (None, None)
        return base.lexicon.get_NNP(token.text)

    frontend = en.G2P(trf=False, british=False, fallback=spell)
    fixture = json.loads(default_fixture.read_text())
    results = []
    for case in fixture["results"]:
        phonemes, tokens = frontend(case["text"])
        results.append({
            "text": case["text"],
            "phonemes": phonemes,
            "tokens": [{"text": t.text, "tag": t.tag, "phonemes": t.phonemes} for t in tokens],
        })
    fixture["results"] = results
    args.output.write_text(json.dumps(fixture, ensure_ascii=True, indent=2) + "\n")
    print(f"Generated {len(results)} cases from pinned Misaki {SOURCE_REVISION}")


if __name__ == "__main__":
    main()
