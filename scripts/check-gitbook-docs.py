#!/usr/bin/env python3
"""Lint the markdown that GitBook Git Sync imports.

Git Sync fails a whole space on one bad page, and the failure only shows up in
GitBook after the push. This catches the two causes seen so far before the push:

* YAML front matter that does not parse (an unquoted ": " inside a value broke
  the Documentation import).
* Files named like "page (1).md", which GitBook and Finder create when a name
  collides. They are orphans that nothing in SUMMARY.md links to.

Synced directories come from gitbook-docs.yaml so a new space is covered
without editing this script. Run from anywhere: `python3 scripts/check-gitbook-docs.py`.
Needs PyYAML.
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DUPLICATE_NAME = re.compile(r" \(\d+\)(\.[^.]+)?$")
FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", re.S)


def synced_directories():
    config = yaml.safe_load((ROOT / "gitbook-docs.yaml").read_text(encoding="utf-8"))
    found = []

    def walk(node):
        if isinstance(node, dict):
            directory = (node.get("content") or {}).get("directory")
            if directory:
                found.append(ROOT / directory)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config)
    return found


def main():
    problems = []
    directories = synced_directories()
    if not directories:
        problems.append("gitbook-docs.yaml: no content directories found")
    checked = 0
    for directory in directories:
        if not directory.is_dir():
            problems.append(f"{directory.relative_to(ROOT)}: synced directory is missing")
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or ".gitbook" in path.relative_to(directory).parts:
                continue
            rel = path.relative_to(ROOT)
            if DUPLICATE_NAME.search(path.name):
                problems.append(f"{rel}: duplicate-style name (delete it or rename it)")
            if path.suffix != ".md":
                continue
            checked += 1
            text = path.read_text(encoding="utf-8")
            if not text.startswith("---"):
                continue
            match = FRONT_MATTER.match(text)
            if not match:
                problems.append(f"{rel}: front matter opens with --- but never closes")
                continue
            try:
                data = yaml.safe_load(match.group(1))
            except yaml.YAMLError as err:
                reason = str(err).splitlines()[0]
                problems.append(f"{rel}: front matter is not valid YAML ({reason}); quote the value")
                continue
            if data is not None and not isinstance(data, dict):
                problems.append(f"{rel}: front matter must be a mapping")

    for problem in problems:
        print(problem)
    print(f"checked {checked} markdown files in {len(directories)} synced directories, "
          f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
