#!/usr/bin/env python3
"""Lint the markdown that GitBook Git Sync imports.

Git Sync fails a whole space on one bad page, and the failure only shows up in
GitBook after the push. This catches the causes seen so far before the push:

* YAML front matter that does not parse (an unquoted ": " inside a value broke
  the Documentation import).
* Files named like "page (1).md", which GitBook and Finder create when a name
  collides.
* Pages that SUMMARY.md never links. GitBook builds a space's navigation from
  SUMMARY.md, so an unlinked page is dead weight, and a second copy of a page
  is how the API Reference ended up documented twice.
* Doubled path segments such as "rust/rust/" or "http/http.md". GitBook writes
  these when a SUMMARY heading group and its first page share a name, and the
  doubling then shows up in the published URL as well.
* Relative links to a .md file that does not exist.

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
FENCED_CODE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.S | re.M)
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"\]\((<[^>]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\)")
# A page that is only named after its folder, with no heading group sharing the
# name: its published URL is /models-and-hardware-acceleration/models, not doubled.
ALLOWED_DOUBLED = {"documentation/models/models.md"}


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


def relative_md_links(text):
    """Yield link targets that point at a local .md file (anchors stripped)."""
    text = INLINE_CODE.sub("", FENCED_CODE.sub("", text))
    for match in LINK.finditer(text):
        target = match.group(1).strip("<>").split("#")[0]
        if target.endswith(".md") and "://" not in target and not target.startswith("/"):
            yield target


def doubled_segment(relative):
    """Return the repeated name if the path nests a directory or page in itself."""
    parts = relative.parts
    for a, b in zip(parts, parts[1:]):
        if a == b:
            return a
    if len(parts) >= 2 and parts[-1] != "README.md" and parts[-1] == parts[-2] + ".md":
        return parts[-2]
    return None


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
        summary = directory / "SUMMARY.md"
        linked = set()
        if summary.is_file():
            for target in relative_md_links(summary.read_text(encoding="utf-8")):
                linked.add((directory / target).resolve())
        else:
            problems.append(f"{summary.relative_to(ROOT)}: SUMMARY.md is missing")
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or ".gitbook" in path.relative_to(directory).parts:
                continue
            rel = path.relative_to(ROOT)
            if DUPLICATE_NAME.search(path.name):
                problems.append(f"{rel}: duplicate-style name (delete it or rename it)")
            doubled = doubled_segment(path.relative_to(directory))
            if doubled and rel.as_posix() not in ALLOWED_DOUBLED:
                problems.append(f"{rel}: doubled path segment '{doubled}' (a heading group and its page share a name)")
            if path.suffix != ".md":
                continue
            checked += 1
            if path != summary and path.resolve() not in linked:
                problems.append(f"{rel}: not linked from {summary.relative_to(ROOT)}")
            text = path.read_text(encoding="utf-8")
            for target in relative_md_links(text):
                resolved = (path.parent / target).resolve()
                # Cross-space links written by GitBook can climb out of the repo; only
                # judge targets that land inside it.
                if ROOT in resolved.parents and not resolved.exists():
                    problems.append(f"{rel}: link to missing page '{target}'")
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
