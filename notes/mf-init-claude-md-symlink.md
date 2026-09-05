---
uuid: "99b84cb7-4ebf-4a70-8170-8d048dcb2ad7"
title: "mf init converts a symlinked CLAUDE.md into an independent file"
summary: "Fix: rm CLAUDE.md && ln -s AGENTS.md CLAUDE.md. mf init patches CLAUDE.md in place, which silently breaks a CLAUDE.md -> AGENTS.md symlink."
status: "active"
tags: ["mf", "claude-md", "agents-md", "symlink"]
created: "2026-09-04"
updated: "2026-09-04"
source: "session 2026-09-04, git status type-change after running mf init"
---

## Fix

This repo's `AGENTS.md` states "CLAUDE.md is a symlink to this file."
`mf init` appends an `<!-- BEGIN AGENT-CONFIG:mf -->` block into CLAUDE.md
directly, which it cannot do to a symlink in place -- it silently replaces
the symlink with a real file (a full copy of AGENTS.md plus the block).
AGENTS.md gets the same block appended too, so right after `mf init` the
two files are byte-identical but no longer linked: any later edit to
AGENTS.md will not show up in CLAUDE.md.

```sh
rm CLAUDE.md
ln -s AGENTS.md CLAUDE.md
```

The mf block is preserved (it's in AGENTS.md already) and CLAUDE.md
inherits it again via the symlink, so mf still works.

**Symptom**: `git status` shows CLAUDE.md as a type change (`T`, mode
120000 -> 100644) with thousands of lines "added," right after `mf init`.

## Don't

- Don't assume a tool that "patches" an agent-instructions file
  (CLAUDE.md, AGENTS.md, .cursorrules) checks whether the target is a
  symlink first. Check with `git ls-tree HEAD <file>` (mode 120000 =
  symlink) before and after running such a tool.
