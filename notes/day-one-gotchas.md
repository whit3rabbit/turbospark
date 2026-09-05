---
uuid: "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e10"
title: "Day-one gotchas: crate aliasing, worktrees, and st"
summary: "Downstream crates import turbospark-core etc. under a per-crate alias (e.g. foundation), not the real package name. st (ripgrep-alike) skips gitignored files and needs st index per tree"
tags: ["gotchas", "day-one"]
depends_on: ["b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e01"]
source: "AGENTS.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What will trip me up in my first week here?

**Crates are imported under an alias, not their real package name.**
Downstream crates depend on `turbospark-core` (and several siblings) under
an alias declared in that crate's own `Cargo.toml`, e.g.
`foundation = { package = "turbospark-core", path = "../core" }`, and refer
to it as `foundation` in `use` statements. The alias is PER CRATE, not
global: `crates/bench` aliases repack to `repack`, but `crates/runtime`'s
dev-dependency on the same crate is unaliased and its tests import
`turbospark_repack` directly. Read the crate's own `Cargo.toml` before
assuming an import name carries over from wherever you were just reading.

**`st` (this repo's ripgrep-alike) skips gitignored files, and two files
are gitignored on purpose:** `ROADMAP.md` and `CLAUDE.local.md`. `st` finds
nothing in either. Read or `grep` them directly.

**Neither of those two files exists at all in a fresh git worktree.**
Gitignored files aren't carried into a worktree. Edits to `ROADMAP.md`
or `CLAUDE.local.md` have to happen at that path in the MAIN checkout, not
relative to whatever worktree you're working in.

**A fresh worktree also starts with no `st` index.** Run `st index` once
per tree before the first query, or every search reports "no index found."

## Don't

- Don't assume a worktree opened for a task whose subject is sitting
  unstaged in the main checkout has that code. It starts EMPTY at `HEAD`,
  with none of the uncommitted work the task description assumes. Run
  `git status` in the MAIN checkout first, and work there when the subject
  is uncommitted.
- Don't `git worktree remove --force` a dirty worktree without saving its
  diff first (`git diff > /tmp/<name>.patch`). It's the only way to remove
  a dirty worktree and it keeps nothing.
- Don't treat a compile error in a crate you didn't touch as yours to fix.
  This tree is routinely worked by more than one session at once. Check
  mtimes before debugging someone else's half-finished edit.
- Don't `git add` broadly and trust it. Re-run `git status` immediately
  before staging: a broad add in a multi-session tree can pick up another
  session's in-progress edit in the same file.
