---
uuid: "f3e5a7c9-2d4b-4f6e-8a9c-1b7d5e3f2c0a"
title: "turbospark-runtime: speculative decoding"
summary: "speculation_policy.rs is the ONE place both crates/cli and turbospark-server resolve MTP/DFlash2 drafter choice. It reads the install's resident index, never assumes a family has a head"
tags: ["crate", "runtime"]
source: "crates/runtime/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## Where does speculative decoding policy live, and why here?

`speculation_policy.rs` owns three decisions in a fixed order:
`resolve_drafter` reads an install's resident index to say which drafter it
carries, `draft_policies` turns that plus the request into what to open
with, and `resolve_speculation` says whether this run may draft at all. It
lives in `turbospark-runtime` (not `crates/cli`, where it started) because
both `turbospark-cli` and `turbospark-server` need the identical three
decisions, and `crates/invocation` (which each front end also has its own
copy of the relevant enums for) is pure and may not read an install or a
machine.

There are two drafters (MTP and DFlash2), and that duplication is not
incidental: several fixed bugs here were "the same mistake, on the other
drafter, a day later."

## Don't

- Don't assume a caller who explicitly names a drafter (`--speculative-drafter
  dflash`) skips the install check that `auto` gets. Passing a drafter
  through untouched is not the same as never looking at the install. A
  named drafter has to be checked against the resident index exactly like
  `auto` does, or a caller pointed at an install with the WRONG drafter
  gets sent chasing an artifact that cannot help (e.g. told to stream an
  MTP shard for a MoE checkpoint no published MTP conversion targets).
- Don't test `install_has_mtp_head` (or its DFlash2 sibling) as a bare
  boolean. It's `Option<bool>`: `None` means "the resident index was
  unreadable," which must keep failing at open with the engine's own
  message. Never get it silently explained away as "no head."
- Don't assume fixing this class of bug on one drafter fixes it on both.
  MTP and DFlash2 each had the identical missing-head routing bug, fixed a
  day apart. Grep for the sibling before calling either fix done.
- Don't assume `speculation_policy_tests.rs`'s 37 passing cases prove the
  blocker's error MESSAGES are right. That file feeds fixture strings into
  `resolve_speculation` and never calls either blocker function, so it pins
  the ROUTING decision, not the TEXT a caller sees. Deleting the pointer
  from the real error message leaves all 37 cases green.
- Don't assume `TURBOSPARK_MTP_DRAFT` or `TURBOSPARK_DFLASH_DRAFT` unset
  means "the feature doesn't run and costs nothing to check." Unset,
  unparsable, or 0 allocates and encodes NOTHING, byte-for-byte identical
  to before either drafter module existed, which is precisely what lets the
  existing memory-oracle frozen rows stand without a new one. Requesting a
  depth on an install with no matching head is an ERROR at open, never a
  silent no-op.

See [[crate-runtime]] for the core architecture and [[crate-runtime-2]] for
the dispatch-order and env-seam constraints.
