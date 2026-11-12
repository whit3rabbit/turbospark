# Stage 8 report: intermediate zoom-out (INDEX.md, full file)

Context: Stage 5 to 7 reports. Verdict: the index passes as an index once the contradictions are fixed. About 60 raw findings reduce to roughly 8 real ones; the rest would turn a 133-line map into a second design document.

## Carry (real, cheap, index-level)

S8-1 Blocker. Lines 83-84, 95, 99. OC-11 is in step 3 and step 7, before OC-10's seam. Fix: delete OC-11 from step 3, keep the pair in step 7. Carry the format-before-diagnose concern only as one clause ("order of the two post-edit stages is decided in OC-10 and OC-11"). Neither draft names the seam, so soften "land OC-10's seam first" to "intended". Do not add a shared stage-list step.

S8-2 Major. Lines 30, 49, 64. 16 + 8 + 11 = 35, not 31. Fix: one sentence under line 28 listing the five split findings (crash recovery, multi-window, provider OAuth, unread, tab strip) and noting deep link is folded into the multi-window row. Reject a per-finding table.

S8-3 Major. Lines 73, 76-78, 94. OC-03 and OC-02 treat ZP-23 and ZP-02 as soft. Fix: reword as "soft dependency, no gating; ZP-23 and ZP-02 enrich"; drop "consumes the same state"; add one line: external prerequisites (ZP-02, ZP-21, ZP-23, shell-hardening import wizard) block only OC-04's tab strip and OC-07's import step. No requires column.

S8-4 Major. Lines 47, 79-80. Two chord collisions missing: OC-05 vs Previous/Next Chat (TurboSparkApp.swift lines 445, 453), OC-04 Cmd+W vs OC-01 reserved list. Fix: two one-line bullets saying each draft must resolve its conflict before shipping; fix the drafts, not the index.

S8-5 Major. Lines 42, 85-87. OC-12 does not reference OC-13. Fix: delete the bullet; change "new Rust crate" to "new crate or server submodule (open)".

S8-6 Major. Lines 53-55, 81-82. .kiro specs absent from this tree; line 54 "1 diagnostics export" should be section 7. Fix: mark specs "not in this tree; confirm before relying"; correct the number.

S8-7 Nit. Lines 30, 33-60. Finding labels are comparison section numbers. Fix: one legend sentence.

S8-8 Nit. Lines 126-131. Branch feat/classlru-replay-arm not in the comparison. Fix: drop the branch name, keep commit and date. Add under line 9: "all drafts are proposals; ZP and .kiro items may be unbuilt".

## Move to the drafts

OC-03 ordering vs ZP-02 line 102 (fix OC-03; index must not assert a shared order). OC-10 vs "no Python or Node runtime": optional clause "language servers are user-installed, not bundled".

## Advise dropping (scope beyond an index)

Glossary; size scale and Size/Note columns; hot-file merge-order line; go/no-go or first-cuts naming; style rewrites (pure integration, largest surface, local green, YAGNI); interface extraction (post-edit stage list, attention-state protocol); status columns and slip rules; per-finding reconciliation table; OC-08 vs OC-10 ordering rationale; duplicate date and link wrap nits; LAN re-verification.

## Still missing

Nothing blocking.
