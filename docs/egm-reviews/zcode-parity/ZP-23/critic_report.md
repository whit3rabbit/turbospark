# Stage 7: Critic report (ZP-23)

One round only (no remediation, so no convergence assessment). Reviewers: egm-design (persona read: agents/design_critic.md, standard feature-spec-standard), egm-structural (persona read: agents/structural.md, standard feature-spec-standard), egm-style (see end of file).

## Design critic (DC)

Verdict: not ready. 3 Blocker, 6 Major, 5 Nit.

DC-B1 Blocker, lines 149-150, 153-155. Alternatives rejected "by author" with no link; B is circular; A is self-defeating (decode serializes anyway, yet the design interleaves on one lane). Missing real options: Rust-side scheduler vs Swift lease, one engine per chat, KV swap/evict for background chats (ZCode idle deactivation), non-generating-only background work.
DC-B2 Blocker, lines 66-74, 111-139. Central mechanism unverified yet presented as settled; no fallback or abort criterion if FFI cannot step sessions independently; Rust-side work unsized.
DC-B3 Blocker, lines 159-171, 75. No Risks and Mitigations section; "no new memory risk" and "2 is safe on 16 GB" unsupported (two KV caches plus expert cache, no numbers, no KV-allocation-failure behavior).
DC-M1 Major, lines 74, 135-137, 167-169. "Round-robin with foreground priority" is two contradictory policies; no weighting, ratio or falsifiable threshold.
DC-M2 Major, lines 48-50, 135-139. Behavior at the cap undefined; subagents sharing parent's cap risk deadlock; title generation, memory extraction, cron/off-peak dispatch and goal timers have no lane treatment.
DC-M3 Major, lines 111-133, 173-208. State refactor underscoped: streaming buffer, context ring, sampling overrides, todos, steering, queue drains, stop, approval cards all selected-chat-bound; five files is not credible for the biggest change in the program.
DC-M4 Major. Tool-execution concurrency and security: same-workspace races, shared permission engine/browser/shell state, approval-prompt confusion when B's approval appears while in A, timeouts, auto-approve unattended. State ladder behavior for background chats.
DC-M5 Major. No error and cancellation isolation, per-chat stop semantics, crash recovery with multiple in-flight chats, flush time budget; "atomic per chat" (line 166) unexplained.
DC-M6 Major, lines 6-8. KV and session lifecycle for idle chats undefined; cited ZCode idle-deactivation parity target not honored; bounded pool or LRU missing.
DC-N1 Nit, line 166. "Sampling owns normalization" unexplained.
DC-N2 Nit, lines 147, 220. "No-unmeasured-gates rule" and "bench gates" unlinked.
DC-N3 Nit, lines 96-102. CUJ 3 gives no way forward; model picker per-chat or global unstated.
DC-N4 Nit. INDEX frames program as "Missing" findings; ZP-23 is "Different"; add a one-line carve-out.
DC-N5 Nit. No success metrics, rollout or flag plan (cap of 1 as kill switch), no fake-engine test strategy, no engine-error or cancellation tests.

## Structural (ST)

Verdict: fails the standard. Heading structure sound (no skipped levels, fence closed). 

ST-1 Blocker, whole document. No Risks and Mitigations H2; nearest is the Resource safety bullet (161-164). Uncovered: KV exhaustion, background starving foreground, lease races, approvals for a non-visible chat (security-relevant), partial flush on crash.
ST-2 Blocker, lines 149-150, 155-156. Rejections by assertion; "the goal is the point" circular; "no-unmeasured-gates rule" unlinked.
ST-3 Major, lines 141-157, 66-74, 117-118, 135-137. Alternatives thin; the real fork (Rust FFI concurrency vs Swift scheduler) is only an Open Question while the plan commits to the Swift lease. Move into Alternatives; mark plan contingent.
ST-4 Major, lines 66-77. Open Questions misplaced, no owner or exit criterion; cap default reads as a decision.
ST-5 Major, lines 49-50, 138-139, 164-166. ResidencyPolicy, "expert cache caps", "runtime rules" undefined or unlinked; "subagents already serialize" unsourced; "sampling owns normalization" without context.
ST-6 Major, lines 37-53, 167-169, 219-220. Goals lack measurable criteria (foreground tok/s regression, switch latency, memory ceiling).
ST-7 Nit, lines 26, 111. Headings differ from standard names ("The Problem", parenthetical in "Technical Plan").
ST-8 Nit, lines 198-208. "Mutation checks" nested as a sibling field of file 6 only.
ST-9 Nit, lines 96-102. "This model" ambiguous (A on M, user picks N); string must be cataloged in 21 languages, so finalize or mark placeholder.
ST-10 Nit (no defect). Metadata, relative link, ZCode links and finding 2.1 classification match INDEX and ZCODE_COMPARISON.

## Style (SY)

Persona read: agents/style.md. ZCODE_COMPARISON.md not read by this reviewer. No Blockers.

SY-1 Major, lines 161-165. "Without new memory risk", "KV per session is the accounting unit", "expert caps continue to apply" are uncited claims. State verified vs to-verify, cite crates/runtime source.
SY-2 Major, lines 167-169. "Without stealing the interactive budget" contradicts round-robin on a serialized GPU and lines 219-220. Rewrite as design intent with a measurable foreground tok/s criterion.
SY-3 Nit, lines 18-24, 33-34. Rhetoric ("not by parity inertia", "biggest single behavioral change"). Drop.
SY-4 Nit, lines 75-77. "Safe default" unsupported; sits under Open Questions while lines 49 and 118 treat it as decided.
SY-5 Nit, lines 68-74 vs 135-139. Decode lane open question vs committed plan; align.
SY-6 Nit, lines 143-150. Alternative A "real throughput gain" conflicts with its cons; "disproportionate" subjective; rule uncited.
SY-7 Nit, lines 149, 157. "Rejected by author" duplicated; "the goal is the point" is filler.
SY-8 Nit, lines 135-139 vs 61. "Subagents already serialize" uncited.
SY-9 Nit, lines 205-208. Mutation checks nested inconsistently.
SY-10 Nit, lines 98-102. Refusal text not parameterized; name the chats and the localization key.
SY-11 Nit, lines 170-171. Accessibility announcement text depends on ZP-02, which depends on this draft; state owner.
SY-12 Nit, lines 215, 170-171. "21 languages" vs INDEX "ten localization parity gates"; use one reference.
SY-13 Nit, line 51. "Quit generalizes" is figurative; rewrite.
SY-14 Nit, lines 28-31. "Parks the whole app" imprecise; say blocks chat switching and creation.
