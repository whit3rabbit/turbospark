# Stage 7: Critic report (INDEX.md, full file, one round, no remediation)

Reviewers: egm-design (DC), egm-structural (ST), egm-style (SY). All completed (structural and style were first refused with "Concurrent subagent limit reached", then rerun one at a time and succeeded). Standard: feature-spec-standard for design and structural. Applicability note (all three): Problem Statement, Alternatives, Proposed Architecture and per-feature Risks sections belong to the ZP drafts and do not apply to an index. Applicable: accuracy against siblings, dependency and sequencing correctness, consistency, link validity, a program-level risk note. ST and SY reviewers did not open every sibling; SY did not open any.

## egm-design

Major
- DC-1. Line 32 (and line 94): "adds Rust-side rollout logging" is false. ZP-03 is Swift-only (RolloutStore.swift, hooks on existing engine callbacks, "no model gates apply"). Fix: "large; Swift-side per-chat JSONL rollout store and generation hooks, no Rust changes"; reword step 4 rationale.
- DC-2. Line 42: "WebM" is not what ZP-15 proposes (lines 62-65 lean MP4/H.264 via AVAssetWriter; container is open). "Reverses an earlier exclusion" is accurate (ZP-90 line 165). Fix: state the container is an open question leaning MP4.
- DC-3. Lines 97-105: ZP-06 missing from sequencing (22 of 23 listed). Fix: add a step after ZP-05 and ZP-07.
- DC-4. Line 99: ZP-17 (step 3) sequenced before ZP-03 (step 4), ZP-08 (step 5) and ZP-06 (none), whose files its typed-error contract modifies (ZP-17 lines 158-164, test line 171). The CrashRecoveryStore half is independent. Fix: split ZP-17 or move it after those three; add a dependency bullet.
- DC-5. Lines 87-88 vs 99-100: ZP-20 and ZP-18 ship through ZP-07's bundled-skill root (ZP-20 lines 45, 91; ZP-18 line 38; ZP-07 line 53), not ZP-05; ZP-05 lines 57-59 list them as day-one candidates. Fix: state ZP-05 is not a prerequisite and they migrate to it.
- DC-6. Line 89: ZP-22 "stands alone" omits its dependency on ZP-03's rollout log (ZP-22 line 106). Fix: add a soft dependency.
- DC-7. Lines 50, 65-74, 51: coverage map inconsistent. 7.1 counted in both ZP-22 and ZP-90; 7.2 "key half" vs "excluded permanently"; 2.1 is "Different" (ZCODE_COMPARISON line 159), outside the 40. The comparison file has exactly 40 Missing headings. Fix: add a tally (24 drafted Missing + 5 kiro + 12 ZP-90 - 1 overlap, plus 2.1 outside the 40) and say 7.1 is adopted.
- DC-8. Line 98 vs ZP-01 line 73: ZP-01 (step 2) depends on the concurrent-chats decision (ZP-23, step 6); ZP-01 cites ZP-02 instead of ZP-23; the index lists no edge. Fix: add a bullet; scope ZP-01 to originating-chat mode or move it after ZP-23.

Nit
- DC-9. Lines 106-108, 134-135: sequencing criteria unstated; stale branch name. Fix: SHA, one-line criteria.
- DC-10. Lines 67-71: "nine excluded" not itemized by reason. Fix: name them or link ZP-90 rows.
- DC-11. Which exclusions are account-gated is not stated. Fix: one clause.

Verified: ZP-02/21/23 wording (lines 78-80) matches those drafts; ZP-09/ZP-08/ZP-19 wording matches; all 23 drafts plus ZP-90 exist.

## egm-structural

Major
- ST-1. Lines 92-105: ZP-06 absent from every step. Fix: add it after ZP-05 or state it is deliberately unsequenced.
- ST-2. Lines 87-88 vs 97-99: dependency vs order contradiction; soft via bundled-skill-root fallback (ZP-07 lines 53-56, ZP-18 line 38). Fix: reorder or state the fallback.
- ST-3. Lines 4-6, 25, 49-51, 63-66: counts do not reconcile (25 + 5 + 12 = 42 rows; works only if 7.1 once and 2.1 excluded). Fix: reconciliation line.

Nit
- ST-4. Lines 36-48: table order neither ZP nor finding order; inconsistent notation for combined findings. Fix: sort by ZP number or document the grouping.
- ST-5. Lines 27-51: size notes inconsistent; ZP-20 row label differs from its H1 (Bundled content plugins). Fix: Size and Note columns, use draft wording.
- ST-6. Lines 76-90: dependencies are prose; ZP-02 line 45 and ZP-21 line 27 may still call ZP-23 future or gated. Fix: dependency table with hard or soft kind; check sibling wording.
- ST-7. Lines 63-74: heading "Excluded or decided" holds adoption, partial inclusion and exclusion; only IDs listed. Fix: Finding | Verdict | Decision table or link ZP-90's table.
- ST-8. Lines 107-135: no program-level risk note (ZP-23 slip, link drift, ZP-22 seam); Provenance holds decisions. Fix: short Program risks subsection; rename Provenance.
- ST-9. Lines 57-61, 69: .kiro links do not resolve in this worktree; depth is correct for a full checkout. No action beyond an optional note.
- ST-10. Line 135: branch name stale. Fix: base commit hash.
- ST-11. Lines 16-17: says drafts follow the EGM template; the index follows none and does not say so. Fix: one clause saying it is exempt.

Verified: heading hierarchy clean; all 24 ZP links resolve; sibling H1 numbers and finding numbers agree; ZCODE_COMPARISON link, commit a60ba6a7, v3.17.1 and audit date match.

## egm-style

Major
- SY-1. Lines 5-7, 23, 25-75: "account for all 40" does not total 40 on the tables (42 entries, 41 unique, 2.1 is Different, 7.2 only in ZP-90). Fix: state the arithmetic.
- SY-2. Lines 63-74 vs 50: 7.1 "remaining" yet drafted; 7.2 excluded yet "key half" in ZP-22. Fix: split wording; say which part of 7.2 ZP-22 takes.
- SY-3. Lines 92-105: ZP-06 omitted though ZP-07's plugin host. Fix: place it or state deferral.
- SY-4. Lines 87-88 vs 99-100: ZP-05 vs ZP-20/ZP-18 order. Fix: reorder or state what can land without it.
- SY-5. Lines 27-51, 97-105: size labels undefined; "some cross-cutting", "app-local" vague; "largest" vs "large" unranked. Fix: define the scale and name the drafts.
- SY-6. Lines 6-7, 18-20: "fit" and "defensible" have no criterion. Fix: refer to the roadmap exclusion policy and program constraints.
- SY-7. Lines 132-135: "link-verified at audit time" and "per repo rules" ungrounded; branch vs commit. Fix: cite rule file, SHA.

Nit
- SY-8. Lines 89-90: "turn loop seam" unexplained. Fix: name module and what is verified.
- SY-9. Line 123: slogan "Local green is not release evidence". Fix: plain sentence plus link.
- SY-10. Lines 98-105: informal wording ("wins", "big behavioral change", "go live"). Fix: neutral terms.
- SY-11. Line 116: "ten localization parity gates" has no pointer. Fix: link or drop the count.
- SY-12. Lines 86-91: loose role wording. Fix: "ZP-19 provides the image search tool ZCode's office plugins call".
- SY-13. Lines 1, 17: terms mixed; EGM template unlinked. Fix: link or drop.
- SY-14. Lines 56-72: table order. Fix: sort by ZP number.
