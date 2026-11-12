# Consolidated review report: ZCode parity INDEX

Target: docs/designs/zcode-parity/INDEX.md (full file, 137 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. Every reviewer completed; none failed. Egm-structural and egm-style were first refused ("Concurrent subagent limit reached"), then rerun one at a time and completed.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.
Standard applicability: this is an index, so Problem Statement, Alternatives, Proposed Architecture and per-feature Risks sections of feature-spec-standard do not apply; review concentrated on accuracy against siblings, sequencing, consistency, completeness. The .kiro links (lines 57-61, 69) do not resolve in this worktree and were not verified.

Counts after deduplication: Blocker 0, Major 9, Nit 12.

Verified by the reviewers against sibling files: the Rust-side claim (line 32) is false; the WebM label (line 42) is stale; ZP-17 is sequenced before ZP-03/08/06, whose files it edits.

## Blocker

None.

## Major

M1. Line 32 (also 94). "Adds Rust-side rollout logging" for ZP-03 is false: ZP-03 is Swift-only on existing engine callbacks, no Rust change. Fix: "large; Swift-side rollout store on existing engine callbacks, no Rust changes"; drop "Rust-side" from line 94. [ZO-4, DC-1, S8-1]

M2. Lines 97-105 (and 34, 84). ZP-06, the "largest draft" and the host for ZP-07's plugins, is in no sequencing step (22 of 23 listed). Fix: add it after ZP-05/ZP-07, beside ZP-08, or state it is unsequenced. [ZO-1, IC-4, CO-5, DC-3, ST-1, SY-3, S8-2]

M3. Line 99 (and 76-90). ZP-17 (step 3) pins the error contract of RolloutStore (ZP-03, step 4), OfficeRenderModel (ZP-08, step 5) and PluginPageWebView (ZP-06, unscheduled); ZP-17 lines 158-164 and test line 171. The recovery half is independent. Fix: land only the recovery marker in step 3, the contract after ZP-03/06/08; add the dependency bullet. [ZO-2, IC-4, DC-4, S8-3]

M4. Lines 87-88 vs 99-100. ZP-05 is called the delivery mechanism for ZP-20 and ZP-18, yet both sit in step 3, before ZP-05 in step 4. The siblings ship first through ZP-07's bundled-skill root (ZP-20 lines 45, 91; ZP-18 line 38; ZP-07 line 53; ZP-09 line 118). Fix: say ZP-05 is a later re-delivery path, not a prerequisite, and add ZP-07's root as the shared dependency. [ZO-7, ZO-8, IC-3, CO-5, DC-5, ST-2, SY-4, S8-5]

M5. Lines 4-6, 25-51, 63-74. Coverage accounting does not reconcile to the stated 40. 7.1 is both drafted (ZP-22) and in ZP-90's "remaining 12"; 7.2 is "excluded permanently" yet "key half" is in ZP-22; 2.1 is "Different", outside the 40 (ZCODE_COMPARISON line 159, which has exactly 40 Missing headings); 4.3-4.5 share one row. Fix: one reconciliation sentence (11 unique to ZP-90, 7.1 cross-listed, ZP-22 takes only the API-key half of 7.2, 2.1 outside the 40). [ZO-6, CO-1, CO-11, DC-7, ST-3, ST-7, SY-1, SY-2, S8-6]

M6. Lines 78-79, 102-104 vs ZP-23 lines 20, 43, 86, 128, 170. ZP-23's row states show through ZP-02 badge slots while ZP-02's live states need ZP-23; step 6 says "before ZP-02 goes live" and numbering puts ZP-23 after its dependents. Unclear whether "before" is merge or release order. Fix: state which; note ZP-02 lands first in non-live form or ship together. [ZO-3, CO-2, ST-6]

M7. Line 98 vs ZP-01 line 70-74. ZP-01 (step 2, "app-local") depends on the concurrent-chats decision (ZP-23, step 6); the edge is absent and ZP-01 cites ZP-02 instead. Fix: add the edge; scope ZP-01 to originating-chat mode or move it after ZP-23. [IC-1, DC-8] Reviewer in Stage 8 advises deferring the redesign to ZP-01; at minimum record the edge.

M8. Lines 89-90 vs ZP-22 line 106, 214 and ZP-23 line 115. "ZP-22 stands alone" is wrong: it logs to ZP-03's rollout, and ZP-22's "verify the turn-loop seam first" conflicts with ZP-23 landing earlier on the same loop; "turn loop seam" is undefined. Fix: "mostly stands alone; writes to ZP-03's rollout log if present"; define or link the seam. [IC-2, IC-5, CO-6, DC-6, SY-8] Stage 8 advises keeping only the wording fix in the index.

M9. Lines 3-7, 18-20, 27-51, 97-105, 132-135. Vague or unverifiable statements: size labels (small, medium, large, largest) undefined; "fit", "defensible", "some cross-cutting", "app-local" without criterion; "link-verified at audit time" and "per repo rules" uncited; drafts "written against branch feat/classlru-replay-arm" while current branch is main; ZCode and "kiro spec" never defined. Fix: define ZCode and kiro spec in one sentence each, name the cross-cutting drafts, cite the rule file, replace the branch with a base commit SHA. [CO-3, CO-4, CO-5, CO-8, DC-9, SY-5, SY-6, SY-7, ZO-9, ST-10, S8-7]

## Nit

N1. Line 42. "WebM recording" is stale; ZP-15 lines 62-65 lean MP4 (AVAssetWriter) with the container open. "Reverses" (42) and "supersedes" (43) mean the same. Fix: note the container is open, leaning MP4; use one verb. [ZO-5, IC-7, DC-2, S8-4]

N2. Lines 87-91. ZP-15's soft dependency on ZP-14 (ZP-15 line 45) and loose role wording ("fills the role", "documents the rest"). Fix: optional bullet; "ZP-19 provides the image search tool ZCode's office plugins call". [IC-7, SY-12]

N3. Lines 25-51, 56-72. Table order follows neither ZP nor finding order (ZP-09 after ZP-17); combined-finding notation inconsistent; ZP-20 row label differs from the draft H1; no tally. Fix: sort by ZP number, align the label. [IC-8, ST-4, ST-5, SY-14]

N4. Lines 67-71. Nine exclusions not itemized by reason. Fix: group by reason or link ZP-90 rows by finding number. [DC-10, DC-11, ST-7]

N5. Lines 107-135. No program-level risk note; Provenance carries decisions. Stage 8 advises rejecting as imposed structure on an index. Fix (optional): one line naming the ZP-23 slip risk. [ST-8]

N6. Lines 16-17. "EGM design document template" unlinked and the index is exempt from it, unstated. Fix: link or drop; one clause. [ST-11, SY-13]

N7. Line 116. "ten localization parity gates" has no pointer. Fix: link or drop the count. [SY-11]

N8. Line 123. "Local green is not release evidence" is a slogan. Fix: "A local passing build does not satisfy the release gates in verification.md". [CO-9, SY-9]

N9. Lines 98-105. Informal wording ("wins", "big behavioral change", "go live", "builds on"). Fix: neutral terms. [SY-10]

N10. Lines 3-12, 126-135. Audit date 2026-10-31 vs draft date 2026-11-11, not stated whether findings were re-verified. Fix: one clause. [CO-10]

N11. Lines 57-61, 69. .kiro briefs and roadmap links absent in this worktree (valid depth for a full checkout). Fix: none; optional note. [ZO-10, ST-9]

N12. Lines 21, 29, 31, 34, 46, 121-122. Undefined project terms (Open Questions, ladder position, off-peak, Gen UI, builtin seeding, CDN zip, no-Python/Node rule). Stage 8 advises rejecting a glossary. Fix: define at first use only where unavoidable. [CO-7]

## Dropped or deferred per Stage 8 (scope-adding, not carried as findings)

IC-5 (standalone ModelStep seam step zero), IC-6 (split ZP-02 into early and live parts), IC simplification (replace sequencing with a dependency graph), ST-3 to ST-8 structural additions (Size/Note columns, dependency table, verdict table, renamed Provenance). These belong in the individual drafts or are imposed structure; decide with the author.

## Traceability

- zoom_out_report.md: ZO-1 to ZO-10, IC-1 to IC-9. Feeds M1-M8, N1-N3, N11.
- comprehension_report.md: CO-1 to CO-11. Feeds M2, M4, M5, M6, M8, M9, N8, N10, N12.
- critic_report.md: DC-1 to DC-11, ST-1 to ST-11, SY-1 to SY-14. Feeds all Majors and N1-N9.
- intermediate_zoom_out_report.md: S8-1 to S8-7 confirm M1-M5, N1, M9 (branch) as open; recommends the drop list above.
- Merge notes: ZO-1, IC-4, DC-3, ST-1, SY-3 merged into M2; ZO-2, DC-4 and the ZP-17 half of IC-4 merged into M3; the ZP-22 seam, rollout and "stands alone" findings merged into M8.
