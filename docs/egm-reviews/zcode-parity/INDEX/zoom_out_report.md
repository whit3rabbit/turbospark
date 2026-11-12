# Stage 5: Zoom Out report (INDEX.md, full file, 137 lines)

Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed. Neither read every sibling end to end; the .kiro files linked by the target are absent from this worktree. Zoom-Out criteria (YAGNI, reuse) mostly do not apply to an index; accuracy, consistency and completeness were checked instead.

## egm-zoom-out

Major
- ZO-1. Lines 97-105: ZP-06 (the "largest draft", line 34) is missing from the sequencing list (22 of 23 drafts listed). Fix: add ZP-06, after ZP-05/ZP-07 and alongside ZP-08.
- ZO-2. Line 99 vs ZP-17 lines 158-164: ZP-17 (step 3) is sequenced before ZP-03 (step 4), ZP-08 (step 5) and ZP-06 (absent), yet its typed-error contract modifies RolloutStore.swift (created by ZP-03), OfficeRenderModel.swift and PluginPageWebView.swift. Fix: split ZP-17 (recovery marker early, typed-error contract after ZP-03/06/08) or move it after step 5; add the edge to Cross-draft dependencies.
- ZO-3. Lines 78-83, 102-104 vs ZP-23 lines 20, 43, 86, 128, 170: circular. ZP-23's row states show through ZP-02 badge slots, while ZP-02's live states need ZP-23. Fix: state that ZP-02 lands first in non-live form, or ship both as one unit.
- ZO-4. Line 32: "adds Rust-side rollout logging" contradicts ZP-03, which is Swift-only (RolloutStore, hooks on existing engine callbacks, no decode/sampling/KV change, ZP-03 lines 100-103, 135-136, 147-151). Line 94 also files ZP-03 under "Rust-side or cross-cutting". Fix: "large; Swift-side rollout store on existing engine callbacks"; drop the Rust claim.

Nit
- ZO-5. Line 42: "WebM recording" is stale; ZP-15 lines 62-64 lean MP4. Fix: note the container is an open question leaning MP4.
- ZO-6. Lines 65-66: "remaining 12 findings ... 7.1" is imprecise; 7.1 is drafted as ZP-22. Count reconciles as 25 + 5 + 12 - 1 = 41 - 1 = 40 only if 7.1 is counted once. Fix: say 11 are unique to ZP-90 and 7.1 is cross-referenced; note 7.2's key half lives in ZP-22.
- ZO-7. Lines 87-88: ZP-05 is called the delivery mechanism for ZP-20 and ZP-18, but they are sequenced before it; the drafts fall back to ZP-07's bundled-skill root. Fix: state that.
- ZO-8. Lines 81-88: dependency list omits ZP-09, ZP-18, ZP-20 relying on ZP-07's bundled-skill root (ZP-09 line 118, ZP-18 line 39, ZP-20 lines 45, 91). Fix: add a bullet.
- ZO-9. Lines 134-135: "written against branch feat/classlru-replay-arm" is stale; current branch is main. Fix: give a commit SHA.
- ZO-10. Lines 57-61, 69: .kiro briefs and roadmap links do not resolve in this worktree, so "already owned" claims were not verified. Fix: note that .kiro is out of tree or confirm.

Verified clean: ZP-02/ZP-21 cite ZP-23 as prerequisite (matches steps 6-7); ZP-09 after ZP-08; titles and numbers in the table match the files; ZP-90 breakdown sums to 12.

## egm-inventive-catalyst

Major
- IC-1. Lines 99-100, 102, 104 vs ZP-01 lines 70-74: ZP-01 sits in step 2 as "app-local", but its open question on where tasks run depends on the concurrent-chats decision (now ZP-23). Edge unrecorded. Fix: land ZP-01 with originating-chat mode only; record ZP-23 as the later relaxation.
- IC-2. Lines 89-90 vs ZP-22 line 106: index says ZP-22 "stands alone", but ZP-22 writes to the rollout log owned by ZP-03. Fix: add the edge, or make the log write a no-op without a logger.
- IC-3. Lines 87-88 vs steps 3-4: same finding as ZO-7/ZO-8 (ZP-07 bundled-skill root is the real shared dependency; ZP-05 is a later re-delivery path).
- IC-4. Sequencing omits ZP-06, and ZP-17 line 163 ("ZP-03/08/06 implement" failure states) precedes all three. Same as ZO-1/ZO-2. Fix: add ZP-06 after ZP-07 and ZP-17; note ZP-17 defines the primitive and ZP-03/06/08 adopt it.
- IC-5. Step 6 and line 78-79 vs ZP-22 line 214 and ZP-23 line 115: both touch the turn loop; ZP-22's "first task verifies the seam" is contradicted by ZP-23 landing earlier. Fix: make ModelStep seam extraction a standalone step zero before ZP-23.
- IC-6. Steps 6-7: ZP-02's pending-approval state exists today, so unread and pending can ship early, with the running state added after ZP-23. Fix: segment ZP-02.

Nit
- IC-7. Lines 32, 42, 43: ZP-15 depends softly on ZP-14 (ZP-15 line 45), unlisted; "reverses" vs "supersedes" used for the same thing.
- IC-8. Lines 25-51 vs 65: table ordered by finding category, not ZP number; no stated count arithmetic. Fix: add a one-line count check, sort by ZP number.
- IC-9. Line 135: stale branch reference (same as ZO-9).

Simplification offered: replace the five-bucket list with a short dependency graph of hard edges only (ZP-07 to ZP-20/18/09; ZP-08 to ZP-09; seam to ZP-23/22; ZP-23 to ZP-21/02 live; ZP-03 to ZP-22; ZP-17 to ZP-03/06/08; ZP-14 to ZP-15). Reviewer advisory only; may add scope.
