# Stage 8: Intermediate Zoom Out report (INDEX.md, full file, unedited since Stage 5)

Reviewer: egm-zoom-out (completed). Read the persona, standard, target, Stage 5-7 reports, and ZP-03, ZP-15, ZP-17 text. Other siblings not re-read. Verdict: passes on simplicity and altitude for an index; three accuracy defects (S8-1 to S8-3) remain open and cheap to fix.

Verified against files: ZP-17 item 4 (lines 158-164) edits RolloutStore.swift (created by ZP-03 line 142), OfficeRenderModel.swift (ZP-08) and PluginPageWebView.swift (ZP-06), but sits in step 3 before them. Line 32 "Rust-side" is false (ZP-03 is Swift-only). Line 42 "WebM" is stale (ZP-15 lines 62-65 lean MP4).

Findings (all repeat earlier open findings)
- S8-1 Major. Lines 32, 94: Rust-side claim false. Fix: "large; Swift-side rollout store on existing engine callbacks, no Rust changes"; reword line 94.
- S8-2 Major. Lines 97-105: ZP-06 absent. Fix: add to step 5 beside ZP-08, or say it is deliberately unsequenced.
- S8-3 Major. Line 99: ZP-17 ordered before the surfaces it pins. Fix: move after step 5 or say only the recovery marker lands in step 3. One clause, do not split the draft.
- S8-4 Nit. Line 42: WebM vs MP4. Fix: say the container is open, leaning MP4.
- S8-5 Nit. Lines 87-88: ZP-05 vs ZP-20/ZP-18 order. Fix: one clause, ZP-05 is a later re-delivery path, not a prerequisite (bundled-skill root first).
- S8-6 Nit. Lines 63-66, 72-74: "remaining 12" count. Fix: one sentence reconciling to 40 (11 unique to ZP-90, 7.1 cross-listed, ZP-22 takes only the API-key half of 7.2, 2.1 outside the 40).
- S8-7 Nit. Line 135: stale branch. Fix: base commit SHA.

Recommended to drop or defer (scope-adding for an index)
- IC-5 (ModelStep seam step zero), IC-6 (split ZP-02), ZO-3 (ZP-02/ZP-23 circularity), IC-1/DC-8 (ZP-01 edge): belong in the ZP-22, ZP-23, ZP-02, ZP-01 drafts as open questions. At most reword "stands alone" for ZP-22 (line 89) to "mostly stands alone; writes to the ZP-03 rollout log if present".
- IC-7, ZO-8: optional; do not enumerate soft edges.
- IC simplification (replace sequencing list with a graph): reject.
- ST-3 to ST-8 (Size/Note column, dependency table, verdict table, program risks subsection, renamed Provenance, table re-sort): reject; structure imposed on an index that works. ST-11 acceptable but unnecessary.
- CO-4, CO-7 to CO-9, SY style nits: reject the glossary; at most one sentence defining ZCode and "kiro spec".
- ZO-10, ST-9 (.kiro links unresolved): worktree artifact, not a document defect.

New issues: none beyond the confirmed ZP-17 ordering.
