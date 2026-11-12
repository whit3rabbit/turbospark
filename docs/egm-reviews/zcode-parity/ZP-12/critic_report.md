# Stage 7 critic report: ZP-12 treemap activity

Round 1 only (convergence not assessed). Reviewers egm-design (DC), egm-structural (ST), egm-style (SY) all completed. Standard: feature-spec-standard. Note: the design reviewer reported agents/design.md missing and used design_critic.md.

## Design (DC)

- DC-1 Blocker. Lines 88-102. Both rejections read "Rejected by author" with no link; the standard requires concrete trade-offs and no unlinked maintainer attribution. Missing alternatives: sunburst/icicle/indented tree, churn badges on the existing Files inspector, stored snapshot; no reason for custom squarify over reuse.
- DC-2 Major. Lines 31-32, 46-51, 80-82, 97-100. Data source unresolved and contradicts Goals. Live diff list is working-tree state: includes user edits and pre-existing dirt, excludes reverted or committed agent edits, cannot supply "viewed". Viewed/pending/written counts unsourced. Fix: close the Open Question, define input type and fields, cut viewed or source it from tool-call history; reconcile with "no new git invocations".
- DC-3 Major. Whole document. No Risks and Limits section: thousands of files and main-actor cost, deep trees, tiny rectangles, zero-area files (renames, binaries), drilled directory disappearing mid-update, paths outside workspace, vendored directories dominating.
- DC-4 Major. Lines 37-38, 60-61, 65. Summary counts undefined; CUJ 1 match claim incompatible with a viewed count; CUJ 2 empty state cannot distinguish tool-driven from manual changes; no success metric.
- DC-5 Major. Lines 106-109, 123, 129-135, 111. No state model for drill-down; Canvas has no per-rectangle accessibility; no VoiceOver plan; tests omit zero-weight, single-file, deep nesting, large N, removed drilled directory; mutation check oddly specified; builder has no file; "21 languages" vs INDEX "ten localization parity gates".
- DC-6 Nit. Lines 117-122. Files under Diagnostics/ while routing is in Generation/; ActivityTreeBuilder missing from file list.
- DC-7 Nit. Lines 16-17, 42-43. "Data the app already holds", ZCode per-task scope, "small" are uncited.
- DC-8 Nit. INDEX first-wave "self-contained" vs dependency on git-write-integration; state fallback and whether "current chat" binds to the selected chat (ZP-23).

## Structural (ST)

- ST-1 Major. Whole document (nearest lines 104-111). Missing Risks and Mitigations section. Fix: add H2 between Alternatives and NFRs.
- ST-2 Major. Lines 95, 102. "Rejected by author" attribution without link; only two alternatives. Fix: concrete trade-offs, add a third alternative.
- ST-3 Nit. Lines 46-51 vs 97-102, 31-32, 44. Open Question partly resolved elsewhere; Goals say "per-turn change records". Fix: mark decided and align wording.
- ST-4 Nit. Lines 53-70, 104, 137. Section order informational only, consistent with folder.
- ST-5 Nit. Lines 134-135. "Mutation checks" is a bare bullet outside the labelled fields. Fix: label or fold in.
- ST-6 Nit. Lines 3, 48. Relative link `../../ZCODE_COMPARISON.md` unverified; `git-write-integration` not linked though INDEX links it.

## Style (SY)

- SY-1 Major. Lines 24-25. "80 percent ... two directories" is a hypothetical stated as fact; "viewed but never written" conflicts with v1 source. Fix: mark as example or drop.
- SY-2 Major. Lines 15-19, 31-39, 48-51. Goals claim "per-turn" records while Open Question says they do not exist. Fix: state v1 source once.
- SY-3 Major. Lines 38-39, 61. Summary terms undefined (written vs modified, pending). Fix: one-line definition or source field each.
- SY-4 Major. Lines 48-51, 97-102. "Lean live diff list" hedge vs Alternative B rejection as decided; no owner. Fix: record decision or keep open; link git-write-integration path.
- SY-5 Major. Lines 95, 102. "Rejected by author" prefix redundant, "structure is the point" rhetorical, line 102 duplicates Cons. Fix: measurable reasons.
- SY-6 Nit. Lines 16-19, 99-100. "small", "cheap surface", "richer data" subjective. Fix: cite file count, say what is richer.
- SY-7 Nit. Line 85. "small deterministic layout function" filler.
- SY-8 Nit. Line 59. "largest rectangle drills into its files" only holds for a directory rectangle.
- SY-9 Nit. Lines 66-68. "without errors" vague; empty-state text unspecified.
- SY-10 Nit. Lines 106-110. "never color alone" fragment; focusability unclear; DESIGN.md unlinked.
- SY-11 Nit. Lines 117-134. Directory choice unexplained; WorktreeModel without path; mutation bullet outside pattern.
- SY-12 Nit. Lines 111, 140-143. "21 languages" vs "ten localization parity gates"; design-check scope unclear.
- SY-13 Nit. Lines 3-5, 26-27. ZCode data-source claim (task chat messages) needs a citation.
