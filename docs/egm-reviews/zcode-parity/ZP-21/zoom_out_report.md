# Stage 5: Zoom out report (ZP-21 workspace tabs)

Target: docs/designs/zcode-parity/ZP-21-workspace-tabs.md (full file, 191 lines). Reviewers: egm-zoom-out (ZO), egm-inventive-catalyst (IC). Both completed.

## egm-zoom-out (ZO)

ZO-1 Blocker, lines 20-29, 33-36, 54-57, 110-113. "Workspace" is never mapped to the existing `AppProject` / `selectedProjectID`; the tab model carries `workspaceID` and risks a second identity. Fix: state a tab is a persisted reference to an existing `AppProject.id` and list which AppModel fields move per tab.
ZO-2 Major, lines 97-113, 152-157. The core refactor (selected chat, draft, scroll moved into a per-tab store) is sized as one extension file with no inventory of AppModel state; drafts already live on chats. Fix: audit what is already per-chat or per-project and shrink the store to what is missing.
ZO-3 Major, lines 31-41, 58-60. Reserved settings tab id is YAGNI and contradicts the "keep the window" lean. Fix: drop it from Goals and the `reservedSettings?` field.
ZO-4 Major, lines 36-38, 106-108, 136-139, 47-48. "Per-tab availability" is never defined. Fix: define in one sentence or delete.
ZO-5 Major, lines 54-57, 61-87, 185-187. Stale conditionals ("if adopted", "both Open Question answers") after ZP-23 was adopted; CUJ 3 is titled "switch guard" but describes no guard; ZP-23 Non-Goal (lines 63-64) says no concurrent chats inside one tab; behavior when running chats across tabs exceed the ZP-23 cap is silent. Fix: remove conditionals, move the resolved item out of Open Questions, add a cap sentence.
ZO-6 Major, lines 80-86, 160-161. Badge reflects "tab's active chat"; a background approval in a non-selected chat is invisible. Fix: aggregate ZP-02 states over all chats in the workspace.
ZO-7 Major, lines 134-137, 36-39. "Lightweight state" NFR is unsupported; scroll restore in lazy transcripts is undesigned. Fix: delete the clause, add one line on scroll anchors or defer.
ZO-8 Major, lines 118-125, 40-47. Alternatives omit improving the existing project switcher with per-project memory; rejections are "by author". Fix: add Alternative C and say why it is insufficient.
ZO-9 Nit, lines 158-164. Strip file under `Generation/`; "above the sidebar" vs window-wide strip is unclear. Fix: state position, place with window chrome.
ZO-10 Nit, lines 138-142, 169-176. Quarantine tests re-test AppJSONStore; "guard routing" test depends on ZP-23 guards. Fix: one line plus isolation test.
ZO-11 Nit, lines 3-5. ZCode link path differs from ZCODE_COMPARISON.md 1.6 (`store/tabStore.ts`); no size or phasing note. Fix: align link, add note.

## egm-inventive-catalyst (IC)

IC-1 Major, lines 88-113, 148-157. A keyed per-tab store is accepted as inevitable; two sources of truth risk (draft in chat and in tab slice). Resolution: tab is a thin saved view {workspaceID, lastSelectedChatID, scrollAnchor} applied through the existing selectChat path; small keyed struct for per-workspace UI state.
IC-2 Major, lines 21-30, 58-60. Need for tabs vs a lighter control is not shown. Resolution: add project-switcher-with-memory as Alternative C or phase it as v0.
IC-3 Major, lines 33-35, 58-60, 98. Settings-tab goal contradicts the Open Question; reserved id is speculative. Resolution: drop from v1 schema.
IC-4 Major, lines 82-86, 137, 187. Failure paths missing: closing a busy tab, relaunch while generating (CUJ 2 vs ZP-23 interrupted rows), restored tab whose workspace is gone. Resolution: close hands off to ZP-23 guard; restore prunes missing workspaces with a notice; add a "was generating" CUJ variant.
IC-5 Major, lines 40-41, 62-80. Model switching is global and refused while any chat generates (ZP-23); tabs imply per-workspace contexts, unaddressed. Resolution: state model is global, add refusal text and a CUJ.
IC-6 Major, lines 40-41, 54-57, 70, 84, 186 (ZP-23 lines 63-64). Dependency statement inconsistent with ZP-23 Non-Goal and stale conditionals. Resolution: state hard dependency on ZP-23 and ZP-02, align wording.
IC-7 Major, lines 15-17, 36-37, 99, 105-108. Availability undefined; tab-to-workspace cardinality and separate tab id unexplained. Resolution: define, or key by workspaceID if one-to-one.
IC-8 Major, lines 158-161, 84-85. Badge aggregation ambiguous (same as ZO-6). Resolution: aggregate over all chats with ZP-02 priority.
IC-9 Nit, lines 96-100, 136-138. Scroll is view state; background-tab retention unstated. Resolution: view-scoped storage keyed by chat id, state what is retained.
IC-10 Nit, lines 4-5. ZCode path differs from comparison doc. Resolution: re-check and align.
IC-11 Nit, lines 158, 162-163. Folder and mount position inconsistent. Resolution: App/ or Navigation folder, state position.
IC-12 Nit, lines 166-168. Shortcut conflicts unchecked. Resolution: name bindings, note catalog conflict check.

Persona files read: ZO read agents/zoom_out.md; IC read agents/inventive_catalyst.md (both under /Users/whit3rabbit/.claude/plugins/cache/egm/egm/0.1.1/).
