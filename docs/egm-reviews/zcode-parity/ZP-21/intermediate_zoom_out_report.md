# Stage 8: Intermediate zoom-out report (ZP-21 workspace tabs)

Reviewer: egm-zoom-out on the full file (191 lines), given the Stage 5 to 7 reports. Completed. The target was not edited, so every earlier finding is still open. Code check by the reviewer: `AppProject` and `selectedProjectID` already exist (45 occurrences), and drafts already live on the chat and composer side.

Verdict: objective clear; fails root-problem focus and simplicity. About 45 earlier findings trace to five root issues. No proposed fix adds a component; the smallest fix set removes scope.

## Blockers

S8-B1 (lines 20-29, 97-113, 148-157). A new keyed per-tab store copying selected chat, draft and scroll off AppModel creates two sources of truth and leaks unmigrated properties. Fix: a tab is a persisted reference to an existing `AppProject.id` plus last selected chat id and scroll anchor, applied through the existing selectChat path; delete the per-tab state container and `AppModel+TabContext.swift`; list which AppModel fields stay global.
S8-B2 (lines 115-132). Both alternatives are layout variants "rejected by author"; the cheapest option (project switcher with per-project memory) is missing. Fix: add Alternative C and say what it fails to deliver; phase it as v0 if sufficient.

## Majors

S8-M1 (lines 33-35, 58-60, 94, 98). Reserved settings tab id contradicts the "keep the window" lean. Fix: delete from Goals, schema and Open Question; list under Non-Goals.
S8-M2 (lines 36-38, 105-108, 138-139, 47-48). "Per-tab availability" undefined. Fix: delete; persist open project ids, selected tab, last chat.
S8-M3 (lines 54-57, 70-71, 84-86, 185-187). Stale conditionals, resolved item under Open Questions, CUJ 3 mis-titled. Fix: remove conditionals, one-line "Depends on ZP-23 and ZP-02", rename CUJ 3, "before implementation".
S8-M4 (lines 33-41, 80-86). Silent on the ZP-23 cap of 2 and global model-switch refusal. Fix: two sentences: cap and model are global.
S8-M5 (lines 80-86, 160-161). Badge uses "active chat"; background approvals invisible. Fix: aggregate ZP-02 states over all chats in the project by ZP-02 priority.
S8-M6 (lines 62-87, 134-139). Missing failure paths. Fix: close hands off to the ZP-23 guard; restore prunes missing projects, falls back to one tab; re-opening a project selects its tab; relaunch shows ZP-23 interrupted rows.
S8-M7 (lines 134-137). "Lightweight state" unsupported; no tab cap. Fix: delete claim, set cap or say unmeasured; scroll best effort per chat id.
S8-M8 (lines 136-139). Quarantine framed as new; profile scoping and vault unlock ordering (1.12) missing. Fix: reuse AppJSONStore, say per profile, persist no drafts.
S8-M9 (lines 40-41). "Without schema changes" unjustified. Fix: drop it.

## Nits

S8-N1 (lines 15-17, 20-29, 98). Workspace, project, tab interchangeable. Fix: "workspace = AppProject".
S8-N2 (lines 3-5, 50). ZCode link path differs from comparison doc; ZP-90 cross-reference may dangle. Fix: align both.
S8-N3 (lines 158-164). Strip under `Generation/`; "above the sidebar" vs window-wide. Fix: state position, place in App/.
S8-N4 (lines 169-176). Tests re-test AppJSONStore and ZP-23. Fix: keep isolation and prune tests; label mutation checks.
S8-N5 (lines 140-142, 166-168). No shortcut conflict (Cmd+W) or VoiceOver note. Fix: name bindings, one-line check.
S8-N6 (lines 26-29). "Largest navigation change" is rhetoric. Fix: use INDEX size note.

Persona file read: /Users/whit3rabbit/.claude/plugins/cache/egm/egm/0.1.1/agents/zoom_out.md
