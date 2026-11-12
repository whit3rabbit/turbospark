# Consolidated review report: ZP-21 workspace tabs

Target: docs/designs/zcode-parity/ZP-21-workspace-tabs.md (full file, 191 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed. Two Stage 7 spawns (structural, style) were first refused with "Concurrent subagent limit reached" and succeeded on retry, run one at a time.
Source IDs: ZO = zoom_out_report (zoom-out), IC = zoom_out_report (inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Persona files read (all under /Users/whit3rabbit/.claude/plugins/cache/egm/egm/0.1.1/agents/): ZO zoom_out.md, IC inventive_catalyst.md, CO comprehension.md, DC design_critic.md, ST structural.md, SY style.md, S8 zoom_out.md.

Counts after deduplication: Blocker 3, Major 11, Nit 6.

## Blocker

B1. Lines 20-29, 90-113, 148-157. A new keyed per-tab store duplicates state that exists (`AppProject`, `selectedProjectID`, per-chat drafts); "workspace" is never mapped to a project; two sources of truth and leak risk. Fix: a tab is a persisted reference to an `AppProject.id` plus last chat and scroll anchor via existing selectChat; drop the per-tab container and list fields that stay global. [ZO-1, ZO-2, IC-1, CO-5, DC-12, SY-8, S8-B1, S8-N1]
B2. Lines 134-144 (missing section). No Risks & Mitigations section, required by the standard. Fix: add it (state leakage, memory, restore failure, ZP-23 dependence, cap interaction, close while generating). [DC-1, ST-2]
B3. Lines 115-132. Both alternatives "Rejected by author" with no linked record; the cheapest alternative (project switcher with per-project memory) is absent. Fix: concrete rationales, add Alternative C, phase as v0 if sufficient. [ZO-8, IC-2, DC-2, ST-1, ST-3, S8-B2]

## Major

M1. Lines 15-17, 33-35, 58-60, 94, 98. Settings tab is a Goal, a reserved schema field and an open question leaning the other way. Fix: drop from Goals and schema, list under Non-Goals. [ZO-3, IC-3, CO-2, DC-4, ST-7, SY-2, S8-M1]
M2. Lines 15, 36-38, 47-48, 105-108. "Per-tab availability" is undefined and absent from the model. Fix: delete it; persist open project ids, selected tab, last chat. [ZO-4, IC-7, CO-1, DC-14, ST-10, SY-3, S8-M2]
M3. Lines 26-29, 54-57, 70-71, 80-86, 185-187. Stale conditionals, resolved item under Open Questions, "before this draft", CUJ 3 titled "switch guard" with no guard. Fix: unconditional "Depends on ZP-23 and ZP-02", rename CUJ 3, define the guard as deferral to ZP-23. [ZO-5, IC-6, CO-3, CO-4, DC-5, ST-5, ST-6, ST-9, SY-1, SY-10, S8-M3]
M4. Lines 33-41, 80-86. Silent on the ZP-23 cap of 2 and global model-switch refusal when several tabs generate. Fix: state cap and model are global. [IC-5, DC-3, ZO-5, S8-M4]
M5. Lines 70-71, 84-86, 160-161. Tab badge reflects "active chat"; background approvals hidden; ZP-02 defines row badges only. Fix: aggregate ZP-02 states over all chats in the project by ZP-02 priority. [ZO-6, IC-8, DC-11, SY-4, S8-M5]
M6. Lines 62-87, 98-103, 134-139. Missing paths: close with running chat or draft, duplicate tab on one project, deleted or moved project on restore, relaunch while generating, restore write timing. Fix: one rule each (close defers to ZP-23 guard; prune on restore; reopen selects existing tab). [IC-4, CO-6, DC-6, DC-8, S8-M6]
M7. Lines 36-39, 134-137. "Lightweight state" unsupported; no tab cap or latency target; scroll restore undesigned. Fix: delete claim or set cap and target; scroll best effort per chat id. [ZO-7, IC-9, CO-9, DC-9, SY-6, S8-M7]
M8. Lines 36-38, 105-108, 136-139. Tabs archive is plain AppJSONStore; per-profile scoping, vault unlock ordering (finding 1.12), and which fields persist are unstated. Fix: reuse AppJSONStore, state per profile, persist no drafts. [DC-7, S8-M8, ZO-10]
M9. Lines 40-41, 110-113. "Without schema changes", "views unchanged in most surfaces" and "like every AppJSONStore file" are uncalibrated. Fix: drop or cite code. [DC-10, CO-7, SY-6, S8-M9]
M10. Lines 38-39, 64-71, 85. CUJ promises "restore exactly" including sidebar state; Goals enumerate only chat, scroll, draft. Fix: enumerate restored state consistently. [SY-7, CO-9]
M11. Lines 3-5, 50. ZCode link path differs from the comparison doc (`store/tabStore.ts`); ZP-90 reference for rekeying may dangle; "session restore" vs "pane session restore". Fix: use verified URL, confirm ZP-90. [ST-4, ST-8, SY-5, ZO-11, IC-10, DC-13, S8-N2]

## Nit

N1. Lines 158-164. Strip under `Generation/`; "above the sidebar" vs window-wide strip is unexplained (CO-8 rated it Major). Fix: state position, place with window chrome. [ZO-9, IC-11, CO-8, DC-15, ST-11, SY-12, S8-N3]
N2. Lines 169-187. Tests re-test AppJSONStore and ZP-23; mutation bullet unlabeled; no tests for missing-workspace restore, cap, strip or first-launch migration. Fix: trim, label, add. [ZO-10, ST-12, SY-14, DC-17, S8-N4]
N3. Lines 140-142, 166-168. No shortcut conflict (Cmd+W) or VoiceOver note. Fix: name bindings, add check. [IC-12, DC-16, S8-N5]
N4. Lines 17, 22-29, 45-46. Rhetoric ("largest navigation change", "mental model", "deliberately single-window", "matching ZCode"). Fix: concrete statements. [ST-13, SY-9, SY-13, S8-N6]
N5. Lines 45, 108, 141. Undefined jargon (AppModel, quarantine, token ratchet, ZP-02 states); "NSWindowTab" likely not the API name; overflow of many tabs. Fix: link or define. [CO-10, CO-11, CO-12, CO-13, SY-12]
N6. Lines 120-132. Alternatives padded or unsupported claims. Fix: state concrete cost. [SY-11]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-11, IC-1 to IC-12. Feeds B1, B3, M1-M7, M11, N1-N3.
- comprehension_report.md: CO-1 to CO-13. Feeds B1, M1-M3, M6, M7, M9, M10, N1, N5.
- critic_report.md: DC-1 to DC-17, ST-1 to ST-13, SY-1 to SY-14. Feeds all Blockers and Majors and N1-N6.
- intermediate_zoom_out_report.md: S8-B1, S8-B2, S8-M1 to S8-M9, S8-N1 to S8-N6. Confirms all earlier findings open; notes the minimal fix set removes scope (B1, B3, M1, M2) and the rest add one-line statements.
- Merge notes: ST-4 kept Major because a broken parity link affects verification; CO-8 kept in N1 despite its Major rating because the fix is a one-line clarification.
