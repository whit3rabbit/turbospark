# Consolidated review report: OC-02 notifications and sounds

Target: docs/designs/opencode2-parity/OC-02-notifications-and-sounds.md (full file, 202 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed (structural and style hit "Concurrent subagent limit reached" on first spawn and succeeded on one-at-a-time retry).
Source IDs: ZO and IC = zoom_out_report (zoom-out, inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.

Counts after deduplication: Blocker 2, Major 11, Nit 10.

## Blocker

B1. Whole document (no section). No Alternatives Considered section, mandatory under the standard; ZP-02 has one. Fix: add 3-5 bullets: content.sound vs NSSound, single master toggle vs four events, observe ZP-02 badge state vs imperative hooks, unfocused-only vs always post. [DC-1, ST-1, SY-16, S8 keep-list]
B2. Whole document (no section; partial at 141-143). No Risks and Mitigations section, mandatory. Fix: short list: flood across chats, stale dedupe slot, revoked authorization, click on deleted chat, double sound, lock-screen chat title. No rate caps. [ST-2, DC-10, S8 keep-list]

## Major

M1. Lines 131-135, 171-173, 56-59. AutomationExecutors "keeps its own use in v1" and "migrates in the same change"; the executor requests authorization at tool-call time (AutomationExecutors.swift line 51), conflicting with Allow-button-only. Fix: defer migration (additive change), or migrate with agent pushes bypassing the focus rule and never prompting; delete one sentence. [ZO-1, IC-3, CO-1, DC-3, ST-4, SY-1, SY-2, S8-M1]
M2. Lines 56-58 vs 104-111. Declined authorization: "all events off" vs "toggle stays on". Fix: keep toggle on with a status row; state once. [ZO-6, CO-4, DC-8, SY-3, S8-M2]
M3. Lines 60-62, 127-128. NSSound ignores Focus and Do Not Disturb, may double-play, plays when authorization declined; alert-volume claim unverified. Fix: set sound on notification content, drop NSSound and picker (or gate on successful post and name the owner). [ZO-5, IC-4, DC-9, SY-5, S8-M3]
M4. Lines 51-55, 91-95, 100-102. Dedupe has no priority and three release conditions; a finished or error event can replace a blocking permission cue. Fix: permission and answer not replaced by finished or error; one release rule; clear on app activation. [IC-2, CO-3, DC-5]
M5. Lines 48-50, 97-102. "NSApp not active" vs "no window focused" vs selected chat; background chat in an active app gets no cue. Fix: one-sentence rule: suppress only when app is active and the event's chat is selected. [ZO-7, IC-5, CO-7, DC-4, SY-11, S8-M7]
M6. Lines 26-29, 70-71, 79-84. Dependency on ZP-02, ZP-23 and desktop-shell-hardening has no interface; brief link does not resolve in this worktree. Fix: one or two sentences: policy observes ChatBadgeState transitions, fallback hooks if ZP-02 slips, v1 fires mainly when user switches away; fix or mark link unverified. [ZO-3, IC-1, DC-6, ST-7, SY-8, S8-M6]
M7. Lines 79-84, 164-170. Hook points unverified; cancelled turn behavior unstated; "Event source wiring" is a decision inside Open Questions. Fix: name the seam or mark unverified, state cancel does not fire turnFinished, move wiring to the architecture. [ZO-4, CO-2, CO-6, DC-7, ST-3, SY-12, SY-15, S8-M5]
M8. Lines 115-129, 116, 164. `AppAttentionEvents.notify` appears only in the diagram, not the file list; two types plus a facade is over-layered. Fix: one type with injected isActive and a small center protocol, or list and name the facade. [ZO-2, CO-11, ST-6, SY-13, S8-M4]
M9. Lines 54-55, 115-130. Click routing undefined: delegate must be set at app start, deleted chat. Fix: one line covering both; leave the rest as implementation detail. [DC-2, S8 keep-list]
M10. Lines 174-176. Toggles persisted "with the archive"; ZP-02 uses MacAppSettings; defaults unstated. Fix: use MacAppSettings, state default toggle states. [IC-6, DC-12, SY-7, ZO-9]
M11. Lines 33-38, 93, 140-142. "Nothing tells the user" overstated (automation already notifies); "chat title only" uncalibrated if titles derive from user text; CUJ 1 text lacks the tool name the privacy bullet promises. Fix: narrow claims; add an optional generic-text note as an open question. [SY-4, SY-6, IC-7, ZO-8]

## Nit

N1. Line 36. Executor link relative to doc folder does not resolve. Fix: ../../../swift/... [ZO-10, DC-13, ST-5, SY-17, S8-N1]
N2. Lines 139-140. "Under 200 ms" unmeasured. Fix: "never blocks the main actor" or drop. [ZO-8, CO-8, DC-11, IC-8, SY-14, S8-N1]
N3. Lines 76-77. "the app already has 18" lacks a noun; uncited. Fix: "18 settings panes" with citation. [CO-10, SY-12]
N4. Lines 38-42, 50. OpenCode claims unsourced; cross-window dedupe irrelevant to a single-window app; filler sentence at 41-42. Fix: cite coordinator.ts or cut. [DC-14, SY-9]
N5. Lines 43-46. Sound set unenumerated; defaults missing. Fix: name values (moot if sounds drop to on/off). [CO-12, SY-10]
N6. Lines 64-71, 77. No Non-Goal for scheduling or quiet hours; "scheduling" undefined. Fix: add Non-Goal. [DC-15, SY-12]
N7. Lines 75-78 vs 160-163. Open Question leans General, plan hard-commits; recorded assumed default. Fix: mark as assumed. [DC-16]
N8. Lines 59, 142-146, 198, 3, 6. Bare references (permission rules, announcements, DESIGN.md section 9, Sections 3 and 9a) and undefined "attention request". Fix: link or define. [CO-9, CO-13, CO-14, CO-15, ST-10, ST-12]
N9. Lines 164-167, 182-198. Four-path file item formatting; no tests for hook emission and persistence; plan items 8-9 low altitude; mock seam not listed. Fix: sub-bullets; trim or add one test line. [ST-9, ST-11, ZO-9, DC-16, S8-N2]
N10. Lines 79-84. Auto-resolve "one notification" lean depends on unresolved ZP-02 option. Fix: defer or record as blocking. [IC-8, DC-17]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-10, IC-1 to IC-8. Feeds M1-M8, M10, M11, N1-N3, N9, N10.
- comprehension_report.md: CO-1 to CO-15 (no Blockers). Feeds M1, M2, M4, M5, M7, M8, N2-N3, N5, N8.
- critic_report.md: DC-1 to DC-17, ST-1 to ST-13, SY-1 to SY-17. Feeds B1, B2 and all Majors, N1-N10.
- intermediate_zoom_out_report.md: S8-M1 to S8-M7, S8-N1, S8-N2. Confirms all Stage 5 findings open. Recommends keeping ambiguity-removal fixes and compact Alternatives and Risks sections; declining occlusion and Spaces definitions, identifier scheme, revocation refresh, persistence migration, acceptance criteria, sound enumeration, extra tests, rate caps, ZP-23 subagent coverage.
- Merge notes: the Alternatives and Risks gaps were rated Blocker by structural and Major by design and style; kept Blocker per the standard's failure rule. The Stage 8 advice to decline some asks is recorded but those items are still listed where they are real defects.
