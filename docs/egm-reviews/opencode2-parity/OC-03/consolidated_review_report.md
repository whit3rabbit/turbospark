# Consolidated review report: OC-03 inbox triage mode

Target: docs/designs/opencode2-parity/OC-03-inbox-triage-mode.md (full file, 159 lines)
Preset: Feature spec only (stages 5, 6, 7, 8). No remediation, target not edited. Stage 7 ran one round only, so convergence was not assessed. All reviewers completed; none failed (spawns hit "Concurrent subagent limit reached" and were retried successfully).
Source IDs: ZO and IC = zoom_out_report (zoom-out, inventive catalyst), CO = comprehension_report, DC/ST/SY = critic_report (design, structural, style), S8 = intermediate_zoom_out_report.
Caveat: the inventive catalyst did not read Section 9a of OPENCODE2_COMPARISON.md (tool cap), so claims about OpenCode's inbox were not checked against it.

Counts after deduplication: Blocker 3, Major 14, Nit 12.

## Blocker

B1. Lines 44-46 (also 36-38 of ZP-02, 145-146). "Same priority order as ZP-02's row slots" is false: ZP-02 ranks pending > unread > running, OC-03 ranks Needs you > Working > Finished; the swap mutation test locks in the wrong order. Fix: pick one order, justify any divergence, test a running chat with unread content. [ZO-1, IC-3, CO-1, DC-3, S8-M1]
B2. Whole file (missing section). No Alternatives Considered. Fix: add a short section with one or two rejected options, including pinned attention bands above the existing time buckets with no toggle. [DC-1, ST-1, IC-2]
B3. Whole file (missing section). No Risks and Mitigations. Fix: add 3 to 4 bullets (ghost chats, row jumping, state drift against ZP-02, delete from a new surface). [DC-2, ST-2]

## Major

M1. Lines 97-114, 129-131. ChatInboxModel plus a "small protocol" duplicates ZP-02 ChatBadgeState and ChatSidebarLayout. Fix: second grouping in ChatSidebarLayout over ChatBadgeState; drop the protocol and pre-ZP-02 shim. [ZO-2, IC-4, DC-4, S8-M2]
M2. Lines 17-27, 47-52, 61, 72; INDEX 76-78 vs 93-95. "Works on states that exist today" is false; only the selected chat can generate or park. Fix: make ZP-02 and ZP-23 prerequisites, drop the approximation, fix INDEX sequencing. [ZO-3, IC-5, CO-5, DC-5, DC-13, SY-3, S8-M3]
M3. Lines 43-46, 83-87. Idle read non-archived chats have no section; CUJ 1 says "normal sections or absent". Fix: state Inbox shows only attention chats or add an everything-else section. [ZO-4, IC-2, CO-4, DC-7, ST-9, SY-4, S8-M4]
M4. Lines 42-43, 61-62, 129-146. "Choice persists" contradicts "no new persistence"; no settings file listed. Fix: reword the non-goal and add MacAppSettings.swift, or always default to Sessions. [IC-1, CO-16, DC-12, ST-3, SY-10, S8-M5]
M5. Lines 49-52, 72-75, 137-138. Finished defined two ways; last-ended value is in memory and resets on relaunch. Fix: one definition, or remove it by depending on ZP-02. [CO-2, DC-6, SY-1]
M6. Lines 49, 52-55, 95-109. Archived ranks below Needs you so archived chats with pending work escape; existing Archived disclosure with Restore and Delete already exists. Fix: archived flag wins; reuse existing section; busy-chat archive and delete follow ZP-23 guards. [ZO-5, IC-6, CO-3, DC-11, S8-M6]
M7. Lines 31-39, 53-55. Search filter, project scope, pins, and ghost chats (title leak) undefined in Inbox mode. Fix: honor filter and scope, exclude ghost chats. [IC-8, DC-10, S8-M4]
M8. Lines 115-118. "O(n), identical to existing cost" ignores per-token body re-runs and per-chat vault decrypt check. Fix: recompute on state transitions only; drop "identical". [IC-7, DC-9, CO-14, SY-5]
M9. Lines 88-93. Row moves under the cursor when a Finished row is opened or an approval resolves. Fix: keep the selected row in place until deselected. [IC-2, DC-8]
M10. Lines 88-93. CUJ 2 has two "When" lines and a nondeterministic "Working or Finished". Fix: one When, Then Working if the turn resumes, Finished if it ends. [IC-15, CO-6, ST-5, SY-4]
M11. Lines 42, 98, 133-135. Toggle placement vs Chat/Projects tabs and filter bar unstated; "Sessions" vs "Chat". Fix: place it and scope Inbox to the Chat tab. [IC-12, ZO-6, DC-12, SY-15]
M12. Lines 47-54, 59-62. Error, interrupted, queued-idle states unmapped; "queued" Working row does not drain for an unselected chat. Fix: add a short membership table. [IC-6, CO-5]
M13. Lines 98-105, 130-132. ZP-02 protocol unnamed, fallback values unspecified, invalidation signals "same as row states" hand-waved. Fix: name observed properties (or remove per M1). [CO-7, CO-8]
M14. Lines 69-75. Open Questions carry Leans that Goals already assume; default mode unstated in Goals. Fix: state Sessions default, resolve or mark pending. [SY-1, SY-2, ZO-10]

## Nit

N1. Line 32 (also 3-9). Repo-root-relative link does not resolve. Fix: ../../../swift/... [ZO-7, IC-14, CO-13, DC-14, ST-4, S8-N2]
N2. Lines 129-147. File list omits ChatSidebarOrganizer.swift and ChatSidebarChatRowView.swift. Fix: list them. [ZO-8, IC-9, DC-16, S8-N3]
N3. Lines 56-57, 121-122. Bucket headers have no `.isHeader` trait today; header counts via Text concatenation are not localization-safe. Fix: say the trait is new; use a localized format string. [ZO-9, DC-15, CO-12, S8-N4]
N4. Lines 42, 107, 155. "Vocabulary pill picker", "fifth chrome band", "DESIGN.md section 9" undefined or unlinked. Fix: one DESIGN.md link. [CO-9, CO-11, ST-7, ST-8, ST-10, SY-14, SY-8]
N5. Lines 49-52, 56-57. In-section ordering unspecified; prev/next order vs inbox order. Fix: one line each; do not add mark-all-read. [IC-10, IC-11]
N6. Lines 6-9, 3-5. OpenCode links use mutable `dev` branch and are unverified. Fix: pin to commit 7ce35879b8, repeat the INDEX note. [SY-11, DC-18]
N7. Lines 125-146. No Depends-on or sequencing note. Fix: add one line. [ST-11]
N8. Lines 67-75. Open Questions placed before CUJs. Fix: optional; reviewer recommends rejecting. [ST-6]
N9. Lines 36-38, 111-114, 53-57. Wording: "glanceable triage pass", "hunt across rows", unnamed "existing patterns". Fix: plain phrasing. [SY-6, SY-7, SY-9]
N10. Lines 22-28, 132, 143-146. Protocol and "before and after ZP-02" repeated; mutation check phrased as instruction. Fix: state once. [SY-12, SY-13]
N11. Lines 20-24, 64-66. Counts sidebar-only not stated; VoiceOver announcement on row moves unspecified. Fix: one sentence each. [CO-15, IC-13]
N12. Whole file. No success metric. Fix: reviewer recommends rejecting as speculative. [DC-17]

## Traceability

- zoom_out_report.md: ZO-1 to ZO-10, IC-1 to IC-15. Feeds B1-B2, M1-M12, N1-N5, N11.
- comprehension_report.md: CO-1 to CO-16. Feeds B1, M2, M3, M4, M5, M6, M8, M10, M12, M13, N1, N3, N4, N11.
- critic_report.md: DC-1 to DC-18, ST-1 to ST-11, SY-1 to SY-15. Feeds all Blockers and most Majors and Nits.
- intermediate_zoom_out_report.md: S8-M1 to S8-M6, S8-N1 to S8-N7. Confirms all Stage 5 findings open. Recommends shrinking OC-03 to a thin second grouping over ZP-02 ChatBadgeState, accepting Alternatives and Risks only in short form, and rejecting DC-17, mark-all-read, the Open Questions reorder, and wording nits.
- Merge notes: DC-1/ST-1 into B2, DC-2/ST-2 into B3; zoom-out rated priority and missing-section items Major or absent, critic rated them Blocker, the higher severity was kept.
