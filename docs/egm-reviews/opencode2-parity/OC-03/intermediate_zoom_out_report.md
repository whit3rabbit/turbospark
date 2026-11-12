# Stage 8: intermediate zoom-out report (OC-03)

Reviewer completed. The target was not edited after Stage 5, so every Stage 5 finding remains open. No Blockers from zoom-out; 5 Majors (S8-M1 to S8-M5, and S8-M6 listed as Major) are confirmed.

## Open Majors

S8-M1 (lines 44-46): priority claim false (ZP-02 pending > unread > running). Open from ZO-1, DC-3. Fix: one order, stated once.
S8-M2 (lines 97-114, 129-131): ChatInboxModel plus a protocol duplicates ZP-02 ChatBadgeState and ChatSidebarLayout. Open from ZO-2, IC-4, DC-4. Fix: second grouping in ChatSidebarLayout over ChatBadgeState; drop the protocol and shim.
S8-M3 (lines 17-27, 47-52, 61, 72; INDEX 76-78 vs 93-95): "works today" unsupported; only the selected chat can generate or park. Open from ZO-3, IC-5, DC-5, DC-13, SY-3. Fix: ZP-02 and ZP-23 as hard prerequisites; delete the pre-ZP-02 approximation; fix INDEX sequencing.
S8-M4 (lines 43-46, 83-87): sections do not cover the list; filter, scope, pins, ghost chats, row leaving under the cursor undefined. Open from ZO-4, DC-7, DC-10, DC-8, IC-8. Fix: Inbox shows only attention chats or add an "everything else" section; define filter and scope; exclude ghost chats; keep the selected row in place.
S8-M5 (lines 42-43, 61-62, 129-146): persistence contradiction, no settings file listed. Open from IC-1, DC-12, ST-3. Fix: reword the non-goal and list MacAppSettings.swift, or always default to Sessions.
S8-M6 (lines 52-55): existing Archived disclosure; Archived ranks below Needs you. Open from ZO-5, DC-11. Fix: reuse it; archived flag wins; busy-chat rules via ZP-23 guards.

## Open Nits

S8-N1 (line 42): picker placement and Projects-mode behavior; "Sessions" vs "Chat". Fix: Chat tab only.
S8-N2 (line 32): broken link. Fix: ../../../swift/...
S8-N3 (lines 129-147): file list omits ChatSidebarOrganizer.swift and ChatSidebarChatRowView.swift.
S8-N4 (lines 56-57, 121-122): .isHeader trait and counts are new, not "existing"; use a localized count format string.
S8-N5 (lines 69-75): open questions have one sensible answer. Fix: resolve.
S8-N6 (lines 88-93): CUJ 2 malformed.
S8-N7 (lines 115-118): "identical cost" claim; recompute on transitions only.

## Critic asks: accept or reject (reviewer recommendation)

Accept: DC-3, DC-4, DC-5, DC-6, DC-7, DC-8, DC-10, DC-11, DC-12, DC-13, SY-1, SY-2, ST-11 (Depends-on line), SY-11 and DC-18 (pin OpenCode links, mark unverified).
Accept in short form only: DC-1/ST-1 Alternatives (one or two options; include pinned bands above time buckets, no toggle, from IC-2); DC-2/ST-2 Risks (3 to 4 bullets: ghost chats, row jumping, state drift against ZP-02).
Reject as scope creep: DC-17 success metric; IC-11 mark-all-read (keep a one-line in-section ordering rule); IC-10 prev/next order (one line at most); ST-6 Open Questions reorder; SY-6, SY-7, SY-8, SY-9, SY-12, SY-13 wording nits. VoiceOver on row moves: one sentence only. DESIGN.md links: fold into one link.

Recommendation: shrink rather than grow. Make OC-03 a thin second grouping over ZP-02 ChatBadgeState inside ChatSidebarLayout, sequenced after ZP-02 and ZP-23.
