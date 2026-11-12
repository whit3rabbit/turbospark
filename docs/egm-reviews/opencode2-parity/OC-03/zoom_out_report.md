# Stage 5: zoom-out and inventive catalyst report (OC-03)

Target: docs/designs/opencode2-parity/OC-03-inbox-triage-mode.md (full file, 159 lines). Both reviewers completed.

## Zoom-out reviewer (ZO)

Verdict: no Blockers, 4 Majors.

ZO-1 (Major, lines 44-46): "same priority order as ZP-02's row slots" is false. ZP-02 ranks pending > unread > running; OC-03 ranks Needs you > Working > Finished (running above unread). Fix: pick one order and state it once.
ZO-2 (Major, lines 97-114, 129-131): ChatInboxModel plus a "small protocol" duplicates ZP-02's ChatBadgeState and the existing ChatSidebarLayout grouping function (ChatSidebarOrganizer.swift). Fix: make the inbox a second grouping of ChatSidebarLayout over ChatBadgeState; drop the protocol; sequence after ZP-02.
ZO-3 (Major, lines 17-27, 47-48, 61, 72): "works on states that exist today" does not hold. Only the selected chat can generate today (sidebar gates on global model.isRunning), so Needs you and Working hold at most one chat; Finished needs new last-ended state, contradicting "no new persistence". Fix: make ZP-02 and ZP-23 hard prerequisites, delete the pre-ZP-02 approximation, correct INDEX sequencing.
ZO-4 (Major, lines 43-46, 83-87): sections do not partition the chat list; idle read chats have no section, CUJ 1 says "normal sections or absent". Search filter, project scope, pins, ghost chats in Inbox mode are undefined. Fix: add an "everything else" section or say Inbox shows only attention chats; define filter and scope behavior.
ZO-5 (Major or Nit, lines 52-55): an Archived disclosure with Restore and Delete already exists (ChatSidebarView.swift 380-431). Archived ranks below Needs you, so an archived chat with a stale approval escapes archive. Fix: reuse the existing archived section; archived flag wins over all states.
ZO-6 (Nit, line 42): header already holds Chat/Projects tabs, ghost button, filter bar; picker placement and Projects-mode behavior unstated. Fix: state placement; Inbox is Chat mode only.
ZO-7 (Nit, line 32): repo-root-relative link will not resolve from this folder. Fix: use ../../../swift/...
ZO-8 (Nit, lines 129-147): file list omits the real seam ChatSidebarOrganizer.swift. Fix: list it as the modify target.
ZO-9 (Nit, lines 56-57, 121-122): section header counts via Text concatenation are not localization-safe. Fix: specify a localized count format string.
ZO-10 (Nit, lines 69-75): default-mode question has only one sensible answer. Fix: resolve it.

## Inventive catalyst reviewer (IC)

Verdict: false compromises, no Blockers. Reviewer did not read Section 9a of OPENCODE2_COMPARISON.md (tool cap), so OpenCode claims were not checked.

IC-1 (Major, lines 42-43, 61-62, 129-146): "choice persists" contradicts "no new persistence"; no settings file in the target list; Finished approximation needs new state. Fix: drop the non-goal sentence and add the settings file, or drop persistence.
IC-2 (Major, lines 42, 69-71, 79-86): a mode toggle forces default-mode question, ships dark, leaves idle chats homeless, and the selected (viewed) row can vanish. Fix: consider attention bands pinned above the existing time buckets, no toggle.
IC-3 (Major, lines 44-45 vs ZP-02 36-37): priority claim false; same chat gets two answers. Fix: one order, or consume ZP-02's resolved slot.
IC-4 (Major, lines 99-113, 129-137): inbox re-derives state ZP-02, ZP-23 and OC-02 already own, with a throwaway pre-ZP-02 shim. Fix: ChatBadgeState as the single projection; sequence ZP-02 first.
IC-5 (Major, lines 18-26, 47-52; INDEX 76-78 vs 93-95): "works today" is nearly empty; INDEX puts OC-03 before ZP-23. Fix: resequence after ZP-23 and ZP-02, or label as automation-only.
IC-6 (Major, lines 47-54, 59-62): error, interrupted, queued-idle, archived-with-pending are not mapped; delete on a parked chat has no cancel semantics. Fix: add a membership matrix and decide archive/delete behavior on a parked turn.
IC-7 (Major, lines 115-118, 108-110): O(n) per invalidation ignores per-token body re-runs and the vault check per chat; "same signals as rows" differs from ZP-02's "store writes only on transitions". Fix: recompute on state transitions only, publish an Equatable snapshot.
IC-8 (Major, lines 31-39): filter bar, project scope, ghost chats (privacy), pins (non-goal says no pinning though pinning exists), duplicate Archived disclosure are ignored. Fix: add an interaction subsection; exclude ghost chats by default.
IC-9 (Nit, lines 97-109): parallel ChatInboxModel; ChatSidebarChatRowView not listed. Fix: extend ChatSidebarLayout or justify.
IC-10 (Nit, lines 56-57): prev/next chat navigation uses time order, disagrees with inbox order. Fix: state which order wins.
IC-11 (Nit, lines 49-52): in-section ordering unspecified; Finished has no clear action. Fix: add ordering line and mark-all-read.
IC-12 (Nit, lines 17-26, 42): "Sessions" vs existing Chat/Projects vocabulary. Fix: align name, scope to Chat tab.
IC-13 (Nit, lines 117-123): "no new surface" untrue if a toggle row is added; VoiceOver announcement on row moves unspecified. Fix: check DESIGN.md; add an announcement sentence.
IC-14 (Nit, line 32): broken repo-root link (same in OC-02). Fix: ../../../.
IC-15 (Nit, lines 88-93): CUJ 2 has two consecutive "When" lines. Fix: merge.
