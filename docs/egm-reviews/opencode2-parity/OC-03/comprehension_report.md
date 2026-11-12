# Stage 6: comprehension report (OC-03)

Reviewer: egm-comprehension. Baseline read: INDEX.md, ZP-02, ZP-23. Completed.

## What this proposes

The draft adds a second way to group the chat sidebar. Instead of time buckets, an "Inbox" mode sorts chats into four sections by what they need from the user: Needs you (open approval or question), Working (generating or queued), Finished (ended and not yet viewed), and Archived. A toggle in the sidebar header switches between Sessions and Inbox and remembers the choice. Each chat shows in exactly one section, using the same priority order as ZP-02's row slots. A new derived model, ChatInboxModel, computes membership from existing chat state, and the sidebar view renders it with the existing row views, context menus and confirmation dialogs. It is meant to work before ZP-02 (unread flags) and ZP-23 (concurrent chats) land and to gain live rows after.

## How the existing system works, per this document

The sidebar today groups chats into time buckets in ChatSidebarView.swift. Only one chat can generate at a time. Approvals and questions appear as in-transcript cards, and a chat parked on one is "derivable today from parked turns". Chats have an existing archived flag and existing archive and delete paths with confirmation dialogs. ZP-02 and ZP-23 are separate, unbuilt drafts that will add persisted unread state, pending pills, and concurrent generation. The doc claims AppChat state and AppModel+Chat.swift may or may not already expose the needed flags.

## Findings

No section was unparaphrasable, so no Blockers.

CO-1 (Major, lines 45-47, 36-38, 145-146): "same priority order as ZP-02's row slots" is false (ZP-02: pending, unread, running; OC-03: Needs you, Working, Finished). The Working/Finished swap mutation check hides this. Fix: state OC-03's order as its own and justify, or adopt ZP-02's.
CO-2 (Major, lines 49-52, 72-75, 61-62, 136-139): Finished before ZP-02 is defined two ways ("ended since you last opened" vs "ended while open and unfocused"); focus tracking is not in the inputs; "last opened" implies stored state, against "no new persistence". Fix: pick one definition and say where the value lives and whether it survives relaunch.
CO-3 (Major, lines 49, 95-109): archived chat with an open approval or running turn: Archived or Needs you? Priority puts Archived last, hiding pending work. Fix: state which wins.
CO-4 (Major, lines 82-86): "normal sections or absent" is vague; idle read non-archived chats have no home in Inbox mode. Fix: define a fifth section, hide them, or Sessions-only, plus an empty-state rule.
CO-5 (Major, lines 21-26, 48-49, 69-71): Working is "generating or queued", but today only the selected chat generates and an unselected queued chat does not drain. Fix: state Working's pre-ZP-23 behavior.
CO-6 (Major, lines 86-93): CUJ 2 has two consecutive "When" lines and a nondeterministic "Working or Finished". Fix: one When (approve), Then Working if the turn resumes, Finished if it ends.
CO-7 (Nit-to-Major, lines 98-105, 130-132): the ZP-02 "small protocol" is not named or shaped; fallback values unknown. Fix: list members and defaults when ZP-02 is absent.
CO-8 (Major, lines 101, 105): "invalidates on the same signals the row states use" is hand-waving; ZP-02 row states are unbuilt. Fix: name the observed properties and how changes reach the model.
CO-9 (Nit, line 42): "vocabulary pill picker" undefined. Fix: link DESIGN.md or the defining file.
CO-10 (Nit, lines 45-46): "row slots" not explained. Fix: inline ZP-02's one-line order.
CO-11 (Nit, lines 97-109): AppChat, @MainActor, "fifth chrome band" undefined. Fix: short definitions or DESIGN.md link.
CO-12 (Nit, lines 56-58, 120-123): "existing patterns" and `.isHeader` assumed knowledge; counts mechanism stated two ways. Fix: name the pattern, cite DESIGN.md section 9, one mechanism.
CO-13 (Nit, lines 4-9, 33): relative links wrong from this folder (swift/... is repo-root relative). Fix: ../../../swift/...
CO-14 (Nit, line 118): "identical to the existing grouping cost" is unmeasured. Fix: soften or cite.
CO-15 (Nit, lines 20-24, 64-66): unstated whether Inbox counts show beyond the sidebar (dock badge is ZP-02/desktop-shell). Fix: state counts are sidebar-only.
CO-16 (Nit, lines 42, 61-62, 138): persisted mode toggle is new persistence, contradicting the non-goal. Fix: clarify the non-goal covers chat data and name where the toggle is stored.
