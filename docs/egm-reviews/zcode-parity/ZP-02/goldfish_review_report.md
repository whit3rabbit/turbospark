Status: resolved

# Goldfish Annotated Review Report

**Target:** `docs/designs/zcode-parity/ZP-02-unread-pending-badges.md` (full file, 202 lines)

Findings source: `consolidated_review_report.md` in this folder (Blocker 2, Major 11, Nit 11). Item IDs below are the consolidated IDs. Severities are unchanged.

## Summary

| Category | Count | Items |
| --- | --- | --- |
| CONTEST | 0 | none |
| ESCALATE | 1 | M10 |
| PROPOSE FIX | 20 | B1, B2, M1-M9, M11, N1, N2, N3, N5, N6, N7, N8, N10 |
| DECLINE | 3 | N4, N9, N11 |
| Total | 24 | |

### Verified-false reviewer claims

None. Every code claim I checked held:

* `rowIcon(isSelected:)` in `Generation/ChatSidebarChatRowView.swift` (line 202) shows the progress flame only when `isSelected && model.isRunning`.
* `AppModel.pendingToolCall` (`State/AppModel.swift:628`) and `pendingToolCallChatID` (line 635) exist.
* `AppModel.PendingUserQuestions` (`State/AppModel+Questions.swift:11`) carries `chatID`, and `pendingUserQuestions` is published at `State/AppModel.swift:677`.
* `AppChat` (`State/AppChat.swift:411`) has `updatedAt` (line 470) and no revision field. `ChatSidebarOrganizer.swift` has no revision reference.
* ZP-23 CUJ 1 (lines 81-87) shows a spinner during generation and the unread dot on completion.
* "21 languages" appears in `Tests/TurboSparkAppTests/LocalizationParityTests.swift` (lines 410, 423, 467). "Ten gates" is stated in the root `CLAUDE.md`.

Two partial points where the reviewers were imprecise, not wrong:

* M3 calls the ZP-23 ordering "hidden". `INDEX.md` lines 78-80 and 102-104 do state ZP-23 before ZP-02. The gap is that ZP-02 itself does not say what ships if ZP-23 slips.
* M10 treats `.kiro` as absent. `.kiro/specs/` exists in this checkout but holds only `audio-workspace`. `desktop-shell-hardening` is absent.

One false claim in the design doc itself: lines 150-151 say the revision token is "wired to the chat archive revision the organizer already reads". No such revision exists (see M1).

## ESCALATE first

* **M10**: the `desktop-shell-hardening` spec (linked from `INDEX.md` line 57 as `.kiro/specs/desktop-shell-hardening/brief.md`) is not in this checkout. The author or user must confirm whether the dock badge half exists, who owns the shared count, or whether to drop the goal.

## Prioritized Findings & AI Suggestions

### [B1] [`design / structural / style / zoom-out`] - Severity: Blocker
*   **Affected Scope:** [Lines 116-130](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L116-L130) (Alternatives Considered)
*   **Discrepancy Description:** Alternatives are "Rejected by author" with no trade-off.
*   **Rationale:** A reviewer cannot judge the choice. A and B are strawmen (do nothing, or a different feature).
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Both alternatives are non-competitors and the rejection reasons are assertions. The real design forks are visible in the doc itself (CAS, countdown, auto-resolve). The text below is drafted from the doc and the findings. Costs and rationale are inferred and need author confirmation.
*   ***Proposed Edit:***
    *   Location: replace the whole `## Alternatives Considered` section (lines 116-130).
    *   Change:
        ```markdown
        ## Alternatives Considered

        ### Alternative A: compare-and-set read with a revision token

        * **Pros**: safe with several writers racing on one chat.
        * **Cons**: chats have no revision field today (`AppChat` carries
          only `updatedAt`), so this adds a counter, a migration, and a
          token plumbed through every append path. The app has one writer
          per chat.
        * **Reason for Rejection**: cost exceeds benefit. A last-read
          watermark (see Technical Plan) gives the same re-raise behavior
          when content lands after the mark.

        ### Alternative B: countdown pill with snooze and auto-resolve

        * **Pros**: matches ZCode's TaskInteractionBadge.
        * **Cons**: needs a policy for resolving an approval or question
          with no user answer. That policy is a permission decision, not a
          badge decision.
        * **Reason for Rejection**: out of scope here. Ship a static pill;
          auto-resolve, if wanted, gets its own design.

        ### Alternative C: in-memory unread flags
        * **Pros**: no persistence or migration.
        * **Cons**: all unread state is lost on app relaunch.
        * **Reason for Rejection**: the cue is only useful if it survives
          relaunch.
        ```
        (Alternative B text and the "app has one writer per chat" claim are invented; author confirms.)

### [B2] [`design / zoom-out / comprehension / inventive-catalyst`] - Severity: Blocker
*   **Affected Scope:** [Lines 17-19, 41-42, 58-62, 85, 163-172](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L17-L19) (Overview, Goals, Open Questions, CUJ 2, files 4-5)
*   **Discrepancy Description:** Countdown, snooze and auto-resolve rest on an unresolved Open Question and default off. Auto-resolving an approval with an empty payload is undefined.
*   **Rationale:** The feature is a badge, but these parts change the approval path.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. This is a security-relevant finding (approvals, permission ladder). Open Question option (b) would resolve a pending approval or question with an empty answers payload, which a downstream tool or hook could read as consent or as a real answer. The conservative edit is to cut it, as the reviewers suggest: ship a static pill and leave pending items pending (deny/ask by default). Any auto-resolve work gets a separate design with its own security review.
*   ***Proposed Edit:***
    *   Location: Overview (line 17-19), Goals bullets 4-5 (lines 41-45), Open Questions first bullet (lines 58-62), CUJ 2 Then (lines 82-85), Technical Plan `MacAppSettings` line (107), Target Files items 4 and 5 (lines 163-172).
    *   Change:
        * Overview: replace "with a countdown fill and click-to-snooze" with "(static, no timer)".
        * Delete Goals bullets "The pending pill appears ... auto-resolve window" and "Click-to-snooze ...". Replace with: "* The pending pill appears while a chat has an open approval or open question, and disappears when it is answered or dismissed. It never answers or dismisses anything itself."
        * Delete the "Auto-resolve window" Open Question. Add to Non-Goals: "* No countdown, snooze, or auto-resolution of approvals or questions. Resolving a prompt without a user answer is a permission decision and needs its own design."
        * CUJ 2 Then: "B's row shows the pending pill. Click navigates to B. The pill clears when the prompt is answered or dismissed."
        * Delete the `MacAppSettings` line from the plan and delete file items 4 and 5 (`MacAppSettings.swift`, `AppModel+Questions.swift`).

### [M1] [`zoom-out / inventive-catalyst / comprehension / design`] - Severity: Major
*   **Affected Scope:** [Lines 38-40, 110-114, 146-155](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L38-L40) (Goals, Technical Plan, file 1)
*   **Discrepancy Description:** CAS has no real revision source, targets multi-writer clients, and may re-raise on a streaming selected chat.
*   **Rationale:** The mechanism is heavier than the single-process app needs.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified. `AppChat` has `updatedAt` but no revision field, and `ChatSidebarOrganizer.swift` has no revision reference, so lines 150-151 ("the chat archive revision the organizer already reads") are wrong. The doc's own line 113-114 admits only the selected chat generates today.
*   ***Proposed Edit:***
    *   Location: Goals bullet 2 (lines 38-40), Technical Plan paragraph (lines 110-114), file 1 changes (lines 148-151).
    *   Change: replace CAS with a watermark. Goals: "* Unread is persisted per chat as a last-read marker. A chat is unread when it has assistant content newer than the marker and is not selected." Technical Plan: "Selecting a chat, or content landing in the selected chat, advances the marker. Unread is derived: content past the marker while not selected. No revision token or compare step is needed." Delete the "revision token wired to the chat archive revision" clause.

### [M2] [`zoom-out / design / structural`] - Severity: Major
*   **Affected Scope:** [Lines 94-108, 142-186](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L94-L108) (Technical Plan, Target Files)
*   **Discrepancy Description:** New `ChatBadgeState.swift` mixes derived and persisted state. Storage layer, decode default for old chats, and file-list gaps are unstated.
*   **Rationale:** An implementer cannot tell where the persisted field lives or what an upgraded user sees.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. `AppChat` (`State/AppChat.swift:411`) is `Codable` and persisted in the chat archive, so one optional field there is the smallest persisted change. The slot priority is a pure function and fits in the organizer.
*   ***Proposed Edit:***
    *   Location: Target Files list (lines 146-185) and Technical Plan block (lines 100-108).
    *   Change: delete file 1 (`ChatBadgeState.swift`). Add a new file 1: "`State/AppChat.swift` (Modify): add optional `lastReadAt: Date?`. A missing value decodes to 'read' for chats that predate the field, so no chat lights up after upgrade. Ghost chats (`isGhost`) never persist it (they are filtered out of the archive)." Reword organizer item: "expose a pure `slot(for chat)` function with priority pending > unread > running > default." Rename the test file to `ChatSidebarOrganizerBadgeTests.swift` or keep the name; author picks.

### [M3] [`zoom-out / inventive-catalyst / comprehension / design / structural / style`] - Severity: Major
*   **Affected Scope:** [Lines 24-32, 44-45, 47-54, 83-85, 113-114](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L44-L54) (Goals, Non-Goals, CUJ 2, Technical Plan)
*   **Discrepancy Description:** Stale and hidden ZP-23 dependency. ZP-02 is inert before ZP-23. "Future concurrent world" wording is stale.
*   **Rationale:** A reader cannot tell what ships alone.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. `INDEX.md` lines 78-80 and 102-104 do order ZP-23 first, but ZP-02 hedges ("future", "future-proofing") while also saying ZP-23 is adopted. Line 113-114 is also true today only at chat-switch boundaries.
*   ***Proposed Edit:***
    *   Location: Non-Goals second bullet (lines 51-54); Goals bullet 5 (lines 44-45); Technical Plan last sentence (lines 113-114).
    *   Change: replace the Non-Goals dependency bullet with a `## Dependencies` note: "ZP-23 (concurrent chats) lands first, per INDEX.md. Before ZP-23, only the selected chat generates, so the unread dot and pending pill can appear only after a chat switch with a prompt open; the feature ships correct but mostly idle. If ZP-23 slips, ZP-02 may still ship in that reduced form." Delete "and any future concurrent-chats world (finding 2.1)" from Goals and "the semantics exist for the concurrent future" from line 114. (The "may still ship" sentence is an author decision.)

### [M4] [`zoom-out / inventive-catalyst / design / structural / comprehension / style`] - Severity: Major
*   **Affected Scope:** [Lines 36-38, 87-92, 100-105](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L36-L38) (Goals, CUJ 3, Technical Plan)
*   **Discrepancy Description:** Priority contradicts ZP-23 CUJ 1. The spinner is scoped to the selected row, and the error state is dropped silently.
*   **Rationale:** Under unread > running, a streaming background chat shows a dot, but ZP-23 CUJ 1 (lines 81-87) says it shows a spinner and the dot appears on completion.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified. `rowIcon` uses `isSelected && model.isRunning`, so a background chat can never show a spinner today. ZP-23 requires one. Defining unread at turn completion (see M5) removes the conflict without reordering the priority.
*   ***Proposed Edit:***
    *   Location: Goals bullet 1 (lines 36-37), CUJ 3 (lines 87-92), Non-Goals.
    *   Change: Goals: add "A chat is 'running' when it is generating, selected or not (per ZP-23). Unread is set only when a reply completes, so a streaming background chat shows the spinner and then the dot." CUJ 3 Given: "generation is running in any chat." Non-Goals: add "* No error badge. A failed turn keeps its current in-transcript treatment." (Whether to add an error state is an author decision.)

### [M5] [`zoom-out / inventive-catalyst / comprehension / design / style`] - Severity: Major
*   **Affected Scope:** [Lines 63-76](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L63-L76) (Open Questions, CUJ 1)
*   **Discrepancy Description:** Unread definition is left open while CUJ 1 picks the broad one.
*   **Rationale:** Tool-only churn would light rows with no reply.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. The Open Question itself recommends the first option ("matches chat conventions"), and CUJ 1 says "appends content", which is the second. Choosing completion while unselected is the minimal, consistent choice.
*   ***Proposed Edit:***
    *   Location: delete the "Unread definition" Open Question (lines 63-65); CUJ 1 When/Then (lines 72-76).
    *   Change: When: "a reply in chat B completes while B is not selected." Then: "B's row shows the unread dot. Selecting B advances B's last-read marker and clears the dot. If another reply completes after that, the dot returns. The dot never shows on the selected row." Add to Goals: "Unread means an assistant reply completed while the chat was unselected. Tool-only activity does not count." Author confirms the choice.

### [M6] [`codebase-details / structural / zoom-out / comprehension`] - Severity: Major
*   **Affected Scope:** [Lines 36, 103-104, 154-155](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L154-L155) (Goals, Technical Plan, file 2)
*   **Discrepancy Description:** Pending source is vague. The pill scope for approvals versus questions and for a selected chat is unclear.
*   **Rationale:** `pendingToolCall` and `pendingToolCallChatID` exist, and the question source is unnamed.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Verified. `AppModel.pendingToolCall` (line 628) and `pendingToolCallChatID` (line 635) exist, and `AppModel.PendingUserQuestions` has an optional `chatID` (`AppModel+Questions.swift:13`, published at `AppModel.swift:677`). Both are single slots, so at most one approval and one question are pending app-wide today. Note `chatID` is optional, so a question without a chat gets no pill.
*   ***Proposed Edit:***
    *   Location: file 2 (organizer) proposed changes, lines 154-155.
    *   Change: replace "observe pending tool and question state from the app model" with: "A chat has a pending prompt when `AppModel.pendingToolCallChatID == chat.id` and `pendingToolCall != nil`, or `pendingUserQuestions?.chatID == chat.id`. The pill shows for these on unselected rows only; on the selected row the in-transcript card is the cue. A question with a nil `chatID` shows no pill." (Selected-row scope is an author decision.)

### [M7] [`comprehension / design / style`] - Severity: Major
*   **Affected Scope:** [Lines 17, 41-42, 83-86, 160](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L41-L42) (Overview, Goals, CUJ 2, file 3)
*   **Discrepancy Description:** Click is both navigate and snooze. Snooze unit and persistence are undefined.
*   **Rationale:** One gesture, two meanings.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid, and fully resolved by the B2 edits, which remove snooze. No separate edit is needed beyond B2. Also drop "snooze action" from file 3 (line 160).
*   ***Proposed Edit:***
    *   Location: file 3 (`ChatSidebarChatRowView.swift`) changes, lines 159-162.
    *   Change: replace "pending pill (countdown fill, snooze action)" with "static pending pill"; delete "Countdown rendering follows Reduce Motion rules."

### [M8] [`design / structural`] - Severity: Major
*   **Affected Scope:** Whole document
*   **Discrepancy Description:** No Risks section.
*   **Rationale:** Rollout and upgrade hazards are unstated.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Zoom-out judged a full section scope growth, so the consolidated report suggests two bullets under Non-Functional Requirements. Do that, after B2 and M1 land.
*   ***Proposed Edit:***
    *   Location: add after the "Design compliance" bullet in Non-Functional Requirements (line 140).
    *   Change:
        ```markdown
        * **Risks**: (1) A chat saved before this change has no marker; it
          decodes as read, so nothing lights up after upgrade. (2) After a
          long offline gap or a profile restore, many rows can show unread
          at once; the dot is the only cue, so the sidebar stays calm.
        ```
        (Risk 2 is inferred from the findings; author confirms.)

### [M9] [`design / style / inventive-catalyst`] - Severity: Major
*   **Affected Scope:** [Lines 136-139](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L136-L139) (Performance)
*   **Discrepancy Description:** "No polling" conflicts with the countdown timer.
*   **Rationale:** A countdown fill needs a tick.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid, and removed by B2. After B2 the sentence is true. Replace "plus one persisted field" with "plus one persisted marker" for consistency with M1.
*   ***Proposed Edit:***
    *   Location: Performance bullet, line 138.
    *   Change: "badge state is derived from existing per-chat data plus one persisted last-read marker; there is no timer or polling, and the store writes only when the marker changes."

### [M10] [`structural / comprehension / zoom-out / design / style`] - Severity: Major
*   **Affected Scope:** [Lines 30, 44, 127](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L30) (The Problem, Goals, Alternative B)
*   **Discrepancy Description:** The dock badge spec is unlinked and not found under `.kiro/specs`. The owner of the shared count is unnamed.
*   **Rationale:** The doc depends on another spec it cannot point to.
*   ***AI Suggestion:*** ESCALATE
*   ***Justification:*** The critique is accurate as to the checkout: `.kiro/specs/` holds only `audio-workspace`, and `desktop-shell-hardening` is absent, so no reader of this repo can verify the dock badge claim. The evidence for or against lives outside readable files (the author's other branch or a missing spec). I cannot decide whether to link, name an owner, or drop the goal. `INDEX.md` line 57 already links `.kiro/specs/desktop-shell-hardening/brief.md`, so the link text is available once the spec is confirmed to exist.
*   ***Proposed Edit:***
    *   Location: The Problem, last sentence (lines 30-32); Goals bullet 5 (line 44); Alternative B (line 127).
    *   Change: none proposed. Needs the author's decision: link the brief (as in `INDEX.md` line 57) and name who owns the shared count, or delete the dock badge references from this doc.

### [M11] [`style / comprehension / design / zoom-out`] - Severity: Major
*   **Affected Scope:** [Lines 25-33, 134-136, 183-185, 195](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L25-L33) (The Problem, NFR, files, Verification)
*   **Discrepancy Description:** Claims about existing behavior, "21 languages" and "ten gates" lack file, symbol or source.
*   **Rationale:** Reader cannot check them.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** The claims are true (see the verified list above) but uncited, and a human reader needs the pointers. Cheap to add.
*   ***Proposed Edit:***
    *   Location: The Problem lines 26-28; NFR Accessibility (line 136); file 7 (lines 183-185).
    *   Change: Problem: "...the only row status cue is the progress flame in `ChatSidebarChatRowView.rowIcon`, shown when `isSelected && model.isRunning`." NFR: "strings are localized for all 21 offered languages (`LocalizationParityTests`)." File 7: "keeping the ten parity gates in `LocalizationParityTests` green (root `CLAUDE.md`)." Also state the source of "searching `unread` hits only MCP/background-shell plumbing" as a search over `swift/TurboSparkApp/Sources`, or delete that sentence. (My search found only unrelated "unreadable" hits, so the sentence is roughly right but its detail is unverified.)

### [N1] [`comprehension / zoom-out`] - Severity: Nit
*   **Affected Scope:** [Lines 134-136](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L134-L136) (Accessibility)
*   **Discrepancy Description:** Non-color cue is unnamed.
*   **Rationale:** "Never color alone" has no stated alternative.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Cheap and a human implementer needs it. Accessibility cue only; no security aspect.
*   ***Proposed Edit:***
    *   Location: Accessibility bullet, line 134-135.
    *   Change: "badge states are never color alone: the dot is a filled circle, the pill carries text, and each has an accessibility label that joins the combined row label."

### [N2] [`style / zoom-out / design`] - Severity: Nit
*   **Affected Scope:** [Lines 9-11, 17-21](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L9-L11) (header, Overview)
*   **Discrepancy Description:** "State machine" without transitions, "sized for" filler, Draft versus "adopted".
*   **Rationale:** Wording overstates.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid and cheap. The doc defines a priority slot, not a state machine.
*   ***Proposed Edit:***
    *   Location: Overview, lines 17-21.
    *   Change: "Give chat sidebar rows one priority slot: a pending-interaction pill, an unread dot, a running spinner, or the default icon. This is the TurboSpark port of ZCode's task-row badges, for concurrent background chats (ZP-23)." (The "adopted" wording is removed with M3.)

### [N3] [`style / comprehension / structural / zoom-out`] - Severity: Nit
*   **Affected Scope:** [Lines 121-130](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L121-L130) (Alternatives)
*   **Discrepancy Description:** Vague pros, and "across generation sessions".
*   **Rationale:** Imprecise wording.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Covered by the B1 rewrite, which uses "across app relaunches". No separate edit.
*   ***Proposed Edit:***
    *   Location: Alternatives section.
    *   Change: none beyond B1.

### [N4] [`style`] - Severity: Nit
*   **Affected Scope:** [Lines 83-92](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L83-L92) (CUJ 2-3)
*   **Discrepancy Description:** "Navigation no longer stops anything", "interval" versus "extension", "yields".
*   **Rationale:** Inconsistent terms.
*   ***AI Suggestion:*** DECLINE
*   ***Justification:*** Nit. Most of these phrases are deleted by B2 (snooze, interval, extension) and M3. The remainder ("yields to") is readable.
*   ***Proposed Edit:***
    *   Location: none.
    *   Change: none.

### [N5] [`structural`] - Severity: Nit
*   **Affected Scope:** [Line 94](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L94) (heading)
*   **Discrepancy Description:** Heading differs from the standard.
*   **Rationale:** The sibling set uses "Proposed Architecture".
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** One-line change. I did not re-check the sibling headings; the reviewer's claim is taken as stated.
*   ***Proposed Edit:***
    *   Location: line 94.
    *   Change: `## Technical Plan (Architecture)` becomes `## Proposed Architecture`.

### [N6] [`structural / comprehension`] - Severity: Nit
*   **Affected Scope:** [Lines 179-180](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L179-L180) (file 6, tests)
*   **Discrepancy Description:** Mutation checks sit outside the item schema and are unexplained.
*   **Rationale:** Reader does not know why they exist.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. Root `CLAUDE.md` asks that guards be mutation-checked. After M1 the CAS mutation no longer applies.
*   ***Proposed Edit:***
    *   Location: file 6 proposed changes, lines 179-180.
    *   Change: fold into the Proposed Changes bullet: "Mutation check (a new guard is only trusted if breaking it fails a test): invert the marker comparison and assert the re-raise test fails; invert slot priority and assert the priority test fails."

### [N7] [`design / zoom-out / implementation-readiness`] - Severity: Nit
*   **Affected Scope:** [Lines 173-180, 194](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L173-L180) (file 6, Verification)
*   **Discrepancy Description:** Tests omit decode default and re-raise on selected-and-streaming. `design-check-full` is unjustified.
*   **Rationale:** Tests should match the simplified design.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. The root `CLAUDE.md` says `design-check-full` adds contrast and ratchet tests, which does apply to new status-adjacent colors, so keep it but say why.
*   ***Proposed Edit:***
    *   Location: file 6 test list (lines 175-178) and Static Analysis bullet (lines 193-194).
    *   Change: tests: "slot priority; unread set on reply completion while unselected; no unread while selected; decode of a chat with no marker is read; persistence round-trip; pending pill for approval and question; spinner on a background running chat." Remove "snooze-once semantics" and "CAS markRead" items. Static analysis: "`make design-check` plus `make design-check-full`, because the dot and pill add status colors covered by the contrast tests."

### [N8] [`design / structural`] - Severity: Nit
*   **Affected Scope:** [Lines 47-54](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L47-L54) (Non-Goals)
*   **Discrepancy Description:** Non-Goals thin, and one bullet is a dependency.
*   **Rationale:** Scope boundaries are unclear.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid. The dependency bullet moves out under M3; the other items come from B2, M4 and M10.
*   ***Proposed Edit:***
    *   Location: Non-Goals.
    *   Change: list "* No error badge.", "* No sidebar reordering by badge state.", "* No countdown, snooze, or auto-resolution (B2).", "* No dock badge work here (M10, pending the author's answer)."

### [N9] [`comprehension / zoom-out`] - Severity: Nit
*   **Affected Scope:** [Lines 96-108](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L96-L108) (Technical Plan code block)
*   **Discrepancy Description:** Pseudo-code block mixes types and logic.
*   **Rationale:** Hard to read as either.
*   ***AI Suggestion:*** DECLINE
*   ***Justification:*** Nit. M1 and M2 already rewrite most of this block. Re-check after those edits; a further change is not needed.
*   ***Proposed Edit:***
    *   Location: none.
    *   Change: none.

### [N10] [`structural`] - Severity: Nit
*   **Affected Scope:** [Line 181](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L181) (file 7)
*   **Discrepancy Description:** Item 7 "Unchanged, must pass" is a gate, not a change.
*   **Rationale:** It does not belong in the file-change list.
*   ***AI Suggestion:*** PROPOSE FIX
*   ***Justification:*** Valid and cheap. The Verification Plan already has a Localization bullet (line 195).
*   ***Proposed Edit:***
    *   Location: delete file item 7 (lines 181-185). Extend the Localization bullet (line 195).
    *   Change: "* **Localization**: every new string enters the catalog for all 21 languages in the same change; all ten parity gates in `LocalizationParityTests` must stay green."

### [N11] [`structural`] - Severity: Nit
*   **Affected Scope:** [Line 94 and sibling set](../../../designs/zcode-parity/ZP-02-unread-pending-badges.md#L94)
*   **Discrepancy Description:** Heading pattern differs from ZP-23.
*   **Rationale:** Same as N5.
*   ***AI Suggestion:*** DECLINE
*   ***Justification:*** Duplicate of N5. The N5 edit already aligns the heading.
*   ***Proposed Edit:***
    *   Location: none.
    *   Change: none.

## Applied

* Applied (20): B1, B2, M1, M2, M3, M4, M5, M6, M7, M8, M9, M11, N1, N2, N3 (covered by B1, no separate edit), N5, N6, N7, N8, N10.
* Unapplied: M10 (ESCALATE, needs author decision; the doc only notes it as pending in Non-Goals), N4, N9, N11 (DECLINE).
* Invented content carries [confirm] markers in the design doc.
