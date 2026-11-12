# Comprehension report (Stage 6): OC-05 back and forward history

Target: docs/designs/opencode2-parity/OC-05-back-forward-history.md (lines 1-132, read alone, no standard applied). Reviewer egm-comprehension completed.

## What this proposes

The app gets a back/forward history for section-level navigation. Each sidebar selection, settings pane open, chat selection or search jump pushes an entry (section, optional chat id, optional settings tab id) onto a stack capped at 100 entries, with the oldest dropped first. Cmd+[ and Cmd+] move back and forward through it, exposed as View menu items. Menu items disable at the ends of the stack with a tooltip saying why. Several settings pane visits in a row collapse into one entry, so one "back" leaves Settings entirely. History does not survive relaunch, has no pick-from-list UI, and ignores in-pane changes.

## How the existing system works, per this document

Section switching is done by directly assigning state in RootView.swift. There is no back-stack and no go-back handling anywhere in the app sources, so every jump is one-way. Scroll and composer state are said to be preserved by "existing per-section state". The document does not describe that state. A keyboard shortcut catalog and an "AlternateShortcutBridge" exist, but the document does not explain either one.

## Findings

CO-1 MAJOR. Lines 38-40, 80-83, 112-114. "AlternateShortcutBridge: no extra chords needed" is unexplained. The reader does not know what the bridge is or why it needs no work. The catalog is never defined: whether registering a chord binds the key or only documents it. Cmd+[ and Cmd+] may already be bound, and no conflict check is mentioned.

CO-2 MAJOR. Lines 41-42, 55-61 (CUJ 1). The collapse rule is under-specified. It is unclear which pane a collapsed run stores (first, last, latest tab id). CUJ 1 implies the last but the Goals do not say so. Unclear what happens when the user navigates back into the middle of a collapsed run or mixes chat ids.

CO-3 MAJOR. Lines 18-20, 63-68 (CUJ 2). Forward-history truncation is never stated: after going back and then making a new jump, is the forward stack discarded? Also not stated: whether re-selecting the current section pushes a duplicate.

CO-4 MAJOR. Lines 86-88. "Scroll and composer state are preserved by the existing per-section state" is unsupported. The doc does not say what that state is, where it lives, or why it survives back-navigation. The "section granularity only" scope relies on it.

CO-5 MAJOR. Lines 34-35, 42-43, 66-67. No behavior for stale targets: a deleted or archived chat id, or a settings tab that no longer exists. Skip, fall back, or error is undefined.

CO-6 NIT. Lines 3-9. "OpenCode" is never introduced beyond lines 20-22 and the origin links are external; the reader gets no summary of the finding.

CO-7 NIT. Lines 26-27, 50-51, 106-108. "Section" is the key concept but the sections are never listed; "section-level" versus "in-pane" (audio sub-surfaces) assumes knowledge of the app layout.

CO-8 NIT. Line 38. The "system View menu" placement is asserted without saying whether a View menu already exists or must be created.

CO-9 NIT. Lines 94, 119-120, 124-125. "21 languages", "the localization gates", `make design-check`, `make swift-lib` and "mutation check" are project jargon, unexplained.

CO-10 NIT. Lines 90-93. "Recording never blocks the transition" is a requirement with no mechanism; for an O(1) in-memory push it is trivially true.

CO-11 NIT. Lines 96-97. "Menu items announce enabled state" is vague beyond default platform behavior.
