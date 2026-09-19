# Large-Paste Handling

Display-side handling of large user input: big pastes in the composer, long
messages in the transcript, and context-window awareness. The underlying
content is never altered by any of this -- only what the VIEW renders and
whether a paste sits in the draft or in a chip.

## The three pieces

1. **Composer conversion** (`State/AppModel+Paste.swift`,
   `State/PromptPastePolicy.swift`): a paste over `attachmentThreshold`
   (4,000 characters) is lifted out of the draft into a `Pasted Text.txt`
   attachment chip. Below it, pastes stay inline as before.
2. **Transcript preview** (`Presentation/MessageContentPreview.swift`):
   message content over `collapseThreshold` (6,000 characters) renders a
   head+tail window while collapsed (user messages: 4,000 head + 500 tail;
   assistant messages: head only, matching what the old frame clamp showed).
   The same gate bounds the attachment preview pane and the split panes.
3. **Context awareness**: at conversion time the paste is priced at the
   app-wide 4-characters-per-token ratio against `resolvedContextTokens -
   maxNewTokens` (the same budget `fitConversationWindow` enforces at send
   time). Over budget: store the head that fits, set
   `wasTruncatedDuringExtraction` (the chip already shows "truncated"), and
   warn. No room at all: refuse the attach, leave the paste inline.
   Imported files warn (toast) when their estimate exceeds the free window;
   the 240k-character extractor cap stays the hard bound and the send-time
   fit check the backstop.

## Detection is post-hoc, not a pasteboard hook

`PromptComposerEditor` is a plain SwiftUI `TextEditor`; it offers no paste
interception. The alternatives were rejected on cost: an NSTextView
representable rewrite drags the autocomplete, the two-stage Escape, the
history walk, and the measured-sizer height pattern with it (Gotcha 18), and
a pasteboard monitor races the editor's own paste. So the policy watches the
draft's before/after values from `PromptComposerView`'s existing
`.onChange(of: model.promptText)` and recovers the EDIT as longest common
UTF-8 prefix + suffix. A paste is the only ordinary gesture that inserts
thousands of characters in one change.

Two details that are load-bearing:

- **The cheap gate is not just the delta.** Pasting OVER a long selection
  SHRINKS the draft while inserting thousands of characters, so the gate is
  `growth >= threshold OR previous.count >= threshold`. After a conversion
  the draft sits near the threshold, so the worst-case scan per keystroke is
  bounded and small.
- **The threshold counts CHARACTERS** (what the chip reports and the
  estimates price), so after the byte-level split the recovered middle
  re-checks `pasted.count >= threshold`.

## Suppression: programmatic draft writes

`writePromptTextDirectly(_:)` (`AppModel+Paste`) wraps every programmatic
`promptText` write: history recall, queued-message restore, autocomplete
acceptance (editor and view copies), plus-menu insert, `/goal`, skill
expansion, cron job dispatch, welcome suggestions. These can legitimately
insert thousands of characters, and converting one corrupts the thing it
belongs to -- a history recall would no longer match
`lastHistoryAppliedText`; a restored queued message would silently become an
attachment. Clearing writes (`promptText = ""`) need no wrapper: a deletion
can never pass the growth gate.

A new programmatic writer must use the wrapper, not the raw binding.

## Undo, mid-turn, ghost

- Cmd+Z after a conversion re-inserts the paste through the editor, which
  reads as a fresh paste; the dedup guard (a newest chip with identical
  text is kept, not stacked) makes that round trip a no-op.
- Mid-turn is legal: pasting while a turn runs converts through
  `appendPromptAttachmentDuringSubmission` (the guard-free tail
  `MentionResolver` uses), so the chip lands on the draft and the queue
  captures it.
- Ghost chats need no special case: attachments ride the row (in memory
  only; the archive filter excludes ghost rows), same as file attachments.

## Renderers that were touched

`CollapsibleMessageContentView` gained `gatedPreview` (string-level, nil
when expanded); its Show-more button shows
"View all (+N characters)" while gated. `SplitChatPanes` previews head-only
with the hidden-count marker inline (read-only pane, no expander; the full
text is one promote-click away). `FilePreviewView`'s extracted-text pane
gates behind a "View all" button. `MessageRowView`'s per-body
`UserMemoryInputMessage.parse` / `ShellMessageContent.parse` calls were
measured by inspection and left alone: both gate on a cheap `hasPrefix`
before scanning, so a huge ordinary message costs them nothing.

## Tests

`PromptPastePolicyTests` (split, fit, AppModel end to end including
suppression, dedup, truncation, and the no-room refusal) and
`MessageContentPreviewTests` (thresholds, head/tail exactness, multibyte).
All four mutation checks redden only their own case: the large-previous
gate clause, the fit boundary, the dedup guard, and the preview threshold.
Localization keys live in `Localizable.xcstrings` for every catalog
language (parity-gated); `make compile-strings` after editing the catalog.
