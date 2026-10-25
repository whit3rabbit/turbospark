# Keyboard Shortcuts and Navigation Reference

TurboSparkApp provides complete keyboard-driven navigation, command menus, and VoiceOver accessibility integration across macOS.

## Global Menu Commands and Shortcuts

### Navigation and View (View Menu)

The app's items join the system View menu (there is one View menu, not two).
The destination rows are generated from `AppNavigationSection`, the same
source as the sidebar and the Keyboard Shortcuts pane, so names and chords
cannot drift between the three.

| Shortcut | Action | Description |
| :--- | :--- | :--- |
| `Cmd+1` | Chat | Activates the chat view and conversation pane. |
| `Cmd+2` | Image Generation | Opens the Images workspace. |
| `Cmd+3` | Files | Opens the Files workspace. |
| `Cmd+4` | Installed | Opens installed text and image models. |
| `Cmd+5` | Discover | Opens the model catalog. |
| `Cmd+6` | API | Opens the API workspace. |
| `Cmd+7` | Audio | Opens Audio Studio. |
| `Cmd+8` | Humanize | Opens the Humanize workspace. |
| `Ctrl+Cmd+S` | Toggle Sidebar | Shows or hides the panel beside the icon rail (the chat list in Chat, the page list in Audio). The rail stays. Disabled in sections without a panel. |
| `Cmd+Shift+I` | Toggle Inspector | Expands or collapses the right-hand model and runtime inspector. |
| `Cmd++` | Make Text Bigger | Increases in-app text readability and Dynamic Type scale. |
| `Cmd+-` | Make Text Smaller | Steps the text size down toward Small (Small, Default, Large, Extra Large). |
| `Cmd+0` | Default Text Size | Resets text size to Default and the base sizes to the factory 14px UI and 12px code. |

### Conversations (Chat Menu)

| Shortcut | Action | Description |
| :--- | :--- | :--- |
| `Cmd+N` | New Chat | Starts a new chat session in the currently active project. |
| `Cmd+[` | Previous Chat | Switches focus to the previous conversation in chronological history. |
| `Cmd+]` | Next Chat | Switches focus to the next conversation in chronological history. |
| `Cmd+L` | Focus Prompt | Moves first-responder focus directly into the prompt editor. |
| `Cmd+Shift+K` | Clear Chat History | Clears all turns from the active conversation transcript. |

### Model Execution and Generation (Generation Menu)

| Shortcut | Action | Description |
| :--- | :--- | :--- |
| `Cmd+Return` | Generate Response / Queue Message | Submits the chat prompt, or queues it while a turn is running. The label reads Queue Message only when the draft would actually queue. |
| `Cmd+.` or `Escape` | Cancel Generation | Interrupts active prefill or token decoding immediately. |
| `Cmd+Shift+.` | Stop All | Stops the turn, every running background agent and shell, and an in-flight model install. The server and the loaded model are untouched. |

Generation menu scope: Generate/Queue acts only while Chat is the active
section, because it sends the chat draft. Images, Audio and Humanize bind
`Cmd+Return` and `Cmd+.` to their own run and stop buttons, so the chat items
are disabled there (`AppNavigationSection.runsChatGenerationCommand` and
`ownsGenerationShortcuts`). Stop All stays global.

On the Voice Cloning page, `Cmd+Return` is the script composer's Generate
speech button and `Cmd+.` is the Audio status strip's Cancel. The progress
card's Cancel, the inline recorder's buttons, and the take rows carry no
chord, so each chord has exactly one owner on the page.

### Audio

| Shortcut | Action | Description |
| :--- | :--- | :--- |
| `Cmd+Shift+R` | Start Recording | Starts a take on the Record page with the chosen recording type and sources, or in the current Speech-to-Text mode with that mode's sources. Bound on the shared record bar's Start button (`AudioRecordBar`), not in a menu. |
| `Cmd+Return` | Transcribe | In Speech-to-Text's Any Audio mode, transcribes the selected file or capture. |

### Model Management (Model Menu)

| Shortcut | Action | Description |
| :--- | :--- | :--- |
| Menu | Choose Model Folder... | Opens an NSOpenPanel to select a local .gturbo model directory. |
| Menu | Load Chat Model | Memory-maps and warms up the selected chat model. |
| Menu | Reload Chat Model | Re-initializes runtime session and clears KV cache allocations. |
| Menu | Unload Chat Model | Drops current runtime session to reclaim physical memory. |
| Menu | Unload Image Model | Shown while an image model is resident; releases it. |

### Help Menu

| Shortcut | Action | Description |
| :--- | :--- | :--- |
| Menu | TurboSpark Help | Opens the project README. |
| Menu | Keyboard Shortcuts... | Opens Settings > Keyboard Shortcuts (also `Cmd+/` in View). |
| Menu | Report an Issue... | Opens a new GitHub issue. |

---

## Editor Shortcuts (Prompt Composer)

| Key Combination | Action |
| :--- | :--- |
| `Return` | Submits the prompt if input is present and model is ready. |
| `Shift+Return` | Inserts a newline character without submitting. |
| `Escape` | Clears focus from editor; cancels if generation is active. |

---

## Accessibility, Display Modes and VoiceOver

### VoiceOver Support (Cmd+F5)

1. **Live Announcements & Toasts**:
   - Token generation start announces "Generation started. Thinking." or "Generation started.".
   - Token generation completion posts non-blocking announcements via `AccessibilityNotification.Announcement` ("Generation finished. N tokens.").
   - Interactive user questions announce "Interactive question from model: [Question text]".
   - Model download, installation, loading, cancellation, and error events trigger spoken announcements and floating visual toasts.

2. **Transcript Rotor & Heading Navigation**:
   - The conversation transcript provides an accessibility rotor titled "Messages". Switch to the Messages rotor (`VO + U` or gesture) to skip between user prompts and assistant answers.
   - Audio Studio adds "Markers" and "Speakers" rotors on a recording, and a "Takes" rotor on the Voice Cloning page. The Voice Cloning steps strip is a row of buttons whose value is each step's state, with the first unfinished step marked selected.
   - User prompts and assistant responses carry `.accessibilityHeading(.h2)`, allowing rapid jump navigation via `VO + Cmd + H`.

3. **Custom Accessibility Actions (No Mouse Hover Required)**:
   - Message actions (previously requiring mouse pointer hover) are exposed directly to VoiceOver via `.accessibilityAction`:
     - Copy message
     - Read message out loud / Stop reading
     - Edit message (for user prompts)
     - Regenerate response (for assistant messages)
     - Branch conversation into a new chat
   - Chat sidebar rows provide custom actions: Pin chat, Rename chat, Duplicate chat, Delete chat.
   - Voice Cloning take rows provide a Reuse reference action.

4. **Custom Labels and Hints**:
   - Icon-only buttons supply explicit `.accessibilityLabel` and `.accessibilityHint` properties across all settings panes, model actions, and file pickers.
   - Status indicators (e.g., pulsing dots, progress spinners) combine into unified accessibility elements.

5. **Colorblind & Differentiate Without Color Support**:
   - Status dots, badges, and recommendation verdicts incorporate distinct glyph shapes (checkmarks, pause, minus, bolts, triangles, octagons) in addition to color.

6. **macOS Display & System Accessibility Integrations**:
   - **Increase Contrast (`colorSchemeContrast == .increased`)**: Theming automatically clamps contrast to at least 95%, elevates subtle borders to high-contrast visible strokes (opacity 0.75), and boosts secondary text opacity to 0.96.
   - **Reduce Transparency (`accessibilityReduceTransparency`)**: Disables frosted glass and translucent materials, replacing sidebar and card backgrounds with solid high-contrast surfaces.
   - **Dynamic Type & Larger Text (`dynamicTypeSize`)**: Font sizes in UI controls and code snippets scale proportionally with system-wide text size adjustments set in macOS Accessibility Settings.

## The model move sheet

"Move to..." in the Installed Models list and the Audio picker opens
`ModelMoveSheet`. `Return` presses the sheet's one primary button (Move, Retry
or Done) and `Escape` closes it, as in every other sheet. **While a move runs
the sheet cannot be dismissed**: Escape and the close gesture do nothing
(`interactiveDismissDisabled`, and no button carries the cancel shortcut),
because one stray key must not abandon a copy that has run for twenty minutes.

Cancel is a button. It asks the engine to stop, and the sheet reads "Cancelling..."
until the engine answers. The progress, the phase and the outcome are one
VoiceOver element each, and an outcome is announced.

## Alternate chords (unsloth studio compatibility)

Added 2026-09-07 (`AlternateShortcutBridge.swift`). unsloth studio's
shortcut scheme shares several actions with this app under different
chords: its new chat is Cmd-Shift-O where the Chat menu says Cmd-N, its
sidebar toggle is Cmd-B where View says Ctrl-Cmd-S, its chat cycling is
Cmd-Shift-[ and Cmd-Shift-] where ours is Cmd-[ and Cmd-], and its
workspace keys are Ctrl-1..9 where the rail is Cmd-1..8.

**The menus show one chord per action, so an alternate chord is a hidden
button, not a second menu item.** A SwiftUI menu item carries exactly one
shortcut, so "also bind Cmd-Shift-O" cannot be a second `.keyboardShortcut`
on the existing item -- the choice is a visible duplicate menu row or a
hidden button, and the hidden button does not clutter the menu. The
reconciliation is ADDITIVE: the menus keep the chords they have always
shown, and the unsloth chords fire as window-level hidden buttons mounted
by `RootView`, the same mechanism the chat search dialog uses
(`swift/docs/SWIFT_CHAT_SEARCH.md`), so they keep working while the prompt
editor is first responder.

Three things the next change can get wrong. Every bridge button's
`.disabled` must mirror the menu command it shadows (`isRunning` for new
chat, `isRunning || orderedChats.isEmpty` for the cycle pair), or the
alternate reaches an action the advertised key refuses. The rows above
carry the alternates in `KeyboardShortcutRow.altKeys`, and a catalog test
holds each pair: a bridge chord without a row, or a row without a bridge
chord, is this page lying again. And the pane's own entry point is real,
not an alternate: "Keyboard Shortcuts..." on the View menu, Cmd-/ (also
the unsloth shortcuts-tab chord), routed through
`openSettings(tab: .shortcuts)`.

