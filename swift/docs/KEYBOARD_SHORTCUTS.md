# Keyboard Shortcuts and Navigation Reference

TurboSparkApp provides complete keyboard-driven navigation, command menus, and VoiceOver accessibility integration across macOS.

## Global Menu Commands and Shortcuts

### Navigation and View (View Menu)

| Shortcut | Action | Description |
| :--- | :--- | :--- |
| `Cmd+1` | Switch to Chat | Activates the chat view and conversation pane. |
| `Cmd+2` | Switch to Model Hub | Opens the on-device Model Hub and catalog browser. |
| `Ctrl+Cmd+S` | Toggle Chat Sidebar | Expands or collapses the left-hand conversation sidebar. |
| `Cmd+Shift+I` | Toggle Inspector | Expands or collapses the right-hand model and runtime inspector. |
| `Cmd++` | Make Text Bigger | Increases in-app text readability and Dynamic Type scale. |
| `Cmd+-` | Make Text Smaller | Decreases in-app text size scale. |
| `Cmd+0` | Default Text Size | Resets text scaling to system standard. |

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
| `Cmd+Return` | Generate Response | Submits the current prompt and initiates token generation. |
| `Cmd+.` or `Escape` | Cancel Generation | Interrupts active prefill or token decoding immediately. |
| `Cmd+Shift+.` | Stop All | Stops the turn, every running background agent and shell, and an in-flight model install. The server and the loaded model are untouched. |

### Model Management (Model Menu)

| Shortcut | Action | Description |
| :--- | :--- | :--- |
| `Cmd+M` / Menu | Choose Model Folder | Opens an NSOpenPanel to select a local .gturbo model directory. |
| Menu | Load Model | Memory-maps and warms up the selected model. |
| Menu | Reload Model | Re-initializes runtime session and clears KV cache allocations. |
| Menu | Unload Model | Drops current runtime session to reclaim physical memory. |

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
   - User prompts and assistant responses carry `.accessibilityHeading(.h2)`, allowing rapid jump navigation via `VO + Cmd + H`.

3. **Custom Accessibility Actions (No Mouse Hover Required)**:
   - Message actions (previously requiring mouse pointer hover) are exposed directly to VoiceOver via `.accessibilityAction`:
     - Copy message
     - Read message out loud / Stop reading
     - Edit message (for user prompts)
     - Regenerate response (for assistant messages)
     - Branch conversation into a new chat
   - Chat sidebar rows provide custom actions: Pin chat, Rename chat, Duplicate chat, Delete chat.

4. **Custom Labels and Hints**:
   - Icon-only buttons supply explicit `.accessibilityLabel` and `.accessibilityHint` properties across all settings panes, model actions, and file pickers.
   - Status indicators (e.g., pulsing dots, progress spinners) combine into unified accessibility elements.

5. **Colorblind & Differentiate Without Color Support**:
   - Status dots, badges, and recommendation verdicts incorporate distinct glyph shapes (checkmarks, pause, minus, bolts, triangles, octagons) in addition to color.

6. **macOS Display & System Accessibility Integrations**:
   - **Increase Contrast (`colorSchemeContrast == .increased`)**: Theming automatically clamps contrast to at least 95%, elevates subtle borders to high-contrast visible strokes (opacity 0.75), and boosts secondary text opacity to 0.96.
   - **Reduce Transparency (`accessibilityReduceTransparency`)**: Disables frosted glass and translucent materials, replacing sidebar and card backgrounds with solid high-contrast surfaces.
   - **Dynamic Type & Larger Text (`dynamicTypeSize`)**: Font sizes in UI controls and code snippets scale proportionally with system-wide text size adjustments set in macOS Accessibility Settings.

## Alternate chords (unsloth studio compatibility)

Added 2026-09-07 (`AlternateShortcutBridge.swift`). unsloth studio's
shortcut scheme shares several actions with this app under different
chords: its new chat is Cmd-Shift-O where the Chat menu says Cmd-N, its
sidebar toggle is Cmd-B where View says Ctrl-Cmd-S, its chat cycling is
Cmd-Shift-[ and Cmd-Shift-] where ours is Cmd-[ and Cmd-], and its
workspace keys are Ctrl-1..9 where the rail is Cmd-1..5.

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

