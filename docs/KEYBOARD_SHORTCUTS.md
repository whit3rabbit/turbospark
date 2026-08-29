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
   - Token generation completion posts non-blocking announcements via `AccessibilityNotification.Announcement` ("Generation finished. N tokens.").
   - Model download, installation, loading, cancellation, and error events trigger spoken announcements and floating visual toasts.

2. **Custom Labels and Hints**:
   - Icon-only buttons supply explicit `.accessibilityLabel` and `.accessibilityHint` properties.
   - Status indicators (e.g., pulsing dots, progress spinners) combine into unified accessibility elements.

3. **Colorblind & Differentiate Without Color Support**:
   - Status dots, badges, and recommendation verdicts incorporate distinct glyph shapes (checkmarks, pause, minus, bolts, triangles, octagons) in addition to color.

4. **Contrast Adaptation (WCAG 2.1 AA & High Contrast)**:
   - Adaptive brand accents meet WCAG 2.1 AA contrast requirements across standard Aqua and DarkAqua appearances (minimum 4.5:1 ratio).
   - Automatically adapts when macOS Increased Contrast / Accessibility High Contrast modes are active (scaling past 9.5:1 contrast).

