# Personalities

Short, app-wide response styles for TurboSparkApp. Read this before changing
the personality model, Settings control, prompt assembly, or built-in text.

This feature is Swift-app only. The CLI and server do not read, store, or
apply personalities.

## Use

Open Settings, then Engine, then Personality. The picker starts at None, so a
fresh or upgraded app sends no personality text. Selecting a row applies its
instructions on the next turn. The selected prompt is shown below the picker.

Use Add Personality to supply a name and short instruction. Adding selects the
new row. Remove Selected removes any row, including a built-in; removing the
active row returns the setting to None.

The built-ins are deliberately compact. They are a starting library, not
protected presets:

| Name | Instructions |
| --- | --- |
| Formal | Be precise, professional, and structured. Use relevant domain terms. Do not critique spelling. |
| Friendly | Be warm, curious, and conversational. Match the user's tone. Be helpful without flattery. |
| Coach | Be direct and constructive. Give practical advice, correct mistakes plainly, and encourage progress. |
| Creative | Be playful and imaginative when appropriate. Use fresh language and light humor. Avoid cliches. |
| Concise | Be concise, clear, and complete. Skip small talk, filler, opinions, and unsolicited commentary. |
| Dry Humor | Be dry, witty, and helpful. Use gentle sarcasm for low-stakes topics; be kind on sensitive ones. |

Keep every built-in short. Its full text is sent on every turn while selected,
so extra prose takes usable context from local models.

## Storage and recovery

`MacAppSettings` stores the library in `personalities` and the selection as
`activePersonalityID`. The IDs of built-ins are fixed, so selection survives a
settings round-trip.

An older settings file with no personality fields receives the built-in library
and keeps None selected. An encoded empty library remains empty, since it means
the user removed every row. A selected ID that no longer exists resolves to
None rather than silently choosing another personality.

## Prompt behavior

`AppModel.buildSystemPromptSections` appends the selected instructions after
the per-chat or default system prompt and before any project-derived section.
The personality remains active when a chat replaces the app-wide default with
its own system prompt.

It is included in the normal system-prompt context slice, so the context ring
and the sent history price the same text. Isolated subagents receive the
app-wide default system prompt plus the selected personality, never a
conversation-specific prompt.

`SYSTEM_PROMPT.md` owns the complete section order and precedence contract.

## Code and tests

- `State/AppPersonality.swift` defines the stored row and built-ins.
- `Components/PersonalitySettingsSection.swift` renders the picker and editor.
- `State/AppModel+Tools.swift` resolves, selects, adds, removes, and injects a
  personality.
- `Tests/TurboSparkAppTests/PersonalityTests.swift` covers defaults, migration,
  round-trip persistence, stale selection, and system-prompt ordering.

When changing prompt assembly, also run the focused personality, system-prompt,
and context-usage tests. The personality assembly assertion is mutation-checked:
removing the resolved personality prompt must fail only that behavior test.
