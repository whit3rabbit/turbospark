# System prompt assembly

The Swift app builds one model-bound system message for a turn. Read this
before changing a prompt source, its order, a project rule, the context meter,
or subagent prompt construction.

This page covers TurboSparkApp. The standalone CLI and server accept their own
system-prompt inputs. The app's in-process server uses the selected app-wide
prompt as its `defaultSystem` but never applies a personality.

## App-wide prompt library

Engine Settings has a named prompt library. `TurboSpark Agent` is selected for
a fresh install, and `Compact Agent` and `Code Reviewer` are short alternatives
for local models. Select a row to load it, or add, edit, save, and delete any
row, including a built-in. Selecting None sends no app-wide default.

`MacAppSettings` stores `systemPrompts` and `activeSystemPromptID`. Its legacy
`defaultSystemPrompt` field remains a compatibility mirror for the selected
row, the in-process server, and copied server commands. A pre-library custom
default becomes an `Imported Default` row; a pre-library empty setting receives
the starter library and selects `TurboSpark Agent`.

The in-process server reads that mirror when it starts. Restart it after
loading, editing, or deleting its selected prompt.

## Global SOUL.md

Engine Settings also exposes a global `SOUL.md` section. TurboSpark resolves
`HERMES_HOME/SOUL.md`, falling back to `~/.hermes/SOUL.md`, before the
per-profile `soulPrompt` setting. An existing Hermes file wins even when it is
empty, while a missing file uses the native setting, which is blank by
default. Saving an active Hermes file updates it atomically. Import copies it
into the native fallback; Create Hermes writes only when the file is absent.
The section detects an existing OpenClaw workspace file at
`~/.openclaw/workspace/SOUL.md`, honoring `OPENCLAW_HOME`,
`OPENCLAW_STATE_DIR`, and `OPENCLAW_WORKSPACE_DIR`, and offers a separate
import action for each detected harness. Load File is always available for a
workspace or framework that uses a different location. All imports copy into
the native fallback and never overwrite the source file.

Nonblank global SOUL content is its own system-prompt section and is included
in `appWideSystemPrompt`, so isolated and background subagents inherit it.
The existing Personality library remains a separate section and storage path.

## Main-turn assembly

`AppModel.buildSystemPromptSections` is the only builder for the main turn.
`buildSystemPrompt` joins its nonempty sections with a blank line. The order is
part of the contract:

| Order | Section | Source | When included |
| --- | --- | --- | --- |
| 1 | User prompt | The chat's nonblank `systemPrompt`, otherwise the selected app-wide prompt | When nonempty |
| 2 | SOUL | Global Hermes `SOUL.md`, otherwise the native profile fallback | When nonblank |
| 3 | Personality | Selected app-wide `AppPersonality` | When selected and its instructions are nonempty |
| 4 | Agent role | Project agent type | With a project |
| 5 | Workspace | Project root path | With a nonempty project root |
| 6 | Environment | TurboSpark macOS and tool-use harness | With a project |
| 7 | Project rules | Live `AGENTS.md` or `CLAUDE.md`, `CONTEXT.md`, `SOUL.md`, and manual project guidance | With nonempty instructions |
| 8 | Memory | Project memory prompt | When memory is enabled and the project has a root |
| 9 | Tools and skills | Project-scoped tool catalog and enabled agents | With a project |
| 10 | MCP servers | Visible project and global MCP servers | When one or more servers are active |

A projectless Chat-mode turn can contain the user prompt plus app-wide identity
sections only. It is not
offered tools, skills, memory, workspace data, or project rules. Tool parsing
has the same independent project gate, so a model cannot execute a call it was
not offered.

The per-chat prompt replaces the selected app-wide prompt. A personality is
independent of that replacement and still applies. None means no personality
section.

## Project instruction boundary

Repository instructions are wrapped as untrusted project content. They provide
context and conventions, but core system instructions, tool-safety limits, and
user directions take precedence. `SOUL.md` at a project root is supplemental
project context, loaded after the selected rules and `CONTEXT.md`, inside the
same untrusted wrapper. The global SOUL section is app-wide identity and is not
part of that wrapper. `PERSONALITY.md` documents the optional style section
that precedes these project-derived sections.

The environment section is trusted app harness text, not repository content. It
states macOS, the listed-tool boundary, inspect-edit-verify behavior, the
system-reminder contract, and no Git-history mutation without an explicit user
request. `AppProject.turboSparkEnvironmentPrompt()` feeds both main and
subagent builders.

## History and context accounting

`buildAppendOnlyHistory` creates exactly one system message at index zero when
the joined content is nonempty. When every section is empty, it creates no
system message.

`buildEstimateParts` calls the same section builder. It uses the joined text
for the exact token measurement and groups user prompt, SOUL, personality, agent
role, workspace, environment, and project rules into the System prompt context
row. A selected personality therefore consumes visible, priced context rather
than becoming an uncounted addendum.

System reminders are different. They are computed at assembly time and appended
to the model-bound copy of the last user message. They are not persisted as a
system section. See `SWIFT_TURN_PIPELINE.md`.

## Isolated subagents

`SubagentRunner.buildSystemPrompt` is a second assembler because it runs
without an `AppModel`. The caller passes `appWideSystemPrompt`, which contains
the selected app-wide prompt, global SOUL, and selected personality but never a per-chat
prompt. The subagent then adds its role, workspace, shared environment block,
project context, memory, allowed tools, and allowed MCP tools in its own
isolated history.

Keep both assemblers aligned when adding an app-wide prompt source. Their tool
presentations intentionally differ: the main prompt carries the skills listing
inside its tool addendum, while the subagent prompt lists only tools it can
actually call.

## Tests

- `SystemPromptTests` pins library migration and selection, default versus
  per-chat resolution, first-message placement, projectless tool exclusion,
  the macOS harness, and subagent user-prompt ordering.
- `PersonalityTests` pins the optional personality section and persistence.
- `ContextUsageTests` pins that the exact history and context breakdown use the
  same assembled system prompt.

Run these together after changing prompt sources or their order:

```sh
cd swift/TurboSparkApp
swift test --filter 'PersonalityTests|SystemPromptTests|ContextUsageTests'
```
