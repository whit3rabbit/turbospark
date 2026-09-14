# Swift auto-memory (persistent project memory)

Auto-memory in TurboSparkApp: a profile-wide `MEMORY.md` plus the existing
per-project directory on disk holding a `MEMORY.md` index and topic files the model itself writes through a
`memory` tool. The index is injected into the system prompt every turn, so
what was learned in an earlier conversation is visible in the next one. The
transcript is never consulted; the memory directory is the whole of it.
Default off. Existing files remain on disk when disabled.

Profile memory is ordinary Markdown at the active profile's user-scope
`memory/MEMORY.md` and is shared across that profile's projects. Project memory
remains partitioned by project root. The optional `.embeddings.json` sidecar is
disposable derived state. The native local encoder API can index an
Arctic-compatible MLX encoder; lexical recall remains available without one.

Read this before touching `MemoryStore`, `MemoryPromptBuilder`, the
`memory` tool, or quoting an index budget. Ports Claude Code's `memdir/`
(the `MEMORY.md` entrypoint plus frontmatter-tagged topic files) at the
scale this app runs at; the deliberate differences are listed at the bottom.

## The layout

Everything hangs off ONE profile-aware base, resolved through the same
`UserProfileStore.userScopeSubdirectory` seam as skills and plugins:

```
<base>/projects/<key>/memory/MEMORY.md     the index, injected every turn
<base>/projects/<key>/memory/<slug>.md     topic files, read on demand
```

`<base>` is `~/.turbospark/memory` for the Default profile and inside the
profile folder for anyone else. `<key>` (`MemoryStore.projectKey`) is the
project root's SYMLINK-RESOLVED path with every character outside
`[A-Za-z0-9._-]` replaced by `-`, plus 8 hex digits of SHA-256 of the
resolved path. The readable prefix is what a user browses in Finder; the
hash is what keeps `/a-b` and `/a.b` -- which sanitize identically -- from
silently sharing one directory. Resolving symlinks is what makes a linked
project and its target one project.

No project means no memory: the section is behind the same nil-project
guard as every other project-derived prompt section, and Chat mode has no
tools to write with anyway.

## The index and the budgets

`MEMORY.md` is an INDEX, not a memory. One row per topic file,
`- [slug](file.md) -- hook`; `MemoryStore.upsertingIndexLine` replaces a
file's row in place, so a save never duplicates a row and hand-written
lines survive a model save untouched.

The prompt injection truncates the index at 200 lines or 25,000 bytes
(Claude Code's `MAX_ENTRYPOINT_LINES` / `MAX_ENTRYPOINT_BYTES`) and the
truncation SAYS SO in the injected text. A silently shortened index reads
as complete memory; a self-describing one reads as a prompt to move detail
into topic files.

The index is memoized in `MemoryStore` against the file's mtime, so per-turn
prompt assembly re-reads nothing that did not change, and an edit made
outside the app is picked up on the next read.

## Topic files

A topic file is markdown with three frontmatter keys, reusing
`SkillParser`'s line handling:

```
---
name: deploy-workflow
description: one line that decides relevance later
type: feedback
---
```

The four types are Claude Code's taxonomy: `user` (who the user is),
`feedback` (guidance they gave, with why and how to apply), `project`
(goals, constraints, decisions the code does not record), `reference`
(pointers outside the workspace).

**The name is the containment boundary.** The tool takes a NAME, never a
path; `MemoryStore.slug` reduces whatever the model sent to a legal kebab
stem (`[a-z0-9-]`, 1..80), so no separator, traversal or case trick can
move the write outside the project's memory directory. The store re-checks
with `isValidTopicName` as defense in depth. This is why the feature needed
no `PathContainment` carve-out and no permission-engine change: the general
`write_file` tool never learns the memory path exists.

A description carrying newlines is folded to one line before it reaches
either the frontmatter or the index. This is not cosmetics: a newline in a
frontmatter value is a parse hazard, and an index line is a ROW -- an
unfolding description could smuggle arbitrary rows into the injected text.

## The memory tool

`memory` (`Tools/Memory/MemoryTool.swift`): `save` (name, type,
description, content; upserts file and index row), `read` (one file, or the
index when `name` is omitted), `forget` (removes both). A missing
description falls back to the body's first line cut at 120 chars, so the
index stays one informative line when the model skips the field.

The enable gate is enforced again by the executor, not only by prompt and
tool advertising. A parsed model-issued call therefore cannot read or mutate
persistent memory after the user disables the feature. The user-authored `#`
quick-save remains separate and writes directly through `MemoryStore`.

Availability follows the per-agent lists in `AppToolCatalog.tools(for:)`
(coder, researcher, general/custom; autonomous gets everything through
`allTools`). Both vocabulary lists carry it: it is in
`supportedToolNames` AND `workspaceRootedToolNames` -- memories key on a
project root, so a projectless chat refuses it by the standard message.

## Where the section is injected

`MemoryPromptBuilder.section(store:projectRoot:)` is the ONE builder, and
BOTH assemblers call it -- `AppModel.buildSystemPrompt` (project branch,
after Project Rules) and `SubagentRunner.buildSystemPrompt` (same branch).
The two assemblers share no code, so a section added to one and not the
other silently applies to half this app's runs; the same builder is the fix
for memory. The section is emitted even with an EMPTY index -- "your memory
directory exists and is empty" is the state a first conversation must be
told, or the model never saves anything and the feature never warms up.

The enable gate reads `MemoryStore.shared.isModelEnabled`, a plain var on
the shared store. `AppModel.memoryEnabled` mirrors it via `didSet`, because
the catalog and the subagent assembler are static surfaces with no
`AppModel` in hand -- the same reason `CommandGate.vetoEnabled` is static.

## The `#` quick-save and /memory

A composer draft starting with `#` plus text (`UserMemoryInputMessage`)
saves a `user`-type memory immediately and appends a transcript row
wrapping the text in `<user-memory-input>` tags -- Claude Code's
representation, rendered here as a "Saved to memory" note row instead of a
chat bubble. No generation turn runs.

**PLACEMENT IS LOAD-BEARING: both memory commands are intercepted ABOVE
`run()`'s `canRun` guard.** They never generate, so they work with no model
loaded; `/compact` stays below the guard because its summarizer needs the
session. `/memory` is dispatched off the table (`isMemoryCommand`), is the
second meta command in `BuiltInSlashCommand` (the drift test pins both the
count and the dispatch), and opens the project's memory directory in
Finder.

A failed quick-save KEEPS the draft: nothing was lost and the user can
retry. Ghost chats append the note through `mutateTurnMessages` (sealed in
the vault, never on the row) but still write the memory file -- ghost hides
the transcript, not the store.

## Testing

- `MemoryFeatureTests` covers the store algebra, slug coercion, truncation,
  both assemblers, the tool, the quick-save (normal and ghost) and the
  settings round-trip. Five mutations were checked to redden only their own
  case (index write dropped, date prefix dropped, line cap disabled, tool
  gate disabled, uppercase names accepted).
- The memory base REDIRECTS under a test runner (`AppStorageRoot.isRunningTests`
  in `defaultBase()`), because prompt ASSEMBLY creates the directory as a
  side effect and a test building a prompt for a project would otherwise
  mkdir inside the user's real `~/.turbospark`. Stores under test take the
  injected `base:` init anyway.
- `swift test` in `swift/TurboSparkApp`; one case with
  `--filter MemoryFeatureTests/testName`.

## Deliberate differences from Claude Code

- **No end-of-turn extraction subagent.** CC reviews each finished turn with
  a forked agent and saves what it finds. Here the model saves through the
  `memory` tool when it decides something is durable, guided by the prompt
  section. The app has the subagent machinery for the other shape if it is
  ever wanted (see DEVIATIONS.md).
- **No relevance side-query.** CC surfaces topic files with a cheap
  selector over the frontmatter manifest, capped per turn. Here the index
  is always in context and `memory read` / the file tools fetch detail.
- **Keyed on the resolved project root, not the canonical git root.** Two
  worktrees of one repository get separate memory directories here; CC
  shares one across all of a repo's worktrees.
- **`#` writes to auto-memory, not to CLAUDE.md.** CC's shortcut edits the
  project's instruction file with a destination picker; here it appends to
  the memory store, so it never modifies a file inside the user's
  repository unbidden.
- **No CLAUDE.md-family discovery upgrade.** `ProjectRuleDetector` still
  snapshots AGENTS.md/CLAUDE.md at project creation; CC's per-directory
  walk with `rules/` dirs and `@path` imports is a separate future feature.
