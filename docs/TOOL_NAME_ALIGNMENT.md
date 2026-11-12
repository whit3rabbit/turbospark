# Tool name alignment

Local models are trained on the tool vocabularies of Claude Code, opencode and
the Pi coding agent. This page records how TurboSpark's model-facing tools
compare, what the app does about the differences, and what it leaves alone.

Sources, read on 2026-10-28: opencode v1.19.5 (`packages/opencode/src/tool`, the
released set), opencode `dev` (`packages/core/src/tool/plugin`, an unreleased
rewrite) and Pi (`coding-agent/src/core/tools`). The Claude Code column is from
memory and was not re-fetched.

## Comparison

| Concept | Claude Code | opencode 1.19.5 | opencode dev | Pi | TurboSpark |
|---|---|---|---|---|---|
| read | Read(file_path) | read(filePath) | read(path) | read(path) | FileRead(file_path, 8 mode params) |
| write | Write | write(filePath) | write(path) | write(path) | FileWrite |
| edit | Edit(old_string, new_string, replace_all) | edit(oldString, newString, replaceAll) | edit(path, ...) with fuzzy retry | edit(path, edits[{oldText,newText}]) | FileEdit(4 commands, 10 params) |
| shell | Bash | bash | shell | bash (timeout in seconds) | Bash (milliseconds) |
| search | Glob, Grep | glob, grep(include) | glob, grep(glob, literal) | find, grep, ls | Glob, Grep, grep_search |
| todo | TodoWrite | todowrite | todowrite | none | TodoWrite |
| subagent | Task / Agent | task | subagent | none | Agent |
| web | WebFetch, WebSearch | webfetch, websearch(numResults) | same | none | WebFetch, WebSearch(num_results) |
| patch | none | apply_patch(patchText) | patch | none | apply_patch |
| ask | AskUserQuestion | question | question | none | AskUserQuestion |
| skill | Skill | skill(name) | skill(id) | none | skill(name) |

opencode `dev` renames `bash` to `shell` and `task` to `subagent`. The released
v1.19.5 keeps `bash`, and its source says the id stays for compatibility until
opencode 2.0. The app does not follow `dev`. Both spellings are accepted as
aliases and neither is advertised.

## What was wrong

The executor already accepted every spelling above (`ToolArgumentResolver`,
`AppToolRegistry` alias sets). The gate in front of it did not.
`TurnAvailableTools.validate` matched the tool name and every required argument
exactly, so against an offered `FileRead(file_path)`:

- a call to `read`, `Read` or `read_file` was refused as "not offered";
- a call with `path` (Pi) or `filePath` (opencode) was refused as missing
  `file_path`;
- a `TodoWrite` call without `activeForm` was refused, although the decoder
  already defaults it to the item text.

A model that answered correctly in its own training vocabulary was told it was
wrong and had to guess again.

## What changed

- `Tools/Core/ToolCallNormalization.swift`: before validation, a call's tool
  name resolves to the offered definition (exact, then case-insensitive, then
  the same family: `read`/`read_file`/`FileRead`, `bash`/`shell`/`run_command`,
  `task`/`subagent`/`Agent`, and a few more), and its argument keys are renamed
  to the properties that definition declares (`filePath`/`path` to `file_path`,
  `oldString` to `old_string`, `background` to `run_in_background`, `include` to
  `glob`, `ignoreCase` to `-i`, `numResults` to `num_results`). The dispatch gate
  then carries the offered name, so the approval card, the permission engine and
  replayed history all see one spelling.
- Normalization only resolves to something the turn offered. A tool removed by a
  tool profile or an agent allowlist stays refused under every spelling, and a
  renamed argument never overwrites a value sent under the canonical name.
- `TodoWrite.activeForm` is optional in the schema.

## Deliberately not changed

- **Advertised names.** `FileRead`, `FileWrite` and `FileEdit` still differ from
  the `Read`, `Write` and `Edit` every reference uses. With normalization a
  model that says `Read` is understood, so a rename now only changes which words
  the prompt teaches. It changes every checkpoint's tool prompt, so it needs a
  real-model comparison first: same prompts, chat template applied, count valid
  first-try calls. That comparison has not been run.
- **Schema size.** FileRead and FileEdit advertise mode and command parameters
  no reference model was trained on. Trimming them would shrink the native-lane
  tool prompt, and would hide features from the Default profile. Measure the
  token cost on the target checkpoints before deciding.
- **Two grep tools.** `Grep` and `grep_search` overlap. They take different
  arguments, so neither resolves to the other. Pick one only after checking which
  built-in agents depend on `grep_search`.
- **Bash `workdir` and timeout units.** Pi counts timeout in seconds and the
  others in milliseconds. Guessing the unit is unsafe, and `workdir` interacts
  with the sandbox and permission engine.
- **Pi `edit` with `edits[]`.** Not accepted. `multiedit` covers the same need.
- **Prompt text.** opencode's usage notes for edit and read (keep the exact
  indentation after the line-number prefix, read before write, prefer edit over
  write) are worth porting once the read output format is confirmed to match.
