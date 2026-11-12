# ZP-07: Plugin creator

Finding 4.10 in [ZCODE_COMPARISON.md](../../ZCODE_COMPARISON.md).
Parity target: ZCode's
[plugin-creator doc](https://github.com/zai-org/ZCode/blob/main/docs/plugin-creator.md)
and its SKILL.md, a builtin plugin whose skill scaffolds plugins.

* Author: whit3rabbit
* Status: Draft
* Date: 2026-11-11

--------------------------------------------------------------------------------

## Overview

Ship an agent-facing plugin creator: a bundled skill that scaffolds a
plugin (`.claude-plugin/plugin.json` plus only the requested components),
validates structure and path containment, generates a dev marketplace.json,
and hands the user exact manual install steps. The plugins pane gains a
"Create Plugin" entry that opens a prefilled chat task using the skill.

## The Problem

TurboSpark's plugins pane can register a local folder but cannot create a
plugin. Its creator surface covers skills only (SkillEditorSheet). Users
who want to package skills + MCP servers into a distributable plugin must
hand-write manifests; the agent cannot be asked to scaffold one.

## Goals

* A bundled `plugin-creator` skill ships with the app and is discoverable
  by the SkillManager (as bundled content, per ZP-05 seeding or the
  bundled-skill root like ZCode's bundled-skills pack).
* The skill scaffolds a plugin at a user-chosen path with only the
  requested components (skills, commands, agents, MCP servers), writes a
  valid `.claude-plugin/plugin.json`, validates structure and path
  containment (no `..`, no absolute escapes), and writes a dev
  marketplace.json listing it.
* The plugins pane Import menu gains "Create Plugin", which opens a
  prefilled chat draft whose prompt instructs the agent to use the skill.
* The skill's final step prints exact manual add/install/update steps
  with the chosen paths.

## Non-Goals

* No GUI plugin editor; the skill plus file tools are the editor.
* No auto-install of the created plugin; the user installs it through
  the existing register-local-folder flow or the dev marketplace.json
  flow.

## Open Questions

* **Bundled skill delivery**: as a seeded builtin plugin (ZP-05) or as a
  bundled skill root entry (the ZCode bundled-skills pattern: unremovable,
  always discovered). Lean the bundled-skill root for day one because it
  avoids depending on ZP-05's ship date.

## Critical User Journeys

### CUJ 1: scaffold from chat

* **Given**: the user opens the plugins pane Import menu.
* **When**: they choose Create Plugin and confirm the target path and
  components in the prefilled draft.
* **Then**: the agent uses the plugin-creator skill, scaffolds the
  plugin, and the final message lists exact install steps; the created
  folder contains a valid manifest and only the requested components.

### CUJ 2: containment validation

* **Given**: the agent proposes a component path that escapes the plugin
  folder.
* **When**: the skill validates.
* **Then**: the skill refuses with a named error and asks for a
  corrected path; nothing is written outside the plugin folder.

## Technical Plan (Architecture)

```text
Bundled skill root (new, day one)
  Plugins/BuiltinSkills/plugin-creator/SKILL.md
     |
     v  discovered by
SkillManager (bundled root, unremovable, always offered)
     |
     v  activated by
Agent (chat) via Skill tool
  scaffolds plugin folder (manifest + requested components)
  validates structure + path containment
  writes dev marketplace.json
  prints manual add/install steps
     |
     v
Plugins pane Import menu "Create Plugin"
  opens prefilled chat draft (skill named in prompt)
```

## Alternatives Considered

### Alternative A: a settings-pane form wizard

* **Pros**: no agent involvement.
* **Cons**: component authoring (skills text, command files, agent
  files) is content work; a form cannot author content. ZCode's own
  creator is a skill, not a form.
* **Reason for Rejection**: Rejected by author: the agent is the
  content author; the UI only needs an entry point.

### Alternative B: extend SkillEditorSheet to full plugins

* **Pros**: existing UI.
* **Cons**: a plugin bundles multiple component kinds plus a manifest;
  the skills sheet is single-file oriented.
* **Reason for Rejection**: Rejected by author: wrong tool for the
  artifact's shape.

## Non-Functional Requirements

* **Security**: path containment validation is mandatory; the skill text
  includes the containment rules; nothing runs the created plugin's
  hooks without the existing SHA-256 plugin-hook trust approval.
* **Localization**: the Create Plugin menu string and the prefilled
  draft prompt are catalog strings (the prefilled draft prompt is a
  user-facing chat message; localize it).

## Detailed Implementation Plan

### Target Files to Create/Modify

1. `swift/TurboSparkApp/Sources/TurboSparkApp/Resources/bundled-skills/plugin-creator/SKILL.md` (Create)
   * **Type / Scope**: bundled skill content.
   * **Proposed Changes**: scaffold instructions, manifest template,
     validation checklist, dev marketplace.json template, manual steps
     template, containment rules.
2. `swift/TurboSparkApp/Sources/TurboSparkApp/State/SkillManager.swift` (Modify)
   * **Type / Scope**: discovery.
   * **Proposed Changes**: add the bundled skill root to discovery
     (unremovable, always offered, named in the listing as system
     content).
3. `swift/TurboSparkApp/Sources/TurboSparkApp/Components/PluginSettingsPaneView.swift` (Modify)
   * **Type / Scope**: UI entry.
   * **Proposed Changes**: "Create Plugin" item in the Import menu,
     opening a prefilled chat draft referencing the skill.
4. `swift/TurboSparkApp/Tests/TurboSparkAppTests/BundledSkillsTests.swift` (Create)
   * **Type / Scope**: unit tests.
   * **Proposed Changes**: the bundled root is discovered; the skill
     activates through the Skill tool; containment rules present in the
     skill text (string assertions, since the rules guide the agent).
5. `swift/docs/SWIFT_PLUGINS.md` (Modify)
   * **ZP-07 note: docs entry; content: creator entry point, bundled
     root, manual steps contract.

### Verification Plan

* **Compilation/Build**: `make swift-lib`, then build the app scheme.
* **Unit Tests**: `swift test` for the bundled-skills tests.
* **Localization**: catalog additions in 21 languages.
* **Manual**: create a plugin through the chat flow, register it via
  the local-folder flow, confirm contributions load.

--------------------------------------------------------------------------------

*Drafted with EGM*