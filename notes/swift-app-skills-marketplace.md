---
uuid: "a0863529-2ab6-4386-ab12-7465356d5bc8"
title: "TurboSparkApp: skills marketplace and discovery"
summary: "SKILL.md files at ~/.turbospark/skills or <project>/.turbospark/skills, precedence project-over-user by name. Discovers Claude Code, Cursor, and OpenClaw skills automatically, no migration"
tags: ["swift", "app", "skills"]
source: "docs/SWIFT_SKILLS.md"
created: "2026-09-05"
updated: "2026-09-05"
depends_on: ["7b3eeef2-d664-48a7-9b54-9b250768029e"]
---

## What is the Skills subsystem, and how is it different from SKILL.state?

Do not confuse this with SKILL.state (see [[swift-skill-state]]), a
different, unrelated feature despite the similar name. This is TurboSpark's
version of a slash-command / reusable-workflow system, the same shape as
Claude Code's skills: a `SKILL.md` file with YAML frontmatter plus Markdown
instructions, invocable by name or auto-suggested from context.

Two scopes: **User** (`~/.turbospark/skills/<name>/SKILL.md`, always
available, including standalone chats) and **Project**
(`<project>/.turbospark/skills/<name>/SKILL.md`, only when that project is
open). `SkillManager.resolveEffectiveSkills` merges both by lowercased
name, with project overriding user. It also discovers skills already
installed for OTHER harnesses (`~/.claude/skills/`, `~/.agents/skills/`,
`~/.cursor/skills/`, `~/.gemini/skills/`, `~/.config/opencode/skills/`,
`~/.codex/skills/`), so a skill installed for Claude Code or Cursor runs
here with no migration step.

Two execution modes when the model calls `skill(name, arguments)`:
**inline** (default: the skill's instructions, with `${arg}`/`${SKILL_DIR}`/
`${SESSION_ID}` substituted, are returned as a tool response the model
reads next) and **fork** (`context: fork` in frontmatter: runs in an
isolated `SubagentRunner` with its own token budget and only the
`allowed-tools` it declares, returning just the synthesized result).

## Don't

- Don't assume a project skill silently wins with no trace. When it shadows
  a same-named user skill, `SkillManager.shadowedUserSkillNames` flags it
  and the Settings UI badges the user skill as shadowed. Precedence without
  that disclosure would make the user skill look broken.
- Don't expect every installed skill's full instructions in the prompt from
  turn one. Only `name` and `when_to_use`/`description` are advertised, and
  the total is capped at 1% of the model's context window. Past that cap,
  non-bundled skill descriptions get truncated.
- Don't assume a skill with a `paths` glob in frontmatter is preloaded. It
  activates dynamically only once a file tool touches a matching path
  (`SkillManager.matchesPath`), for the rest of that conversation.
- Don't forget to call `invalidateResolutionCache()` after any disk change
  to a skill file or a scope toggle. `SkillManager` memoizes resolution
  behind an `NSLock`, and stale results read as "I edited the SKILL.md and
  nothing changed."
- Don't assume `context: fork`'s subagent result folds tool-call history
  back into the main conversation. Only the final synthesized text returns,
  by design, to avoid cluttering the primary history.
- Don't hand-author `installed_skills.json` entries. It tracks scope,
  version, and `gitCommitSha` per install for the marketplace flow, and a
  hand-edited row that disagrees with what's actually on disk is what a
  reinstall or update check would trust.
