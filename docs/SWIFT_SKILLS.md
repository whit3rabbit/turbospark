# Swift skills: architecture, scopes, file structure, and marketplace

This document describes the design and specification of the skills subsystem in
`swift/TurboSparkApp`. It covers scope management (user vs project), file
structure, discovery and precedence, remote fetching via Git and HTTPS,
marketplace manifests, prompt budgeting, and chat integration.

It aligns with the upstream agent interoperability contracts documented in
`https://github.com/whit3rabbit/agent-config/blob/main/docs/support-matrix.md`
and the Claude Code and Codex skill conventions. Keep all code, comments, and
docs ASCII: no emojis and no em dashes (project rule).

All Swift paths on this page are relative to
`swift/TurboSparkApp/Sources/TurboSparkApp/` unless qualified.

---

## 1. System architecture

The skills subsystem is organized into five functional layers:

```
+-------------------------------------------------------------------------+
|                        TurboSpark Skills System                         |
+------------------------------------+------------------------------------+
|             User Scope             |           Project Scope            |
|   ~/.turbospark/skills/<name>/     |   <project>/.turbospark/skills/    |
|   (plus ~/.claude, ~/.agents)      |   (plus .claude, .agents)          |
|   * Available in standalone chats  |   * Available when project is open |
|   * Cross-project defaults         |   * Precedence over matching user  |
+------------------------------------+------------------------------------+
|                           SkillManager Core                             |
|   * Thread-safe memoized resolution cache (NSLock)                      |
|   * Precedence merge: project overrides user by normalized name        |
|   * Shadowed skill tracking (disclosure when user skill is replaced)    |
|   * Variable substitution: ${arg}, ${SKILL_DIR}, ${SESSION_ID}          |
|   * Path containment security: sandbox boundaries enforced              |
+-------------------------------------------------------------------------+
|                      Marketplace & Remote Fetching                      |
|   * Sources: github, git (HTTPS/SSH), url (HTTPS JSON), local file/dir  |
|   * Shallow clone (--depth 1) + git sparse-checkout for monorepos       |
|   * Target scope selector: Install to User Scope or Project Scope       |
|   * Multi-scope install tracking and cache storage                      |
+-------------------------------------------------------------------------+
|                        Execution & Prompt Budget                        |
|   * Skill catalog character budget (1% of model context window)         |
|   * Conditional skill activation: paths frontmatter matching file edits |
|   * Inline expansion vs Subagent fork (isolated token budget)           |
+-------------------------------------------------------------------------+
|                            Chat Integration                             |
|   * Slash and @ autocomplete in the composer (type "/" or "@")          |
|   * Direct tool invocation via AppToolRegistry ("skill")                |
|   * Session capture ("skillify" / propose_skills)                       |
+-------------------------------------------------------------------------+
```

---

## 2. Scopes and file structure

### A. User scope vs Project scope

Skills are partitioned into two user-facing scopes:

1. **User Scope (`SkillScope.userGlobal`)**:
   - Location: `~/.turbospark/skills/<name>/SKILL.md` (canonical).
   - Compatibility roots: also discovers existing skills from other installed
     agents defined in the agent-config matrix:
     `~/.claude/skills/`, `~/.agents/skills/`, `~/.cursor/skills/`,
     `~/.gemini/skills/`, `~/.config/opencode/skills/`, `~/.codex/skills/`.
   - Availability: always visible and executable in any chat session, including
     standalone (projectless) chats.
   - Purpose: general-purpose workflows that follow the developer across all
     repositories (e.g. general commit formatting, summarizing documents).

2. **Project Scope (`SkillScope.projectLocal`)**:
   - Location: `<project_root>/.turbospark/skills/<name>/SKILL.md` (canonical).
   - Compatibility roots: `<project_root>/.claude/skills/`,
     `<project_root>/.agents/skills/`, `<project_root>/.cursor/skills/`.
   - Availability: visible and executable only when the respective project is
     selected as the active project.
   - Precedence: when a project skill shares a name with a user skill (compared
     case-insensitively), the project skill overrides the user skill.
   - Purpose: repository-specific workflows, deployment scripts, CI/CD routines,
     and project coding guidelines.

3. **Bundled Scope (`SkillScope.bundled`)**:
   - Read-only skills packaged with the application binary (e.g. system
     diagnostic skills, initial starter templates).

### B. On-disk directory and file layout

Skills must adhere to the standard directory-based layout:

```
~/.turbospark/
|-- settings.json                      # User preferences, enabled marketplaces
|-- marketplaces/                      # Cached marketplace repositories
|   `-- community-skills/
|       `-- marketplace.json
|-- plugins/
|   `-- installed_skills.json          # Scope, version, commit SHA tracking
`-- skills/
    |-- commit-helper/
    |   |-- SKILL.md                   # Main entry point (YAML + Markdown)
    |   |-- commit_rules.json          # Optional reference file
    |   `-- scripts/
    |       `-- check_format.sh        # Executable helper script
    `-- pr-review/
        `-- SKILL.md
```

For project-local skills:

```
<project_root>/
|-- .turbospark/
|   `-- skills/
|       `-- deploy-staging/
|           |-- SKILL.md
|           `-- config.env
`-- src/
    `-- ...
```

Single-file `.md` skills (e.g. `~/.turbospark/skills/simple.md`) are supported
as fallback on read, but all new creations and imports standardize on
`<name>/SKILL.md`.

### C. SKILL.md schema and frontmatter

Every `SKILL.md` starts with a YAML frontmatter block between `---` markers,
followed by Markdown instructions.

```yaml
---
name: deploy-staging
description: Deploys current git branch to the staging Kubernetes cluster
when_to_use: Use when the user asks to deploy, push to staging, or test cluster rollout
allowed-tools:
  - Terminal(kubectl:*)
  - Terminal(helm:*)
  - FileRead
arguments:
  - name: namespace
    description: Target kubernetes namespace
    placeholder: "[namespace]"
    defaultValue: "staging"
argument-hint: "[namespace]"
context: inline                  # "inline" (default) or "fork" (subagent runner)
paths:                           # Conditional activation glob patterns
  - "deploy/**/*.yaml"
  - "k8s/**"
user-invocable: true             # Shows in / slash command menu
disable-model-invocation: false  # If true, model cannot call it via tool
shell: bash                      # bash or powershell
---

# Deploy Staging Workflow

## Inputs
- `$namespace`: Destination namespace.

## Steps
1. Verify kubernetes context.
2. Run lint on Helm charts located at `${SKILL_DIR}/charts`.
3. Apply deployment.
```

---

## 3. Discovery, precedence, and shadowing disclosure

### A. Precedence resolution
In `State/SkillManager.swift`, `resolveEffectiveSkills(projectURL:)` combines
user and project skills:
1. Scan user skills (`~/.turbospark/skills` and external user agent roots).
2. If `projectURL` is nil (standalone chat), return user skills.
3. If `projectURL` is provided, scan project skills (`.turbospark/skills` etc).
4. Insert user skills into a dictionary keyed by lowercased name.
5. Insert project skills into the dictionary, overwriting any matching user
   skill name.
6. Return sorted array.

### B. Shadowing disclosure
Precedence without disclosure makes user skills appear broken. `SkillManager`
tracks `shadowedUserSkillNames(projectURL:)`. When a project overrides a user
skill:
- The Settings UI (`Components/SkillsSettingsPaneView.swift`) displays a badge
  alerting that the user skill is currently shadowed by the active project.
- In chat diagnostics, active skill listings identify whether the loaded skill
  originates from project scope or user scope.

### C. Thread safety and memoization
`SkillManager` is marked `@unchecked Sendable`. The resolution cache is
protected by an `NSLock` (`cacheLock`). Resolution compute runs outside the
lock to prevent lock contention across concurrent background tool executions
and UI updates. Any disk modification or toggle calls
`invalidateResolutionCache()`.

---

## 4. Marketplace and remote fetching (Git & HTTPS)

To acquire skills from external repositories (like Claude Code plugins, Codex
skills, or GitHub repositories), the engine provides `SkillMarketplaceManager`.

### A. Marketplace manifest (`marketplace.json`)

Marketplaces publish a manifest describing their skills collection:

```json
{
  "name": "developer-essentials",
  "description": "Curated productivity and coding skills",
  "owner": {
    "name": "TurboSpark Community",
    "url": "https://github.com/turbospark"
  },
  "skills": [
    {
      "name": "docker-build",
      "description": "Optimized multi-stage container build assistance",
      "source": {
        "source": "github",
        "repo": "whit3rabbit/agent-skills",
        "path": "skills/docker-build",
        "ref": "main"
      }
    },
    {
      "name": "security-audit",
      "description": "Scans dependencies and codebase for known CVEs",
      "source": {
        "source": "url",
        "url": "https://raw.githubusercontent.com/whit3rabbit/agent-skills/main/skills/security-audit/SKILL.md"
      }
    }
  ]
}
```

### B. Remote fetch mechanisms

1. **HTTPS URL Fetcher**:
   - Downloads raw `marketplace.json` or individual `SKILL.md` files using
     `URLSession.shared.data(from:)`.
   - Used for quick direct imports without requiring a local git checkout.

2. **Git Cloner with Sparse Checkout**:
   - For repository sources (`github` or `git`), uses `/usr/bin/git`.
   - To minimize bandwidth and disk consumption on large monorepos, fetches are
     executed with shallow clone and cone-mode sparse checkout:
     ```sh
     git clone --depth 1 --filter=blob:none --no-checkout <git_url> <cache_dir>
     git sparse-checkout set --cone -- <skill_path>
     git checkout HEAD
     ```
   - Target branch/ref pinning and SHA verification are supported.

### C. Installation target selector

When installing a skill from a marketplace or Git URL, the user is prompted to
select the target scope:
- **User Scope**: Installs to `~/.turbospark/skills/<name>/`. Available across
  all projects and in standalone chat.
- **Project Scope**: Installs to `<project_root>/.turbospark/skills/<name>/`.
  Available only in the current workspace, checked into source control if
  desired.

### D. Installed skills ledger (`installed_skills.json`)

Installations are recorded in `~/.turbospark/plugins/installed_skills.json`
supporting multi-scope tracking:

```json
{
  "version": 1,
  "skills": {
    "docker-build": [
      {
        "scope": "user",
        "installPath": "/Users/user/.turbospark/skills/docker-build",
        "version": "1.2.0",
        "gitCommitSha": "d4e2f1...",
        "installedAt": "2026-09-04T12:00:00Z"
      }
    ]
  }
}
```

---

## 5. Execution pipeline and prompt context management

### A. Context character budget (1% rule)
Exposing full skill instructions at the start of a conversation wastes model
context tokens. TurboSpark applies Claude Code's prompt budgeting policy:
- Skills are summarized in the tool catalog: only name and `when_to_use`
  (or `description`) are exposed initially.
- The total length of advertised skill summaries is constrained to **1% of the
  model context window** (e.g. ~8,000 characters for a 200,000 token context).
- If the count of skills exceeds the budget, non-bundled skills have their
  descriptions truncated to stay within limits.

### B. Dynamic conditional activation (`paths`)
Skills that declare a `paths` list in their frontmatter (e.g. `paths: ["**/*.swift"]`)
are not preloaded. When file tools (`file_read`, `file_edit`, `apply_patch`,
`grep_search`) access files matching the glob pattern:
1. `SkillManager.matchesPath(skill:filePath:)` evaluates the glob pattern.
2. Matching conditional skills are dynamically activated for the remaining turns
   of the conversation.

### C. Execution modes
When the model proposes `skill(name: "...", arguments: {...})`:

1. **Inline Mode (`context: inline`)**:
   - In `AppToolRegistry.swift`, arguments are substituted via
     `SkillManager.substituteArguments`.
   - Variables replaced:
     - `${arg_name}`: parameter from tool call.
     - `${SKILL_DIR}` or `${CLAUDE_SKILL_DIR}`: absolute directory containing
       the skill.
     - `${SESSION_ID}` or `${CLAUDE_SESSION_ID}`: active conversation turn ID.
   - The full instruction text is returned as a `.tool` response, guiding the
     model in the next generation step.

2. **Fork Mode (`context: fork`)**:
   - Invokes `SubagentRunner.run` with an isolated conversation context,
     its own token budget, and specific tool permissions from `allowed-tools`.
   - The final synthesized result from the subagent is returned to the main
     conversation without cluttering the primary history.

---

## 6. Chat and user interface integration

### A. Slash and `@` autocomplete
In `Generation/PromptComposerEditor.swift`, `ComposerAutocompleteController`
and `ComposerAutocompleteEngine`:
- Typing `/` as the draft's trailing token triggers an inline autocomplete
  list; typing `@` does the same for files and folders under the open
  project's root (`Files/ProjectFileIndex.swift`, a bounded walk with the
  attachment importer's skip list, directories included, refreshed on a
  short TTL). Detection is TRAILING TOKEN ONLY: `TextEditor` exposes no
  caret, so a trigger typed mid-draft does not open the list. An unterminated
  `@"partial path` stays a trigger across spaces; the closed `@"..."` form
  is not.
- Slash rows are the `BuiltInSlashCommand` table (the SAME table the plus
  menu and the submit-time parsers read; `/compact`, `/explore`, `/plan`,
  `/review`, `/agent`) plus `model.effectiveSkills` filtered by
  `isEnabled == true && userInvocable == true`. A skill named like a built-in
  command is dropped: the agent parser runs first at submit time, so the
  skill is unreachable and must not be offered.
- Up/down arrows move the selection, Tab and Return accept, Escape dismisses
  only the popup (a second Escape still cancels a running turn), and a click
  accepts. Accepting replaces the trigger token with `/name ` or
  `@relative/path ` (quoted when the path has spaces) and closes the list.
- `@` with no project open shows a hint row instead of scanning.
- At SEND time (`Files/MentionResolver.swift`, inside `AppModel.run()`'s
  submission task, before the `UserPromptSubmit` hook): every `@path` that
  resolves becomes an ordinary attachment -- a file through
  `DocumentTextExtractor`, a folder through the same bounded walk the folder
  picker uses -- and the `@path` token STAYS in the message text. A token
  that resolves to nothing stays as prose, silently; the missing chip is the
  feedback.

### B. Session capture ("Skillify" / propose_skills)
- When a complex task completes successfully, the user or model can invoke the
  skill capture flow (`propose_skills` in `Tools/Planning/`).
- The assistant extracts:
  1. The repeatable goal and ordered steps.
  2. Concrete success criteria for each step.
  3. Necessary tool permission patterns.
- The user is asked via `AskUserQuestion` whether to save to **User Scope**
  or **Project Scope**, and the `SKILL.md` is generated and saved to disk.

---

## 7. Agent-config support matrix compatibility

TurboSpark skills are designed for cross-harness compatibility with the
contracts in `https://github.com/whit3rabbit/agent-config/blob/main/docs/support-matrix.md`:

| Agent Harness | User Scope Directory | Project Scope Directory | Skill File Format |
|---|---|---|---|
| **TurboSpark** | `~/.turbospark/skills/<name>/` | `<root>/.turbospark/skills/<name>/` | `SKILL.md` |
| **Claude Code** | `~/.claude/skills/<name>/` | `<root>/.claude/skills/<name>/` | `SKILL.md` |
| **Cursor** | `~/.cursor/skills/<name>/` | `<root>/.cursor/skills/<name>/` | `SKILL.md` |
| **OpenClaw** | `~/.openclaw/skills/<name>/` | `<root>/.agents/skills/<name>/` | `SKILL.md` |
| **Gemini CLI** | `~/.gemini/skills/<name>/` | `<root>/.gemini/skills/<name>/` | `SKILL.md` |
| **OpenCode** | `~/.config/opencode/skills/<name>/`| `<root>/.opencode/skills/<name>/` | `SKILL.md` |

Because TurboSpark discovers across all these locations in both user home and
project roots, skills installed for Claude Code, Cursor, or OpenClaw are
automatically discovered and runnable in TurboSpark without manual migration.
