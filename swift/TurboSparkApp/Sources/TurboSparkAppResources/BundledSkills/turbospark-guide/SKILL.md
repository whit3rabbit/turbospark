---
name: turbospark-guide
description: Diagnose and explain TurboSpark's own configuration, covering hooks, slash commands, skills, MCP servers and plugins, by reading the real stores and naming the Settings pane that fixes each problem.
when_to_use: Use when the user asks why a hook, slash command, skill, MCP server or plugin is not working or not showing up, or asks how TurboSpark settings, profiles or shortcuts are organized. Do not use it for problems in the user's own code.
---

# TurboSpark guide

This file guides you. It enforces nothing: the app's tool permission prompts do.
You inspect state with your file tools and tell the user what to change. You
never edit these stores yourself unless the user asks, because the app rewrites
its own settings and a hand edit can be overwritten or quarantined.

Rules for every diagnosis:

- Reads outside the open project (the app store, `~/.turbospark`) may prompt the
  user. Say what you are about to read and why, then read only that file.
- Never repeat secrets. MCP `env` values, `headers`, tokens and API keys stay
  redacted in your report: name the key, write `<redacted>` for the value.
- Report one failing step, the evidence (file and field), and the pane that
  fixes it. Do not guess: if you could not read a file, say so.
- Settings panes are named exactly as in Settings: General, Profiles,
  Appearance, Keyboard Shortcuts, Files & Permissions, Models & Storage,
  Engine, Safety & Steering, MCP Servers, Tools, Skills, Memory, Soul,
  Agents & Subagents, Plugins, Scheduled Tasks, Hooks & Lifecycle, Browser,
  Computer Use, Network.

## Where state lives

App store root, the "app store" below: `~/Library/Application Support/TurboSpark/`.
A non-default profile keeps its own copy under `profiles/<id>/` inside it, and
`TURBOSPARK_STATE_DIR` redirects the whole root. Ask which profile is active
(Settings > Profiles) before concluding a file is missing.

| What | File or folder |
| --- | --- |
| App preferences and the plugin enable switches | `settings.json` in the app store |
| Per-project settings and project MCP servers | `projects_archive.json` in the app store |
| Global MCP servers | `global_mcp_servers.json` in the app store |
| Items the user switched off (skills and others) | `disabled_items.json` in the app store |
| Hook trust and custom hooks | `Hooks/trusted_hook_hashes.json`, `Hooks/custom_hooks.json` in the app store |
| Plugins (installed, registered folders, marketplaces) | `~/.turbospark/plugins/installed_plugins.json`, `local_plugins.json`, `marketplaces/known_marketplaces.json` |
| User skills | `~/.turbospark/skills/<name>/SKILL.md` |
| Project skills | `<project>/.turbospark/skills/<name>/SKILL.md` |

Shortcuts and profiles: Settings > Keyboard Shortcuts lists and rebinds every
shortcut. Settings > Profiles switches profiles (a relaunch) and each profile
owns its own stores. When the user asks how these work, name the pane, then
read the relevant file to answer from the real state.

## Diagnose hooks

A hook runs a command at a lifecycle point (for example before a tool runs).

1. Sources, scanned in this order: user `~/.turbospark/hooks.json` and
   `~/.turbospark/settings.json`, then `settings.local.json` beside them; the
   project's `.turbospark/hooks.json`, `.turbospark/settings.json`,
   `.claude/settings.json`, `hooks/hooks.json` and the `settings.local.json`
   files; then enabled plugins. A non-default profile reads only its own
   `hooks.json` in the app store. Another tool's `~/.claude/settings.json` is
   read only when the cross-agent opt-in in Settings > General is on.
2. Read each existing source. Malformed JSON is reported by the app as
   "Issues found while scanning hook config files" at the top of
   Settings > Hooks & Lifecycle. Open that file and find the syntax error.
3. A hook from a file or plugin does not run until the user reviews it:
   Hooks & Lifecycle shows "Needs review". Hooks the user creates in the app
   are trusted already. Trust is stored by content hash in
   `Hooks/trusted_hook_hashes.json`, so editing a command un-trusts it.
4. Check the matcher and event name against the tool the user expects, then
   the command: does the executable exist, is it executable, does it read
   stdin. Ask the user to run the command by hand if you cannot.
5. Fix path: Settings > Hooks & Lifecycle (Refresh, trust, edit, disable).

## Diagnose commands

TurboSpark has no separate command store. A slash command is one of:

- a built-in command (the slash popup in the composer lists them);
- a skill: `/<skill-name>` runs a skill, so a missing command is usually a
  missing, disabled or shadowed skill (see Diagnose skills);
- a plugin command: `<plugin>/commands/**/*.md`, run as
  `/<plugin>:<command>`; nested folders add name segments
  (`commands/review/lint.md` is `/<plugin>:review:lint`).

Steps: ask for the exact text typed; type `/` in the composer and check the
popup; if it is a plugin command, confirm the plugin is enabled and the file
has valid frontmatter (Settings > Plugins, then Diagnose plugins). A command
file is plain Markdown with optional frontmatter (`description`,
`argument-hint`, `allowed-tools`). Fix path: Settings > Skills or Plugins.

## Diagnose skills

1. Locations: `~/.turbospark/skills/<name>/SKILL.md`, the project's
   `.turbospark/skills/<name>/SKILL.md` (other agents' folders such as
   `.claude/skills` count only with the cross-agent opt-in in Settings >
   General), plugin skills (`<plugin>:<name>`), and skills shipped with the
   app (bundled, read-only, lowest precedence).
2. The folder must contain `SKILL.md` directly (not nested deeper) with
   frontmatter between `---` lines holding at least `name` and `description`.
   A file that fails to parse is skipped silently: check it first.
3. Precedence by lowercased name: project beats user beats bundled. Settings >
   Skills marks a user skill "Shadowed" while a project skill replaces it.
4. A skill the user switched off is listed in `disabled_items.json` and is
   not offered to you. Fix path: Settings > Skills (toggle, Refresh).

## Diagnose MCP

1. Global servers are in `global_mcp_servers.json`; project servers are in
   `projects_archive.json` (each project's `mcpServers`); plugin servers come
   from the plugin's `.mcp.json` or manifest. Settings > MCP Servers lists
   global ones, the project settings sheet lists project ones. A server whose
   switch is off, or which was imported from another agent (imported servers
   arrive disabled), is never started.
2. Two transports: `stdio` (a local command with `args`, `env`, `cwd`,
   `envPassthrough`) and `sse` (a `url` plus `headers`).
3. For stdio, the child gets a minimal environment: only `PATH`, `HOME`,
   `LANG`, `TMPDIR`, names listed in `envPassthrough` and keys in `env`. A
   server that needs a token from the shell fails with an auth error until the
   key is added to `envPassthrough` or `env`. That is deliberate, not a bug.
   `PATH` is extended with the executable's folder and the usual Homebrew
   folders, so a missing `npx` or `node` means it is not installed there.
4. There is no log file. The evidence is the result of the connection test:
   ask the user to press "Test connection and discover tools" on the server
   card and read you the message. A spawn failure names the command; a
   handshake failure usually means the program printed non-JSON on stdout.
5. Check the command exists (`command -v` in your shell tool), the
   `cwd` exists, and the URL is reachable for sse. Fix path: Settings > MCP Servers (edit, Test, toggle).

## Diagnose plugins

1. A plugin is a folder with `.claude-plugin/plugin.json`. Layout and
   precedence: `~/.turbospark/plugins/<name>/` flat folders, then registered
   local folders (`local_plugins.json`), then marketplace installs under
   `~/.turbospark/plugins/cache/`, then Claude interop (opt-in only).
2. A plugin load problem is shown as a diagnostic line in Settings > Plugins.
   Ask the user to read it to you; common causes are invalid manifest JSON, a
   name collision (the lower-precedence copy is dropped and reported), and
   fields this client does not support (reported, loading continues).
3. Enabled state is `enabledPlugins` keyed `<name>@<origin>` in `settings.json`
   (user) and in the project's entry of `projects_archive.json`; a project
   entry wins, and absent means enabled. Toggle in Settings > Plugins only.
4. Plugin hooks stay unreviewed until approved in Settings > Hooks &
   Lifecycle. Plugin MCP servers appear in the MCP list once the plugin is on.
5. To create a plugin, use the `plugin-creator` skill instead of this one.
