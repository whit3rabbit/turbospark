---
name: plugin-creator
description: Scaffold a TurboSpark plugin (.claude-plugin/plugin.json plus only the components the user asks for) and a dev marketplace.json, then print the exact manual install steps.
when_to_use: Use only when the user asks to create, scaffold or package a plugin. Do not use it to install, edit or debug an existing plugin.
---

# Plugin creator

You scaffold a plugin folder with your file tools. Nothing here installs the
plugin: the user does that, with the steps you print at the end. This file
guides you, it does not enforce anything. The enforcement is the app's tool
permission prompts, so never ask the user to approve a write outside the
folders named below.

File writes need an open project. If you have no file write tool this turn,
say so, ask the user to open a project, and stop.

## 1. Gather inputs

Ask once for anything the user left out, then restate the plan and wait for a
yes before writing:

- Plugin name: lowercase letters, digits and hyphens, no spaces. Never
  `inline` or `builtin`.
- Target folder: an absolute path to an empty or new folder. This becomes the
  marketplace root.
- Components wanted: any of skills, commands, agents, MCP servers. Create
  ONLY the ones requested. Do not add hooks: a plugin's hooks stay untrusted
  until the user approves them in the app, so leave them out unless asked,
  and then tell the user about the approval prompt.
- One-line description, and optionally author and version (default `0.1.0`).

## 2. Layout

```text
<root>/
  .claude-plugin/marketplace.json     dev marketplace listing the plugin
  plugins/<name>/
    .claude-plugin/plugin.json        the manifest
    skills/<skill>/SKILL.md           only if skills requested
    commands/<command>.md             only if commands requested
    agents/<agent>.md                 only if agents requested
    .mcp.json                         only if MCP servers requested
```

Skills, commands and agents sit in the conventional directories, so the
manifest does not need to list them.

## 3. plugin.json

```json
{
  "name": "<name>",
  "version": "0.1.0",
  "description": "<one line>",
  "author": { "name": "<author>" }
}
```

Add `"skills"`, `"commands"` or `"agents"` only for a non-conventional
location, as `./relative/path` strings. Valid JSON only: no comments, no
trailing commas. Re-read the file after writing it.

Component files:

- Skill: `skills/<skill>/SKILL.md` with `name`, `description` and optional
  `when_to_use` frontmatter, then the instructions.
- Command: `commands/<command>.md` with a `description` frontmatter line and
  the prompt body. Nested folders add a namespace segment.
- Agent: `agents/<agent>.md` with `name` and `description` frontmatter and the
  system prompt body.
- MCP servers: `.mcp.json` at the plugin root (template below). The command
  is executed on the user's machine, so write only a command the user gave you
  or confirmed, and repeat it back in your final message. Never invent a
  package or download URL.

```json
{
  "mcpServers": {
    "<server>": { "command": "<executable>", "args": [] }
  }
}
```

Loaded names are namespaced as `<plugin>:<skill>` for skills and commands and
`plugin:<plugin>:<server>` for MCP servers.

## 4. Containment rules (mandatory)

Every path you write, and every path inside the manifest, must stay inside
`<root>/plugins/<name>/` (the marketplace file is the one exception, at
`<root>/.claude-plugin/`).

- No `..` segment anywhere, in any path you write or put in a file.
- No absolute path inside the manifest or marketplace entries. Manifest paths
  start with `./`.
- Do not create symlinks. Do not follow one that already exists in the target.
- Do not overwrite an existing file. If the target folder is not empty or a
  file already exists, stop and ask the user for another folder or an explicit
  go-ahead for that one file.
- Never write to the user's home plugin folder (`~/.turbospark/plugins`), the
  plugin ledger, settings files, or any other app state. Registration is the
  user's step.

If a requested path violates a rule, refuse with this exact error name and
ask for a corrected path, writing nothing:

`PLUGIN_PATH_ESCAPE: '<path>' leaves the plugin folder`

## 5. Dev marketplace.json

At `<root>/.claude-plugin/marketplace.json`:

```json
{
  "name": "<name>-dev",
  "owner": { "name": "<author or the user>" },
  "plugins": [
    { "name": "<name>", "source": "./plugins/<name>", "description": "<one line>" }
  ]
}
```

The marketplace name must be ASCII, with no spaces, `/`, `\` or `..`.

## 6. Validate before you report

Read the files back and check each item; fix and re-check any failure:

1. `plugin.json` and `marketplace.json` parse as JSON.
2. The plugin name has no whitespace and is not `inline` or `builtin`.
3. Every manifest path starts with `./` and has no `..`.
4. Each requested component exists, and no unrequested component does.
5. Each `SKILL.md`, command and agent file starts with frontmatter.

## 7. Final message: manual steps

Print these with the real paths filled in, and say that nothing has been
installed yet:

1. Register the folder directly: Settings > Plugins > Import > Add Local
   Folder..., choose `<root>/plugins/<name>`.
2. Or use the dev marketplace: Settings > Plugins > Browse Marketplace >
   Manage sources, add a Directory source named `<name>-dev` with path
   `<root>`, then install `<name>` from it.
3. Enable the plugin and open a new chat. Check that its skills, commands,
   agents and MCP servers appear under their namespaced names.
4. After you edit the plugin, use Refresh in the Plugins pane.
