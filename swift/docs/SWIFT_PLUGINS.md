# Swift plugins

The TurboSparkApp plugin system, ported from Claude Code's
(`claude-code-build1/src/plugins`). A plugin is a directory whose manifest
lives at `.claude-plugin/plugin.json`; it contributes skills, slash
commands, agents, hooks and MCP servers, and it arrives either by being
placed on disk or installed from a marketplace. This page is the HOME for
the plugin semantics: what is supported, what is parsed-and-reported, and
the two deliberate deviations from Claude Code.

Read this page before adding a contribution surface or re-deriving the
enable cascade.

## Layout on disk

```
~/.turbospark/plugins/            THIS APP'S ROOT (writable)
+-- <name>/                       flat plugin dirs (hand-placed; also what the
|                                 pre-plugin hook stub scanned)
+-- cache/<marketplace>/<plugin>/<version>/
|                                 versioned install cache (marketplace installs)
+-- marketplaces/known_marketplaces.json
+-- marketplaces/<cached-source>/ marketplace checkouts
+-- installed_plugins.json        v2 install ledger
+-- local_plugins.json            locally-registered folders
+-- data/<sanitized-id>/          persistent per-plugin data (${CLAUDE_PLUGIN_DATA})

~/.claude/plugins/                CLAUDE CODE'S ROOT (READ-ONLY)
+-- (same shapes)                 discovered for interop, never written
```

Precedence on a lowercased name collision, first match wins: turbospark
flat dirs, then local folders, then the versioned caches, then Claude
interop. The shadowed entries are reported in `pluginLoadDiagnostics`, not
silently dropped. Within one cache pair, the newest version directory wins.

## plugin.json

Same location and mostly the same fields as Claude Code's manifest
(`PluginModels.swift` / `PluginManifestParser.swift`):

| Field | Support |
|---|---|
| `name`, `version`, `description`, `author`, `homepage`, `repository`, `license`, `keywords` | read; `name` must be non-empty, no whitespace, not `inline` or `builtin` |
| `commands` | single path, array of paths, or object map (`source` xor `content`; both keeps source with a diagnostic) |
| `agents`, `skills` | single relative path or array; each falls back to the conventional `agents/` / `skills/` directory when absent |
| `hooks` | `./file.json` paths and the inline schema, parsed by the SAME code path as `hooks.json` |
| `mcpServers` | `./file.json` paths, the inline `Record<name, config>`, and a root `.mcp.json` |
| `userConfig` | option specs; surfaced in the plugin settings pane and the hooks options store |
| `dependencies` | parsed, informational only (no closure resolution, see below) |
| `lspServers`, `outputStyles`, `channels`, `settings`, `.mcpb` | parsed, recorded as per-plugin diagnostics ("not supported by this client"), loading continues |

Validation rules, per Claude Code: a MISSING manifest is fine (one is
synthesized: the directory name plus a description naming the source, when
a convention directory exists); UNPARSEABLE JSON is fatal for that plugin
alone and its siblings load; a field whose value has the wrong shape is
dropped with a diagnostic while unknown top-level keys are ignored
entirely. Relative paths that climb out (`..`, absolute) are refused at the
parse layer.

Marketplace entries add `source` and `strict` (default true). Strict means
the plugin's own manifest must exist and the entry only fills gaps;
non-strict synthesizes from the entry; BOTH defining the same surface is a
conflict error, matching Claude Code.

## Contributions

All four surfaces resolve through the same parsers the first-party
surfaces use, so a plugin contribution is shaped exactly like its
hand-written counterpart and differs only in its NAME:

- **Skills and slash commands**: `skills/<name>/SKILL.md` and
  `commands/**/*.md` load through `SkillParser` as `AppSkill` named
  `<plugin>:<skill>` -- nested command directories add segments
  (`plugin:ns:name`), which is Claude Code's namespace rule. They appear in
  the system prompt tagged `[Plugin]`, run through the `skill` tool, and
  answer `/plugin:skill` slash commands.
- **Agents**: `agents/**/*.md` through `AgentParser`, named
  `<plugin>:<name>`, scope `.plugin`, lowest precedence (namespaced names
  cannot collide with anything).
- **Hooks**: `hooks/hooks.json`, a bare `hooks.json` (the pre-plugin stub's
  layout), manifest-declared `./file.json` paths, and inline manifest
  hooks. Disabled plugins contribute nothing on every surface.
- **MCP servers**: named `plugin:<plugin>:<server>` -- the namespacing IS
  the collision rule, and the colon survives the `mcp__<server>__<tool>`
  splitter because it splits on `__`. Plugin servers never auto-approve:
  `autoApprove` is forced false and every call passes the permission
  engine.

## Enable cascade

`enabledPlugins: Record<String, Bool>` keyed `<name>@<origin>`, in BOTH
`settings.json` (user scope) and `projects_archive.json` (project scope,
inside `AppProject`). Resolution order in `PluginManager.isResolvedEnabled`:

1. project entry (a per-project off wins in that project),
2. user entry,
3. for `.claudeInterop` plugins only, Claude Code's own `enabledPlugins`
   from `~/.claude/settings.json` (bool and version-array forms),
4. absent means ENABLED.

Enable writes always go through `AppModel` (`pluginEnableState` published
property, `persistSettings`), never through `PluginManager` -- a direct
write to the file would be clobbered by the model's own settings save. The
toggle path persists synchronously (not debounced) BEFORE invalidating, so
the next file read cannot see stale state.

## Marketplace

`.claude-plugin/marketplace.json`: `name` (no whitespace, `/`, `\`, `..`,
non-ASCII, or the reserved `inline`/`builtin`), `owner`,
`metadata.pluginRoot`, and `plugins[]` entries. Sources are the shared
`MarketplaceSource` / `MarketplaceGit` pair: `github`, `git`, and
`directory` install; a bare `./relative` path installs from the
marketplace checkout; `url` can list but cannot install relative entries
(it never materializes a checkout) and says so. Entry paths and
`metadata.pluginRoot` are resolved through symlinks and must remain inside
the materialized checkout. Only a locally configured `directory`
marketplace may install an entry whose source is another local directory.

Installs land in the versioned cache with the version resolved manifest >
entry > git sha12 > `"unknown"`, and are recorded in the v2 ledger
`installed_plugins.json` (`{"version": 2, "plugins": {"<id>": [records]}}`,
one record per scope). Uninstall removes one scope; the cache directory is
deleted only when the LAST scope goes, and only ever paths INSIDE the
cache root are deleted -- a hand-edited ledger row pointing elsewhere is
left on disk while its record is removed.

## Variables

`${CLAUDE_PLUGIN_ROOT}` (the version-scoped install directory), 
`${CLAUDE_PLUGIN_DATA}` (persistent, lazily created, deleted on last-scope
uninstall), and `${user_config.KEY}` (saved option values, stored under the
hooks options store with sourceID `plugin_<name>`). Rules from Claude
Code, enforced in `PluginVariableExpander`:

- Sensitive options are substituted only in process ENVIRONMENTS (hooks,
  MCP children, `preserveSensitive: true`); skill and agent content gets
  `[sensitive option not available in content]` instead, because content is
  shown and copied to places the user cannot see.
- Unknown `${user_config.KEY}` keys stay literal.
- In shell hook commands the two path variables are shell-quoted for the
  same reason `${CLAUDE_PROJECT_DIR}` is (hooks state#60).
- Hooks and MCP children also receive `CLAUDE_PLUGIN_ROOT`,
  `CLAUDE_PLUGIN_DATA` and `CLAUDE_PLUGIN_OPTION_<KEY>` (plus
  `TURBOSPARK_PLUGIN_*` aliases) in their environment.

## The two deviations from Claude Code

1. **Plugin hooks are UNTRUSTED until approved.** Claude Code runs
   marketplace plugin hooks the moment they are installed; this app's
   SHA-256 trust gate applies to them exactly as to hooks discovered in a
   cloned repository (`contentHash` includes the plugin origin, so
   reinstalling or editing a plugin re-prompts). A clean run of the hooks
   through `AppHookStore` requires a trust decision first.
2. **`userConfig` entries are dropped with a diagnostic when malformed**
   rather than failing the plugin. Claude Code rejects the manifest; this
   app draws the line where `TolerantDecoding` does: a file that will not
   parse is fatal, a field that will not convert is a field.

## Adding a contribution surface

Wire it through `PluginManager+Contributions` and give it the namespaced
name (`<plugin>:<name>`, or `plugin:<plugin>:<server>` for MCP). A bare
name would collide with user content, and `executeMcpCall`'s precedence
rule assumes namespacing is what prevents collisions there. Every manager
takes injectable roots, so a new surface is testable without touching
`~`; do not introduce a hardcoded plugin path.

`McpClientEngine.resolveExecutablePath` is the one answer to "can this
command be launched," called both by the spawn and by
`McpMarketplaceManager` before it will install a catalog entry (see
`swift/docs/SWIFT_TOOLS.md` section 10). A second lookup would drift, and
the one that drifted would accept a server the spawn then refuses --
reported at the first tool call rather than at install time.

## Out of scope (parsed or absent, on purpose)

Dependency closure and demotion, auto-update, enterprise policy/managed
plugins, builtin plugin registry, seed dirs, LSP servers, output styles,
channels, MCPB bundles, npm/pip sources. `dependencies` is recorded
informationally; the unsupported surfaces are listed per plugin in the
settings pane.

## Where things live

`State/PluginModels.swift` (types), `State/PluginManifestParser.swift`
(parsing, validation, strict merge), `State/PluginManager.swift`
(discovery, cascade, memoized resolution), `State/PluginManager+Contributions.swift`
(skills/agents/MCP accessors), `State/PluginVariableExpander.swift`
(substitution contract), `State/PluginLedgerStore.swift` (v2 ledger,
local folders), `State/PluginMarketplaceManager.swift` (fetch, install,
uninstall), `State/AppModel+Plugins.swift` (UI actions, toasts, settings
writes), `Components/PluginSettingsPaneView.swift` +
`Components/PluginMarketplaceSheet.swift` (UI). Tests:
`PluginSystemTests`, `PluginManagerTests`, `PluginHookTests`,
`PluginMarketplaceTests`, all over injectable temp roots.
