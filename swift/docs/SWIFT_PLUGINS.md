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

The Claude interop root is read only under the cross-agent opt-in
(`MacAppSettings.autoLoadExternalAgentContent`, Settings > General,
default OFF; mirrored into `PluginManager.includeClaudeInterop`). With the
setting off, another tool's installs do not appear at all -- the Import
wizard (Settings > General > "Import from Other Agents...") is the opt-in
path for cross-agent content.

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
plugins, the `cdnZip` source (deferred, see ZP-05), LSP servers, output styles,
channels, MCPB bundles, npm/pip sources. `dependencies` is recorded
informationally; the unsupported surfaces are listed per plugin in the
settings pane.

## Plugin creator (ZP-07)

Plugins are authored by the agent, not a form. The bundled `plugin-creator`
skill (`Sources/TurboSparkAppResources/BundledSkills/plugin-creator/SKILL.md`)
scaffolds `.claude-plugin/plugin.json` plus only the requested components, a
dev `marketplace.json`, and prints the manual steps. The Plugins pane Import
menu has "Create Plugin...", which calls `AppModel.composeInChat` with a
localized draft naming the skill. The draft is editable and never sent for the
user.

- **Delivery.** A `SkillScope.bundled` skill read from the resource bundle by
  `SkillManager.discoverBundledSkills()`. It is the lowest precedence (a user
  or project skill of the same name replaces it), read-only (`ownsSkill` is
  false), and disablable through the persisted skill state. The folder is a
  `.copy` resource in `Package.swift` because `.process` flattens directories.
  This differs from ZCode, which ships a builtin plugin; the builtin plugin
  registry is out of scope here, and a later seeded-plugin path (ZP-05) can
  take over the same skill.
- **Containment is advisory.** The skill text forbids `..`, absolute paths,
  symlinks and overwrites and names the refusal `PLUGIN_PATH_ESCAPE`, but only
  the agent follows it. The enforced controls are the manifest parser (which
  drops `..` and absolute paths) and the tool permission prompts on file
  writes. The skill writes no hooks by default, so the plugin-hook trust gate
  is untouched; MCP commands in `.mcp.json` run only once the user registers
  and enables the plugin.
- **Needs a project.** File writes are workspace-rooted tools, so the agent
  can scaffold only in a chat with an open project and says so otherwise.
- **Tests.** `PluginCreatorSkillTests`: real-bundle discovery, precedence,
  read-only, templates through the real parsers, localized draft.

## Builtin plugins (ZP-05)

`State/BuiltinPluginSeeder.swift` seeds plugins shipped in the app bundle.
The bundle holds a read-only directory marketplace
(`BuiltinPlugins/.claude-plugin/marketplace.json`, one `./relative` folder
per plugin) named `turbospark-plugins-builtin`. The app ships NO builtin
today, so the row is hidden and launch seeding is a no-op. To ship the first
one: add the folder, add `.copy("BuiltinPlugins")` to the
`TurboSparkAppResources` target in `Package.swift`, and run
`make swift-app-build` (read the resource rules in `swift/AGENTS.md`).

- **Same installer.** Seeding calls `PluginMarketplaceManager.install` as a
  `directory` source, so staging, the versioned cache, the v2 ledger and the
  hook trust gate are unchanged. Seeding copies files and never runs code; a
  seeded plugin's hooks stay untrusted until approved.
- **Builtin row.** Injected by `AppModel.marketplaceSources(kind: .plugins)`,
  never persisted, injected as a `directory` source (the `builtin` case
  exists for hand-written rows but the injection does not use it, so the
  seeded path is unchanged), and it wins a name clash so a hand-edited
  `known_marketplaces.json` cannot repoint it. Hidden when the bundle is empty.
- **State.** One file, `builtin_seed.json` beside the ledger, one record per
  builtin (`seeded` with version, or `suppressed`). `seed()` reconciles it
  with the ledger each launch: suppressed is never reinstalled; seeded but
  missing from the ledger becomes suppressed; installed but unrecorded is
  adopted; an older installed version is reinstalled when the bundle
  carries a newer one. A corrupt file fails closed (nothing seeded).
- **Uninstall sticks, Install restores.** `uninstallPluginID` (user scope)
  records suppression; installing from the builtin row records `seeded`,
  clearing it. A project-scope install or removal touches neither.
- **Seed lock.** A `flock` on `.builtin-seed.lock` in the plugin root,
  held for the whole reconcile. The kernel drops it when the holder dies, so
  there is no stale-lock case; a waiter gives up after 10 s.
- **Refusals.** Per entry: only a `./relative` source strictly inside the
  bundle, no symlink anywhere in the tree, at most 2,000 files and 64 MiB.
  One refused entry does not stop the others. The bundle's code signature is
  the integrity anchor; there is no second digest.
- **Profiles.** The state file and seeded copies live under the profile's
  plugin root, so profiles stay isolated.
- **`cdnZip` source (ZP-05).** A marketplace entry whose `source` is
  `{"type": "cdnZip", "url": "https://...", "sha256": "<64 lowercase hex>"}`
  installs one plugin from a zip. `MarketplaceSource` also gained `builtin`
  (resolves to the bundle directory) and `cdnZip`; decoding is tolerant, so a
  row with no digest still loads and an unknown `type` keeps the old `url`
  fallback. It is an install source only (a catalog cannot be a `cdnZip`) and
  the skills and MCP marketplaces refuse both new kinds.
  - **User-initiated only.** The download runs inside
    `PluginMarketplaceManager.install`, the Install action. Listing a
    marketplace, launch, and seeding never fetch it.
  - **Download** (`State/CdnZipDownloader.swift`): https only, no userinfo;
    the digest must be exactly 64 lowercase hex or the install is refused
    before any request; redirects are followed at most 3 times and only to
    https; the body streams to a temp file under a 64 MiB cap (early refusal
    from Content-Length, hard stop for a server that omits it) with an
    idle and a total timeout; the SHA-256 is computed over exactly those bytes
    and a mismatch deletes the file and throws before extraction.
  - **Extraction** (`State/ZipSafeExtractor.swift`): in-process (Compression
    framework, no subprocess, no dependency). `unzip` and `ditto` were not
    reused because they write to disk themselves and would create symlinks,
    read names as wildcards, and cannot report a lying size table. Stored and
    deflate only; no encryption or zip64; entry names through
    `ProfileBackupImport.isSafeArchiveEntry` plus no `.` component and
    case-folded uniqueness; regular files and directories only (a symlink,
    device, fifo or socket mode is refused); at most 2,000 entries, 16 MiB
    per entry, 64 MiB total, ratio 100:1 (declared and actual); overlapping
    members, a local header that disagrees with the directory, and a CRC
    mismatch are refused. Files land 0644 (0755 if the archive marked them
    owner-executable), never setuid.
  - **Same installer.** The zip is extracted into the install's staging
    directory and goes through the existing stage-move-rollback path, so
    version resolution, the ledger and the hook trust gate are unchanged:
    nothing in the archive runs at install, and its hooks stay untrusted until
    the user approves them. The plugin must sit at the zip root (no wrapper
    directory is stripped).
  - **Trust anchor.** The digest sits in the same row as the URL, so it
    proves the bytes are the ones the row names, not that the row is
    trustworthy. The marketplace the user added is the anchor; a signed
    catalog is a separate decision. The ledger does not record the source kind
    (its format is shared with Claude Code).
  - **Tests.** `CdnZipSourceTests`: a loopback fixture server (good zip, bad
    and malformed digest, zip-slip, symlink, bomb, lying size table, oversize,
    redirect to http), extractor limits at unit level. Plain http to
    127.0.0.1 is reachable only through the internal test policy.
- **Tests.** `BuiltinPluginSeederTests` (fixture bundle, temp root).

## Where things live

`State/PluginModels.swift` (types), `State/PluginManifestParser.swift`
(parsing, validation, strict merge), `State/PluginManager.swift`
(discovery, cascade, memoized resolution), `State/PluginManager+Contributions.swift`
(skills/agents/MCP accessors), `State/PluginVariableExpander.swift`
(substitution contract), `State/PluginLedgerStore.swift` (v2 ledger,
local folders), `State/BuiltinPluginSeeder.swift` (bundled plugins), `State/PluginMarketplaceManager.swift` (fetch, install,
uninstall), `State/AppModel+Plugins.swift` (UI actions, toasts, settings
writes), `Components/PluginSettingsPaneView.swift` +
`Components/PluginMarketplaceSheet.swift` (UI). Tests:
`PluginSystemTests`, `PluginManagerTests`, `PluginHookTests`,
`PluginMarketplaceTests`, all over injectable temp roots.

## UI plugins, page sandbox, and Gen UI (ZP-06)

A plugin can ship HTML pages that open in the right column. Design and
decisions: `docs/designs/zcode-parity/ZP-06-ui-plugins-sandbox-gen-ui.md`.
Code: `Sources/TurboSparkApp/Plugins/UI/`.

### Manifest

```json
{ "ui": { "surfaces": [ { "id": "chart", "title": "Chart", "resourceUri": "ui/chart.html",
    "resources": ["ui/chart.js", "ui/chart.css"],
    "capabilities": ["invokeTool", "widgetState", "sendFollowUp"],
    "tools": ["myserver/run_report"], "placement": "sidePane" } ] } }
```

- `resourceUri` and `resources` are bundle-relative paths (no `..`, no
  absolute path, no dot segment, no scheme). Only the page and the listed
  resources can ever be served; everything else is `not_declared`.
- Served types: html, js, css, json, txt, png, jpg, gif, webp, svg, woff2.
- `capabilities`: `invokeTool`, `widgetState`, `sendFollowUp`. An unknown
  one is dropped with a diagnostic. `tools` is `"<server>/<tool>"` over the
  plugin's OWN servers; the app builds `mcp__plugin:<plugin>:<server>__<tool>`
  itself, so a page cannot name another server.
- A malformed surface is dropped with a named diagnostic in the plugin's
  "Not applied by this client" list; siblings load. At most 8 surfaces and 64
  resources per surface.
- `placement` is parsed; v1 opens every plugin page in the side pane.

### Opt-in and opening

Pages are OFF for every plugin. The plugin detail pane shows a "UI pages"
block (only when surfaces exist) with an "Allow this plugin to show UI
pages" switch, stored in `plugin_ui_optin.json` (`PluginUiOptInStore`; a
corrupt file means nothing is opted in). Open buttons are disabled until it
is on. Disabling the plugin, removing it, or switching the opt-in off closes
its pages (`PluginUiHost.reconcile`). The plugin hook trust gate is not
involved and is unchanged.

### Sandbox contract

| Layer | Rule | Where |
| --- | --- | --- |
| Origin | every byte is served by a `WKURLSchemeHandler` under `tsplugin://<random token>/`; no `file:` or network origin exists; two pages never share an origin | `PluginPageSchemeHandler` |
| Files | declared paths only, no symlink on any component, `O_NOFOLLOW`, real path under the real root, 2 MiB per file | `PluginPageResolver.load` |
| CSP | derived per surface: `default-src 'none'`, `connect-src 'none'`, `frame-src 'none'`, no inline script, `script-src 'self'` only when a script is declared | `PluginPageCSP` |
| Network | the offline content rule list (shared with `ArtifactWebView`) blocks http(s) and ws(s) before the page loads; a missing rule list refuses to load | `PluginPageSession.load` |
| Navigation | only the page's own initial load; links, forms, `location`, subframes, new windows, `file:`, `data:`, `javascript:`, `blob:` are cancelled | `PluginPageNavigationPolicy` |
| Permissions | camera, microphone, geolocation denied; alert, confirm, prompt, file chooser and popups answered with nothing | `PluginPageSession` (WKUIDelegate) |
| Storage | `WKWebsiteDataStore.nonPersistent()` per page; widget state in memory only, cleared with the page | `PluginPageSession`, `PluginWidgetStateStore` |
| Menu | no context menu (its items are navigation and download paths) | `PluginGestureWebView` |

### Bridge: `window.turbospark`

Injected at document start, main frame only. The native handler is the
authority; the JavaScript shim is a convenience.

| Method | Needs | Rules |
| --- | --- | --- |
| `invokeTool(name, args)` | `invokeTool` + the tool in `tools` | recent real click or key press in the page (5 s), at most 2 in flight, 12 per minute, runs through `executeDeferredMcpCall` (hooks, deny rules, session approvals, permission engine). An `ask` verdict cannot show a card from a page, so it is refused as `permission_denied` |
| `getWidgetState()`, `setWidgetState(state)` | `widgetState` | keyed by page identity (plugin + surface + chat), 16 KiB per message |
| `sendFollowUp(text)` | `sendFollowUp` | recent gesture, 6 per minute, 4000 characters, control characters stripped; fills the composer draft labeled with the page title and plugin name. NEVER sends |

Errors reach the page as `error.code`: `undeclared_method`,
`tool_not_declared`, `permission_denied`, `payload_too_large`,
`rate_limited`, `gesture_required`, `too_many_pending`, `invalid_request`,
`bad_frame`, `tool_failed`. Messages from a subframe or another origin are
`bad_frame`. A genuine gesture is an AppKit mouse-down or key-down reaching
the page's web view (`PluginGestureWebView`); a page cannot fake it.

### Lifecycle

Page identity is plugin id + surface id + chat id. `PluginUiHost.maxLivePages`
is 8 (provisional: the app shares unified memory with model weights).
Opening past it tears down the least recently visible page that has had no
click or key press for 30 s and no tool call in flight; if every page is in
use, the new one is refused with a toast. Closing the pane, switching chat
and disabling the plugin tear the page down (web view released, widget state
cleared). A web content process loss reloads once from the stored page; a
second loss, a refused page document, or a missing offline rule list shows a
bounded failure card with Try again (`PluginPageFailureCard`, the contract
ZP-17 deferred here).

### Gen UI

`gen_ui_render` takes a JSON `spec` and draws a native SwiftUI card in the
chat. It is NOT HTML: the component set is closed (text, slider, stepper,
toggle, picker, readout, table, bars, follow_up), every field is validated
(`GenUiParser`), an unknown type or field rejects the whole spec naming its
path, and all model text is drawn verbatim. There is no web view, script,
network or tool authority on this path; a `follow_up` button only fills the
composer draft, labeled with the card title. Limits: 16 KiB, 40 components,
20 table rows, 24 bars. Ladder: category `automation`, risk `safe`, works
without a project; any future component with an effect must move the tool
up the ladder first.

### Tests

`PluginPageSandboxTests` (real WKWebView through the real scheme handler,
loopback fixture that must record zero requests), `PluginUiHostTests`,
`GenUiTests`.
