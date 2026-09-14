# Swift user profiles

User profiles in TurboSparkApp: multiple named users on one machine, each
with their own settings, chat history, projects, global MCP servers, model
favorites, appearance, hooks, custom tools, skills, agents, plugins, and
marketplace installs. There are no passwords and no accounts: a profile is a
FOLDER plus a row in a registry file, and nothing else.

Read this before adding a store (where does it live?) or touching
`AppStorageRoot`, `UserProfileStore`, or any of the `~/.turbospark` path
constants in the app.

## The two contracts

**The Default user is the machine's existing setup, not a folder.** Its
stores stay at `~/Library/Application Support/TurboSpark/` directly and its
user-scope content stays in the shared `~/.turbospark` tree, exactly where
every installation that predates profiles put them. There is no migration:
an empty registry IS the default-only state, which is what every existing
install already is on disk. The Default user also keeps reading the
cross-agent roots (`~/.claude/skills`, `~/.cursor/skills`, ...) that make
the shared tree interoperate with other harnesses, and it cannot be deleted.

**Every other profile is self-contained.** A non-default profile holds its
stores under `profiles/<id>/` and its user-scope content under that same
folder; it never reads `~/.turbospark` or any cross-agent root. That
isolation is the point of a profile, so do not "helpfully" merge the shared
roots back in for non-default profiles.

## Layout

```
~/Library/Application Support/TurboSpark/     machine root (AppStorageRoot.machineRoot)
  profiles.json                               the registry: profiles + activeProfileID
  settings.json, chats_archive.json, ...      the DEFAULT user's stores (all of them)
  profiles/<uuid>/                            one folder per additional user
    settings.json, chats_archive.json, projects_archive.json,
    global_mcp_servers.json, mcp_marketplaces.json + mcp-marketplaces/,
    disabled_items.json, model_organization.json, excluded_scan_paths.json,
    appearance.json, granted_folders.json, hooks.json,
    Hooks/*.json, tools/*.json,
    skills/, agents/, marketplaces/, plugins/installed_skills.json
```

Shared across profiles by design: downloaded model weights and the install
registry (`~/.turbospark/models`, `installed.json`, the Rust catalog), the
Keychain server API key, and the UI language (a `@AppStorage` key). Model
favorites, nicknames and tags ARE per profile (`model_organization.json`).
Sensitive plugin hook options use Keychain accounts derived from the
profile-aware hook storage path, so unlike the server key they remain isolated
between profiles.

## How the seam works

`AppStorageRoot.directory` is the one place every first-party store derives
its paths from, so pointing it at the profile folder is the whole
per-profile mechanism for settings, history, projects, MCP, favorites,
appearance, hooks and app-scope tools. It resolves as:

1. `TURBOSPARK_STATE_DIR` override or a test host: the pinned root, profiles
   bypassed entirely (this is what keeps the test suite out of real data,
   and why profile tests can exercise only the pure helpers).
2. Default user active: `machineRoot`, unchanged.
3. Otherwise: `machineRoot/profiles/<activeProfileID>/`.

Content outside that root follows `UserProfileStore.userScopeSubdirectory`
instead: `~/.turbospark/<relative>` for the Default user,
`profiles/<id>/<relative>` for anyone else. SkillManager, AgentManager,
CustomToolManager, SkillMarketplaceManager, PluginManager,
PluginMarketplaceManager, PluginLedgerStore and hook discovery all route
through it.

**Which profile a run belongs to is resolved ONCE per process**, because
`AppStorageRoot.directory` is a cached `static let` every store hangs off.
Precedence: `-TurboSparkProfile <id>` on the command line, then
`TURBOSPARK_PROFILE` in the environment, then the persisted
`activeProfileID`, then the Default user. An id that names no row in the
registry (deleted elsewhere, hand-edited file, the Default id itself)
resolves to the Default user; it never invents a folder.

## Switching

Switching is SAVE AND RELAUNCH, deliberately. `switchToProfile` refuses
while a generation or install is running, saves the registry choice, runs
the same ordered flush as quit (`AppModel.shutdown()`), re-execs
`Bundle.main.executableURL` with `TURBOSPARK_PROFILE` pinned in the child
environment, and exits. A live in-app switch would mean re-pointing the
cached root and re-creating every `.shared` store; nothing here tries. The
Settings pane asks for confirmation first and says what a switch means.

Deletion moves the profile's folder to the Trash (`FileManager.trashItem`)
BEFORE saving the registry, so a failed trash leaves the profile named
rather than forgotten. The active profile and the Default user cannot be
deleted; switch away first. The pane's delete confirmation enumerates the
inventory so the cost is read before it is paid: the folder holds the
user's settings, chat history, projects and agents, global MCP servers and
marketplaces, skills, plugins, custom tools, hooks, memory, and model
favorites, while downloaded models, the install registry, the Keychain
server API key, and the UI language stay shared. Restoring a trashed
folder does not re-register the profile; the registry row is gone with the
save, so the Trash is for recovering files by hand only.

## Entry points

| File | Holds |
|---|---|
| `State/UserProfile.swift` | `UserProfile`, `UserProfileRegistry`, `UserProfileStore` (registry IO, resolution precedence, path math, mutation rules) |
| `State/AppStorageRoot.swift` | `machineRoot` vs profile-aware `directory` |
| `State/AppModel+Profiles.swift` | the UI-facing half: create/rename/delete/switch, toasts, relaunch |
| `Components/ProfilesSettingsPaneView.swift` | the Settings pane |
| `Tests/TurboSparkAppTests/UserProfileTests.swift` | registry semantics, precedence, path math, mutation rules (all against the pure helpers) |

## Out of scope in this version

Passwords or auth of any kind; live switching without a relaunch;
per-profile copies of downloaded models; a per-profile Keychain server key;
copying an existing profile's settings at creation (new profiles start
fresh, which every store already handles as a first run); migrating the
Default user's `~/.turbospark` content into a folder of its own.
