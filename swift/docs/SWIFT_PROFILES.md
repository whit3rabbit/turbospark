# Swift user profiles

User profiles in TurboSparkApp are bootstrap registry rows plus encrypted
private vaults. Read [PROFILE_VAULT.md](PROFILE_VAULT.md) for the canonical
storage, protection, migration, and export contracts. Shared models, skills,
plugins, hooks, and tool executables keep their existing filesystem layout.

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
  private-vault/                              the Default user's private vault
    security.json, profile.sqlite3, assets/, recovery/
  profiles/<uuid>/                            one folder per additional user
    private-vault/
      security.json, profile.sqlite3, assets/, recovery/
    shared-component configuration and installs (outside the private vault):
    mcp-marketplaces/, hooks.json,
    Hooks/*.json, tools/*.json,
    skills/, agents/, marketplaces/, plugins/installed_skills.json
```

Shared across profiles by design: downloaded model weights and the install
registry (`~/.turbospark/models/{text,image,audio}`, `installed.json`, the Rust catalog), the
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

## Backup export and import

This section describes the legacy version 1 plaintext backup format, retained
only for importing older archives. New exports are either an exact encrypted
`.turbospark-profile` backup or an explicit plaintext open ZIP. Their current
contracts are in [PROFILE_VAULT.md](PROFILE_VAULT.md).

Export writes a plain `.zip` (Finder-openable, no app needed to read it):
the payload plus `turbospark-backup-manifest.json` at the archive root. The
manifest carries a format version, the profile's identity, the layout, the
export time and app version, the sorted contents list, and the category
selection (nil for a whole-profile backup), so a backup is self-describing
and a future restore can refuse what it cannot read.

The legacy export sheet lists the categories a backup can carry -- settings
(including SOUL and personality, which are settings keys), chat history,
projects, model favorites and scan paths, MCP servers and marketplaces,
skills, agents, custom tools, plugins, hooks, memory, and automation data
(cron jobs, steering vectors, tool observations) -- with every category
selected by default and Select All / Clear All at hand. Category selection is
a strict allowlist. Unknown files and directories never enter a partial
backup, which prevents generated images and future private stores from leaking
through an unrelated category. Import reads none of this -- its contents list
already says what arrived -- and restores whatever the archive holds.

The payload depends on the user:

- A non-default profile archives the one folder (`layout:
  profile-folder`); the archive root holds the folder's contents beside the
  manifest.
- The Default user spans two roots (`layout: default-two-root`):
  `app-support/` holds the machine-root stores and `dot-turbospark/` the
  `~/.turbospark` content, each minus the top-level entries that are
  machine-level by design: `profiles.json`, `profiles/`, `models/`, and
  `installed.json`. Each root is copied per top-level entry, so the
  multi-GB shared model downloads are never even read for copy.

Exporting the profile the current run belongs to first flushes the same
store writes the quit path ends with (`persistChats` + `persistSettings`),
so the backup cannot miss the last keystroke; other profiles have no live
writer. Import is the inverse under the same rules: the archive is listed
with `zipinfo` and refused before extraction if the listing is truncated or
if any entry is absolute, starts with `..`, or carries a backslash (zip-slip);
the manifest must be
present, kind-correct, and version-matched; and the payload always restores
into a NEW identity -- a freshly minted UUID and a user-chosen name -- so a
Default backup's `"default"` id can never reach the registry and a
collision with a live profile cannot happen. The two-root layout merges
into that one folder, `dot-turbospark/` first and `app-support/` second, so
the first-party stores win file collisions (`tools/` is the one directory
both roots carry today). The folder is filled before the registry row is
saved, mirroring delete's trash-first ordering: any failure leaves no row
pointing at a partial folder.

Profile names may contain emoji; they survive everywhere (registry, UI,
manifest, and the sanitized suggested file name, which strips path-hostile
punctuation and control characters and caps at 60 whole graphemes). The
one name no profile can take is the reserved "Default" (case-insensitive),
which is what keeps a Default-user backup importable as a distinct row.
Not part of a backup, ever: the Keychain-stored hook secrets (isolated by
design), the shared model downloads, and the install registry.

## Entry points

| File | Holds |
|---|---|
| `State/UserProfile.swift` | `UserProfile`, `UserProfileRegistry`, `UserProfileStore` (registry IO, resolution precedence, path math, mutation rules, the reserved-name rule) |
| `State/AppStorageRoot.swift` | `machineRoot` vs profile-aware `directory` |
| `State/AppModel+Profiles.swift` | the UI-facing half: create/rename/delete/switch, toasts, relaunch |
| `State/ProfileVault.swift` | security manifest, vault session, repository, legacy migration |
| `State/ProfileDatabase.swift` | SQLCipher schema, normalized persistence, FTS, online backup |
| `State/ManagedAssetStore.swift` | chunked encrypted managed assets |
| `State/EncryptedProfileBackup.swift` | exact encrypted backup and verified restore |
| `State/OpenProfileExport.swift` | allowlisted streaming plaintext export |
| `State/ProfileBackup.swift` | legacy v1 backup plus shared category and name helpers |
| `State/ProfileBackupImport.swift` | the zip-slip validator, manifest validation, restore/merge |
| `State/AppModel+ProfileBackup.swift` | the backup panels: export, import pick/sheet/name suggestions, flush-before-export, folder-first registry-last |
| `Components/ProfilesSettingsPaneView.swift` | the Settings pane |
| `Tests/TurboSparkAppTests/UserProfileTests.swift` | registry semantics, precedence, path math, mutation rules (all against the pure helpers) |
| `Tests/TurboSparkAppTests/ProfileBackupTests.swift` | export/import round-trips through real temp dirs and the real archive tools, exclusions, sanitization, zip-slip refusals, merge precedence |

## Out of scope in this version

Live switching without a relaunch; per-profile copies of downloaded models;
a per-profile Keychain server key;
copying an existing profile's settings at creation (new profiles start
fresh, which every store already handles as a first run); migrating the
Default user's `~/.turbospark` content into a folder of its own; live
progress reporting inside a backup run (the pane disables the buttons and
toasts the outcome); excluding `.git` directories inside marketplace
clones from backups (an archive is a faithful copy, so a profile with
cloned marketplaces produces a large one).
