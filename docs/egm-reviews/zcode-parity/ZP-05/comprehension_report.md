## What this proposes

The Swift app would get an official, built-in marketplace row named `turbospark-plugins-builtin`. At app startup, a new BuiltinPluginSeeder copies plugins shipped inside the app bundle into the per-profile plugin root. It does this under a per-profile file lock and records a seed marker for each plugin. If a user uninstalls a builtin plugin, a suppression record is saved in settings.json so the next launch does not re-seed it. Choosing Install on the builtin row clears that record and reinstalls from the bundle. Separately, a new `cdnZip(url, sha256)` source kind would download a zip from a static host, check its sha256 against a declared digest, and then install it through the existing stage, move and rollback installer. A digest mismatch refuses the install with a named error and leaves no partial state. TurboSpark hosts no CDN or catalog itself.

## How the existing system works, per this document

`MarketplaceSource` currently supports only url, github, git and directory sources. There is no builtin registry or seed directory, and the docs list those as out of scope. Installs always come from marketplaces the user adds. An existing atomic installer stages files, moves them into place and rolls back on failure. A per-plugin install ledger records installs. A plugin trust model requires SHA-256 approval for plugin hooks. Plugin auto-update does not exist, so updating means reinstalling by hand. A marketplace sheet UI renders marketplace rows. The document never says how "installed" is represented, so the claim that a restart "would re-seed it" is the only description of seeding today.

## Findings

CO-1 (Major). Technical Plan and Goals (seed marker "in the ledger", "seed lock"). The ledger is never defined: existing per-plugin install ledger or a new one, its format and location. The lock file location and behavior when held by another process are not stated. Interaction of marker and suppression is unclear (marker present, no suppression, files manually deleted).

CO-2 (Major). Seeding behavior, Goals, CUJ 1. Nothing says what happens on app upgrade when the bundled version of a seeded plugin changes. The marker does not say whether it records a version. A bundled plugin has no marketplace to reinstall from except the builtin row, so stale seeded copies may stay forever.

CO-3 (Major). Technical Plan diagram versus Alternative B. The diagram shows an expanded directory `Plugins/Builtin/<id>/...`; Alternative B's rejection says "bundle zip resources are lighter and digest-verifiable". Whether builtins are directories or zips, and whether seeding verifies them, is unclear.

CO-4 (Major). Open Questions (CDN zip manifest) versus the `cdnZip(url, sha256)` signature. The signature puts the digest on the source; the lean answer puts it in the marketplace row per plugin. A row listing several plugins cannot use a source with one digest. The question is listed open but the plan and tests build on the lean option.

CO-5 (Major). Target Files item 2 versus Problem and Goals. A `builtin` source kind is added but nothing explains how it differs from the seeder, what data it carries, or how Install on the builtin row reaches the seeder instead of the normal installer.

CO-6 (Major). NFR Security and Goals. The declared digest comes from the same row as the URL, so it only protects against transport tampering or corruption, not a compromised entry. Unspecified: download size limit, zip extraction safety (path traversal, symlinks), the named error (never named), partial-download behavior. The pointer to the Rust `ranged_download` crate does not say which patterns apply to a Swift component.

CO-7 (Major). Goals and Technical Plan (suppression in settings.json). Not stated whether settings are per profile or global while seeding is per profile. Behavior when settings are deleted, and what the UI shows for a suppressed builtin, are unspecified.

CO-8 (Minor). Problem, Open Questions, Overview. References to ZP-18, ZP-20, "the fleet", "guide skills pack", finding 4.2 and 4.9 are not summarized; CUJ 1 relies on "guide skills" anyway.

CO-9 (Nit). Terms used without definition: seed lock, profile, official segment, stage-move-rollback installer, ledger, marketplace sheet, PluginManager, trust gates.

CO-10 (Nit). Verification Plan. The concurrent-seed test appears only in a mutation check, not in the listed cases. `BuiltinPluginSeederTests.swift` also holds cdnZip tests, which are not seeder behavior. The local static server for the manual check has no setup described.

CO-11 (Nit). Rollout. An always-present empty official row ships with no stated user reason; unclear whether intended UX or a side effect.
