# Stage 5: Initial Zoom Out Review (ZP-05)

Target: docs/designs/zcode-parity/ZP-05-builtin-marketplace-seeding.md (full file, 182 lines)

## Part A: egm-zoom-out (ZO)

Verdict: context is sufficient, but the design adds more machinery than the problem needs and one reuse claim does not hold. No Blockers.

ZO-1 (Major). Lines 42-46, 62-65, 98-102, 146-149. cdnZip has no stated consumer. Day-one builtins (ZP-20, ZP-18) ship in the bundle and use seeding; line 50 says there is no first-party CDN; existing url/github/git/directory kinds cover user-added marketplaces. Open Question 2 is undecided because nothing needs it. Fix: split; ship seeding now, defer cdnZip or name a concrete consumer.

ZO-2 (Major). Lines 99-100. "Reuse ranged-download patterns from crates/repack" points at a Rust crate built for multi-GB weights, not callable from Swift. State which Swift code is reused (`SteeringVectorDownload.swift`, `ManagedAssetStore.swift`) or a plain size-capped URLSession download plus sha256. Zip extraction safety (zip-slip, size and entry caps, symlinks) is not mentioned.

ZO-3 (Major). Lines 104-107, 142-145. Seeding is not routed through the existing install path. `MarketplaceSource.directory(path:)` and the stage-move-ledger flow in `PluginMarketplaceManager.swift` already exist. Make the builtin row a directory-source marketplace on the bundle resources. The spec never says why the ledger cannot record a seeded flag. Also unstated: whether `builtin` is a persisted source kind, and whether the name collides with the name reservation in SWIFT_PLUGINS.md.

ZO-4 (Major). Lines 35-41, 94-97, 104-106. Seed lock, suppression storage and version policy are under-specified. A file lock for a single-process app needs a named reason (single-instance is owned by desktop-shell-hardening). Suppression in settings.json is a second store beside the ledger. No policy when a newer bundled plugin version ships (overwrite, skip, compare).

ZO-5 (Nit). Lines 119-125. Alternative B (bundled git checkout) is a strawman; the real alternative is the directory-source approach.

ZO-6 (Nit). Lines 158-164. Mutation-check bullets are test-case detail, not design altitude.

ZO-7 (Nit). Lines 135-136; INDEX sequencing. INDEX puts ZP-05 at step 4, after ZP-20 and ZP-18 (its consumers). An always-present empty official row has no user value before then. Fix: note a test fixture as first builtin, or reorder; hide the row when empty.

Checked and fine: non-goals exclude a first-party CDN and auto-update; symlink alternative correctly rejected; trust-gate statement is sound.

## Part B: egm-inventive-catalyst (IC)

No Blockers.

IC-1 (Major). Line 99; Target Files item 3. Downloader reuse names a Rust crate in a Swift-only plan. Fix: URLSession download to temp file, stream sha256 with CryptoKit, hard size cap, and add zip extraction bounds (entry count, uncompressed size, traversal rejection); digest check before extraction.

IC-2 (Major). Lines 62-65; NFR Security. If the catalog row and the zip share an origin, the digest only detects corruption. Fix: say what the digest defends against; require HTTPS, refuse cross-host redirects; digest counts when the row comes from a trusted source (bundled row or user-added marketplace); optionally pin at add time.

IC-3 (Major). Lines 92-96, 104-107. Split-brain state: seed marker in the ledger, suppression in settings.json. A crash between writes leaves a plugin re-seeded or permanently suppressed. Restore with a changed bundled version is undefined. Fix: one durable per-builtin ledger record (seeded, suppressed, version) with idempotent startup reconcile and an explicit version-bump policy.

IC-4 (Major). Line 91; NFR Rollout. The seed lock has no stated scope or failure path (stale lock, bad bundle resource, disk full, partial copy). Fix: seed via the stage-move-rollback installer, off the launch critical path, bounded lock timeout, visible retryable state on the row.

IC-5 (Nit). Line 148; line 98. Persisting `builtin` and `cdnZip` cases changes the settings schema and older versions fail to decode. Fix: keep builtin as an app-owned non-persisted row; add only cdnZip; add a downgrade note.

IC-6 (Nit). Lines 17-22, 57-61. Mechanism built before any consumer; cdnZip leans YAGNI; sequencing contradicts "ZP-05 delivers for ZP-20". Fix: ship seeding first, name a cdnZip consumer, fix the order.

IC-7 (Nit). Line 36; SWIFT_PLUGINS.md line 51. Confirm the marketplace name `turbospark-plugins-builtin` does not collide with the reserved plugin names `builtin` and `inline`; state whether seeded builtins with hooks still need SHA-256 approval on first launch.
