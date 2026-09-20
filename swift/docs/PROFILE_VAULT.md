# Encrypted profile vault

TurboSpark stores private profile data in an always-encrypted vault. Profile
protection controls how the vault key is unlocked. It does not switch
encryption on and off.

## Layout

```text
Application Support/TurboSpark/
  profiles.json
  private-vault/                         Default profile
    security.json
    profile.sqlite3
    assets/aa/<opaque-id>.tsasset
    recovery/<legacy-name>.legacy.enc
  profiles/<profile-uuid>/
    private-vault/
      security.json
      profile.sqlite3
      assets/aa/<opaque-id>.tsasset
      recovery/<legacy-name>.legacy.enc
```

Vault directories use mode `0700`; files use `0600`. `profiles.json` is a
bootstrap registry. A protected row contains an opaque profile ID, a
protection flag, and a public label. The real display name is stored in the
encrypted database.

Models, skills, plugins, hooks, tool executables, external repositories, and
external files stay outside the vault. API credentials stay in Keychain. An
external artifact enters the vault only when the user chooses Copy into
profile. Prompt attachments are copied automatically because the chat needs a
stable private copy.

## Keys and authentication

Each profile has a random 256-bit master key. HKDF-SHA256 with separate
purpose labels derives the SQLCipher key, asset encryption keys, keyed asset
IDs, recovery key, and backup keys.

Protected profiles wrap the master key with AES-256-GCM. The wrapping key is
PBKDF2-HMAC-SHA256 with 600,000 iterations and a random 16-byte salt. Recovery
passphrases are normalized to NFC, are not trimmed, and must contain 15 to
1,024 Unicode characters. Changing or disabling protection rewraps the master
key. It does not rewrite the database or assets.

Optional quick unlock stores a second master-key copy in the Data Protection
Keychain with `userPresence` and `WhenUnlockedThisDeviceOnly`. macOS chooses
Touch ID, Apple Watch, or the login password through its system UI. If an
ad-hoc signature cannot create the item, passphrase protection still succeeds
and quick unlock remains off. A Developer ID is not required for local
encryption. It is needed for normal trusted direct distribution and
notarization.

The app relocks protected profiles at launch, session resignation or lock,
and screen sleep. Locking closes SQLCipher, cancels profile work, wipes the
in-memory master key, drops the `AppModel`, and removes decrypted cache files.
The cache is also purged before a vault is opened after launch.

## Database

`ProfileDatabase` opens the official SQLCipher Apple package with a raw
256-bit key. It enables foreign keys, WAL, `synchronous=FULL`, secure deletion,
and SQLCipher memory security. The schema has normalized tables for chats,
messages, alternates, drafts, projects, attachments, artifacts, memory,
embeddings, todos, profile settings, and managed asset metadata. Versioned
JSON payload columns preserve fields that evolve faster than relational
indexes.

FTS5 indexes are inside SQLCipher. Chat search content covers titles,
messages, reasoning, drafts, and extracted attachment text. Memory has its
own encrypted FTS index. Draft writes are delayed by 300 ms. Chat database
writes run on a serialized utility queue, with explicit durability flushes
before lock, migration cleanup, backup, and shutdown. Unchanged chat payloads
and FTS rows are not rewritten.

Profile and project memory use `private_records` plus normalized memory rows.
The Markdown shape remains the portable logical format, but protected content
is no longer left as Markdown files in the profile directory.

SQLCipher Community is used for consumer privacy. TurboSpark makes no FIPS or
regulated-compliance claim. Its BSD notice is bundled at
`Resources/Licenses/SQLCipher.txt`.

## Managed assets

`ManagedAssetStore` encrypts files in 1 MiB AES-256-GCM chunks. Each asset has
a per-asset HKDF key. A random 8-byte prefix plus a 32-bit chunk counter forms
the nonce. The header, keyed content ID, and chunk index are authenticated.
Truncation, appended bytes, header edits, and chunk edits fail authentication.

On-disk names are HMAC-SHA256 content IDs. Original names, MIME types, sizes,
and reference counts are inside SQLCipher. Duplicate content shares one
ciphertext. The database reference change commits before an unreferenced file
is removed.

## Migration

The first unlocked launch imports legacy JSON, profile and project memory,
generated images, and readable attachment sources. JSON and memory imports
are written and verified before any plaintext source is removed. Recovery
copies of legacy private JSON are AES-GCM encrypted inside `recovery/`.

Asset migration rewrites chat references only after every copy succeeds. A
failed or interrupted pass leaves its sources in place. Missing external files
retain their existing metadata and extracted text.

## Export formats

Exact backup uses `.turbospark-profile`. It contains an authenticated,
chunk-encrypted standard ZIP with a consistent SQLCipher online-backup
snapshot, encrypted managed assets, encrypted recovery data, schema versions,
and SHA-256 checksums. The export password wraps the profile master key for
that backup. Device Keychain data is never included. Restore verifies safe
paths, every checksum, SQLCipher integrity, asset inventory, and every asset
authentication tag before adding the new locked profile to `profiles.json`.

Open export uses a standard plaintext ZIP. Settings presents a prominent
warning and requires recent authentication for a protected profile. The ZIP
contains only explicitly selected allowlisted categories. Clear All exports
only the manifest and empty checksum inventory. The stable layout is:

```text
manifest.json
checksums.sha256
chats/<uuid>/chat.json
chats/<uuid>/messages.jsonl
chats/<uuid>/transcript.md
chats/<uuid>/tool-observations/<uuid>.bin
projects/projects.json
settings/settings.json
settings/appearance.json
settings/<other-allowlisted-profile-record>.json
memory/profile/...
memory/projects/...
assets/index.json
assets/generated-images/...
assets/attachments/...
```

The ZIP writer streams each entry. Managed files decrypt directly into the
archive and never enter a plaintext staging directory. Open export does not
fetch URLs and does not include Keychain data, shared components, external
repositories, or external referenced files.

The version 1 plaintext profile backup importer remains for recovery of old
archives. New exact backups and open exports are separate operations.

## Limits

The vault protects private data at rest while locked. It does not protect
against malware, administrators, screenshots, an unlocked process, or files
the user keeps outside the vault. FileVault is still recommended for full-disk
protection. TurboSpark does not promise secure erasure of SSD remnants.

Touch ID availability, login-password fallback, session-lock behavior,
accessibility, save panels, codesigning, and notarization remain real-device
or release gates. Automated tests use injectable stores and do not prove those
system interactions.
