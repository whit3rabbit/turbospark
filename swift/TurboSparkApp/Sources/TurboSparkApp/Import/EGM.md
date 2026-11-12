# Import

## Overview
Two small import helpers. `ChromeBookmarkImporter` reads Chrome bookmarks in a scan, preview, confirm flow and returns bookmark trees for the in-app browser to persist; it is built to read only the exact `Bookmarks` file of profiles found in the current scan, with caps on size and tree shape. `MarkdownFrontmatterParser` is a shared, stateless YAML-frontmatter splitter used for Markdown documents such as Skills, Agents and Prompts [MarkdownFrontmatterParser.swift:MarkdownFrontmatterParser]. Neither type persists anything itself.

## Interface
- Chrome bookmark import [ChromeBookmarkImporter.swift:ChromeBookmarkImporter]:
  - `enumerateProfiles() -> ChromeBookmarkProfileEnumeration`: lists profile directories that hold a regular `Bookmarks` file.
  - `preview(profileIDs:) -> [ChromeBookmarkProfilePreview]`: parses and caches a preview per profile ID.
  - `importConfirmed(selectionsByProfileID:) -> ChromeBookmarkImportResult`: returns `BrowserBookmarkTree` values for the confirmed folder and bookmark IDs.
  - Limits: `maximumBookmarkFileBytes` (= `BrowserSettings.bookmarksSizeLimit`), `maximumProfileCount` 128, `maximumTreeDepth` 32, `maximumTreeNodeCount` 20,000, `maximumBookmarkTitleLength` 1,024, `maximumBookmarkURLLength` 8,192.
  - `init(profileRootURL:fileManager:readFile:)`: root defaults to `~/Library/Application Support/Google/Chrome`; `readFile` is injectable for tests.
- Value types: `ChromeBookmarkProfileCandidate`, `ChromeBookmarkProfileEnumeration`, `ChromeBookmarkProfilePreview`, `ChromeBookmarkImportResult`, and the `ChromeBookmarkImportIssue` enum (profileRootUnavailable, noProfilesFound, bookmarksFileMissing, bookmarksFileUnreadable, bookmarksFileTooLarge, malformedBookmarks, profileUnavailable, unknownSelection) [ChromeBookmarkImporter.swift:ChromeBookmarkImportIssue].
- Frontmatter parsing [MarkdownFrontmatterParser.swift:MarkdownFrontmatterParser]: `normalizedLines`, `stripQuotes`, `parseListField`, `isBlockScalarMarker`, `extractFrontmatterAndBody(from:)`.
- Child contracts: none (no subdirectories).

## Invariants
- Only a file named exactly `Bookmarks` is ever opened, checked by `isAllowedProfileFile`. Profile directories and the file must be regular, non-symlink entries [ChromeBookmarkImporter.swift:enumerateProfiles].
- `preview` and `importConfirmed` accept only profile IDs found by the latest `enumerateProfiles`, which clears both caches. No arbitrary path is accepted; unknown IDs yield `.profileUnavailable` [ChromeBookmarkImporter.swift:preview].
- `importConfirmed` works from the cached preview only (no re-read of disk). Selected IDs not in `selectableIdentifiers` are dropped, counted as skipped and reported as `.unknownSelection`; a profile whose preview carried an issue is skipped [ChromeBookmarkImporter.swift:importConfirmed].
- Selecting a folder includes its whole subtree; a partial selection keeps only the matching bookmarks and subfolders [ChromeBookmarkImporter.swift:filter].
- File size is checked twice, by metadata before reading and by `data.count` after, and over-limit files fail with `.bookmarksFileTooLarge` [ChromeBookmarkImporter.swift:preview].
- Parsing fails closed: malformed JSON or a missing `roots` object returns nil (`.malformedBookmarks`). Depth, node-count, empty or over-long IDs, duplicate IDs, non-folder/url types and unreadable nodes are skipped and counted, never thrown [ChromeBookmarkImporter.swift:ChromeBookmarkTreeParser].
- A bookmark URL is accepted only when it is within the length cap and `BrowserOrigin(url:)` is non-nil; the stored value is `url.absoluteString`. Titles and labels have control characters stripped, and titles are length-capped. An empty title falls back to the host, then the URL [ChromeBookmarkImporter.swift:safeBookmarkURL].
- Only roots `bookmark_bar`, `other`, `synced` are read, with IDs prefixed `root:`, `folder:`, `bookmark:`, so Chrome IDs cannot collide across kinds [ChromeBookmarkImporter.swift:parse].
- Profile display labels strip control characters and `./\`, and are capped at 64 characters [ChromeBookmarkImporter.swift:sanitizedDirectoryLabel].
- The importer is a non-`Sendable` class with mutable caches, so use it from one actor or thread.
- Frontmatter extraction returns `(nil, text)` unless the first line is exactly `---` after trimming and a closing `---` line exists. CR and CRLF are normalized to LF [MarkdownFrontmatterParser.swift:extractFrontmatterAndBody].

## Side Effects
- Disk reads: lists the Chrome profile root, reads resource values, and reads each selected profile's `Bookmarks` file via `readFile` (default `Data(contentsOf:)`) [ChromeBookmarkImporter.swift:preview].
- In-memory caches `profileDirectories` and `cachedPreviews` [ChromeBookmarkImporter.swift:ChromeBookmarkImporter].
- No disk writes, network, Keychain, UserDefaults, process spawn or FFI. `MarkdownFrontmatterParser` has no side effects.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
