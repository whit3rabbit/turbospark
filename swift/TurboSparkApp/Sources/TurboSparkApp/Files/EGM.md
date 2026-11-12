# Files

## Overview
Everything about turning files, folders and typed `@` references into prompt attachments, and showing them back. `AttachmentImporter` extracts documents off the main actor and appends `AppPromptAttachment` values to a chat. `MentionResolver` and `ContextReferenceParser` turn `@path`, `@file:`, `@folder:`, `@diff`, `@staged`, `@git:N` and `@url:` tokens in a submitted draft into attachments. `ProjectFileIndex` feeds the `@` autocomplete. The remaining files render attachments: the Files section list, a preview pane, shared leaf renderers, and a cached thumbnail store. `FileSystemScanRules` is the one skip list shared by both directory walks [FileSystemScanRules.swift:FileSystemScanRules].

## Interface
- Import [AttachmentImporter.swift:AttachmentImporter]:
  - `importDocuments(_:into:chatID:allowDuringSubmission:)` returns an `Outcome` (`importedCount`, `failures`, `importedIDs`, `errorText`).
  - `importFolder(_:into:chatID:maxFiles:allowDuringSubmission:)` walks a folder (default 80 files) and delegates to `importDocuments`.
  - `collectFolderURLs(from:maxFiles:)` is the nonisolated, bounded walk.
- Mentions:
  - `MentionResolver` [MentionResolver.swift:MentionResolver]: `draftContainsMentions`, `mentionTokens(in:)`, `mentionTokensDetailed(in:)` (returns `MentionToken`), `resolveURL(for:projectRoot:)`, `resolveMentions(in:projectRoot:chatID:into:)` (returns failure lines), `sliceImportedAttachments(ids:to:chatID:into:)`.
  - `ContextReference`, `ParsedContextReference` and `ContextReferenceParser.parse(token:quoted:)`, `stripTrailingPunctuation`, `splitLineRange`, `gitCommitLimit` [ContextReferenceParser.swift:ContextReferenceParser].
  - `resolveGitReference` and `resolveWebReference`, extensions on `MentionResolver` [ContextReferenceContent.swift:resolveGitReference].
- Index: `ProjectFileEntry`, and `ProjectFileIndex.shared` with `cachedEntries(for:)`, `entries(for:)`, `invalidate()`, `scan(root:maxEntries:maxDepth:)`, `freshnessInterval` (20 s), `maxEntries` (2000), `maxDepth` (8) [ProjectFileIndex.swift:ProjectFileIndex].
- Scan rules: `FileSystemScanRules.skipDirectoryNames`, `skipFileNames` [FileSystemScanRules.swift:FileSystemScanRules].
- Views and stores:
  - `FilesSectionView` is the all-chats attachment list with search and Add files; `FileRowView` is private [FilesSectionView.swift:FilesSectionView].
  - `FilePreviewView` is the right-hand preview pane with a Model-text toggle and a maximize sheet [FilePreviewView.swift:FilePreviewView].
  - `ImagePreviewView`, `PDFDocumentView`, `QuickLookPreviewView` are the shared leaf renderers [FilePreviewLeaves.swift:ImagePreviewView].
  - `AttachmentThumbnailStore.shared` (`thumbnail(for:pixelSize:)`, `cacheKey(path:modified:pixelSize:)`) and `AttachmentThumbnailView` [AttachmentThumbnailStore.swift:AttachmentThumbnailStore].
- Child contracts: none (no subdirectories).

## Invariants
- Extraction runs on a detached task and only the attachment append touches the main actor. Images are never extracted: they are stored by path with empty text and `formatLabel` "Image" [AttachmentImporter.swift:importDocuments].
- The over-context check at import is a warning toast, not a gate. The hard bounds are the extractor cap and the send-time fit check [AttachmentImporter.swift:importDocuments].
- `allowDuringSubmission` must be true only for `MentionResolver`, which imports while a submission is already in flight. It routes to `appendPromptAttachmentDuringSubmission` instead of the guarded `addPromptAttachment` [AttachmentImporter.swift:importDocuments].
- The folder walk is bounded and contained: max depth 12, 40 MB cumulative (an oversized file is skipped, not a walk stop, though the first file is always taken), an extension allow-list, and `maxFiles`. Paths whose canonical form escapes the root via `PathContainment` are skipped and not descended. Symlinked directories are never descended or recorded as visited. The shared skip list is checked against canonical and raw path components. `ToolRiskClassifier.isSensitivePath` refuses credentials and keys on both raw and canonical paths [AttachmentImporter.swift:collectFolderURLs].
- `ProjectFileIndex.scan` applies the same containment, symlink, skip-list and sensitive-path rules, plus a depth cap and entry cap. Hidden files are offered, and symlinks are skipped before their target is marked visited [ProjectFileIndex.swift:scan]. The autocomplete must not offer what the resolver refuses.
- `ProjectFileIndex` is `@MainActor`. It caches by canonical root path with a 20 s window, and concurrent callers share one in-flight `Task` per root [ProjectFileIndex.swift:entries].
- The `@token` stays in the message text, and an unresolved bare mention is silent. Only explicit scheme failures toast [MentionResolver.swift:MentionResolver].
- Mention matching requires start-of-text or whitespace before `@` (so emails do not match). `@@` is skipped, tokens are deduplicated, and quoted tokens are always literal paths, never schemes [MentionResolver.swift:mentionTokensDetailed].
- Mention paths must resolve inside the project root via `PathContainment.resolvedIfContained`. Absolute paths, traversal and symlinks out of the project are refused. The sensitive-path check runs on both the typed text and the resolved path [MentionResolver.swift:resolveURL].
- Parser order is fixed: quoted means path, then `diff`/`staged`, `git:N` (clamped 1...10), `url:` (http or https only), `file:`, `folder:`, bare path. Trailing `,.;!?` is stripped before scheme matching. An invalid line range drops to the full file [ContextReferenceParser.swift:parse].
- When a line-sliced path does not exist, resolution falls back to `unslicedPath`, so a file named `notes:2024` still mentions [MentionResolver.swift:resolvePathReference].
- Line slicing is 1-indexed and inclusive. A start past the last line leaves the full text, an end past the last line is clamped and sets `wasTruncatedDuringExtraction`. It then calls `persistChats()` and `updateTokenEstimate()` [MentionResolver.swift:sliceImportedAttachments].
- `@url:` is refused when `model.webSearchEnabled` is false. Fetched text is capped at `DocumentTextExtractor.maximumExtractedCharacters` [ContextReferenceContent.swift:resolveWebReference].
- Git references run only with a non-empty project root, always with `--no-ext-diff --no-textconv`, and output is capped by `AppModel.limitDiffLines` [ContextReferenceContent.swift:appendGitAttachment].
- Thumbnails request only `.thumbnail` and `.lowQualityThumbnail`, never `.icon`, so nil means nil and the caller's themed SF Symbol stays. The cache key includes path, modification date and pixel size. Failures are remembered in `failedKeys`. `NSCache` count limit is 256 [AttachmentThumbnailStore.swift:thumbnail].
- `QuickLookPreviewView` must close its `QLPreviewView` in `dismantleNSView`, or the preview session and helper process leak. It returns a container view so a failable `QLPreviewView` init degrades to an empty area [FilePreviewLeaves.swift:QuickLookPreviewView].
- The preview shows the QuickLook alternative only when extracted text is empty and the source exists. The Model-text toggle appears whenever `content` renders something other than extracted text (`hasRenderedAlternative`). Long extractions are gated behind a head-and-tail `MessageContentPreview` until "View all" [FilePreviewView.swift:hasRenderedAlternative].
- Add files and Remove/Detach are disabled while `model.isRunning`. Add files is also disabled while extracting [FilesSectionView.swift:addFilesButton].
- UI strings use `Text(..., bundle: .module)` or `AppStrings.localized` per the app localization gates.

## Side Effects
- Disk reads: document extraction via `DocumentTextExtractor.extract`, folder and project enumeration through `FileManager`, resource values, `NSImage(contentsOf:)`, PDF loading [AttachmentImporter.swift:importDocuments].
- Disk writes: copies sources into `ManagedAssetStore.shared` [AttachmentImporter.swift:importDocuments].
- Security-scoped access: `startAccessingSecurityScopedResource` on an imported folder, released in `defer` [AttachmentImporter.swift:importFolder].
- Process spawn: `AppModel.runGit` for `@diff`, `@staged` and `@git:N` [ContextReferenceContent.swift:appendGitAttachment].
- Network: `WebFetchExecutor.fetch` for `@url:` [ContextReferenceContent.swift:resolveWebReference].
- Model and UI state: mutates chat attachments (`addPromptAttachment`, `appendPromptAttachmentDuringSubmission`, `draftAttachments`, `persistChats`, `updateTokenEstimate`), shows toasts, calls `selectChat`, `showPreview`, `dismissPreview`, `revealAttachmentInFinder`, `openAttachmentExternally`, `removeAttachment`, and sets `activeSection` [FilesSectionView.swift:FileRowView].
- Tasks: detached tasks for extraction and for the project walk [AttachmentImporter.swift:importDocuments].
- In-memory caches: `NSCache` of thumbnails, `failedKeys`, and the `ProjectFileIndex` cache with its in-flight task table. Thumbnails use `QLThumbnailGenerator.shared` [AttachmentThumbnailStore.swift:AttachmentThumbnailStore].
- Presents a `fileImporter` and a maximize sheet [FilesSectionView.swift:FilesSectionView].
- No Keychain, UserDefaults or NotificationCenter use in this directory.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
