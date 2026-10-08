import Foundation

/// Unified diff and patch parser and applicator for Swift workspace tools.
public enum ApplyPatchExecutor {
    /// Applies a unified diff patch string sequentially against files in the workspace root.
    ///
    /// Async because of the freshness bookkeeping, not the file I/O: the
    /// snapshot store is an actor, and each successful operation must
    /// record (or forget) its hash there or the NEXT tool call on the same
    /// file reads stale -- `edit_file`'s check would then refuse every
    /// post-patch edit until the model re-read a file this patch had just
    /// written.
    public static func apply(patchText: String, rootURL: URL) async throws -> ApplyPatchOutput {
        var appliedOps: [AppliedPatchOperation] = []
        var diffSummaries: [GitDiffSummary] = []

        // Split on "\n" only, dropping one trailing "\r" per line. Splitting on
        // CharacterSet.newlines treated "\r\n" as two separators, which
        // doubled the line array of a CRLF file and replaced form feeds and
        // U+2028 with newlines on rewrite.
        let lines = patchText.components(separatedBy: "\n").map { line in
            line.hasSuffix("\r") ? String(line.dropLast()) : line
        }
        var currentFile: String?
        var currentHunkLines: [String] = []
        var isNewFile = false
        var isDeleteFile = false

        // Two-phase apply: every file's result is computed (and every
        // staleness and context check run) before anything is written, so a
        // stale hunk in file 3 cannot leave files 1 and 2 already rewritten
        // for the model to trip over when it resends the whole patch.
        var pending: [PendingChange] = []
        var snapshotActions: [(url: URL, content: String?)] = []
        // Planned state of files touched earlier in this same patch, so a
        // second section for one path builds on the first. nil = deleted.
        var planned: [String: String?] = [:]

        func plannedExists(_ url: URL) -> Bool {
            if let entry = planned[url.path] { return entry != nil }
            return FileManager.default.fileExists(atPath: url.path)
        }

        func flushCurrentFile() async throws {
            // Reset ALL per-file state on every exit path, including the
            // early return below. Without this, `isDeleteFile` from a file
            // whose `currentFile` was never set (e.g. because the delete
            // header wasn't recognized) survived into the NEXT file's
            // flush, which then got `removeItem`'d instead of edited.
            defer {
                currentFile = nil
                currentHunkLines.removeAll()
                isNewFile = false
                isDeleteFile = false
            }
            guard let fileRelPath = currentFile else { return }
            let secureURL = try AppToolRegistry.resolveSecurePath(relPath: fileRelPath, rootURL: rootURL)
            // A delete is exactly as destructive as a write; both go through
            // the same sandbox check `writeFile`/`editFile` already use.
            try AppToolSandbox.validateWritePath(secureURL, rootURL: rootURL)

            if isDeleteFile {
                if plannedExists(secureURL) {
                    // The delete branch is the one place a patch has NO
                    // content-level protection: there are no context lines
                    // to verify, so the removal happens whatever is on
                    // disk. The snapshot store is the only freshness check
                    // available, so a file the model has read and that has
                    // since changed on disk is refused, mirroring
                    // `edit_file`/`writeFile`. An untracked file (never
                    // read this session) still deletes: refusing that
                    // would make `git diff | apply_patch` flows impossible
                    // to express, and the patch itself is the user's
                    // instruction there.
                    if await FileSnapshotStore.shared.isTracked(url: secureURL),
                        await FileSnapshotStore.shared.isStale(url: secureURL) {
                        throw NSError(domain: "TurboSparkTool", code: 24, userInfo: [
                            NSLocalizedDescriptionKey: "File '\(fileRelPath)' has been modified on disk "
                                + "since it was last read, so the patch's delete was refused. "
                                + "Re-read the file and regenerate the patch."
                        ])
                    }
                    pending.append(.delete(url: secureURL))
                    planned[secureURL.path] = .some(nil)
                    snapshotActions.append((secureURL, nil))
                    appliedOps.append(AppliedPatchOperation(type: "delete", resource: fileRelPath, target: secureURL.path))
                    diffSummaries.append(GitDiffSummary(
                        filename: fileRelPath,
                        status: "deleted",
                        additions: 0,
                        deletions: 0,
                        changes: 0,
                        patch: currentHunkLines.joined(separator: "\n")
                    ))
                }
            } else if isNewFile {
                // A `--- /dev/null` header asserts the file does not exist.
                // Overwriting here would skip the read-before-write gate
                // `write_file` enforces and lose user edits while reporting
                // success; `git apply` refuses, so do the same. `lstat` via
                // attributesOfItem so a dangling symlink also counts.
                if planned[secureURL.path].map({ $0 != nil })
                    ?? ((try? FileManager.default.attributesOfItem(atPath: secureURL.path)) != nil)
                {
                    throw NSError(domain: "TurboSparkTool", code: 26, userInfo: [
                        NSLocalizedDescriptionKey: "Patch creates '\(fileRelPath)' but it already exists. "
                            + "Refusing to overwrite; use a normal update hunk against the current content."
                    ])
                }
                // Collect added lines
                var contentLines: [String] = []
                for hunkLine in currentHunkLines {
                    if hunkLine.hasPrefix("+") {
                        contentLines.append(String(hunkLine.dropFirst()))
                    }
                }
                let text = contentLines.joined(separator: "\n")
                pending.append(.write(url: secureURL, text: text))
                planned[secureURL.path] = .some(text)
                snapshotActions.append((secureURL, text))
                appliedOps.append(AppliedPatchOperation(type: "add", resource: fileRelPath, target: secureURL.path))
                diffSummaries.append(GitDiffSummary(
                    filename: fileRelPath,
                    status: "added",
                    additions: contentLines.count,
                    deletions: 0,
                    changes: contentLines.count,
                    patch: currentHunkLines.joined(separator: "\n")
                ))
            } else {
                // Update file
                var originalText = ""
                if let entry = planned[secureURL.path] {
                    originalText = entry ?? ""
                } else if FileManager.default.fileExists(atPath: secureURL.path) {
                    originalText = try String(contentsOf: secureURL, encoding: .utf8)
                }
                let updatedText = try applyHunkLines(original: originalText, hunkLines: currentHunkLines)
                pending.append(.write(url: secureURL, text: updatedText))
                planned[secureURL.path] = .some(updatedText)
                // No staleness gate on the update branch, on purpose: its
                // hunk context lines are verified against the file
                // (`applyHunkLines` throws on any mismatch), which is a
                // STRONGER freshness check than the snapshot hash -- it
                // proves the patched region matches what the patch was
                // generated from, whatever else changed elsewhere. Record
                // the result so the next tool call on this file compares
                // against the post-patch bytes.
                snapshotActions.append((secureURL, updatedText))
                appliedOps.append(AppliedPatchOperation(type: "update", resource: fileRelPath, target: secureURL.path))
                diffSummaries.append(GitDiffSummary(
                    filename: fileRelPath,
                    status: "modified",
                    additions: currentHunkLines.filter { $0.hasPrefix("+") }.count,
                    deletions: currentHunkLines.filter { $0.hasPrefix("-") }.count,
                    changes: currentHunkLines.count,
                    patch: currentHunkLines.joined(separator: "\n")
                ))
            }
        }

        // Lines still owed to the current hunk, from its `@@ -a,b +c,d @@`
        // header. Inside a hunk EVERY line is body: a removed SQL/Lua comment
        // `-- old` is spelled `--- old` and an added `++ x` is `+++ x`, and
        // neither is a file header. nil outside a hunk or for a header the
        // regex cannot read (then the old prefix-based behavior applies).
        var remainingOld = 0
        var remainingNew = 0
        func inHunkBody() -> Bool { remainingOld > 0 || remainingNew > 0 }

        for (lineIndex, line) in lines.enumerated() {
            // Models miscount hunk lengths. A `--- x` / `+++ y` / `@@` run is
            // a file header whatever the counts still owe, so a too-large
            // count cannot swallow the next file's header as body.
            if inHunkBody(), line.hasPrefix("--- "), lineIndex + 2 < lines.count,
               lines[lineIndex + 1].hasPrefix("+++ "), lines[lineIndex + 2].hasPrefix("@@") {
                remainingOld = 0
                remainingNew = 0
            }
            if inHunkBody() {
                currentHunkLines.append(line)
                if line.hasPrefix("\\") {
                    // "\ No newline at end of file" is not a content line.
                } else if line.hasPrefix("-") {
                    remainingOld -= 1
                } else if line.hasPrefix("+") {
                    remainingNew -= 1
                } else {
                    // Context (a blank line is context with its space stripped).
                    remainingOld -= 1
                    remainingNew -= 1
                }
                continue
            }
            if line.hasPrefix("diff --git ") {
                try await flushCurrentFile()
            } else if line.hasPrefix("--- ") {
                // A plain unified diff (no `diff --git` lines) starts the next
                // file with `--- `: finish the previous file first, or its
                // hunks merge into this one.
                if !currentHunkLines.isEmpty { try await flushCurrentFile() }
                var pathPart = line.replacingOccurrences(of: "--- ", with: "").trimmingCharacters(in: .whitespaces)
                // `diff -u` appends a tab and timestamp after the path.
                if let tab = pathPart.firstIndex(of: "\t") { pathPart = String(pathPart[..<tab]) }
                if pathPart == "/dev/null" {
                    isNewFile = true
                } else {
                    // Also the ONLY place a delete hunk's path is stated
                    // (its `+++` line is `/dev/null`), so a delete with no
                    // `currentFile` ever set is what let a stale
                    // `isDeleteFile` flag attach itself to the next file.
                    let cleaned = pathPart.hasPrefix("a/") ? String(pathPart.dropFirst(2)) : pathPart
                    currentFile = cleaned
                }
            } else if line.hasPrefix("+++ ") {
                var pathPart = line.replacingOccurrences(of: "+++ ", with: "").trimmingCharacters(in: .whitespaces)
                if let tab = pathPart.firstIndex(of: "\t") { pathPart = String(pathPart[..<tab]) }
                if pathPart == "/dev/null" {
                    isDeleteFile = true
                } else {
                    let cleaned = pathPart.hasPrefix("b/") ? String(pathPart.dropFirst(2)) : pathPart
                    currentFile = cleaned
                }
            } else if line.hasPrefix("@@") || currentFile != nil {
                currentHunkLines.append(line)
                if line.hasPrefix("@@"),
                   let match = hunkCountsRegex.firstMatch(
                    in: line, options: [], range: NSRange(location: 0, length: (line as NSString).length)) {
                    func count(_ group: Int) -> Int {
                        let range = match.range(at: group)
                        guard range.location != NSNotFound else { return 1 }
                        return Int((line as NSString).substring(with: range)) ?? 1
                    }
                    remainingOld = count(1)
                    remainingNew = count(2)
                }
            }
        }

        try await flushCurrentFile()

        // A patch in a format this parser does not understand (notably the
        // `*** Begin Patch` form) yields no operations. Reporting "Applied
        // patch to 0 file(s)" as success made the model believe edits landed.
        guard !appliedOps.isEmpty else {
            throw NSError(domain: "TurboSparkTool", code: 27, userInfo: [
                NSLocalizedDescriptionKey: "The patch contained no file operations and NOTHING was applied. "
                    + "apply_patch expects a unified diff with `--- a/path` and `+++ b/path` headers "
                    + "(use `--- /dev/null` to add a file and `+++ /dev/null` to delete one); "
                    + "the `*** Begin Patch` format is not supported."
            ])
        }

        try commit(pending)
        for action in snapshotActions {
            if let content = action.content {
                await FileSnapshotStore.shared.recordSnapshot(url: action.url, content: content)
            } else {
                await FileSnapshotStore.shared.removeSnapshot(url: action.url)
            }
        }

        let summary = "Applied patch to \(appliedOps.count) file(s):\n" + appliedOps.map { "  \($0.type.uppercased()) \($0.resource)" }.joined(separator: "\n")
        return ApplyPatchOutput(applied: appliedOps, files: diffSummaries, summary: summary)
    }
    
    /// Writes the planned changes, restoring the prior bytes of every
    /// already-touched path if a later one fails (as MultiEditExecutor does).
    private static func commit(_ changes: [PendingChange]) throws {
        let fm = FileManager.default
        // path -> original bytes (nil = did not exist), first touch only.
        var originals: [(url: URL, data: Data?)] = []
        var seen: Set<String> = []
        func remember(_ url: URL) {
            guard seen.insert(url.path).inserted else { return }
            originals.append((url, try? Data(contentsOf: url)))
        }
        do {
            for change in changes {
                switch change {
                case .write(let url, let text):
                    remember(url)
                    try fm.createDirectory(
                        at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try text.write(to: url, atomically: true, encoding: .utf8)
                case .delete(let url):
                    remember(url)
                    if fm.fileExists(atPath: url.path) { try fm.removeItem(at: url) }
                }
            }
        } catch {
            for entry in originals.reversed() {
                if let data = entry.data {
                    try? data.write(to: entry.url, options: .atomic)
                } else {
                    try? fm.removeItem(at: entry.url)
                }
            }
            throw error
        }
    }

    private enum PendingChange {
        case write(url: URL, text: String)
        case delete(url: URL)
    }

    /// Matches a unified diff hunk header: `@@ -oldStart[,oldCount] +newStart[,newCount] @@`,
    /// tolerating trailing context text some diff tools append after the closing `@@`.
    private static let hunkHeaderRegex = try! NSRegularExpression(
        pattern: #"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@"#
    )

    /// Line counts of a hunk header: groups are old count and new count
    /// (each defaulting to 1 when omitted).
    private static let hunkCountsRegex = try! NSRegularExpression(
        pattern: #"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@"#
    )

    /// Applies one or more hunks against `original`, honoring each hunk's
    /// stated starting line rather than assuming every hunk starts at the
    /// top of the file, and verifying that every context/removal line
    /// actually matches the file before touching it.
    ///
    /// The previous version walked the WHOLE original file sequentially from
    /// index 0 regardless of what a hunk's `@@` header said, so any patch
    /// whose hunk did not start at line 1 (i.e. almost all of them) applied
    /// its changes at the wrong offset and silently scrambled the file while
    /// still reporting success.
    private static func applyHunkLines(original: String, hunkLines: [String]) throws -> String {
        // Split on "\n" only. A CRLF file (every line break CRLF) is
        // normalized for matching and restored on output; any other "\r",
        // form feed or U+2028 stays inside its line untouched.
        let usesCRLF = original.contains("\r\n")
            && !original.replacingOccurrences(of: "\r\n", with: "").contains("\n")
        let normalized = usesCRLF ? original.replacingOccurrences(of: "\r\n", with: "\n") : original
        let origLines = normalized.components(separatedBy: "\n")
        var resultLines: [String] = []
        var origIdx = 0
        var i = 0

        while i < hunkLines.count {
            let headerLine = hunkLines[i]
            guard headerLine.hasPrefix("@@") else { i += 1; continue }

            let nsHeader = headerLine as NSString
            guard let match = hunkHeaderRegex.firstMatch(
                in: headerLine, options: [], range: NSRange(location: 0, length: nsHeader.length)
            ), let oldStart = Int(nsHeader.substring(with: match.range(at: 1))) else {
                throw NSError(domain: "TurboSparkTool", code: 20, userInfo: [
                    NSLocalizedDescriptionKey: "Malformed patch hunk header: '\(headerLine)'"
                ])
            }
            let oldCountRange = match.range(at: 2)
            let oldCount = oldCountRange.location != NSNotFound ? (Int(nsHeader.substring(with: oldCountRange)) ?? 1) : 1

            // A hunk with a zero old-side count is a pure insertion; per the
            // unified diff convention its `oldStart` already IS the count of
            // preceding lines (not a 1-indexed line number to convert), e.g.
            // `@@ -0,0 +1,3 @@` inserts before line 1. A nonzero count means
            // `oldStart` is the ordinary 1-indexed line of the hunk's first
            // context/removed line.
            let targetIdx = oldCount == 0 ? oldStart : max(0, oldStart - 1)

            guard targetIdx >= origIdx else {
                throw NSError(domain: "TurboSparkTool", code: 21, userInfo: [
                    NSLocalizedDescriptionKey: "Patch hunk '\(headerLine)' is out of order or overlaps the previous hunk."
                ])
            }
            guard targetIdx <= origLines.count else {
                throw NSError(domain: "TurboSparkTool", code: 22, userInfo: [
                    NSLocalizedDescriptionKey: "Patch hunk '\(headerLine)' starts past the end of the file (\(origLines.count) lines)."
                ])
            }

            // Copy everything between the previous hunk (or file start) and
            // this hunk's start verbatim.
            while origIdx < targetIdx {
                resultLines.append(origLines[origIdx])
                origIdx += 1
            }

            i += 1
            while i < hunkLines.count, !hunkLines[i].hasPrefix("@@") {
                let hLine = hunkLines[i]
                if hLine.hasPrefix(" ") {
                    let expected = String(hLine.dropFirst())
                    guard origIdx < origLines.count, origLines[origIdx] == expected else {
                        throw NSError(domain: "TurboSparkTool", code: 23, userInfo: [
                            NSLocalizedDescriptionKey: "Patch context mismatch at line \(origIdx + 1): the file has changed since the patch was generated."
                        ])
                    }
                    resultLines.append(origLines[origIdx])
                    origIdx += 1
                } else if hLine.hasPrefix("-") {
                    let expected = String(hLine.dropFirst())
                    guard origIdx < origLines.count, origLines[origIdx] == expected else {
                        throw NSError(domain: "TurboSparkTool", code: 23, userInfo: [
                            NSLocalizedDescriptionKey: "Patch removal mismatch at line \(origIdx + 1): the file has changed since the patch was generated."
                        ])
                    }
                    origIdx += 1
                } else if hLine.hasPrefix("+") {
                    resultLines.append(String(hLine.dropFirst()))
                }
                // Any other line (e.g. "\ No newline at end of file") is skipped.
                i += 1
            }
        }

        while origIdx < origLines.count {
            resultLines.append(origLines[origIdx])
            origIdx += 1
        }

        let joined = resultLines.joined(separator: "\n")
        return usesCRLF ? joined.replacingOccurrences(of: "\n", with: "\r\n") : joined
    }
}
