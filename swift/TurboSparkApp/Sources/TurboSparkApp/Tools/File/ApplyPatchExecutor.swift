import Foundation

/// Unified diff and patch parser and applicator for Swift workspace tools.
public enum ApplyPatchExecutor {
    /// Applies a unified diff patch string sequentially against files in the workspace root.
    public static func apply(patchText: String, rootURL: URL) throws -> ApplyPatchOutput {
        var appliedOps: [AppliedPatchOperation] = []
        var diffSummaries: [GitDiffSummary] = []
        
        let lines = patchText.components(separatedBy: .newlines)
        var currentFile: String?
        var currentHunkLines: [String] = []
        var isNewFile = false
        var isDeleteFile = false
        
        func flushCurrentFile() throws {
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
                if FileManager.default.fileExists(atPath: secureURL.path) {
                    try FileManager.default.removeItem(at: secureURL)
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
                // Collect added lines
                var contentLines: [String] = []
                for hunkLine in currentHunkLines {
                    if hunkLine.hasPrefix("+") && !hunkLine.hasPrefix("+++") {
                        contentLines.append(String(hunkLine.dropFirst()))
                    }
                }
                let parentDir = secureURL.deletingLastPathComponent()
                try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)
                let text = contentLines.joined(separator: "\n")
                try text.write(to: secureURL, atomically: true, encoding: .utf8)
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
                if FileManager.default.fileExists(atPath: secureURL.path) {
                    originalText = try String(contentsOf: secureURL, encoding: .utf8)
                }
                let updatedText = try applyHunkLines(original: originalText, hunkLines: currentHunkLines)
                try updatedText.write(to: secureURL, atomically: true, encoding: .utf8)
                appliedOps.append(AppliedPatchOperation(type: "update", resource: fileRelPath, target: secureURL.path))
                diffSummaries.append(GitDiffSummary(
                    filename: fileRelPath,
                    status: "modified",
                    additions: currentHunkLines.filter { $0.hasPrefix("+") && !$0.hasPrefix("+++") }.count,
                    deletions: currentHunkLines.filter { $0.hasPrefix("-") && !$0.hasPrefix("---") }.count,
                    changes: currentHunkLines.count,
                    patch: currentHunkLines.joined(separator: "\n")
                ))
            }
        }

        for line in lines {
            if line.hasPrefix("diff --git ") {
                try flushCurrentFile()
            } else if line.hasPrefix("--- ") {
                let pathPart = line.replacingOccurrences(of: "--- ", with: "").trimmingCharacters(in: .whitespaces)
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
                let pathPart = line.replacingOccurrences(of: "+++ ", with: "").trimmingCharacters(in: .whitespaces)
                if pathPart == "/dev/null" {
                    isDeleteFile = true
                } else {
                    let cleaned = pathPart.hasPrefix("b/") ? String(pathPart.dropFirst(2)) : pathPart
                    currentFile = cleaned
                }
            } else if line.hasPrefix("@@") || currentFile != nil {
                currentHunkLines.append(line)
            }
        }
        
        try flushCurrentFile()
        
        let summary = "Applied patch to \(appliedOps.count) file(s):\n" + appliedOps.map { "  \($0.type.uppercased()) \($0.resource)" }.joined(separator: "\n")
        return ApplyPatchOutput(applied: appliedOps, files: diffSummaries, summary: summary)
    }
    
    /// Matches a unified diff hunk header: `@@ -oldStart[,oldCount] +newStart[,newCount] @@`,
    /// tolerating trailing context text some diff tools append after the closing `@@`.
    private static let hunkHeaderRegex = try! NSRegularExpression(
        pattern: #"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@"#
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
        let origLines = original.components(separatedBy: .newlines)
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

        return resultLines.joined(separator: "\n")
    }
}
