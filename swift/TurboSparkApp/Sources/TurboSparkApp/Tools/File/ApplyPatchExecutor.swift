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
            guard let fileRelPath = currentFile else { return }
            let secureURL = try resolveSecurePath(relPath: fileRelPath, rootURL: rootURL)
            
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
            
            currentFile = nil
            currentHunkLines.removeAll()
            isNewFile = false
            isDeleteFile = false
        }
        
        for line in lines {
            if line.hasPrefix("diff --git ") {
                try flushCurrentFile()
            } else if line.hasPrefix("--- ") {
                let pathPart = line.replacingOccurrences(of: "--- ", with: "").trimmingCharacters(in: .whitespaces)
                if pathPart == "/dev/null" {
                    isNewFile = true
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
    
    private static func applyHunkLines(original: String, hunkLines: [String]) throws -> String {
        let origLines = original.components(separatedBy: .newlines)
        var resultLines: [String] = []
        var origIdx = 0
        var insideHunk = false
        
        for hLine in hunkLines {
            if hLine.hasPrefix("@@") {
                insideHunk = true
                continue
            }
            if !insideHunk { continue }
            
            if hLine.hasPrefix(" ") {
                let expected = String(hLine.dropFirst())
                if origIdx < origLines.count {
                    resultLines.append(origLines[origIdx])
                    origIdx += 1
                } else {
                    resultLines.append(expected)
                }
            } else if hLine.hasPrefix("-") {
                if origIdx < origLines.count {
                    origIdx += 1
                }
            } else if hLine.hasPrefix("+") {
                resultLines.append(String(hLine.dropFirst()))
            }
        }
        
        while origIdx < origLines.count {
            resultLines.append(origLines[origIdx])
            origIdx += 1
        }
        
        return resultLines.joined(separator: "\n")
    }
    
    private static func resolveSecurePath(relPath: String, rootURL: URL) throws -> URL {
        let cleaned = relPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.hasPrefix("/"), !cleaned.hasPrefix("~") else {
            throw NSError(domain: "TurboSparkTool", code: 1, userInfo: [NSLocalizedDescriptionKey: "Refusing absolute or tilde path: '\(relPath)'"])
        }
        let candidate = rootURL.appendingPathComponent(cleaned).standardizedFileURL
        let realCandidate = candidate.resolvingSymlinksInPath().path
        let realRoot = rootURL.standardizedFileURL.resolvingSymlinksInPath().path
        let rootPrefix = realRoot.hasSuffix("/") ? realRoot : realRoot + "/"
        guard realCandidate == realRoot || realCandidate.hasPrefix(rootPrefix) else {
            throw NSError(domain: "TurboSparkTool", code: 1, userInfo: [NSLocalizedDescriptionKey: "Path '\(relPath)' resolves outside workspace root."])
        }
        return candidate
    }
}
