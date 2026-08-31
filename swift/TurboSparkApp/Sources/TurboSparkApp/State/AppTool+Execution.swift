import Foundation

/// Size ceilings for every file this app reads on a model's instruction.
///
/// **A path from a model is an unbounded read until something bounds it.**
/// `String(contentsOf:)` loads the whole file, `components(separatedBy:)`
/// makes a second copy as an array of lines, and `FileSnapshotStore` used to
/// read the same bytes a THIRD time to hash them. Point `read_file` at a
/// multi-gigabyte log, a core dump, or a `.gturbo` weight file (every one of
/// which is inside a normal project root) and the app takes three copies of
/// it into memory before deciding it is not text.
///
/// The Office and PDF extraction paths already had real limits
/// (`DocumentTextExtractor.Limits`, `OfficeArchive`'s byte budgets). These
/// are the same idea for the paths that had none.
public enum AppFileReadLimits {
    /// Largest file a tool will read into memory whole. Generous next to any
    /// real source file and far under what makes the app unresponsive.
    public static let maximumBytes = 16 * 1_024 * 1_024

    /// The size of the file at `url`, or nil when it cannot be determined.
    public static func fileSize(of url: URL) -> Int? {
        (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
    }

    /// Reads `url` as UTF-8, refusing anything over `maximumBytes`.
    ///
    /// The size is checked BEFORE the read, which is the whole point: a cap
    /// applied to the string afterwards has already paid the allocation it
    /// exists to prevent (`DocumentTextExtractor`'s plain-text branch did
    /// exactly that, truncating to 240k characters after loading the file
    /// whole).
    public static func readTextFile(at url: URL, describing relPath: String) throws -> String {
        if let size = fileSize(of: url), size > maximumBytes {
            throw NSError(
                domain: "TurboSparkTool", code: 17,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "\(relPath) is \(size / 1_024 / 1_024) MB, over the "
                        + "\(maximumBytes / 1_024 / 1_024) MB limit for a single read. "
                        + "Use start_line and end_line, or a shell command, to read part of it."
                ])
        }
        return try String(contentsOf: url, encoding: .utf8)
    }
}

// MARK: - Native Tool Execution Helpers

extension AppToolRegistry {
    /// Resolves a caller-supplied relative path against the project root, and
    /// REFUSES anything that leaves it.
    static func resolveSecurePath(relPath: String, rootURL: URL) throws -> URL {
        let cleaned = relPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.hasPrefix("/"), !cleaned.hasPrefix("~") else {
            throw NSError(
                domain: "TurboSparkTool", code: 13,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Path must be relative to the project root: \(relPath)"
                ])
        }

        let root = rootURL.standardizedFileURL.resolvingSymlinksInPath()
        let targetURL = root.appendingPathComponent(cleaned)
            .standardizedFileURL
            .resolvingSymlinksInPath()

        guard targetURL.path == root.path || targetURL.path.hasPrefix(root.path + "/") else {
            throw NSError(
                domain: "TurboSparkTool", code: 14,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Path escapes the project root: \(relPath)"
                ])
        }
        return targetURL
    }

    static func listDirectory(relPath: String, rootURL: URL) throws -> String {
        let targetURL = try resolveSecurePath(relPath: relPath, rootURL: rootURL)
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: targetURL.path, isDirectory: &isDir), isDir.boolValue else {
            throw NSError(domain: "TurboSparkTool", code: 10, userInfo: [NSLocalizedDescriptionKey: "Path is not a directory: \(relPath)"])
        }

        let items = try fm.contentsOfDirectory(atPath: targetURL.path)
            .filter { !$0.hasPrefix(".") && $0 != "node_modules" && $0 != "target" && $0 != ".build" }
            .sorted()

        var results: [String] = ["Directory listing for: \(relPath)"]
        for item in items.prefix(80) {
            let itemURL = targetURL.appendingPathComponent(item)
            var itemIsDir: ObjCBool = false
            fm.fileExists(atPath: itemURL.path, isDirectory: &itemIsDir)
            let typeLabel = itemIsDir.boolValue ? "[DIR] " : "[FILE]"
            results.append("\(typeLabel) \(item)")
        }
        if items.count > 80 {
            results.append("... and \(items.count - 80) more entries.")
        }
        return results.joined(separator: "\n")
    }

    static func readFile(relPath: String, rootURL: URL, startLine: Int?, endLine: Int?) async throws -> String {
        let targetURL = try resolveSecurePath(relPath: relPath, rootURL: rootURL)
        guard FileManager.default.fileExists(atPath: targetURL.path) else {
            throw NSError(domain: "TurboSparkTool", code: 11, userInfo: [NSLocalizedDescriptionKey: "File not found: \(relPath)"])
        }

        let content = try AppFileReadLimits.readTextFile(at: targetURL, describing: relPath)
        // The content is already in hand, so the snapshot hashes THAT rather
        // than re-reading the file: the old call took a second full copy of
        // every file the model read, for a hash of bytes this frame was
        // already holding.
        await FileSnapshotStore.shared.recordSnapshot(url: targetURL, content: content)
        let allLines = content.components(separatedBy: "\n")

        let sLine = max(1, startLine ?? 1)
        if sLine > allLines.count {
            return "File has \(allLines.count) lines. Requested start line \(sLine) is out of bounds."
        }
        // `endLine` is aliased from `limit`, which callers pass as a COUNT
        // (Claude/OpenAI convention) rather than an absolute line number, so
        // it is resolved relative to `sLine` rather than to line 1. Clamping
        // here (rather than trusting a model-supplied `end_line`/`limit`)
        // is what keeps `(sLine - 1)..<eLine` a valid, non-empty range: an
        // unclamped `eLine < sLine - 1` previously trapped the process.
        let requestedCount = endLine ?? 120
        let eLine = min(allLines.count, max(sLine, sLine + requestedCount - 1))

        let slice = allLines[(sLine - 1)..<eLine]
        var outputLines: [String] = ["File: \(relPath) (lines \(sLine)-\(eLine) of \(allLines.count))"]
        for (offset, line) in slice.enumerated() {
            outputLines.append(String(format: "%4d | %@", sLine + offset, line))
        }
        return outputLines.joined(separator: "\n")
    }

    static func writeFile(relPath: String, content: String, rootURL: URL) async throws -> String {
        let targetURL = try resolveSecurePath(relPath: relPath, rootURL: rootURL)
        try AppToolSandbox.validateWritePath(targetURL, rootURL: rootURL)
        let dirURL = targetURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
        try content.write(to: targetURL, atomically: true, encoding: .utf8)
        await FileSnapshotStore.shared.recordSnapshot(url: targetURL, content: content)
        return "Successfully wrote \(content.count) characters to \(relPath)."
    }

    static func editFile(relPath: String, oldString: String, newString: String, replaceAll: Bool, rootURL: URL) async throws -> String {
        let targetURL = try resolveSecurePath(relPath: relPath, rootURL: rootURL)
        try AppToolSandbox.validateWritePath(targetURL, rootURL: rootURL)
        guard FileManager.default.fileExists(atPath: targetURL.path) else {
            throw NSError(domain: "TurboSparkTool", code: 11, userInfo: [NSLocalizedDescriptionKey: "File not found: \(relPath)"])
        }

        if await FileSnapshotStore.shared.isStale(url: targetURL) {
            throw NSError(domain: "TurboSparkTool", code: 16, userInfo: [
                NSLocalizedDescriptionKey: "File '\(relPath)' has been modified on disk since it was last read. Please re-read the file before editing."
            ])
        }

        let content = try AppFileReadLimits.readTextFile(at: targetURL, describing: relPath)
        guard content.contains(oldString) else {
            throw NSError(domain: "TurboSparkTool", code: 15, userInfo: [NSLocalizedDescriptionKey: "Target old_string not found in \(relPath)."])
        }

        let updatedContent: String
        if replaceAll {
            updatedContent = content.replacingOccurrences(of: oldString, with: newString)
        } else {
            if let range = content.range(of: oldString) {
                updatedContent = content.replacingCharacters(in: range, with: newString)
            } else {
                updatedContent = content
            }
        }

        try updatedContent.write(to: targetURL, atomically: true, encoding: .utf8)
        await FileSnapshotStore.shared.recordSnapshot(url: targetURL, content: updatedContent)
        return "Successfully replaced occurrences in \(relPath)."
    }

    static func searchCode(pattern: String, relPath: String, rootURL: URL) throws -> String {
        let targetURL = try resolveSecurePath(relPath: relPath, rootURL: rootURL)
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: targetURL, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles, .skipsPackageDescendants]) else {
            throw NSError(domain: "TurboSparkTool", code: 12, userInfo: [NSLocalizedDescriptionKey: "Cannot search path."])
        }

        var matches: [String] = []
        let lowerPattern = pattern.lowercased()
        let rootPrefix = rootURL.standardizedFileURL.resolvingSymlinksInPath().path + "/"

        for case let fileURL as URL in enumerator {
            let path = fileURL.path
            if path.contains("/node_modules/") || path.contains("/target/") || path.contains("/.build/") {
                continue
            }
            // Skipped rather than refused: a search walks a whole tree, and
            // one oversized file in it is not a reason to fail the search.
            // Without this the walk loaded every binary, log and checkpoint
            // it met into memory whole just to find out it was not text.
            if let size = AppFileReadLimits.fileSize(of: fileURL),
                size > AppFileReadLimits.maximumBytes
            {
                continue
            }
            guard let text = try? String(contentsOf: fileURL, encoding: .utf8) else { continue }
            let lines = text.components(separatedBy: "\n")
            for (lineIdx, line) in lines.enumerated() {
                if line.lowercased().contains(lowerPattern) {
                    let rel = fileURL.path.replacingOccurrences(of: rootPrefix, with: "")
                    matches.append("\(rel):\(lineIdx + 1): \(line.trimmingCharacters(in: .whitespaces))")
                    if matches.count >= 40 { break }
                }
            }
            if matches.count >= 40 { break }
        }

        if matches.isEmpty {
            return "No matches found for '\(pattern)'."
        }
        return "Found \(matches.count) matches:\n" + matches.joined(separator: "\n")
    }

    static func runCommand(command: String, rootURL: URL, timeoutMs: Int? = nil) async throws -> String {
        // `timeoutMs` is the model-supplied `BashInput.timeout` (milliseconds);
        // 0 or absent falls back to a generous default rather than blocking
        // forever. `ProcessExecutor` drains stdout/stderr concurrently with
        // the wait, which is what makes the deadline actually fire: the
        // previous `waitUntilExit()` + `readDataToEndOfFile()` pair hung
        // indefinitely the moment a command wrote more than one pipe buffer
        // (~64KB) before exiting, since nothing was reading it meanwhile.
        let timeoutSeconds: TimeInterval
        if let timeoutMs, timeoutMs > 0 {
            timeoutSeconds = TimeInterval(timeoutMs) / 1000.0
        } else {
            timeoutSeconds = ProcessExecutor.defaultTimeoutSeconds
        }

        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/zsh"),
            arguments: ["-c", command],
            currentDirectoryURL: rootURL,
            timeoutSeconds: timeoutSeconds
        )

        if result.timedOut {
            let combined = [result.stdout, result.stderr].filter { !$0.isEmpty }.joined(separator: "\n")
            let prefix = "(Command timed out after \(Int(timeoutSeconds))s and was terminated.)"
            return combined.isEmpty ? prefix : "\(prefix)\n\(combined)"
        }

        let combined = [result.stdout, result.stderr].filter { !$0.isEmpty }.joined(separator: "\n")
        if combined.isEmpty {
            let interp = TerminalCommandClassifier.interpretExitCode(result.exitCode)
            return "(Command finished: \(interp))"
        }
        return combined
    }
}
