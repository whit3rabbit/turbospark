import Foundation
import CryptoKit

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
    ///
    /// **ONE ABSOLUTE-PATH EXCEPTION: THE SPILL ROOT.** Spilled shell
    /// output (`ShellOutputFormatting.compactWithSpill`) lives under
    /// Application Support, never inside the project, and its whole point
    /// is that the model can go back to read it -- so an absolute path is
    /// accepted when, and only when, it resolves under the spill root.
    /// Symlinks are resolved on BOTH sides before the prefix compare, the
    /// same discipline `PathContainment` uses for the project root, so a
    /// symlink planted in the spill directory cannot pivot the check
    /// elsewhere.
    static func resolveSecurePath(relPath: String, rootURL: URL) throws -> URL {
        let cleaned = relPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if cleaned.hasPrefix("/") {
            let candidate = URL(fileURLWithPath: cleaned)
                .standardizedFileURL
                .resolvingSymlinksInPath()
            if ShellOutputFormatting.isUnderSpillRoot(candidate) {
                return candidate
            }
        }
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

    private static func generateSimpleDiff(pathA: String, contentA: String, pathB: String, contentB: String) -> String {
        let linesA = contentA.components(separatedBy: "\n")
        let linesB = contentB.components(separatedBy: "\n")
        var diffLines: [String] = [
            "--- \(pathA)",
            "+++ \(pathB)"
        ]
        let maxLines = max(linesA.count, linesB.count)
        var diffCount = 0
        for i in 0..<maxLines {
            let lineA = i < linesA.count ? linesA[i] : nil
            let lineB = i < linesB.count ? linesB[i] : nil
            if lineA != lineB {
                diffCount += 1
                if let la = lineA {
                    diffLines.append(String(format: "-%4d | %@", i + 1, la))
                }
                if let lb = lineB {
                    diffLines.append(String(format: "+%4d | %@", i + 1, lb))
                }
            }
        }
        if diffCount == 0 {
            return "Files \(pathA) and \(pathB) are identical."
        }
        return compactOutput(diffLines.joined(separator: "\n"), maxLines: 150)
    }

    static func readFile(
        relPath: String,
        rootURL: URL,
        startLine: Int? = nil,
        endLine: Int? = nil,
        limit: Int? = nil,
        mode: String? = nil,
        searchPattern: String? = nil,
        contextLines: Int? = nil,
        comparisonPath: String? = nil,
        numRevisions: Int? = nil
    ) async throws -> String {
        let targetURL = try resolveSecurePath(relPath: relPath, rootURL: rootURL)
        guard FileManager.default.fileExists(atPath: targetURL.path) else {
            throw NSError(domain: "TurboSparkTool", code: 11, userInfo: [NSLocalizedDescriptionKey: "File not found: \(relPath)"])
        }

        let content = try AppFileReadLimits.readTextFile(at: targetURL, describing: relPath)
        // The content is already in hand, so the snapshot hashes THAT rather
        // than re-reading the file: the old call took a second full copy of
        // every file the model read, for a hash of bytes this frame was
        // already holding.
        //
        // Spill files are skipped: they are not project files, they are
        // read-only references, and a session that greps a few big logs
        // through read_file must not evict half the project's real hashes
        // from the store's 512-entry cache.
        if !ShellOutputFormatting.isUnderSpillRoot(targetURL) {
            await FileSnapshotStore.shared.recordSnapshot(url: targetURL, content: content)
        }

        let selectedMode = mode?.lowercased().trimmingCharacters(in: .whitespacesAndNewlines) ?? "lines"
        switch selectedMode {
        case "stats":
            let attrs = (try? FileManager.default.attributesOfItem(atPath: targetURL.path)) ?? [:]
            let fileSize = (attrs[.size] as? Int64) ?? 0
            let modDate = (attrs[.modificationDate] as? Date)?.description ?? "unknown"
            let rawLines = content.components(separatedBy: "\n")
            let lineCount = content.isEmpty ? 0 : (content.hasSuffix("\n") ? rawLines.count - 1 : rawLines.count)
            let wordCount = content.split { $0.isWhitespace || $0.isNewline }.count
            let sha = SHA256.hash(data: Data(content.utf8)).map { String(format: "%02x", $0) }.joined()
            return """
            File Statistics for: \(relPath)
            - Path: \(relPath)
            - Size: \(fileSize) bytes
            - Lines: \(lineCount)
            - Words: \(wordCount)
            - Characters: \(content.count)
            - SHA256: \(sha)
            - Last Modified: \(modDate)
            """

        case "preview":
            let allLines = content.components(separatedBy: "\n")
            let previewCount = min(allLines.count, max(1, limit ?? 50))
            let slice = allLines.prefix(previewCount)
            var previewLines: [String] = ["Preview of \(relPath) (first \(previewCount) of \(allLines.count) lines):"]
            for (idx, line) in slice.enumerated() {
                previewLines.append(String(format: "%4d | %@", idx + 1, line))
            }
            if allLines.count > previewCount {
                previewLines.append("... [\(allLines.count - previewCount) more lines omitted] ...")
            }
            return previewLines.joined(separator: "\n")

        case "diff":
            guard let compRel = comparisonPath, !compRel.isEmpty else {
                throw NSError(domain: "TurboSparkTool", code: 12, userInfo: [NSLocalizedDescriptionKey: "Missing 'comparison_path' for diff mode."])
            }
            let compURL = try resolveSecurePath(relPath: compRel, rootURL: rootURL)
            guard FileManager.default.fileExists(atPath: compURL.path) else {
                throw NSError(domain: "TurboSparkTool", code: 12, userInfo: [NSLocalizedDescriptionKey: "Comparison file not found: \(compRel)"])
            }
            let compContent = try AppFileReadLimits.readTextFile(at: compURL, describing: compRel)
            return generateSimpleDiff(pathA: relPath, contentA: content, pathB: compRel, contentB: compContent)

        case "time_machine":
            let revCount = max(1, min(20, numRevisions ?? 5))
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
            process.arguments = ["log", "-p", "-n", "\(revCount)", "--", targetURL.path]
            process.currentDirectoryURL = rootURL
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            try process.run()
            process.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let gitOutput = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            if process.terminationStatus != 0 || gitOutput.isEmpty || gitOutput.contains("not a git repository") {
                return "No git history found for \(relPath)."
            }
            return compactOutput(gitOutput, maxLines: 150)

        case "search":
            guard let pattern = searchPattern, !pattern.isEmpty else {
                throw NSError(domain: "TurboSparkTool", code: 13, userInfo: [NSLocalizedDescriptionKey: "Missing 'search_pattern' for search mode."])
            }
            let regex = try NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
            let lines = content.components(separatedBy: "\n")
            let ctx = max(0, min(10, contextLines ?? 3))
            var matchIndices = Set<Int>()
            for (idx, line) in lines.enumerated() {
                let range = NSRange(location: 0, length: (line as NSString).length)
                if regex.firstMatch(in: line, options: [], range: range) != nil {
                    matchIndices.insert(idx)
                }
            }
            if matchIndices.isEmpty {
                return "No matches found for pattern '\(pattern)' in \(relPath)."
            }
            var outputLines: [String] = ["Found \(matchIndices.count) matching line(s) for '\(pattern)' in \(relPath):"]
            var shownIndices = Set<Int>()
            for matchIdx in matchIndices.sorted() {
                let start = max(0, matchIdx - ctx)
                let end = min(lines.count - 1, matchIdx + ctx)
                for i in start...end {
                    if !shownIndices.contains(i) {
                        shownIndices.insert(i)
                        let marker = matchIndices.contains(i) ? ">" : " "
                        outputLines.append(String(format: "%@%4d | %@", marker, i + 1, lines[i]))
                    }
                }
                outputLines.append("---")
            }
            return outputLines.joined(separator: "\n")

        default:
            // "lines" mode
            let allLines = content.components(separatedBy: "\n")
            let sLine = max(1, startLine ?? 1)
            if sLine > allLines.count {
                return "File has \(allLines.count) lines. Requested start line \(sLine) is out of bounds."
            }
            // **`end_line` AND `limit` ARE DIFFERENT QUESTIONS AND ARE NO LONGER
            // ALIASED.** The schema advertises "start_line and end_line bounds"
            // while the two keys were collapsed into one value treated as a
            // COUNT, so `read_file(start_line: 500, end_line: 520)` -- exactly
            // what the schema's own example invites -- returned 520 lines instead
            // of 21. `end_line` is absolute (the schema's word), `limit` is a
            // count (the Claude/OpenAI convention), and each is read from its own
            // key. Clamping rather than trusting a model-supplied value is what
            // keeps `(sLine - 1)..<eLine` valid and non-empty: an unclamped
            // `eLine < sLine - 1` previously trapped the process.
            let eLine: Int
            if let endLine {
                eLine = min(allLines.count, max(sLine, endLine))
            } else {
                let requestedCount = limit ?? 120
                eLine = min(allLines.count, max(sLine, sLine + requestedCount - 1))
            }

            let slice = allLines[(sLine - 1)..<eLine]
            var outputLines: [String] = ["File: \(relPath) (lines \(sLine)-\(eLine) of \(allLines.count))"]
            for (offset, line) in slice.enumerated() {
                outputLines.append(String(format: "%4d | %@", sLine + offset, line))
            }
            return outputLines.joined(separator: "\n")
        }
    }

    static func writeFile(relPath: String, content: String, rootURL: URL) async throws -> String {
        let targetURL = try resolveSecurePath(relPath: relPath, rootURL: rootURL)
        try AppToolSandbox.validateWritePath(targetURL, rootURL: rootURL)
        // **AN OVERWRITE OF AN EXISTING FILE IS GATED ON HAVING READ IT.**
        // `edit_file` refuses through the snapshot hash and its own
        // old_string match; `write_file` replaces the whole content, so
        // without a gate here it is the one tool that can silently clobber
        // a file the user changed (or that the model never saw) this
        // session. A file that does not exist is the ordinary create path
        // and is never gated.
        if FileManager.default.fileExists(atPath: targetURL.path) {
            let tracked = await FileSnapshotStore.shared.isTracked(url: targetURL)
            let stale = tracked
                ? await FileSnapshotStore.shared.isStale(url: targetURL)
                : true
            if !tracked || stale {
                throw NSError(domain: "TurboSparkTool", code: 25, userInfo: [
                    NSLocalizedDescriptionKey: "File '\(relPath)' exists but has not been read this "
                        + "session, or has changed since it was last read. Use read_file on it first, "
                        + "then write."
                ])
            }
        }
        let dirURL = targetURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dirURL, withIntermediateDirectories: true)
        try content.write(to: targetURL, atomically: true, encoding: .utf8)
        await FileSnapshotStore.shared.recordSnapshot(url: targetURL, content: content)
        return "Successfully wrote \(content.count) characters to \(relPath)."
    }

    /// Compacts lengthy tool output preserving both head (initial context) and tail (errors/summary).
    public static func compactOutput(_ raw: String, maxLines: Int = 120) -> String {
        let lines = raw.components(separatedBy: "\n")
        guard lines.count > maxLines else { return raw }
        let headCount = 25
        let tailCount = 65
        let head = lines.prefix(headCount).joined(separator: "\n")
        let tail = lines.suffix(tailCount).joined(separator: "\n")
        let omitted = lines.count - (headCount + tailCount)
        return "\(head)\n\n... [\(omitted) lines truncated] ...\n\n\(tail)"
    }

    /// Normalizes Unicode curly quotes to standard straight quotes.
    public static func normalizeQuotes(_ str: String) -> String {
        return str
            .replacingOccurrences(of: "\u{2018}", with: "'")
            .replacingOccurrences(of: "\u{2019}", with: "'")
            .replacingOccurrences(of: "\u{201C}", with: "\"")
            .replacingOccurrences(of: "\u{201D}", with: "\"")
    }

    /// Strips copied line prefixes (e.g. ' 12 | ' or '12: ') from model input.
    public static func stripLinePrefixes(_ str: String) -> String {
        let lines = str.components(separatedBy: "\n")
        var strippedLines: [String] = []
        var hadPrefixes = false

        for line in lines {
            if let regex = try? NSRegularExpression(pattern: "^\\s*\\d+\\s*[|:]\\s?(.*)$", options: []) {
                let ns = line as NSString
                if let match = regex.firstMatch(in: line, options: [], range: NSRange(location: 0, length: ns.length)), match.numberOfRanges > 1 {
                    let rest = ns.substring(with: match.range(at: 1))
                    strippedLines.append(rest)
                    hadPrefixes = true
                    continue
                }
            }
            strippedLines.append(line)
        }
        return hadPrefixes ? strippedLines.joined(separator: "\n") : str
    }

    /// Strips trailing whitespace per line.
    public static func stripTrailingWhitespace(_ str: String) -> String {
        let lines = str.components(separatedBy: "\n")
        return lines.map { $0.replacingOccurrences(of: "\\s+$", with: "", options: .regularExpression) }.joined(separator: "\n")
    }

    /// Normalizes lines by trimming leading and trailing whitespace and collapsing internal runs.
    public static func normalizeLineWhitespace(_ str: String) -> [String] {
        return str.components(separatedBy: "\n").map { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            return trimmed.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        }
    }

    static func editFile(
        relPath: String,
        oldString: String = "",
        newString: String = "",
        replaceAll: Bool = false,
        rootURL: URL,
        command: String? = nil,
        insertLine: String? = nil,
        position: String? = nil,
        regexPattern: String? = nil
    ) async throws -> String {
        let targetURL = try resolveSecurePath(relPath: relPath, rootURL: rootURL)
        try AppToolSandbox.validateWritePath(targetURL, rootURL: rootURL)
        guard FileManager.default.fileExists(atPath: targetURL.path) else {
            throw NSError(domain: "TurboSparkTool", code: 11, userInfo: [NSLocalizedDescriptionKey: "File not found: \(relPath)"])
        }

        let selectedCommand = command?.lowercased().trimmingCharacters(in: .whitespacesAndNewlines) ?? "str_replace"
        if selectedCommand == "undo_edit" || selectedCommand == "undo" {
            guard let previous = await FileSnapshotStore.shared.restoreBackup(url: targetURL) else {
                throw NSError(domain: "TurboSparkTool", code: 17, userInfo: [
                    NSLocalizedDescriptionKey: "No previous backup found to undo for \(relPath)."
                ])
            }
            try previous.write(to: targetURL, atomically: true, encoding: .utf8)
            await FileSnapshotStore.shared.recordSnapshot(url: targetURL, content: previous)
            return "Successfully rolled back \(relPath) to previous snapshot."
        }

        if await FileSnapshotStore.shared.isStale(url: targetURL) {
            throw NSError(domain: "TurboSparkTool", code: 16, userInfo: [
                NSLocalizedDescriptionKey: "File '\(relPath)' has been modified on disk since it was last read. Please re-read the file before editing."
            ])
        }

        let content = try AppFileReadLimits.readTextFile(at: targetURL, describing: relPath)
        // Record backup for undo_edit capability
        await FileSnapshotStore.shared.recordBackup(url: targetURL, content: content)

        if selectedCommand == "insert" {
            guard let target = insertLine, !target.isEmpty else {
                throw NSError(domain: "TurboSparkTool", code: 18, userInfo: [
                    NSLocalizedDescriptionKey: "Missing 'insert_line' argument for insert command."
                ])
            }
            var lines = content.components(separatedBy: "\n")
            var targetIndex: Int?
            if let lineNum = Int(target) {
                targetIndex = max(0, min(lines.count - 1, lineNum - 1))
            } else {
                targetIndex = lines.firstIndex(where: { $0.contains(target) })
            }
            guard let idx = targetIndex else {
                throw NSError(domain: "TurboSparkTool", code: 18, userInfo: [
                    NSLocalizedDescriptionKey: "Could not find line or text matching '\(target)' in \(relPath)."
                ])
            }
            let insertionPos = (position ?? "after").lowercased()
            let insertionIdx = insertionPos == "before" ? idx : idx + 1
            let newLines = newString.components(separatedBy: "\n")
            lines.insert(contentsOf: newLines, at: min(lines.count, max(0, insertionIdx)))
            let updated = lines.joined(separator: "\n")
            try updated.write(to: targetURL, atomically: true, encoding: .utf8)
            await FileSnapshotStore.shared.recordSnapshot(url: targetURL, content: updated)
            return "Successfully inserted \(newLines.count) line(s) \(insertionPos) line \(idx + 1) in \(relPath)."
        }

        if selectedCommand == "pattern_replace" {
            guard let pattern = regexPattern, !pattern.isEmpty else {
                throw NSError(domain: "TurboSparkTool", code: 19, userInfo: [
                    NSLocalizedDescriptionKey: "Missing 'regex_pattern' argument for pattern_replace command."
                ])
            }
            let regex = try NSRegularExpression(pattern: pattern, options: [])
            let nsContent = content as NSString
            let matches = regex.matches(in: content, options: [], range: NSRange(location: 0, length: nsContent.length))
            if matches.isEmpty {
                throw NSError(domain: "TurboSparkTool", code: 19, userInfo: [
                    NSLocalizedDescriptionKey: "Pattern '\(pattern)' did not match any content in \(relPath)."
                ])
            }
            let updated = regex.stringByReplacingMatches(in: content, options: [], range: NSRange(location: 0, length: nsContent.length), withTemplate: newString)
            try updated.write(to: targetURL, atomically: true, encoding: .utf8)
            await FileSnapshotStore.shared.recordSnapshot(url: targetURL, content: updated)
            return "Successfully replaced \(matches.count) pattern match(es) in \(relPath)."
        }

        let isMarkdown = relPath.lowercased().hasSuffix(".md") || relPath.lowercased().hasSuffix(".mdx")

        // 1. Exact match attempt
        var targetOld = oldString
        var matchedSpan: Range<String.Index>? = content.range(of: targetOld)

        // 2. Line prefix stripping attempt (if model accidentally copied '12 | ' from read_file)
        if matchedSpan == nil {
            let strippedOld = stripLinePrefixes(targetOld)
            if strippedOld != targetOld {
                targetOld = strippedOld
                matchedSpan = content.range(of: targetOld)
            }
        }

        // 3. Quote normalization attempt
        if matchedSpan == nil {
            let normOld = normalizeQuotes(targetOld)
            let normContent = normalizeQuotes(content)
            if let normRange = normContent.range(of: normOld) {
                let startOffset = normContent.distance(from: normContent.startIndex, to: normRange.lowerBound)
                let length = normContent.distance(from: normRange.lowerBound, to: normRange.upperBound)
                let targetStart = content.index(content.startIndex, offsetBy: startOffset)
                let targetEnd = content.index(targetStart, offsetBy: length)
                targetOld = String(content[targetStart..<targetEnd])
                matchedSpan = targetStart..<targetEnd
            }
        }

        // 4. Trailing whitespace tolerance attempt (non-markdown)
        if matchedSpan == nil && !isMarkdown {
            let strippedOld = stripTrailingWhitespace(targetOld)
            let strippedContent = stripTrailingWhitespace(content)
            if strippedContent.contains(strippedOld) {
                let oldLines = strippedOld.components(separatedBy: "\n")
                let contentLines = content.components(separatedBy: "\n")
                if !oldLines.isEmpty && contentLines.count >= oldLines.count {
                    for i in 0...(contentLines.count - oldLines.count) {
                        let window = contentLines[i..<(i + oldLines.count)].map { $0.replacingOccurrences(of: "\\s+$", with: "", options: .regularExpression) }
                        if window == oldLines {
                            let actualWindow = contentLines[i..<(i + oldLines.count)].joined(separator: "\n")
                            if let r = content.range(of: actualWindow) {
                                targetOld = actualWindow
                                matchedSpan = r
                                break
                            }
                        }
                    }
                }
            }
        }

        // 5. Line-based normalized indentation and whitespace matching attempt (non-markdown)
        if matchedSpan == nil && !isMarkdown {
            let oldNormLines = normalizeLineWhitespace(targetOld)
            if !oldNormLines.isEmpty && oldNormLines.contains(where: { !$0.isEmpty }) {
                let contentLines = content.components(separatedBy: "\n")
                if contentLines.count >= oldNormLines.count {
                    var candidateWindows: [Range<String.Index>] = []
                    var candidateTexts: [String] = []
                    let targetCount = oldNormLines.count

                    let contentNormLines = contentLines.map { line in
                        line.trimmingCharacters(in: .whitespaces)
                            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
                    }

                    for i in 0...(contentLines.count - targetCount) {
                        let windowSlice = contentNormLines[i..<(i + targetCount)]
                        if Array(windowSlice) == oldNormLines {
                            let rawWindowText = contentLines[i..<(i + targetCount)].joined(separator: "\n")
                            if let r = content.range(of: rawWindowText) {
                                candidateWindows.append(r)
                                candidateTexts.append(rawWindowText)
                            }
                        }
                    }

                    if candidateWindows.count == 1 {
                        targetOld = candidateTexts[0]
                        matchedSpan = candidateWindows[0]
                    }
                }
            }
        }

        guard let firstRange = matchedSpan else {
            throw NSError(domain: "TurboSparkTool", code: 15, userInfo: [
                NSLocalizedDescriptionKey: "Target old_string not found in \(relPath). Ensure exact indentation, or re-read the file to inspect the latest contents."
            ])
        }

        // Count total matches in file
        var matchCount = 0
        var searchRange = content.startIndex..<content.endIndex
        while let nextRange = content.range(of: targetOld, range: searchRange) {
            matchCount += 1
            if nextRange.upperBound >= content.endIndex { break }
            searchRange = nextRange.upperBound..<content.endIndex
        }

        if matchCount > 1 && !replaceAll {
            throw NSError(domain: "TurboSparkTool", code: 15, userInfo: [
                NSLocalizedDescriptionKey: "Target old_string appears \(matchCount) times in \(relPath). Please provide more surrounding context to uniquely identify the target or set replace_all: true."
            ])
        }

        let updatedContent: String
        if replaceAll {
            updatedContent = content.replacingOccurrences(of: targetOld, with: newString)
        } else {
            updatedContent = content.replacingCharacters(in: firstRange, with: newString)
        }

        try updatedContent.write(to: targetURL, atomically: true, encoding: .utf8)
        await FileSnapshotStore.shared.recordSnapshot(url: targetURL, content: updatedContent)
        return "Successfully replaced \(replaceAll ? "\(matchCount) occurrence(s)" : "occurrence") in \(relPath)."
    }

    /// The bound on one `search_code` call.
    ///
    /// **A PATTERN WITH NO HITS USED TO READ THE WHOLE TREE.** The 40-match
    /// break is the only stop the search had, and it never fires when nothing
    /// matches -- so every file under the size cap was read whole, with three
    /// directory names skipped. On a large repository that is a multi-second
    /// main-actor-adjacent stall per call, and the model can make one per
    /// step.
    private enum SearchBudget {
        static let maxFilesVisited = 5_000
        static let maxBytesRead = 64 * 1024 * 1024
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
        var filesVisited = 0
        var bytesRead = 0
        var exhaustedBudget = false

        for case let fileURL as URL in enumerator {
            let path = fileURL.path
            if path.contains("/node_modules/") || path.contains("/target/") || path.contains("/.build/")
                || path.contains("/.git/")
            {
                continue
            }
            let size = AppFileReadLimits.fileSize(of: fileURL)
            if let size, size > AppFileReadLimits.maximumBytes {
                continue
            }
            filesVisited += 1
            if filesVisited > SearchBudget.maxFilesVisited || bytesRead > SearchBudget.maxBytesRead {
                exhaustedBudget = true
                break
            }
            guard let text = try? String(contentsOf: fileURL, encoding: .utf8) else { continue }
            bytesRead += text.utf8.count
            let lines = text.components(separatedBy: "\n")
            for (lineIdx, line) in lines.enumerated() {
                if line.lowercased().contains(lowerPattern) {
                    // `dropFirst`, not `replacingOccurrences`: the latter
                    // strips EVERY occurrence of the root prefix, so a path
                    // that repeats it (a nested checkout, a symlinked
                    // vendor directory) came out mangled.
                    let rel =
                        path.hasPrefix(rootPrefix) ? String(path.dropFirst(rootPrefix.count)) : path
                    matches.append("\(rel):\(lineIdx + 1): \(line.trimmingCharacters(in: .whitespaces))")
                    if matches.count >= 40 { break }
                }
            }
            if matches.count >= 40 { break }
        }

        if matches.isEmpty {
            return exhaustedBudget
                ? "No matches found for '\(pattern)' in the first \(filesVisited - 1) files "
                    + "searched. Narrow the path and try again."
                : "No matches found for '\(pattern)'."
        }
        let formatted = "Found \(matches.count) matches:\n" + matches.joined(separator: "\n")
        return compactOutput(formatted)
    }
}

