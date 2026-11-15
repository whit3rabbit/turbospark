import Foundation

/// What the formatter pass did to one file.
enum FormatterOutcome: Equatable {
    /// Not run, or run with nothing to say: no formatter, a refused path, an
    /// unreadable file, a missing binary. Adds nothing to the tool result.
    case skipped
    /// Ran and left the file as the agent wrote it.
    case unchanged
    /// Ran and rewrote the file.
    case formatted(FormatterChange)
    /// Failed, timed out or lost a race; the file keeps the agent's content.
    case failed(String)
}

/// The visible part of a rewrite: which lines of the NEW text changed.
struct FormatterChange: Equatable {
    var added: [(line: Int, text: String)]
    var removedCount: Int
    var lineDiffAvailable: Bool

    static func == (lhs: FormatterChange, rhs: FormatterChange) -> Bool {
        lhs.removedCount == rhs.removedCount && lhs.lineDiffAvailable == rhs.lineDiffAvailable
            && lhs.added.count == rhs.added.count
            && zip(lhs.added, rhs.added).allSatisfy { $0.line == $1.line && $0.text == $1.text }
    }
}

/// OC-11: runs one formatter over ONE file the agent just wrote.
///
/// The formatter runs on a temporary sibling copy (same directory, so the
/// formatter's config discovery sees the project's config), never on the real
/// file. The real file is replaced only when the formatter exits 0 AND the file
/// still holds the exact bytes that were copied, so a failing, hanging or racing
/// formatter cannot lose the agent's work or a user's concurrent save.
struct FormatterRunner: Sendable {
    /// The hard timeout. A formatter on one file should take well under a second.
    static let defaultTimeoutSeconds: TimeInterval = 10
    /// The pipe cap: bounds memory, the note shows less.
    static let outputCapBytes = 16 * 1024
    static let errorTailChars = 2_048
    /// Bigger files are skipped: the copy, the diff and the note all scale with it.
    static let maximumFileBytes = 1_000_000
    /// Above this many lines the note says how many changed rather than which.
    static let maximumDiffLines = 4_000
    static let maximumNoteLines = 30
    static let tempPrefix = ".tsfmt-"

    var timeoutSeconds: TimeInterval = FormatterRunner.defaultTimeoutSeconds
    var parentEnvironment: [String: String] = ProcessInfo.processInfo.environment

    func format(
        path: String, row: FormatterRow, rootURL: URL, searchPath: String?
    ) async -> FormatterOutcome {
        let lexical = path.hasPrefix("/") ? URL(fileURLWithPath: path) : rootURL.appendingPathComponent(path)
        // Never a sensitive path, on either spelling.
        if ToolRiskClassifier.isSensitivePath(path) || ToolRiskClassifier.isSensitivePath(lexical.path) { return .skipped }
        // Symlinks resolved on both sides: a link that leaves the project is skipped.
        guard let target = PathContainment.resolvedIfContained(lexical, in: rootURL) else { return .skipped }
        if ToolRiskClassifier.isSensitivePath(target.path) { return .skipped }
        // Repository metadata is never rewritten, whatever its extension.
        if target.pathComponents.contains(".git") { return .skipped }
        // The covering row is chosen on the resolved name too, so a symlink
        // named a.swift that points at a.txt is not formatted as Swift.
        guard FormatterRegistry.row(forPath: target.path, in: [row]) != nil else { return .skipped }

        var info = stat()
        guard lstat(target.path, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG,
              info.st_size > 0, info.st_size <= off_t(Self.maximumFileBytes),
              let original = try? Data(contentsOf: target), String(data: original, encoding: .utf8) != nil
        else { return .skipped }

        let temp = target.deletingLastPathComponent()
            .appendingPathComponent(Self.tempPrefix + String(UUID().uuidString.prefix(8)).lowercased()
                + "." + target.pathExtension)
        guard Self.createTemp(at: temp, contents: original, mode: info.st_mode & 0o777) else { return .skipped }
        defer { try? FileManager.default.removeItem(at: temp) }

        let launch: FormatterLaunchPolicy.Launch
        do {
            launch = try FormatterLaunchPolicy.resolve(
                row: row, rootURL: rootURL, searchPath: searchPath, filePath: temp.path)
        } catch {
            // Missing or refused binary: a silent skip (CUJ 2). The settings
            // row shows whether the command resolves.
            return .skipped
        }

        let output: ProcessExecutor.Output
        do {
            output = try await ProcessExecutor.run(
                executableURL: URL(fileURLWithPath: launch.executable),
                arguments: launch.arguments,
                currentDirectoryURL: PathContainment.canonical(rootURL),
                environment: FormatterLaunchPolicy.environment(
                    parent: parentEnvironment, executable: launch.executable, searchPath: searchPath),
                timeoutSeconds: timeoutSeconds,
                outputCapBytes: Self.outputCapBytes,
                mergeStreams: true)
        } catch {
            return Task.isCancelled ? .skipped : .failed("could not start: \(error.localizedDescription)")
        }
        if output.timedOut {
            return .failed("timed out after \(Int(timeoutSeconds)) s and was stopped")
        }
        guard output.exitCode == 0 else {
            let tail = Self.errorTail(output.combinedText)
            return .failed("exit \(output.exitCode)" + (tail.isEmpty ? "" : ": " + tail))
        }
        guard let formatted = try? Data(contentsOf: temp) else { return .skipped }
        if formatted == original { return .unchanged }
        // A formatter that empties or mangles the file is a failure, not a style pass.
        guard !formatted.isEmpty, String(data: formatted, encoding: .utf8) != nil else {
            return .failed("returned empty or non-text output; the file was left as written")
        }
        // The race check: only replace the bytes that were formatted.
        guard let current = try? Data(contentsOf: target), current == original else {
            return .failed("the file changed while it was being formatted; it was left as is")
        }
        do {
            _ = try FileManager.default.replaceItemAt(target, withItemAt: temp)
        } catch {
            return .failed("could not replace the file: \(error.localizedDescription)")
        }
        // The next edit must not be rejected as stale: refresh every spelling
        // of the path the edit tools may have recorded.
        for url in [lexical, target] {
            await FileSnapshotStore.shared.refreshAfterFormat(url: url, data: formatted)
        }
        return .formatted(Self.change(from: original, to: formatted))
    }

    /// Creates the temporary copy exclusively and without following a
    /// symlink, owner-only until the mode is restored.
    static func createTemp(at url: URL, contents: Data, mode: mode_t) -> Bool {
        let fd = open(url.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0o600)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        let written = contents.withUnsafeBytes { raw -> Bool in
            var offset = 0
            while offset < raw.count {
                let n = write(fd, raw.baseAddress! + offset, raw.count - offset)
                if n <= 0 { return false }
                offset += n
            }
            return true
        }
        guard written, fchmod(fd, mode) == 0 else {
            try? FileManager.default.removeItem(at: url)
            return false
        }
        return true
    }

    static func errorTail(_ text: String) -> String {
        let clean = ShellOutputFormatting.stripANSI(text).trimmingCharacters(in: .whitespacesAndNewlines)
        let tail = String(clean.suffix(errorTailChars))
        return tail.replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;")
    }

    static func change(from old: Data, to new: Data) -> FormatterChange {
        let oldLines = String(decoding: old, as: UTF8.self).components(separatedBy: "\n")
        let newLines = String(decoding: new, as: UTF8.self).components(separatedBy: "\n")
        guard oldLines.count <= maximumDiffLines, newLines.count <= maximumDiffLines else {
            return FormatterChange(added: [], removedCount: 0, lineDiffAvailable: false)
        }
        var added: [(line: Int, text: String)] = []
        var removed = 0
        for change in newLines.difference(from: oldLines) {
            switch change {
            case .insert(let offset, let element, _): added.append((offset + 1, element))
            case .remove: removed += 1
            }
        }
        added.sort { $0.line < $1.line }
        return FormatterChange(added: added, removedCount: removed, lineDiffAvailable: true)
    }
}
