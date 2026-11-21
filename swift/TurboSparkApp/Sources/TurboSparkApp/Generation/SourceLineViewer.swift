import Darwin
import Foundation

/// Pure decisions behind the read-only source viewer a Read tool card opens
/// (component audit, source-viewer). The viewer shows the current file with
/// the range the tool read highlighted. It never writes, never attaches the
/// file, and sends nothing to the model, so everything here is display math
/// plus the gates that decide whether a path may be read at all.
enum SourceLineViewer {
    /// Larger files are refused rather than read: the sheet is for looking
    /// at a range, and a multi-megabyte string costs memory on a machine
    /// that is usually also holding a loaded model.
    static let maxFileBytes = 2 * 1024 * 1024
    /// Lines shown before and after the highlighted range.
    static let contextLines = 30
    /// One `Text` per line, so the rendered window is capped.
    static let maxRenderedLines = 400
    /// A minified single-line file would otherwise build one huge `Text`.
    static let maxDisplayedLineLength = 2000

    /// The spellings the card's "Lines N-M" capsule reads, so the button and
    /// the capsule can never disagree about which range was read.
    static let startKeys = ["StartLine", "start_line", "startLine", "start"]
    static let endKeys = ["EndLine", "end_line", "endLine", "end"]

    /// The 1-based inclusive range a read call asked for, or nil when either
    /// bound is missing, not a number, below 1, or reversed.
    static func requestedRange(_ arguments: [String: String]) -> ClosedRange<Int>? {
        func value(_ keys: [String]) -> Int? {
            for key in keys {
                if let raw = arguments[key] {
                    return Int(raw.trimmingCharacters(in: .whitespacesAndNewlines))
                }
            }
            return nil
        }
        guard let start = value(startKeys), let end = value(endKeys), start >= 1, end >= start else {
            return nil
        }
        return start...end
    }

    /// `range` trimmed to the file as it is now, or nil when the file is
    /// empty or now ends before the range starts.
    static func clamp(_ range: ClosedRange<Int>, lineCount: Int) -> ClosedRange<Int>? {
        guard lineCount > 0, range.lowerBound <= lineCount else { return nil }
        return range.lowerBound...min(range.upperBound, lineCount)
    }

    /// The lines to render for a clamped `range`: some context either side,
    /// capped at `maxRenderedLines` from the top so the start line (the
    /// scroll target) is always inside the window.
    static func window(for range: ClosedRange<Int>, lineCount: Int) -> ClosedRange<Int> {
        let lower = max(1, range.lowerBound - contextLines)
        var upper = min(lineCount, range.upperBound + contextLines)
        if upper - lower + 1 > maxRenderedLines {
            upper = lower + maxRenderedLines - 1
        }
        return lower...max(lower, upper)
    }

    enum Resolution: Equatable {
        case file(URL)
        /// No project is selected, so there is no root to contain the read.
        case noProject
        case outsideProject
        /// Credentials and keys stay out of the viewer even inside the
        /// project, the same rule mention resolution applies.
        case sensitive
    }

    /// Where `path` points when it stays inside `projectRoot` after
    /// symlinks are resolved on both sides (`PathContainment`). A `~` path is
    /// refused like the read tool refuses it, not read as a folder named `~`.
    static func resolve(path: String, projectRoot: URL?) -> Resolution {
        guard let projectRoot else { return .noProject }
        let cleaned = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty, !cleaned.hasPrefix("~") else { return .outsideProject }
        let candidate = cleaned.hasPrefix("/")
            ? URL(fileURLWithPath: cleaned)
            : projectRoot.appendingPathComponent(cleaned)
        guard let resolved = PathContainment.resolvedIfContained(candidate, in: projectRoot) else {
            return .outsideProject
        }
        if ToolRiskClassifier.isSensitivePath(cleaned) || ToolRiskClassifier.isSensitivePath(resolved.path) {
            return .sensitive
        }
        return .file(resolved)
    }

    struct Loaded: Equatable {
        /// Split on "\n" exactly like the read tool, so line numbers match
        /// what the model was shown.
        var lines: [String]
        var modifiedAt: Date?
    }

    enum LoadFailure: Error, Equatable {
        case notFound, notAFile, tooLarge, notText
    }

    /// Reads `url` as UTF-8 text. Opened with `O_NOFOLLOW` and checked with
    /// `fstat` on the open descriptor, so a symlink swapped in after the
    /// containment check is not followed and a FIFO or device never blocks
    /// the read. At most `maxBytes + 1` bytes are read.
    static func load(_ url: URL, maxBytes: Int = maxFileBytes) -> Result<Loaded, LoadFailure> {
        let fd = open(url.path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK)
        guard fd >= 0 else { return .failure(errno == ENOENT ? .notFound : .notAFile) }
        let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        var info = stat()
        guard fstat(fd, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else { return .failure(.notAFile) }
        guard info.st_size <= Int64(maxBytes) else { return .failure(.tooLarge) }
        let data: Data
        do {
            data = try handle.read(upToCount: maxBytes + 1) ?? Data()
        } catch {
            return .failure(.notAFile)
        }
        guard data.count <= maxBytes else { return .failure(.tooLarge) }
        guard let text = String(data: data, encoding: .utf8) else { return .failure(.notText) }
        let modified = Date(
            timeIntervalSince1970: TimeInterval(info.st_mtimespec.tv_sec)
                + TimeInterval(info.st_mtimespec.tv_nsec) / 1_000_000_000)
        return .success(Loaded(lines: text.components(separatedBy: "\n"), modifiedAt: modified))
    }

    /// Whether the file was written after the tool finished reading it. One
    /// second of slack absorbs filesystem timestamp granularity.
    static func mayHaveChanged(modifiedAt: Date?, readAt: Date) -> Bool {
        guard let modifiedAt else { return false }
        return modifiedAt.timeIntervalSince(readAt) > 1
    }

    /// One line as drawn: a trailing CR dropped, very long lines cut.
    static func displayLine(_ line: String) -> String {
        let trimmed = line.hasSuffix("\r") ? String(line.dropLast()) : line
        guard trimmed.count > maxDisplayedLineLength else { return trimmed }
        return String(trimmed.prefix(maxDisplayedLineLength)) + " ..."
    }

    /// The highlighted lines in full (not the display truncation), joined
    /// with "\n". `range` is 1-based and already clamped.
    static func copyText(lines: [String], range: ClosedRange<Int>) -> String {
        lines[(range.lowerBound - 1)..<range.upperBound]
            .map { $0.hasSuffix("\r") ? String($0.dropLast()) : $0 }
            .joined(separator: "\n")
    }
}
