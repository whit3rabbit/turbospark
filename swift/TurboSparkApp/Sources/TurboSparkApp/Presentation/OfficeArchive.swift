import Foundation

/// Safe, resource-bounded reader for OpenXML Office document archives (.docx, .xlsx, .pptx).
///
/// Wraps system `/usr/bin/unzip` with explicit timeout budgets, maximum output size limits,
/// and streaming pipe polling to protect against decompression bombs or hung processes.
final class OfficeArchive {
    /// File URL of the Office archive on disk.
    let url: URL

    /// Extraction limits governing timeout, per-entry byte budget, and total document bytes.
    let limits: DocumentTextExtractor.Limits

    /// Set of entry relative paths discovered within the archive.
    let entries: Set<String>

    /// Running counter of bytes decompressed across all entries read so far.
    private var consumedBytes = 0

    /// Inspects the archive table of contents and verifies entries exist within limits.
    ///
    /// - Parameters:
    ///   - url: File URL of the zip archive.
    ///   - limits: Extraction resource limits.
    /// - Throws: `DocumentTextExtractionError` if unreadable, invalid, or exceeding size/timeout bounds.
    init(url: URL, limits: DocumentTextExtractor.Limits) throws {
        self.url = url
        self.limits = limits
        let result = try Self.runUnzip(
            arguments: ["-l", url.path],
            maximumOutputBytes: limits.maximumEntryBytes,
            timeout: limits.timeout,
            fileName: url.lastPathComponent)
        try Self.check(result, fileName: url.lastPathComponent)
        guard result.status == 0,
              let listing = String(data: result.output, encoding: .utf8) else {
            throw DocumentTextExtractionError.invalidArchive(url.lastPathComponent)
        }

        var parsedEntries: Set<String> = []
        for line in listing.split(separator: "\n") {
            let fields = line.split(whereSeparator: \.isWhitespace)
            guard fields.count >= 4, Int(fields[0]) != nil else { continue }
            parsedEntries.insert(fields.dropFirst(3).joined(separator: " "))
        }
        guard !parsedEntries.isEmpty else {
            throw DocumentTextExtractionError.invalidArchive(url.lastPathComponent)
        }
        entries = parsedEntries
    }

    /// Extracts uncompressed bytes for a named entry path inside the archive.
    ///
    /// - Parameter entry: Relative path of the entry (e.g. `word/document.xml`).
    /// - Returns: Decompressed data bytes.
    /// - Throws: `DocumentTextExtractionError` if entry is missing or limits are exceeded.
    func data(for entry: String) throws -> Data {
        guard entries.contains(entry) else {
            throw DocumentTextExtractionError.invalidArchive(url.lastPathComponent)
        }
        let budget = min(
            limits.maximumEntryBytes,
            max(0, limits.maximumSelectedBytes - consumedBytes))
        let result = try Self.runUnzip(
            arguments: ["-p", url.path, entry],
            maximumOutputBytes: budget,
            timeout: limits.timeout,
            fileName: url.lastPathComponent)
        try Self.check(result, fileName: url.lastPathComponent)
        guard result.status == 0 else {
            throw DocumentTextExtractionError.invalidArchive(url.lastPathComponent)
        }
        consumedBytes += result.output.count
        return result.output
    }

    private static func check(_ result: UnzipResult, fileName: String) throws {
        switch result.outcome {
        case .completed:
            return
        case .exceededLimit:
            throw DocumentTextExtractionError.documentTooLarge(fileName)
        case .timedOut:
            throw DocumentTextExtractionError.extractionTimedOut(fileName)
        }
    }

    private enum UnzipOutcome {
        case completed
        case exceededLimit
        case timedOut
    }

    private struct UnzipResult {
        var status: Int32
        var output: Data
        var outcome: UnzipOutcome
    }

    /// Executes unzip subprocess with timeout and byte capping via descriptor polling.
    private static func runUnzip(
        arguments: [String],
        maximumOutputBytes: Int,
        timeout: TimeInterval,
        fileName: String
    ) throws -> UnzipResult {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = arguments
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            throw DocumentTextExtractionError.unreadableFile(fileName)
        }

        let readEnd = pipe.fileHandleForReading
        let descriptor = readEnd.fileDescriptor
        let deadline = Date().addingTimeInterval(timeout)
        var buffer = [UInt8](repeating: 0, count: 64 * 1_024)
        var output = Data()
        var outcome = UnzipOutcome.completed

        while true {
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else {
                outcome = .timedOut
                break
            }
            var poller = pollfd(
                fd: descriptor,
                events: Int16(POLLIN),
                revents: 0)
            let ready = poll(&poller, 1, Int32(clamping: Int(remaining * 1_000) + 1))
            if ready < 0 {
                guard errno == EINTR else { break }
                continue
            }
            if ready == 0 {
                outcome = .timedOut
                break
            }
            let count = read(descriptor, &buffer, buffer.count)
            if count < 0 {
                guard errno == EINTR else { break }
                continue
            }
            if count == 0 { break }
            guard output.count + count <= maximumOutputBytes else {
                outcome = .exceededLimit
                break
            }
            output.append(contentsOf: buffer[0..<count])
        }

        if outcome != .completed {
            process.terminate()
        }
        try? readEnd.close()
        process.waitUntilExit()
        return UnzipResult(
            status: process.terminationStatus,
            output: output,
            outcome: outcome)
    }
}
