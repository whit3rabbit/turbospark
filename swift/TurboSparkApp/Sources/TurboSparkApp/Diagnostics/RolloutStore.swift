import Foundation

/// One line of a chat's trajectory log (ZP-03): a model call or a tool
/// execution, appended after it finished.
///
/// The shape is deliberately small. Every field but `v`, `ts` and `kind` is
/// optional so a reader never fails on a line written by a build that knew
/// fewer or more fields. What is NOT here on purpose: the prompt (the
/// transcript already holds it), reasoning text, sampling parameters, and any
/// file path other than a model file's base name.
public struct RolloutEntry: Codable, Equatable, Sendable {
    public static let currentVersion = 1

    public enum Kind: String, Codable, Sendable {
        case modelCall
        case toolExec
    }

    /// Schema version. A line with a version above `currentVersion` is
    /// skipped and counted, never half-read.
    public var v: Int
    public var ts: Date
    public var kind: Kind
    /// `assistant` for a model call, `tool` for a tool execution.
    public var role: String
    /// The agent step the call belongs to, when the writer knew it.
    public var step: Int?
    /// Model file base name (never a directory).
    public var model: String?
    public var promptTokens: Int?
    public var completionTokens: Int?
    /// Prompt tokens served from the previous turn's KV instead of prefilled.
    public var cachedTokens: Int?
    public var latencyMs: Int?
    public var finishReason: String?
    /// Tool name, for `toolExec`.
    public var name: String?
    public var args: [String: String]?
    /// Model output (`modelCall`) or tool output (`toolExec`), clipped.
    public var result: String?
    public var error: String?
    /// True when any text field was clipped by the size caps.
    public var truncated: Bool?
    /// Set only for a call that went to a remote provider (ZP-22): the host
    /// the content was sent to. Absent for every local call.
    public var remoteHost: String?

    public init(
        v: Int = RolloutEntry.currentVersion, ts: Date = Date(), kind: Kind, role: String,
        step: Int? = nil, model: String? = nil, promptTokens: Int? = nil,
        completionTokens: Int? = nil, cachedTokens: Int? = nil, latencyMs: Int? = nil,
        finishReason: String? = nil, name: String? = nil, args: [String: String]? = nil,
        result: String? = nil, error: String? = nil, truncated: Bool? = nil,
        remoteHost: String? = nil
    ) {
        self.v = v
        self.ts = ts
        self.kind = kind
        self.role = role
        self.step = step
        self.model = model
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.cachedTokens = cachedTokens
        self.latencyMs = latencyMs
        self.finishReason = finishReason
        self.name = name
        self.args = args
        self.result = result
        self.error = error
        self.truncated = truncated
        self.remoteHost = remoteHost
    }

    /// Case-insensitive match over every text field, for the pane's search.
    public func matches(_ query: String) -> Bool {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !needle.isEmpty else { return true }
        var haystack: [String] = [role, kind.rawValue]
        haystack.append(contentsOf: [model, finishReason, name, result, error].compactMap { $0 })
        if let args { haystack.append(contentsOf: args.keys); haystack.append(contentsOf: args.values) }
        return haystack.contains { $0.range(of: needle, options: [.caseInsensitive, .diacriticInsensitive]) != nil }
    }
}

/// A parsed line plus where it starts, so the pane can page back from it.
public struct RolloutRecord: Equatable, Sendable, Identifiable {
    public var offset: UInt64
    public var entry: RolloutEntry
    public var id: UInt64 { offset }
}

/// A window of a log. `skippedLines` counts complete lines that did not
/// parse (corrupt, or a newer schema); `truncatedFinalLine` is a last line
/// with no terminating newline, which a crash mid-append leaves behind. Both
/// are counted, never parsed and never an error.
public struct RolloutPage: Equatable, Sendable {
    public var records: [RolloutRecord]
    /// Offset of the first line this window covers; `> 0` means earlier
    /// lines exist.
    public var startOffset: UInt64
    public var skippedLines: Int
    public var truncatedFinalLine: Bool

    public var hasEarlier: Bool { startOffset > 0 }
    /// What the pane's notice reports: lines that exist on disk but are not
    /// shown as entries.
    public var unreadableCount: Int { skippedLines + (truncatedFinalLine ? 1 : 0) }

    public static let empty = RolloutPage(
        records: [], startOffset: 0, skippedLines: 0, truncatedFinalLine: false)
}

public struct RolloutSearchResult: Equatable, Sendable {
    public var records: [RolloutRecord]
    /// More matches existed than `maxResults`.
    public var hitLimit: Bool
    public var skippedLines: Int
}

/// Redaction for trajectory text. Pure and best effort: it removes the
/// shapes of credentials this app knows to look for, and says so. It is run
/// on every line BEFORE it is written (so a secret the model saw never
/// reaches the file) and again on export.
public enum TrajectoryRedactor {
    public static let marker = "[redacted]"

    private static let patterns: [(NSRegularExpression, String)] = {
        func make(_ pattern: String, _ template: String) -> (NSRegularExpression, String)? {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else { return nil }
            return (regex, template)
        }
        let m = NSRegularExpression.escapedTemplate(for: marker)
        return [
            make(#"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\z)"#, m),
            // Headers: keep the name, drop everything to the end of the line.
            make(#"(?i)\b((?:proxy-)?authorization|x-api-key|api-key|x-auth-token|cookie|set-cookie)(\s*[:=]\s*)[^\r\n]+"#, "$1$2" + m),
            make(#"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"#, "Bearer " + m),
            // Secret-named environment variables and shell assignments.
            make(#"(?i)\b([A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|PRIVATE_?KEY|CREDENTIAL)[A-Z0-9_]*)(\s*=\s*)("[^"\n]*"|'[^'\n]*'|[^\s"']+)"#, "$1$2" + m),
            // JSON members with a secret-shaped name.
            make(#"(?i)("[A-Za-z0-9_-]*(?:secret|token|password|passwd|api[_-]?key|authorization|credential)[A-Za-z0-9_-]*"\s*:\s*)"(?:[^"\\]|\\.)*""#, "$1\"" + m + "\""),
            // Credentials in a URL.
            make(#"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"#, "$1" + m + "@"),
            // Well-known token shapes.
            make(#"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"#, m),
            make(#"\bgh[pousr]_[A-Za-z0-9]{20,}"#, m),
            make(#"\bgithub_pat_[A-Za-z0-9_]{20,}"#, m),
            make(#"\bhf_[A-Za-z0-9]{20,}"#, m),
            make(#"\bAKIA[0-9A-Z]{16}\b"#, m),
            make(#"\bxox[abprs]-[A-Za-z0-9-]{10,}"#, m),
            make(#"\bAIza[0-9A-Za-z_-]{30,}"#, m),
            make(#"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"#, m),
        ].compactMap { $0 }
    }()

    private static let sensitiveKey = try? NSRegularExpression(
        pattern: #"(?i)secret|token|passw|api[_-]?key|authorization|credential|cookie|private[_-]?key"#)

    public static func redact(_ text: String) -> String {
        var output = text
        for (regex, template) in patterns {
            let range = NSRange(output.startIndex..., in: output)
            output = regex.stringByReplacingMatches(in: output, options: [], range: range, withTemplate: template)
        }
        return output
    }

    /// A key that names a secret loses its whole value, whatever the value
    /// looks like.
    public static func isSensitiveKey(_ key: String) -> Bool {
        guard let sensitiveKey else { return false }
        return sensitiveKey.firstMatch(in: key, range: NSRange(key.startIndex..., in: key)) != nil
    }
}

/// Per-chat, local-only, append-after trajectory logs (ZP-03).
///
/// **THE FILES ARE PLAINTEXT, SO THE GATES ARE STRICT.** One JSON object per
/// line under `trajectories/rollout-<chat id>.jsonl` in the profile's store
/// root, mode 0600 in a 0700 folder. The vault's SQLCipher database is the
/// encrypted home for chats; a JSONL file cannot live there, and encrypting
/// lines needs a key and a crypto decision this draft does not make
/// (`docs/designs/zcode-parity/ZP-03-model-trajectory-viewer.md`). So the
/// feature is off by default, never records a Ghost chat, and records nothing
/// for a passphrase-protected profile (the same posture as
/// `ChatRecallStore`). Text is redacted and size-capped BEFORE it is written.
/// Nothing is ever sent anywhere.
///
/// **CRASH SAFETY.** A line is appended whole, after the call finished, never
/// during streaming. A crash can still leave an unterminated last line. The
/// next append first closes that line with a newline so the new entry does
/// not fuse with it, and readers count it as `truncatedFinalLine` instead of
/// parsing it.
///
/// **CORRUPTION.** A file whose first complete line is not a JSON object with
/// a schema version is quarantined (`corrupt-...`) before anything is
/// appended, the same rule as `AppJSONStore`. A bad line in the middle is
/// skipped and counted; the file is never rewritten to remove it, except
/// that the size cap drops the OLDEST whole lines.
///
/// Every operation runs on one private serial queue, which is also the
/// per-chat append lock: an append, a read and a trim never interleave.
public final class RolloutStore: @unchecked Sendable {
    public struct Limits: Sendable {
        /// Per text field (output, tool result, error).
        public var maxFieldChars = 4_000
        /// Per tool-argument value, and the number of arguments kept.
        public var maxArgValueChars = 1_000
        public var maxArgs = 16
        /// Per chat. Past it the oldest lines are dropped down to
        /// `trimToBytes`.
        public var maxFileBytes = 4 * 1_024 * 1_024
        public var trimToBytes = 3 * 1_024 * 1_024
        /// A log untouched for this long is deleted by `pruneExpired`.
        public var retentionDays = 30
        public init() {}
    }

    public static let defaultTailEntries = 500
    public static let fileExtension = "jsonl"
    static let filePrefix = "rollout-"
    static let quarantinePrefix = "corrupt-"
    /// Most bytes a single page read may buffer.
    static let maxPageBytes = 2 * 1_024 * 1_024
    private static let chunkBytes = 64 * 1_024
    private static let newline = UInt8(ascii: "\n")

    public static let shared = RolloutStore(
        directory: AppStorageRoot.directory.appendingPathComponent("trajectories", isDirectory: true))

    public let directory: URL
    public let limits: Limits
    private let isProtected: @Sendable () -> Bool
    private let queue = DispatchQueue(label: "com.turbospark.rollout-store", qos: .utility)
    /// Chats whose file passed the first-append check this process.
    private var verified: Set<UUID> = []

    public init(
        directory: URL, limits: Limits = Limits(),
        isProtected: (@Sendable () -> Bool)? = nil
    ) {
        self.directory = directory
        self.limits = limits
        self.isProtected = isProtected ?? { ProfileVaultStore.shared.isProtected }
    }

    public func fileURL(chatID: UUID) -> URL {
        directory.appendingPathComponent("\(Self.filePrefix)\(chatID.uuidString).\(Self.fileExtension)")
    }

    // MARK: - Writing

    /// Sanitizes (redact, then clip) and appends one entry. Returns at once;
    /// the work runs on the store's queue. Failures are silent: a diagnostic
    /// log must never surface an error into a chat turn.
    public func append(_ entry: RolloutEntry, chatID: UUID) {
        queue.async { self.appendLocked(entry, chatID: chatID) }
    }

    /// Blocks until every queued operation finished. For tests, a quit-time
    /// flush and any caller that reads right after an append.
    public func flush() { queue.sync {} }

    private func appendLocked(_ entry: RolloutEntry, chatID: UUID) {
        guard !isProtected() else { return }
        let clean = Self.sanitize(entry, limits: limits)
        guard var line = try? Self.encoder.encode(clean) else { return }
        line.append(Self.newline)
        let fm = FileManager.default
        do {
            try ensureDirectory()
            let url = fileURL(chatID: chatID)
            if !verified.contains(chatID) {
                quarantineIfCorrupt(url)
                verified.insert(chatID)
            }
            if !fm.fileExists(atPath: url.path) {
                guard fm.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600])
                else { return }
            }
            let handle = try FileHandle(forUpdating: url)
            defer { try? handle.close() }
            let size = try handle.seekToEnd()
            if size > 0 {
                try handle.seek(toOffset: size - 1)
                if try handle.read(upToCount: 1)?.first != Self.newline {
                    // A crash left an unterminated line. Close it so this
                    // entry starts on its own line; the reader counts the
                    // closed fragment as skipped.
                    try handle.seekToEnd()
                    try handle.write(contentsOf: Data([Self.newline]))
                }
            }
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
            if size + UInt64(line.count) > UInt64(limits.maxFileBytes) {
                try? handle.close()
                trimLocked(url)
            }
        } catch {
            // See `append`: never surfaced.
        }
    }

    private func ensureDirectory() throws {
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
    }

    /// Renames a file whose first complete line is not a versioned JSON
    /// object instead of appending to (and so burying) it.
    private func quarantineIfCorrupt(_ url: URL) {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return }
        defer { try? handle.close() }
        let head = (try? handle.read(upToCount: Self.chunkBytes)) ?? Data()
        guard !head.isEmpty, let end = head.firstIndex(of: Self.newline) else { return }
        let firstLine = head[head.startIndex..<end]
        if let object = try? JSONSerialization.jsonObject(with: firstLine) as? [String: Any],
           object["v"] is Int { return }
        let stamp = Self.quarantineStamp.string(from: Date())
        let target = directory.appendingPathComponent(
            "\(Self.quarantinePrefix)\(stamp)-\(url.lastPathComponent)")
        try? FileManager.default.moveItem(at: url, to: target)
    }

    /// Drops the oldest whole lines so the file ends near `trimToBytes`.
    private func trimLocked(_ url: URL) {
        guard let data = try? Data(contentsOf: url), data.count > limits.trimToBytes else { return }
        var cut = data.count - limits.trimToBytes
        guard let newlineIndex = data[cut...].firstIndex(of: Self.newline) else { return }
        cut = newlineIndex + 1
        let kept = data[cut...]
        let temporary = url.appendingPathExtension("tmp")
        guard FileManager.default.createFile(
            atPath: temporary.path, contents: kept, attributes: [.posixPermissions: 0o600])
        else { return }
        _ = try? FileManager.default.replaceItemAt(url, withItemAt: temporary)
        try? FileManager.default.removeItem(at: temporary)
    }

    // MARK: - Reading

    /// The last `maxEntries` lines of the log.
    public func readTail(chatID: UUID, maxEntries: Int = defaultTailEntries) -> RolloutPage {
        queue.sync {
            let url = fileURL(chatID: chatID)
            guard let size = Self.fileSize(url) else { return .empty }
            return readBackwardLocked(url, end: size, maxEntries: maxEntries, isTail: true)
        }
    }

    /// The `maxEntries` lines before `offset` (a `startOffset` from an
    /// earlier page).
    public func readBefore(
        chatID: UUID, offset: UInt64, maxEntries: Int = defaultTailEntries
    ) -> RolloutPage {
        queue.sync {
            let url = fileURL(chatID: chatID)
            guard Self.fileSize(url) != nil else { return .empty }
            return readBackwardLocked(url, end: offset, maxEntries: maxEntries, isTail: false)
        }
    }

    private func readBackwardLocked(
        _ url: URL, end: UInt64, maxEntries: Int, isTail: Bool
    ) -> RolloutPage {
        guard end > 0, maxEntries > 0, let handle = try? FileHandle(forReadingFrom: url) else {
            return .empty
        }
        defer { try? handle.close() }
        var buffer = Data()
        var position = end
        // Read backward until there are enough line breaks, the byte cap is
        // reached, or the start of the file is.
        while position > 0 {
            let chunk = UInt64(min(Self.chunkBytes, Int(position)))
            let start = position - chunk
            guard (try? handle.seek(toOffset: start)) != nil,
                  let part = try? handle.read(upToCount: Int(chunk)), part.count == Int(chunk)
            else { break }
            buffer.insert(contentsOf: part, at: 0)
            position = start
            if buffer.count >= Self.maxPageBytes { break }
            if buffer.reduce(0, { $1 == Self.newline ? $0 + 1 : $0 }) > maxEntries + 1 { break }
        }
        var base = position
        var truncatedFinal = false
        var segments = buffer.split(separator: Self.newline, omittingEmptySubsequences: false)
            .map { Data($0) }
        // The buffer ends at `end`. For the tail that is end of file; an
        // unterminated last line is a crash fragment. For an earlier page
        // `end` is a line start, so the final segment is empty.
        if let last = segments.last {
            if last.isEmpty {
                segments.removeLast()
            } else if isTail {
                segments.removeLast()
                truncatedFinal = true
            }
        }
        // Unless the read reached the start of the file, the first segment
        // may be the end of a longer line and is not a line of its own.
        if base > 0, !segments.isEmpty {
            base += UInt64(segments.removeFirst().count + 1)
        }
        // Offsets of every remaining line, then keep the last `maxEntries`.
        var offsets: [UInt64] = []
        var cursor = base
        for segment in segments { offsets.append(cursor); cursor += UInt64(segment.count + 1) }
        let keep = min(maxEntries, segments.count)
        let firstKept = segments.count - keep
        var records: [RolloutRecord] = []
        var skipped = 0
        for index in firstKept..<segments.count {
            if let entry = Self.parse(segments[index]) {
                records.append(RolloutRecord(offset: offsets[index], entry: entry))
            } else {
                skipped += 1
            }
        }
        let startOffset = keep == 0 ? base : offsets[firstKept]
        return RolloutPage(
            records: records, startOffset: startOffset, skippedLines: skipped,
            truncatedFinalLine: truncatedFinal)
    }

    /// Scans the whole log, oldest to newest, for `query`.
    public func search(chatID: UUID, query: String, maxResults: Int = 200) -> RolloutSearchResult {
        queue.sync {
            var found: [RolloutRecord] = []
            var skipped = 0
            var hitLimit = false
            forEachLineLocked(fileURL(chatID: chatID)) { offset, line in
                guard let entry = Self.parse(line) else { skipped += 1; return true }
                guard entry.matches(query) else { return true }
                if found.count >= maxResults { hitLimit = true; return false }
                found.append(RolloutRecord(offset: offset, entry: entry))
                return true
            }
            return RolloutSearchResult(records: found, hitLimit: hitLimit, skippedLines: skipped)
        }
    }

    /// Streams complete lines forward in bounded chunks. An unterminated
    /// final line is never handed to `body`.
    private func forEachLineLocked(_ url: URL, _ body: (UInt64, Data) -> Bool) {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return }
        defer { try? handle.close() }
        var carry = Data()
        var carryStart: UInt64 = 0
        while let chunk = try? handle.read(upToCount: Self.chunkBytes), !chunk.isEmpty {
            carry.append(chunk)
            var lineStart = carry.startIndex
            while let end = carry[lineStart...].firstIndex(of: Self.newline) {
                let offset = carryStart + UInt64(lineStart - carry.startIndex)
                if !body(offset, Data(carry[lineStart..<end])) { return }
                lineStart = end + 1
            }
            carryStart += UInt64(lineStart - carry.startIndex)
            carry = Data(carry[lineStart...])
        }
    }

    // MARK: - Export, delete, retention

    /// Writes the whole log to `destination` with redaction applied again.
    /// Lines that do not parse are not exported. Returns the entry count.
    /// Best effort by design: see `TrajectoryRedactor`.
    @discardableResult
    public func exportRedacted(chatID: UUID, to destination: URL) throws -> Int {
        try queue.sync {
            var output = Data()
            var count = 0
            forEachLineLocked(fileURL(chatID: chatID)) { _, line in
                guard let entry = Self.parse(line),
                      var encoded = try? Self.encoder.encode(Self.sanitize(entry, limits: limits))
                else { return true }
                encoded.append(Self.newline)
                output.append(encoded)
                count += 1
                return true
            }
            try output.write(to: destination, options: .atomic)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: destination.path)
            return count
        }
    }

    public func hasLog(chatID: UUID) -> Bool {
        queue.sync { Self.fileSize(fileURL(chatID: chatID)) ?? 0 > 0 }
    }

    public func byteCount(chatID: UUID) -> UInt64 {
        queue.sync { Self.fileSize(fileURL(chatID: chatID)) ?? 0 }
    }

    /// Deletes one chat's log. Queued behind earlier appends, so a delete
    /// right after an append leaves nothing.
    public func delete(chatID: UUID) {
        queue.async {
            self.verified.remove(chatID)
            try? FileManager.default.removeItem(at: self.fileURL(chatID: chatID))
        }
    }

    /// Deletes every log and every quarantined file.
    public func deleteAll() {
        queue.sync {
            verified.removeAll()
            for url in storedFiles() { try? FileManager.default.removeItem(at: url) }
        }
    }

    /// Deletes logs (and quarantined files) not modified within the
    /// retention window. Returns how many files went.
    @discardableResult
    public func pruneExpired(now: Date = Date()) -> Int {
        queue.sync {
            let cutoff = now.addingTimeInterval(-Double(limits.retentionDays) * 86_400)
            var removed = 0
            for url in storedFiles() {
                let modified = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
                    .contentModificationDate ?? Date()
                if modified < cutoff, (try? FileManager.default.removeItem(at: url)) != nil {
                    removed += 1
                }
            }
            verified.removeAll()
            return removed
        }
    }

    private func storedFiles() -> [URL] {
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)) ?? []
        return urls.filter {
            let name = $0.lastPathComponent
            return name.hasPrefix(Self.filePrefix) || name.hasPrefix(Self.quarantinePrefix)
        }
    }

    // MARK: - Pure helpers

    /// Redacts, then clips. Redaction runs first on a bounded prefix so a
    /// secret cannot be cut in half by the clip and survive as a fragment.
    static func sanitize(_ entry: RolloutEntry, limits: Limits) -> RolloutEntry {
        var out = entry
        var clipped = false
        func text(_ value: String?, _ max: Int) -> String? {
            guard let value else { return nil }
            let (clean, wasClipped) = clip(TrajectoryRedactor.redact(String(value.prefix(64_000))), max)
            clipped = clipped || wasClipped || value.count > 64_000
            return clean
        }
        out.result = text(entry.result, limits.maxFieldChars)
        out.error = text(entry.error, limits.maxFieldChars)
        out.name = text(entry.name, 200)
        out.model = text(entry.model, 200)
        out.remoteHost = text(entry.remoteHost, 200)
        out.finishReason = text(entry.finishReason, 100)
        if let args = entry.args {
            var kept: [String: String] = [:]
            for key in args.keys.sorted().prefix(limits.maxArgs) {
                let safeKey = String(key.prefix(100))
                if TrajectoryRedactor.isSensitiveKey(key) {
                    kept[safeKey] = TrajectoryRedactor.marker
                } else {
                    kept[safeKey] = text(args[key], limits.maxArgValueChars)
                }
            }
            clipped = clipped || args.count > limits.maxArgs
            out.args = kept
        }
        if clipped { out.truncated = true }
        return out
    }

    private static func clip(_ text: String, _ max: Int) -> (String, Bool) {
        guard text.count > max else { return (text, false) }
        return (String(text.prefix(max)) + " [truncated]", true)
    }

    private static func parse(_ line: Data) -> RolloutEntry? {
        guard !line.isEmpty, let entry = try? decoder.decode(RolloutEntry.self, from: line),
              entry.v >= 1, entry.v <= RolloutEntry.currentVersion
        else { return nil }
        return entry
    }

    private static func fileSize(_ url: URL) -> UInt64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? UInt64) ?? nil
    }

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    private static let quarantineStamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd'T'HHmmss"
        return formatter
    }()
}
