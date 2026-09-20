import CryptoKit
import Foundation

/// Durable metadata for exact, oversized normal-chat tool output.
/// The bytes themselves live outside the transcript so normal archive rewrites
/// stay bounded. Ghost output is kept by `GhostChatPayload` instead.
public struct ToolObservationRef: Codable, Equatable, Sendable {
    public var id: UUID
    public var sourceHash: String
    public var byteCount: Int

    public init(id: UUID = UUID(), sourceHash: String, byteCount: Int) {
        self.id = id
        self.sourceHash = sourceHash
        self.byteCount = byteCount
    }
}

/// Displayable, model-facing facts about an efficiency transformation.
public struct ToolEfficiencyOutcome: Codable, Equatable, Sendable {
    public var validation: String?
    public var archivedOutput: Bool
    public var verifiedReduction: Bool
    public var boundaryCompaction: Bool

    public init(
        validation: String? = nil,
        archivedOutput: Bool = false,
        verifiedReduction: Bool = false,
        boundaryCompaction: Bool = false
    ) {
        self.validation = validation
        self.archivedOutput = archivedOutput
        self.verifiedReduction = verifiedReduction
        self.boundaryCompaction = boundaryCompaction
    }

    public var hasAnyOutcome: Bool {
        validation != nil || archivedOutput || verifiedReduction || boundaryCompaction
    }
}

/// The exact, profile-aware archive for normal chat observations.
public final class ToolObservationStore: @unchecked Sendable {
    public static let shared = ToolObservationStore()

    private let fileManager: FileManager
    private let rootURL: URL
    private let vaultBacked: Bool
    private let repository: ProfileRepository

    public init(
        rootURL: URL? = nil,
        fileManager: FileManager = .default,
        repository: ProfileRepository = .shared
    ) {
        self.vaultBacked = rootURL == nil
        self.rootURL = rootURL ?? AppStorageRoot.subdirectory("tool-observations")
        self.fileManager = fileManager
        self.repository = repository
    }

    private func chatDirectory(_ chatID: UUID) -> URL {
        rootURL.appendingPathComponent(chatID.uuidString, isDirectory: true)
    }

    private func fileURL(chatID: UUID, observationID: UUID) -> URL {
        chatDirectory(chatID).appendingPathComponent(observationID.uuidString + ".bin")
    }

    private func recordKey(chatID: UUID, observationID: UUID) -> String {
        Self.recordKey(relativePath: "\(chatID.uuidString)/\(observationID.uuidString).bin")
    }

    static func recordKey(relativePath: String) -> String {
        "tool-observation:\(relativePath)"
    }

    public func archive(_ bytes: Data, chatID: UUID) throws -> ToolObservationRef {
        let reference = ToolObservationRef(
            sourceHash: Self.hash(bytes), byteCount: bytes.count)
        if vaultBacked {
            try repository.saveRawRecord(
                bytes, key: recordKey(chatID: chatID, observationID: reference.id))
            return reference
        }
        let directory = chatDirectory(chatID)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        try bytes.write(to: fileURL(chatID: chatID, observationID: reference.id), options: .atomic)
        return reference
    }

    public func load(_ reference: ToolObservationRef, chatID: UUID) throws -> Data {
        let data: Data
        if vaultBacked {
            guard let stored = try repository.rawRecord(
                key: recordKey(chatID: chatID, observationID: reference.id))
            else { throw ObservationError.missing }
            data = stored
        } else {
            data = try Data(contentsOf: fileURL(chatID: chatID, observationID: reference.id))
        }
        guard data.count == reference.byteCount, Self.hash(data) == reference.sourceHash else {
            throw ObservationError.integrityFailed
        }
        return data
    }

    public func recall(
        _ reference: ToolObservationRef, chatID: UUID, offsetBytes: Int, maxBytes: Int
    ) throws -> String {
        guard offsetBytes >= 0, maxBytes > 0 else { throw ObservationError.invalidRange }
        let data = try load(reference, chatID: chatID)
        guard offsetBytes < data.count || (offsetBytes == 0 && data.isEmpty) else {
            throw ObservationError.invalidRange
        }
        let end = min(data.count, offsetBytes + maxBytes)
        let page = data.subdata(in: offsetBytes..<end)
        return String(decoding: page, as: UTF8.self)
    }

    public func delete(chatID: UUID) {
        if vaultBacked {
            let prefix = Self.recordKey(relativePath: "\(chatID.uuidString)/")
            guard let rows = try? repository.rawRecords(prefix: prefix) else { return }
            for (key, _) in rows { try? repository.deleteRecord(key: key) }
            return
        }
        try? fileManager.removeItem(at: chatDirectory(chatID))
    }

    public static func hash(_ bytes: Data) -> String {
        SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }

    public enum ObservationError: LocalizedError {
        case invalidRange
        case integrityFailed
        case missing

        public var errorDescription: String? {
            switch self {
            case .invalidRange: return "The requested observation byte range is invalid."
            case .integrityFailed: return "The archived observation did not pass its integrity check."
            case .missing: return "The archived observation is missing."
            }
        }
    }
}

/// In-memory source bytes for Ghost chats. The payload containing this map is
/// AES-GCM sealed by `GhostChatVault` and is never sent to disk.
public struct GhostToolObservation: Codable, Equatable {
    public var reference: ToolObservationRef
    public var bytes: Data

    public init(reference: ToolObservationRef, bytes: Data) {
        self.reference = reference
        self.bytes = bytes
    }
}

/// Action-fusion input and path fingerprint helpers. The agent loop owns
/// policy and hooks; this type deliberately contains no permission bypass.
enum ToolActionFusion {
    struct Validation: Equatable, Sendable {
        var command: String
        var timeoutMs: Int?
    }

    struct Fingerprint: Equatable, Sendable {
        var path: String
        var exists: Bool
        var hash: String?
    }

    static func validation(for call: AppToolCall) -> Validation? {
        guard mutationToolNames.contains(call.name.lowercased()),
              let command = call.arguments["validate_command"]?
                  .trimmingCharacters(in: .whitespacesAndNewlines),
              !command.isEmpty
        else { return nil }
        let timeout = Int(call.arguments["validate_timeout_ms"] ?? "")
        return Validation(command: command, timeoutMs: timeout)
    }

    static func validationCall(for validation: Validation) -> AppToolCall {
        var arguments = ["command": validation.command]
        if let timeout = validation.timeoutMs, timeout > 0 {
            arguments["timeout_ms"] = String(timeout)
        }
        return AppToolCall(
            name: "run_command", arguments: arguments, rawInvocation: "fused validation",
            category: .terminal)
    }

    static func targetPaths(for call: AppToolCall) -> [String] {
        let name = call.name.lowercased()
        if name == "apply_patch" || name == "applypatch" {
            let patch = call.arguments["patch_text"] ?? call.arguments["patchText"]
                ?? call.arguments["patch"] ?? ""
            return patch.split(separator: "\n").compactMap { line in
                let prefixes = ["*** Add File: ", "*** Update File: ", "*** Delete File: "]
                let text = String(line)
                guard let prefix = prefixes.first(where: { text.hasPrefix($0) }) else { return nil }
                return String(text.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
            }
        }
        if let path = call.arguments["path"] ?? call.arguments["file_path"]
            ?? call.arguments["filePath"] ?? call.arguments["file"] {
            return [path]
        }
        return []
    }

    static func fingerprints(for call: AppToolCall, rootURL: URL) -> [Fingerprint] {
        targetPaths(for: call).map { path in fingerprint(path: path, rootURL: rootURL) }
    }

    static func postMutationVerified(
        before: [Fingerprint], after: [Fingerprint], for call: AppToolCall
    ) -> Bool {
        guard !after.isEmpty, before.count == after.count else { return false }
        let patchOperations = patchOperationKinds(call)
        for (index, current) in after.enumerated() {
            let previous = before[index]
            switch patchOperations[current.path] {
            case "add":
                guard !previous.exists, current.exists else { return false }
            case "delete":
                guard previous.exists, !current.exists else { return false }
            case "update":
                guard current.exists, previous.hash != current.hash else { return false }
            default:
                guard current.exists, previous.hash != current.hash || !previous.exists else { return false }
            }
        }
        return true
    }

    private static let mutationToolNames: Set<String> = [
        "write_file", "save_file", "filewrite", "write", "create_file", "write_to_file",
        "edit_file", "fileedit", "edit", "replace_file_content", "editor", "apply_patch", "applypatch",
    ]

    private static func fingerprint(path: String, rootURL: URL) -> Fingerprint {
        let cleaned = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.hasPrefix("/"), !cleaned.hasPrefix("~") else {
            return Fingerprint(path: path, exists: false, hash: nil)
        }
        let candidate = rootURL.appendingPathComponent(cleaned)
        guard let url = PathContainment.resolvedIfContained(candidate, in: rootURL) else {
            return Fingerprint(path: path, exists: false, hash: nil)
        }
        guard let data = try? Data(contentsOf: url) else {
            return Fingerprint(path: path, exists: false, hash: nil)
        }
        return Fingerprint(path: path, exists: true, hash: ToolObservationStore.hash(data))
    }

    private static func patchOperationKinds(_ call: AppToolCall) -> [String: String] {
        let patch = call.arguments["patch_text"] ?? call.arguments["patchText"]
            ?? call.arguments["patch"] ?? ""
        var result: [String: String] = [:]
        for line in patch.split(separator: "\n") {
            let text = String(line)
            for (prefix, kind) in [
                ("*** Add File: ", "add"),
                ("*** Update File: ", "update"),
                ("*** Delete File: ", "delete"),
            ] where text.hasPrefix(prefix) {
                result[String(text.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)] = kind
            }
        }
        return result
    }
}

enum ToolOutputProjection {
    static let fullSendLimit = 2
    static let headBytes = 1_024
    static let tailBytes = 1_024
    static let archiveThresholdBytes = 10 * 1_024

    static func placeholder(for reference: ToolObservationRef) -> String {
        "[Archived tool output: \(reference.byteCount) bytes, sha256 \(reference.sourceHash). "
            + "Use recall_tool_output(observation_id: \"\(reference.id.uuidString)\", "
            + "offset_bytes: 0, max_bytes: 4096) to read an exact byte range.]"
    }

    static func headTail(_ bytes: Data, reference: ToolObservationRef) -> String {
        guard bytes.count > headBytes + tailBytes else { return String(decoding: bytes, as: UTF8.self) }
        let head = String(decoding: bytes.prefix(headBytes), as: UTF8.self)
        let tail = String(decoding: bytes.suffix(tailBytes), as: UTF8.self)
        return head + "\n... [archived middle omitted] ...\n" + tail + "\n" + placeholder(for: reference)
    }
}

enum TodoBoundaryCompactionPolicy {
    static func completedTransition(previous: [TodoItem], current: [TodoItem]) -> Bool {
        let oldByID = Dictionary(uniqueKeysWithValues: previous.map { ($0.id, $0) })
        return current.contains { item in item.isCompleted && oldByID[item.id]?.isCompleted != true }
    }

    static func shouldCompact(
        previous: [TodoItem], current: [TodoItem], messageCount: Int, boundary: Int, keepRecent: Int
    ) -> Bool {
        guard completedTransition(previous: previous, current: current),
              current.contains(where: { !$0.isCompleted && !$0.isCancelled }),
              let newBoundary = AppChatCompaction.newBoundary(messageCount: messageCount, keepRecent: keepRecent),
              newBoundary > boundary
        else { return false }
        // A completed boundary must remove enough history to pay for the
        // compaction request and its replacement summary. Two rows is the
        // smallest useful saving and avoids churning a one-row transcript.
        return newBoundary - boundary >= 2
    }
}
