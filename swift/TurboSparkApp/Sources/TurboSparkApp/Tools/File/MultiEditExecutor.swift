import Foundation

/// Executor for atomic multi-file edits with automated rollback on failure.
public enum MultiEditExecutor {
    public struct EditSpec: Codable, Sendable {
        public var filePath: String
        public var oldString: String
        public var newString: String
        public var replaceAll: Bool

        enum CodingKeys: String, CodingKey {
            case filePath = "file_path"
            case path
            case oldString = "old_string"
            case newString = "new_string"
            case replaceAll = "replace_all"
        }

        public init(filePath: String, oldString: String, newString: String, replaceAll: Bool = false) {
            self.filePath = filePath
            self.oldString = oldString
            self.newString = newString
            self.replaceAll = replaceAll
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.filePath = (try? container.decode(String.self, forKey: .filePath))
                ?? (try? container.decode(String.self, forKey: .path))
                ?? ""
            self.oldString = try container.decode(String.self, forKey: .oldString)
            self.newString = try container.decode(String.self, forKey: .newString)
            self.replaceAll = (try? container.decode(Bool.self, forKey: .replaceAll)) ?? false
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(filePath, forKey: .filePath)
            try container.encode(oldString, forKey: .oldString)
            try container.encode(newString, forKey: .newString)
            try container.encode(replaceAll, forKey: .replaceAll)
        }
    }

    public static func parseEdits(from arguments: [String: String]) throws -> [EditSpec] {
        let decoder = JSONDecoder()
        if let raw = arguments["edits"], let data = raw.data(using: .utf8) {
            if let items = try? decoder.decode([EditSpec].self, from: data) {
                return items
            }
        }
        if let raw = arguments["edits_json"], let data = raw.data(using: .utf8) {
            if let items = try? decoder.decode([EditSpec].self, from: data) {
                return items
            }
        }
        throw NSError(
            domain: "TurboSparkTool",
            code: 40,
            userInfo: [NSLocalizedDescriptionKey: "Missing or invalid 'edits' parameter for multiedit."]
        )
    }

    public static func execute(arguments: [String: String], rootURL: URL) async throws -> String {
        let edits = try parseEdits(from: arguments)
        guard !edits.isEmpty else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 41,
                userInfo: [NSLocalizedDescriptionKey: "No edits provided to multiedit."]
            )
        }

        // Phase 1: Pre-flight validation on all files
        struct PlannedEdit {
            let spec: EditSpec
            let targetURL: URL
            let originalContent: String
            let updatedContent: String
        }

        var plannedEdits: [PlannedEdit] = []
        plannedEdits.reserveCapacity(edits.count)

        // Track content across edits in case the same file is edited multiple times in one batch
        var currentFileContents: [URL: String] = [:]

        for edit in edits {
            let targetURL = try AppToolRegistry.resolveSecurePath(relPath: edit.filePath, rootURL: rootURL)
            try AppToolSandbox.validateWritePath(targetURL, rootURL: rootURL)

            guard FileManager.default.fileExists(atPath: targetURL.path) else {
                throw NSError(
                    domain: "TurboSparkTool",
                    code: 42,
                    userInfo: [NSLocalizedDescriptionKey: "Pre-flight check failed: file not found: \(edit.filePath)"]
                )
            }

            if await FileSnapshotStore.shared.isStale(url: targetURL) {
                throw NSError(
                    domain: "TurboSparkTool",
                    code: 43,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Pre-flight check failed: file '\(edit.filePath)' has been modified on disk since it was last read. Please re-read before editing."
                    ]
                )
            }

            let initialContent: String
            if let cached = currentFileContents[targetURL] {
                initialContent = cached
            } else {
                initialContent = try AppFileReadLimits.readTextFile(at: targetURL, describing: edit.filePath)
                currentFileContents[targetURL] = initialContent
            }

            // Verify oldString match in the current file content
            guard let matchedRange = initialContent.range(of: edit.oldString) else {
                throw NSError(
                    domain: "TurboSparkTool",
                    code: 44,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Pre-flight check failed in '\(edit.filePath)': target old_string not found. Transaction aborted without making any changes."
                    ]
                )
            }

            // Uniqueness check unless replace_all is true
            if !edit.replaceAll {
                var searchRange = initialContent.startIndex..<initialContent.endIndex
                var count = 0
                while let r = initialContent.range(of: edit.oldString, range: searchRange) {
                    count += 1
                    if r.upperBound >= initialContent.endIndex { break }
                    searchRange = r.upperBound..<initialContent.endIndex
                }
                if count > 1 {
                    throw NSError(
                        domain: "TurboSparkTool",
                        code: 45,
                        userInfo: [
                            NSLocalizedDescriptionKey: "Pre-flight check failed in '\(edit.filePath)': target old_string appears \(count) times. Please provide more surrounding context or set replace_all: true. Transaction aborted."
                        ]
                    )
                }
            }

            let updatedContent: String
            if edit.replaceAll {
                updatedContent = initialContent.replacingOccurrences(of: edit.oldString, with: edit.newString)
            } else {
                updatedContent = initialContent.replacingCharacters(in: matchedRange, with: edit.newString)
            }

            currentFileContents[targetURL] = updatedContent
            plannedEdits.append(
                PlannedEdit(
                    spec: edit,
                    targetURL: targetURL,
                    originalContent: initialContent,
                    updatedContent: updatedContent
                )
            )
        }

        // Phase 2: Record backups for all distinct modified URLs
        var originalDiskContents: [URL: String] = [:]
        for planned in plannedEdits {
            if originalDiskContents[planned.targetURL] == nil {
                let diskText = try AppFileReadLimits.readTextFile(at: planned.targetURL, describing: planned.spec.filePath)
                originalDiskContents[planned.targetURL] = diskText
                await FileSnapshotStore.shared.recordBackup(url: planned.targetURL, content: diskText)
            }
        }

        // Phase 3: Apply changes to disk with rollback on failure
        var successfullyWrittenURLs: [URL] = []
        do {
            // Write each unique file's final updated content
            for (targetURL, finalContent) in currentFileContents {
                try finalContent.write(to: targetURL, atomically: true, encoding: .utf8)
                successfullyWrittenURLs.append(targetURL)
                await FileSnapshotStore.shared.recordSnapshot(url: targetURL, content: finalContent)
            }
        } catch {
            // Rollback all written files
            for writtenURL in successfullyWrittenURLs {
                if let orig = originalDiskContents[writtenURL] {
                    try? orig.write(to: writtenURL, atomically: true, encoding: .utf8)
                    await FileSnapshotStore.shared.recordSnapshot(url: writtenURL, content: orig)
                }
            }
            throw NSError(
                domain: "TurboSparkTool",
                code: 46,
                userInfo: [
                    NSLocalizedDescriptionKey: "Failed writing files during multiedit: \(error.localizedDescription). All modified files have been rolled back to their pre-transaction state."
                ]
            )
        }

        var reportLines: [String] = []
        reportLines.append("Atomic multiedit completed successfully across \(currentFileContents.count) file(s) (\(edits.count) total edits applied):")
        for (idx, planned) in plannedEdits.enumerated() {
            reportLines.append("- [\(idx + 1)] `\(planned.spec.filePath)`: replaced \(planned.spec.oldString.count) chars with \(planned.spec.newString.count) chars.")
        }
        return reportLines.joined(separator: "\n")
    }
}
