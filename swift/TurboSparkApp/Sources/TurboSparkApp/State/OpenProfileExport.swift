import CryptoKit
import Foundation

public struct ProfileExportSnapshot: Sendable {
    public struct MemoryFile: Sendable {
        public var archivePath: String
        public var sourceURL: URL?
        public var data: Data?
    }

    public var profile: UserProfile
    public var exportedAt: Date
    public var appVersion: String
    public var modelAlias: String?
    public var chats: [AppChat]
    public var projects: AppProjectArchive
    /// Allowlisted per-profile JSON records, keyed by original file name.
    public var settingsFiles: [String: Data]
    public var chatFiles: [MemoryFile]
    public var memoryFiles: [MemoryFile]
}

enum OpenProfileExport {
    struct Category: Identifiable, Equatable {
        let id: String
    }

    struct Manifest: Codable, Equatable {
        var formatVersion: Int
        var kind: String
        var profileID: String
        var profileName: String
        var exportedAt: Date
        var appVersion: String
        var includedCategories: [String]
    }

    struct AssetIndexRow: Codable, Equatable {
        var managedReference: String
        var exportedPath: String
        var originalFileName: String
        var mimeType: String?
        var byteCount: Int64
        var category: String
    }

    static let categories: [Category] = [
        .init(id: "settings"),
        .init(id: "chats"),
        .init(id: "generated-images"),
        .init(id: "attachments"),
        .init(id: "projects"),
        .init(id: "memory"),
    ]
    static let allCategoryIDs = Set(categories.map(\.id))

    static func export(
        snapshot: ProfileExportSnapshot,
        included: Set<String>,
        destination: URL,
        assets: ManagedAssetStore = .shared
    ) throws {
        let unknown = included.subtracting(allCategoryIDs)
        guard unknown.isEmpty else {
            throw EncryptedProfileBackup.BackupError.unexpectedEntry(unknown.sorted().joined(separator: ", "))
        }
        FileManager.default.createFile(atPath: destination.path, contents: nil)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: destination.path)
        let handle = try FileHandle(forWritingTo: destination)
        defer { try? handle.close() }
        let zip = ProfileZipStreamWriter { try handle.write(contentsOf: $0) }
        do {
            var checksums: [ProfileZipStreamWriter.EntryDigest] = []
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let compactEncoder = JSONEncoder()
            compactEncoder.dateEncodingStrategy = .iso8601
            compactEncoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]

            let assetRows = try buildAssetIndex(
                chats: snapshot.chats, included: included, store: assets)
            let assetPathByReference = Dictionary(
                uniqueKeysWithValues: assetRows.map { ($0.managedReference, $0.exportedPath) })

            if included.contains("chats") {
                for original in snapshot.chats where !original.isGhost {
                    let chat = portableChat(original, assetPaths: assetPathByReference)
                    let base = "chats/\(chat.id.uuidString.lowercased())"
                    checksums.append(try zip.add(
                        path: "\(base)/chat.json", data: encoder.encode(chat)))
                    checksums.append(try zip.add(path: "\(base)/messages.jsonl") { consume in
                        for message in chat.messages {
                            try consume(compactEncoder.encode(message))
                            try consume(Data([0x0a]))
                        }
                    })
                    checksums.append(try zip.add(
                        path: "\(base)/transcript.md",
                        data: Data(AppChatExport.markdown(
                            for: chat, modelAlias: snapshot.modelAlias).utf8)))
                }
                for file in snapshot.chatFiles {
                    if let data = file.data {
                        checksums.append(try zip.add(path: file.archivePath, data: data))
                    } else if let sourceURL = file.sourceURL {
                        checksums.append(try zip.add(path: file.archivePath, fileURL: sourceURL))
                    }
                }
            }
            if included.contains("projects") {
                checksums.append(try zip.add(
                    path: "projects/projects.json", data: encoder.encode(snapshot.projects)))
            }
            if included.contains("settings") {
                for name in snapshot.settingsFiles.keys.sorted() {
                    guard let data = snapshot.settingsFiles[name] else { continue }
                    checksums.append(try zip.add(path: "settings/\(name)", data: data))
                }
            }
            if included.contains("memory") {
                for memory in snapshot.memoryFiles {
                    if let data = memory.data {
                        checksums.append(try zip.add(path: memory.archivePath, data: data))
                    } else if let sourceURL = memory.sourceURL {
                        checksums.append(try zip.add(
                            path: memory.archivePath, fileURL: sourceURL))
                    }
                }
            }
            if !assetRows.isEmpty {
                checksums.append(try zip.add(
                    path: "assets/index.json", data: encoder.encode(assetRows)))
                for row in assetRows {
                    let digest = try zip.add(path: row.exportedPath) { consume in
                        try assets.streamDecrypted(
                            reference: row.managedReference, consume: consume)
                    }
                    checksums.append(digest)
                }
            }

            let manifest = Manifest(
                formatVersion: 1,
                kind: "turbospark-open-profile-export",
                profileID: snapshot.profile.id,
                profileName: snapshot.profile.name,
                exportedAt: snapshot.exportedAt,
                appVersion: snapshot.appVersion,
                includedCategories: included.sorted())
            _ = try zip.add(path: "manifest.json", data: encoder.encode(manifest))
            let checksumText = checksums.sorted { $0.path < $1.path }
                .map { "\($0.sha256)  \($0.path)" }
                .joined(separator: "\n") + (checksums.isEmpty ? "" : "\n")
            _ = try zip.add(path: "checksums.sha256", data: Data(checksumText.utf8))
            try zip.finish()
            try handle.synchronize()
        } catch {
            try? FileManager.default.removeItem(at: destination)
            throw error
        }
    }

    private static func buildAssetIndex(
        chats: [AppChat],
        included: Set<String>,
        store: ManagedAssetStore
    ) throws -> [AssetIndexRow] {
        var generated: Set<String> = []
        var attachments: Set<String> = []
        for chat in chats where !chat.isGhost {
            for artifact in chat.artifacts {
                guard let path = artifact.path, ManagedAssetStore.assetID(from: path) != nil else { continue }
                if artifact.origin == .imageGeneration { generated.insert(path) }
                else { attachments.insert(path) }
            }
            for attachment in chat.draftAttachments {
                if let path = attachment.sourcePath, ManagedAssetStore.assetID(from: path) != nil {
                    attachments.insert(path)
                }
            }
            func collectMessageAssets(_ message: AppChatMessage) {
                for path in message.imagePaths where ManagedAssetStore.assetID(from: path) != nil {
                    if !generated.contains(path) { attachments.insert(path) }
                }
                message.alternates.forEach(collectMessageAssets)
            }
            chat.messages.forEach(collectMessageAssets)
        }
        if !included.contains("generated-images") { generated.removeAll() }
        if !included.contains("attachments") { attachments.removeAll() }
        attachments.subtract(generated)
        var rows: [AssetIndexRow] = []
        for (category, references) in [
            ("generated-images", generated), ("attachments", attachments),
        ] {
            for reference in references.sorted() {
                guard let descriptor = try store.descriptor(for: reference) else { continue }
                let safe = ProfileBackup.sanitizedFileName(
                    descriptor.fileName, fallback: "asset")
                let path = "assets/\(category)/\(descriptor.id.prefix(12))-\(safe)"
                rows.append(AssetIndexRow(
                    managedReference: reference,
                    exportedPath: path,
                    originalFileName: descriptor.fileName,
                    mimeType: descriptor.mimeType,
                    byteCount: descriptor.byteCount,
                    category: category))
            }
        }
        return rows.sorted { $0.exportedPath < $1.exportedPath }
    }

    private static func portableChat(
        _ original: AppChat,
        assetPaths: [String: String]
    ) -> AppChat {
        var chat = original
        chat.draftAttachments = chat.draftAttachments.map { source in
            var copy = source
            if let reference = copy.sourcePath, let path = assetPaths[reference] {
                copy.sourcePath = "../../../\(path)"
            } else if copy.sourcePath.flatMap(ManagedAssetStore.assetID(from:)) != nil {
                copy.sourcePath = nil
            }
            return copy
        }
        func portableMessage(_ source: AppChatMessage) -> AppChatMessage {
            var copy = source
            copy.imagePaths = source.imagePaths.compactMap { reference in
                assetPaths[reference].map { "../../../\($0)" }
            }
            copy.alternates = source.alternates.map(portableMessage)
            return copy
        }
        chat.messages = chat.messages.map(portableMessage)
        chat.artifacts = chat.artifacts.map { source in
            var copy = source
            if let reference = copy.path, ManagedAssetStore.assetID(from: reference) != nil {
                copy.path = assetPaths[reference].map { "../../../\($0)" }
            }
            return copy
        }
        return chat
    }
}

extension AppModel {
    func makeProfileExportSnapshot() throws -> ProfileExportSnapshot {
        var settingsFiles: [String: Data] = [:]
        for name in ProfileRepository.protectedFileNames
        where name != "chats_archive.json" && name != "projects_archive.json" {
            let url = AppStorageRoot.directory.appendingPathComponent(name)
            guard let key = ProfileRepository.protectedRecordKey(for: url),
                  let data = try ProfileRepository.shared.rawRecord(key: key)
            else { continue }
            settingsFiles[name] = data
        }
        var memoryFiles: [ProfileExportSnapshot.MemoryFile] = []
        var chatFiles: [ProfileExportSnapshot.MemoryFile] = []
        if ProfileRepository.shared.isAvailable {
            for (key, data) in try ProfileRepository.shared.rawRecords(prefix: "memory:file:") {
                let relative = String(key.dropFirst("memory:file:".count))
                memoryFiles.append(.init(
                    archivePath: "memory/\(relative)", sourceURL: nil, data: data))
            }
            for (key, data) in try ProfileRepository.shared.rawRecords(prefix: "tool-observation:") {
                let relative = String(key.dropFirst("tool-observation:".count))
                let components = relative.split(separator: "/", omittingEmptySubsequences: true)
                guard components.count == 2,
                      let chatID = UUID(uuidString: String(components[0])),
                      components[1].hasSuffix(".bin"),
                      UUID(uuidString: String(components[1].dropLast(4))) != nil
                else { continue }
                chatFiles.append(.init(
                    archivePath: "chats/\(chatID.uuidString.lowercased())/tool-observations/\(components[1])",
                    sourceURL: nil,
                    data: data))
            }
        } else {
            let manager = FileManager.default
            let roots: [(URL, String)] = [
                (ProfileMemoryStore.shared.directory, "memory/profile"),
                (MemoryStore.defaultBase(), "memory/projects"),
            ]
            for (root, prefix) in roots where manager.fileExists(atPath: root.path) {
                let enumerator = manager.enumerator(
                    at: root, includingPropertiesForKeys: [.isRegularFileKey],
                    options: [.skipsHiddenFiles, .skipsPackageDescendants])
                while let url = enumerator?.nextObject() as? URL,
                      (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
                    let relative = String(url.path.dropFirst(root.path.count))
                        .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                    memoryFiles.append(.init(
                        archivePath: "\(prefix)/\(relative)", sourceURL: url, data: nil))
                }
            }
        }
        return ProfileExportSnapshot(
            profile: currentProfile,
            exportedAt: Date(),
            appVersion: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "",
            modelAlias: selected?.alias,
            chats: chats.filter { !$0.isGhost },
            projects: AppProjectArchive(selectedProjectID: selectedProjectID, projects: projects),
            settingsFiles: settingsFiles,
            chatFiles: chatFiles.sorted { $0.archivePath < $1.archivePath },
            memoryFiles: memoryFiles.sorted { $0.archivePath < $1.archivePath })
    }
}
