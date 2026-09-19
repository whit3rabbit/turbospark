import Foundation

extension AppModel {
    /// Copies every private file currently referenced by chat state into the
    /// encrypted asset store before profile protection removes plaintext.
    func migratePrivateAssetsIntoVault() async throws {
        struct Candidate: Hashable, Sendable {
            var reference: String
            var path: String
            var fileName: String
        }

        var candidates: Set<Candidate> = []
        func add(_ reference: String?, fileName: String? = nil) {
            guard let reference, ManagedAssetStore.assetID(from: reference) == nil else { return }
            if let url = URL(string: reference), url.scheme != nil, url.scheme != "file" { return }
            let path = AppStorageRoot.resolveStoredPath(reference)
            guard !path.isEmpty, FileManager.default.fileExists(atPath: path) else { return }
            let name = fileName ?? URL(fileURLWithPath: path).lastPathComponent
            candidates.insert(Candidate(reference: reference, path: path, fileName: name))
        }

        for chat in chats {
            for attachment in chat.draftAttachments {
                add(attachment.sourcePath, fileName: attachment.fileName)
            }
            func collect(_ message: AppChatMessage) {
                message.imagePaths.forEach { add($0) }
                message.alternates.forEach(collect)
            }
            chat.messages.forEach(collect)
            for artifact in chat.artifacts where artifact.origin == .imageGeneration {
                add(artifact.path)
            }
        }

        let legacyImageDirectory = AppStorageRoot.directory
            .appendingPathComponent("image-artifacts", isDirectory: true)
        if let entries = try? FileManager.default.contentsOfDirectory(
            at: legacyImageDirectory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]) {
            for entry in entries where (try? entry.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
                add(entry.path)
            }
        }

        let migrated = try await Task.detached(priority: .utility) {
            var result: [String: ManagedAssetDescriptor] = [:]
            for candidate in candidates {
                result[candidate.reference] = try ManagedAssetStore.shared.store(
                    fileURL: URL(fileURLWithPath: candidate.path),
                    fileName: candidate.fileName)
            }
            return result
        }.value

        func migrate(_ message: inout AppChatMessage) {
            message.imagePaths = message.imagePaths.map {
                migrated[$0]?.storedReference ?? $0
            }
            for index in message.alternates.indices {
                migrate(&message.alternates[index])
            }
        }
        for chatIndex in chats.indices {
            for attachmentIndex in chats[chatIndex].draftAttachments.indices {
                let old = chats[chatIndex].draftAttachments[attachmentIndex].sourcePath
                guard let old, let descriptor = migrated[old] else { continue }
                chats[chatIndex].draftAttachments[attachmentIndex].sourcePath = descriptor.storedReference
                chats[chatIndex].draftAttachments[attachmentIndex].sourceByteSize = Int(descriptor.byteCount)
            }
            for messageIndex in chats[chatIndex].messages.indices {
                migrate(&chats[chatIndex].messages[messageIndex])
            }
            for artifactIndex in chats[chatIndex].artifacts.indices
            where chats[chatIndex].artifacts[artifactIndex].origin == .imageGeneration {
                guard let old = chats[chatIndex].artifacts[artifactIndex].path,
                      let descriptor = migrated[old]
                else { continue }
                chats[chatIndex].artifacts[artifactIndex].path = descriptor.storedReference
                chats[chatIndex].artifacts[artifactIndex].lastKnownByteSize = Int(descriptor.byteCount)
            }
        }
        if let old = imageJob?.savedPath, let descriptor = migrated[old] {
            imageJob?.savedPath = descriptor.storedReference
        }
        try ProfileRepository.shared.save(
            Array(Set(migrated.values)).sorted { $0.id < $1.id },
            key: "migration:legacy-assets")
        persistChats()
    }
}
