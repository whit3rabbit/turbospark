import Foundation

extension AppModel {
    /// Only generated files owned by this profile can be trashed. A failed move
    /// leaves its history intact, and missing files can still be removed as rows.
    @discardableResult
    func trashGeneratedImages(
        ids: Set<UUID>,
        moveToTrash: (URL) throws -> Void = { url in
            try FileManager.default.trashItem(at: url, resultingItemURL: nil)
        }
    ) -> Set<UUID> {
        let directory = AppStorageRoot.subdirectory("image-artifacts").resolvingSymlinksInPath()
        var removed: Set<UUID> = []
        var paths: Set<String> = []
        for artifact in savedImageArtifacts where ids.contains(artifact.id) {
            guard let path = artifact.path else { continue }
            if ManagedAssetStore.assetID(from: path) != nil {
                do {
                    try ManagedAssetStore.shared.release(reference: path)
                    paths.insert(path)
                    removed.insert(artifact.id)
                } catch {
                    showToast(error.localizedDescription, style: .error)
                }
                continue
            }
            let url = URL(fileURLWithPath: path).standardizedFileURL
            guard url.deletingLastPathComponent().resolvingSymlinksInPath() == directory else {
                showToast(String(localized: "Only images saved by this profile can be removed.", bundle: .module), style: .error)
                continue
            }
            do {
                if !paths.contains(path), FileManager.default.fileExists(atPath: path) {
                    try moveToTrash(url)
                }
                paths.insert(path)
                removed.insert(artifact.id)
            } catch {
                showToast(error.localizedDescription, style: .error)
            }
        }
        guard !removed.isEmpty else { return [] }
        for chatIndex in chats.indices {
            chats[chatIndex].artifacts.removeAll { artifact in
                let matches = artifact.origin == .imageGeneration
                    && artifact.path.map(paths.contains) == true
                if matches { removed.insert(artifact.id) }
                return matches
            }
            for messageIndex in chats[chatIndex].messages.indices {
                chats[chatIndex].messages[messageIndex].imagePaths.removeAll {
                    paths.contains($0) || paths.contains(AppStorageRoot.resolveStoredPath($0))
                }
            }
        }
        if let id = openArtifactID, removed.contains(id) { dismissArtifact() }
        if let path = imageJob?.savedPath, paths.contains(path) { imageJob = nil }
        persistChats()
        return removed
    }

    func discardUnsavedImage() {
        guard imageGenerationTask == nil, imageJob?.savedPath == nil else { return }
        imageJob = nil
    }
}
