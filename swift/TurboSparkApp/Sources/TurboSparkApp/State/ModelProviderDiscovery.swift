import Foundation

/// A local model provider with a conventional on-disk library.
enum ModelProvider: String, CaseIterable, Codable, Sendable {
    case huggingFace
    case lmStudio
    case ollama

    var displayName: String {
        switch self {
        case .huggingFace: return "Hugging Face"
        case .lmStudio: return "LM Studio"
        case .ollama: return "Ollama"
        }
    }
}

/// A provider library candidate found without copying or modifying files.
struct ModelProviderCandidate: Identifiable, Hashable, Sendable {
    let provider: ModelProvider
    let path: String
    let supportedModelCount: Int
    let unsupportedArtifactCount: Int

    var id: String { "\(provider.rawValue):\(path)" }

    var statusText: String {
        if supportedModelCount > 0 {
            return "\(supportedModelCount) supported model\(supportedModelCount == 1 ? "" : "s")"
        }
        if unsupportedArtifactCount > 0 {
            return "Contains unsupported provider artifacts"
        }
        return "No directly runnable models found"
    }
}

extension ModelStorageManager {
    /// Returns conventional provider roots that exist on this machine.
    /// The scan is intentionally narrow: it never walks arbitrary home or
    /// volume folders and it never treats provider blobs as runnable models.
    static func knownProviderPaths(lmStudioPath: String?) -> [(ModelProvider, String)] {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        var paths: [(ModelProvider, String)] = []

        func append(_ provider: ModelProvider, _ path: String) {
            let expanded = expandPath(path)
            guard !expanded.isEmpty,
                  !paths.contains(where: { expandPath($0.1) == expanded }) else { return }
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory),
                  isDirectory.boolValue else { return }
            paths.append((provider, expanded))
        }

        if let cache = ProcessInfo.processInfo.environment["HF_HUB_CACHE"] {
            append(.huggingFace, cache)
        }
        if let hfHome = ProcessInfo.processInfo.environment["HF_HOME"] {
            append(.huggingFace, (hfHome as NSString).appendingPathComponent("hub"))
        }
        append(.huggingFace, (home as NSString).appendingPathComponent(".cache/huggingface/hub"))

        append(.lmStudio, lmStudioPath ?? defaultLMStudioModelsDirectory)

        if let ollama = ProcessInfo.processInfo.environment["OLLAMA_MODELS"] {
            append(.ollama, ollama)
        }
        append(.ollama, (home as NSString).appendingPathComponent(".ollama/models"))
        return paths
    }

    /// Inspects one known provider root using the same supported-artifact
    /// scanner used by the model list.
    static func inspectProvider(_ provider: ModelProvider, path: String) -> ModelProviderCandidate {
        let supported = scanModels(in: path, sourceTag: provider.displayName).count
        let expanded = expandPath(path)
        let unsupportedExtensions: Set<String> = provider == .huggingFace
            ? ["safetensors", "bin", "json"]
            : []
        var unsupported = 0
        if !unsupportedExtensions.isEmpty,
           let enumerator = FileManager.default.enumerator(
               at: URL(fileURLWithPath: expanded),
               includingPropertiesForKeys: [.isDirectoryKey],
               options: [.skipsHiddenFiles]) {
            for case let url as URL in enumerator where unsupportedExtensions.contains(url.pathExtension.lowercased()) {
                unsupported += 1
            }
        }
        return ModelProviderCandidate(
            provider: provider,
            path: expanded,
            supportedModelCount: supported,
            unsupportedArtifactCount: unsupported)
    }
}
