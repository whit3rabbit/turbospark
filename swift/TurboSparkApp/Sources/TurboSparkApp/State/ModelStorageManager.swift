import AppKit
import Foundation
import TurboSpark

/// Discovers, validates, and indexes local model storage directories, including
/// the primary TurboSpark store (~/.turbospark/models), LM Studio (~/.lmstudio/models),
/// and custom external scan folders without copying weight files.
public enum ModelStorageManager {
    /// Default TurboSpark primary storage directory for downloaded models.
    public static var defaultTurboSparkModelsDirectory: String {
        if let home = ProcessInfo.processInfo.environment["TURBOSPARK_HOME"], !home.isEmpty {
            return (home as NSString).appendingPathComponent("models")
        }
        let homeDir = FileManager.default.homeDirectoryForCurrentUser.path
        return (homeDir as NSString).appendingPathComponent(".turbospark/models")
    }

    /// Default LM Studio models directory on macOS.
    public static var defaultLMStudioModelsDirectory: String {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser.path
        return (homeDir as NSString).appendingPathComponent(".lmstudio/models")
    }

    /// Expands leading tildes and standardizes file path.
    public static func expandPath(_ path: String) -> String {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "" }
        return (trimmed as NSString).expandingTildeInPath
    }

    /// Checks whether the LM Studio models directory exists on disk.
    public static func isLMStudioDirectoryPresent(customPath: String? = nil) -> Bool {
        let path = expandPath(customPath?.isEmpty == false ? customPath! : defaultLMStudioModelsDirectory)
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: path, isDirectory: &isDir) && isDir.boolValue
    }

    /// Calculates the total size in bytes of a directory tree.
    public static func directorySize(at path: String) -> UInt64 {
        let expanded = expandPath(path)
        guard let enumerator = FileManager.default.enumerator(
            at: URL(fileURLWithPath: expanded),
            includingPropertiesForKeys: [.fileSizeKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }

        var total: UInt64 = 0
        for case let url as URL in enumerator {
            // **THE WALK IS WHERE CANCELLATION HAS TO LAND** (state#87). A
            // `Task.cancel()` on the scan is cooperative and this loop had no
            // suspension point and no check, so a library on a slow external
            // drive kept walking to the end after the setting that asked for
            // it was turned off.
            if Task.isCancelled { return total }
            if let resourceValues = try? url.resourceValues(forKeys: [.fileSizeKey, .isDirectoryKey]),
               resourceValues.isDirectory != true,
               let fileSize = resourceValues.fileSize {
                total += UInt64(fileSize)
            }
        }
        return total
    }

    /// Scans a directory for runnable models (.gturbo bundles, .gguf files, or model folders)
    /// without copying any bytes.
    public static func scanModels(in directoryPath: String, sourceTag: String = "External") -> [InstalledModel] {
        let expanded = expandPath(directoryPath)
        let rootURL = URL(fileURLWithPath: expanded)
        let fm = FileManager.default

        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: expanded, isDirectory: &isDir), isDir.boolValue else {
            return []
        }

        var discovered: [InstalledModel] = []
        var visitedPaths = Set<String>()

        guard let enumerator = fm.enumerator(
            at: rootURL,
            includingPropertiesForKeys: [.isDirectoryKey, .contentModificationDateKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"

        for case let fileURL as URL in enumerator {
            // Per entry, for `directorySize`'s reason (state#87). Returning
            // what was found so far rather than throwing: the caller drops
            // the result on cancellation anyway, and a partial list costs
            // nothing where an error would need a branch nobody reads.
            if Task.isCancelled { return discovered }
            let path = fileURL.path
            if visitedPaths.contains(path) { continue }

            let pathExtension = fileURL.pathExtension.lowercased()
            let isDirectory = (try? fileURL.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false

            // Case 1: .gturbo directory bundle
            if isDirectory && (pathExtension == "gturbo" || hasManifest(at: fileURL)) {
                enumerator.skipDescendants()
                visitedPaths.insert(path)

                let alias = fileURL.deletingPathExtension().lastPathComponent
                let family = detectFamily(from: fileURL, fallbackName: alias)
                let bytes = directorySize(at: path)
                let modDate = (try? fileURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? Date()

                discovered.append(
                    InstalledModel(
                        alias: alias,
                        repo: "\(sourceTag.lowercased())/\(alias)",
                        revision: "local",
                        path: path,
                        family: family,
                        installBytes: bytes,
                        installedOn: dateFormatter.string(from: modDate)
                    )
                )
                continue
            }

            // Case 2: .gguf model file
            if !isDirectory && pathExtension == "gguf" {
                visitedPaths.insert(path)
                let fileName = fileURL.deletingPathExtension().lastPathComponent
                let family = inferFamilyFromName(fileName)
                let bytes = UInt64((try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
                let modDate = (try? fileURL.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? Date()

                discovered.append(
                    InstalledModel(
                        alias: fileName,
                        repo: "\(sourceTag.lowercased())/\(fileName)",
                        revision: "local",
                        path: path,
                        family: family,
                        installBytes: bytes,
                        installedOn: dateFormatter.string(from: modDate)
                    )
                )
            }
        }

        return discovered
    }

    /// Checks if a directory contains a gturbo manifest.json file.
    private static func hasManifest(at url: URL) -> Bool {
        let manifestURL = url.appendingPathComponent("manifest.json")
        return FileManager.default.fileExists(atPath: manifestURL.path)
    }

    /// Detects model family from manifest if present, or infers from name.
    private static func detectFamily(from url: URL, fallbackName: String) -> String {
        let manifestURL = url.appendingPathComponent("manifest.json")
        if let data = try? Data(contentsOf: manifestURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let arch = json["arch"] as? [String: Any], let family = arch["family"] as? String {
                return family
            }
            if let family = json["family"] as? String {
                return family
            }
        }
        return inferFamilyFromName(fallbackName)
    }

    /// Infers architecture family from filename or alias string.
    public static func inferFamilyFromName(_ name: String) -> String {
        let lower = name.lowercased()
        if lower.contains("gemma") { return "gemma4" }
        if lower.contains("qwen") {
            if lower.contains("moe") || lower.contains("a3b") || lower.contains("a14b") || lower.contains("30b-a") || lower.contains("35b-a") {
                return "qwen3moe"
            }
            return "qwen36"
        }
        if lower.contains("mistral") || lower.contains("mixtral") { return "mistral" }
        if lower.contains("llama") || lower.contains("tinyllama") { return "llama" }
        if lower.contains("ornith") { return "ornith" }
        if lower.contains("gptoss") { return "gptoss" }
        if lower.contains("museglimmer") { return "museglimmer" }
        return "custom"
    }

    /// Reveals a file or directory in Finder.
    public static func revealInFinder(path: String) {
        let expanded = expandPath(path)
        let url = URL(fileURLWithPath: expanded)
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: url.path)
    }
}

extension ModelStorageManager {
    /// Scans the external model directories, off the main actor.
    ///
    /// `nonisolated` and taking plain values so `AppModel.refreshModels` can
    /// hand it to a detached task: every walk here is recursive, and
    /// `directorySize(at:)` is a second full walk per bundle found, so on a
    /// large library this is hundreds of milliseconds to seconds of
    /// filesystem work that used to run on the main actor at launch.
    public static func scanExternal(
        lmStudioPath: String?, customPaths: [String]
    ) -> [(model: InstalledModel, sourceTag: String)] {
        var results: [(model: InstalledModel, sourceTag: String)] = []
        if let lmStudioPath {
            results += scanModels(in: lmStudioPath, sourceTag: "LM Studio").map { ($0, "LM Studio") }
        }
        for dir in customPaths {
            if Task.isCancelled { return results }
            results += scanModels(in: dir, sourceTag: "Custom").map { ($0, "Custom") }
        }
        return results
    }
}
