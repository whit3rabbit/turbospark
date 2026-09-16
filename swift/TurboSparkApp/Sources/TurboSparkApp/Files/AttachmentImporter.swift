import Foundation
import SwiftUI

/// Shared document import used by both the composer and the Files section.
///
/// Extraction is the expensive half (a 240k-character PDF walk), so it runs on
/// a detached task and only the resulting attachments touch the main actor.
@MainActor
enum AttachmentImporter {
    /// Result of importing a batch of documents.
    struct Outcome: Sendable {
        /// Number of documents that extracted cleanly.
        var importedCount: Int
        /// One line per document that failed, ready to show verbatim.
        var failures: [String]
        /// IDs of the attachments this import appended, in import order, so
        /// a caller that post-processes content (the mention resolver's
        /// line ranges) can find them on the chat row.
        var importedIDs: [UUID] = []

        /// A single newline-joined error string, or nil when nothing failed.
        var errorText: String? {
            failures.isEmpty ? nil : failures.joined(separator: "\n")
        }
    }

    /// Extracts every URL and attaches the successes to the given chat.
    ///
    /// `allowDuringSubmission` is for `MentionResolver`, which imports while
    /// the submission it belongs to is already `submitting`; every
    /// interactive caller leaves it false and keeps the guard.
    static func importDocuments(
        _ urls: [URL],
        into model: AppModel,
        chatID: UUID?,
        allowDuringSubmission: Bool = false
    ) async -> Outcome {
        guard !urls.isEmpty else { return Outcome(importedCount: 0, failures: []) }

        let outcomes = await Task.detached(priority: .userInitiated) {
            urls.map { url -> (URL, Int?, Result<ExtractedPromptDocument, Error>) in
                let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? Int
                // **A PICTURE IS NOT EXTRACTED, IT IS CARRIED BY PATH.**
                // `DocumentTextExtractor` throws `unsupportedFormat` on every
                // image type, which is correct for its own job and is why
                // images could not be attached at all before the engine took
                // them: the pixels go to the vision tower, and the only thing
                // the prompt needs is where to find them.
                if AppPromptAttachment.imageFileExtensions.contains(
                    url.pathExtension.lowercased())
                {
                    return (
                        url, size,
                        .success(
                            ExtractedPromptDocument(
                                fileName: url.lastPathComponent,
                                formatLabel: "Image",
                                text: "",
                                wasTruncated: false)))
                }
                do {
                    return (url, size, .success(try DocumentTextExtractor.extract(from: url)))
                } catch {
                    return (url, size, .failure(error))
                }
            }
        }.value

        var imported = 0
        var failures: [String] = []
        var importedIDs: [UUID] = []
        for (url, size, outcome) in outcomes {
            switch outcome {
            case .success(let document):
                let attachment = AppPromptAttachment(
                    fileName: document.fileName,
                    formatLabel: document.formatLabel,
                    extractedText: document.text,
                    wasTruncatedDuringExtraction: document.wasTruncated,
                    sourcePath: url.path,
                    sourceByteSize: size)
                if allowDuringSubmission {
                    model.appendPromptAttachmentDuringSubmission(attachment, toChatID: chatID)
                } else {
                    model.addPromptAttachment(attachment, toChatID: chatID)
                }
                importedIDs.append(attachment.id)
                imported += 1
            case .failure(let error):
                failures.append("\(url.lastPathComponent): \(error.localizedDescription)")
            }
        }
        return Outcome(importedCount: imported, failures: failures, importedIDs: importedIDs)
    }

    /// Recursively scans a directory for supported text, code, document, and image files.
    static func importFolder(
        _ folderURL: URL,
        into model: AppModel,
        chatID: UUID?,
        maxFiles: Int = 80,
        allowDuringSubmission: Bool = false
    ) async -> Outcome {
        let didAccess = folderURL.startAccessingSecurityScopedResource()
        defer {
            if didAccess {
                folderURL.stopAccessingSecurityScopedResource()
            }
        }

        let collectedURLs = await Task.detached(priority: .userInitiated) { () -> [URL] in
            collectFolderURLs(from: folderURL, maxFiles: maxFiles)
        }.value

        guard !collectedURLs.isEmpty else {
            return Outcome(
                importedCount: 0,
                failures: ["No supported code, text, or document files found in \(folderURL.lastPathComponent)."]
            )
        }

        return await importDocuments(
            collectedURLs, into: model, chatID: chatID,
            allowDuringSubmission: allowDuringSubmission)
    }

    private nonisolated static func collectFolderURLs(
        from folderURL: URL,
        maxFiles: Int
    ) -> [URL] {
        let fileManager = FileManager.default
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: folderURL.path, isDirectory: &isDir), isDir.boolValue else {
            return []
        }

        let canonicalRoot = PathContainment.canonical(folderURL)
        let rootDepth = canonicalRoot.pathComponents.count
        let maxDepth = 12
        var visitedCanonicalPaths: Set<String> = [canonicalRoot.path]

        let skipDirectoryNames = FileSystemScanRules.skipDirectoryNames

        let supportedExtensions: Set<String> = [
            "pdf", "docx", "pptx", "xlsx", "txt", "md", "markdown", "json",
            "swift", "py", "rs", "c", "cpp", "h", "hpp", "js", "ts", "tsx", "jsx",
            "html", "css", "yaml", "yml", "toml", "sh", "sql", "xml", "csv",
            "png", "jpg", "jpeg", "webp", "gif", "heic", "tiff", "bmp"
        ]

        var results: [URL] = []
        var totalBytes: Int64 = 0
        let maxTotalBytes: Int64 = 40 * 1024 * 1024 // 40 MB max cumulative size

        guard let enumerator = fileManager.enumerator(
            at: folderURL,
            includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey, .isDirectoryKey, .fileSizeKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return []
        }

        while let fileURL = enumerator.nextObject() as? URL {
            // Check containment and resolve symlinks
            let canonical = PathContainment.canonical(fileURL)
            guard PathContainment.isContained(canonical, in: canonicalRoot) else {
                // Escaped the folder via symlink: do not process and do not descend
                enumerator.skipDescendants()
                continue
            }

            let resourceValues = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .isDirectoryKey, .fileSizeKey])
            let isSymlink = resourceValues?.isSymbolicLink ?? false
            let isDirectory = resourceValues?.isDirectory ?? fileURL.hasDirectoryPath

            if isDirectory {
                // Symlinked directories are never descended into to prevent infinite recursion
                if isSymlink {
                    enumerator.skipDescendants()
                    continue
                }

                // Depth limit from root
                let currentDepth = canonical.pathComponents.count - rootDepth
                if currentDepth > maxDepth {
                    enumerator.skipDescendants()
                    continue
                }

                if skipDirectoryNames.contains(fileURL.lastPathComponent) {
                    enumerator.skipDescendants()
                    continue
                }

                if visitedCanonicalPaths.contains(canonical.path) {
                    enumerator.skipDescendants()
                    continue
                }
                visitedCanonicalPaths.insert(canonical.path)
                continue
            }

            // Regular file check
            if visitedCanonicalPaths.contains(canonical.path) {
                continue
            }
            visitedCanonicalPaths.insert(canonical.path)

            let ext = fileURL.pathExtension.lowercased()
            if supportedExtensions.contains(ext) {
                let fileSize = Int64(resourceValues?.fileSize ?? 0)
                if totalBytes + fileSize > maxTotalBytes && !results.isEmpty {
                    break
                }
                totalBytes += fileSize
                results.append(fileURL)
                if results.count >= maxFiles {
                    break
                }
            }
        }
        return results
    }
}

