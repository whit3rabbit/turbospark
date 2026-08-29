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

        /// A single newline-joined error string, or nil when nothing failed.
        var errorText: String? {
            failures.isEmpty ? nil : failures.joined(separator: "\n")
        }
    }

    /// Extracts every URL and attaches the successes to the given chat.
    static func importDocuments(
        _ urls: [URL],
        into model: AppModel,
        chatID: UUID?
    ) async -> Outcome {
        guard !urls.isEmpty else { return Outcome(importedCount: 0, failures: []) }

        let outcomes = await Task.detached(priority: .userInitiated) {
            urls.map { url -> (URL, Int?, Result<ExtractedPromptDocument, Error>) in
                let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? Int
                do {
                    return (url, size, .success(try DocumentTextExtractor.extract(from: url)))
                } catch {
                    return (url, size, .failure(error))
                }
            }
        }.value

        var imported = 0
        var failures: [String] = []
        for (url, size, outcome) in outcomes {
            switch outcome {
            case .success(let document):
                model.addPromptAttachment(
                    AppPromptAttachment(
                        fileName: document.fileName,
                        formatLabel: document.formatLabel,
                        extractedText: document.text,
                        wasTruncatedDuringExtraction: document.wasTruncated,
                        sourcePath: url.path,
                        sourceByteSize: size),
                    toChatID: chatID)
                imported += 1
            case .failure(let error):
                failures.append("\(url.lastPathComponent): \(error.localizedDescription)")
            }
        }
        return Outcome(importedCount: imported, failures: failures)
    }
}
