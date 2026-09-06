import Foundation

/// Resolves `@path` tokens in a submitted draft into prompt attachments.
///
/// The token STAYS in the message text: the attachment inlines the content,
/// and the token keeps telling the model which of its files the human was
/// pointing at. A token that resolves to nothing is left as ordinary prose,
/// silently -- the chip that fails to appear is the feedback, and a toast on
/// every casual `@word` in prose would be noise.
@MainActor
enum MentionResolver {
    /// A mention opens a token (start of text or whitespace before the `@`,
    /// so an email address never matches), and the quoted form carries paths
    /// with spaces. Group 1 is the token body with `@` and any quotes
    /// stripped.
    private static let mentionPattern = "(?:^|\\s)@((?:\"[^\"]*\")|[^\\s]+)"

    /// Cheap pre-check so the regex runs only on drafts that could hold one.
    nonisolated static func draftContainsMentions(_ draft: String) -> Bool {
        draft.contains("@") && !mentionTokens(in: draft).isEmpty
    }

    /// Mention tokens in draft order, `@` and quotes stripped, `@@` treated
    /// as an escaped literal and skipped, deduplicated.
    nonisolated static func mentionTokens(in draft: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: mentionPattern) else {
            return []
        }
        let range = NSRange(draft.startIndex..., in: draft)
        var tokens: [String] = []
        var seen = Set<String>()
        for match in regex.matches(in: draft, range: range) {
            guard let full = Range(match.range(at: 1), in: draft) else { continue }
            var token = String(draft[full])
            if token.hasPrefix("\""), token.count >= 2, token.hasSuffix("\"") {
                token = String(token.dropFirst().dropLast())
            }
            if token.isEmpty || token.hasPrefix("@") { continue }
            if seen.insert(token).inserted {
                tokens.append(token)
            }
        }
        return tokens
    }

    /// The URL a typed token points at: absolute paths as-is, everything
    /// else against the project root. No containment clamp -- the user typed
    /// the path on their own machine, and the plus menu offers the same
    /// reach through a picker.
    nonisolated static func resolveURL(for token: String, projectRoot: URL?) -> URL? {
        if token.hasPrefix("/") {
            return URL(fileURLWithPath: token)
        }
        guard let projectRoot else { return nil }
        return projectRoot.appendingPathComponent(token)
    }

    /// Resolves every token and appends what exists: a directory expands
    /// through the bounded folder walk, a file imports directly. Returns one
    /// failure line per existing path that failed to import (missing paths
    /// are not failures -- see the header). The caller may discard it: the
    /// resolution is deliberately silent in production.
    @discardableResult
    static func resolveMentions(
        in draft: String,
        projectRoot: URL?,
        chatID: UUID?,
        into model: AppModel
    ) async -> [String] {
        var failures: [String] = []
        var seenPaths = Set<String>()
        for token in mentionTokens(in: draft) {
            guard let url = resolveURL(for: token, projectRoot: projectRoot) else { continue }
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
                continue
            }
            guard seenPaths.insert(url.path).inserted else { continue }

            let outcome: AttachmentImporter.Outcome
            if isDir.boolValue {
                outcome = await AttachmentImporter.importFolder(
                    url, into: model, chatID: chatID, allowDuringSubmission: true)
            } else {
                outcome = await AttachmentImporter.importDocuments(
                    [url], into: model, chatID: chatID, allowDuringSubmission: true)
            }
            failures.append(contentsOf: outcome.failures)
        }
        return failures
    }
}
