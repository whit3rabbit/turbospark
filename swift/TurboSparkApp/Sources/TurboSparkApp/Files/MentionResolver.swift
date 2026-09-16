import Foundation

/// Resolves `@path` tokens and typed `@scheme:value` context references in a
/// submitted draft into prompt attachments.
///
/// The token STAYS in the message text: the attachment inlines the content,
/// and the token keeps telling the model which of its files the human was
/// pointing at. A token that resolves to nothing is left as ordinary prose,
/// silently -- the chip that fails to appear is the feedback, and a toast on
/// every casual `@word` in prose would be noise. An EXPLICIT scheme token
/// (`@file:`, `@diff`, `@url:` ...) is different: the user asked for that
/// content by name, so its failures toast.
@MainActor
enum MentionResolver {
    /// A mention opens a token (start of text or whitespace before the `@`,
    /// so an email address never matches), and the quoted form carries paths
    /// with spaces. Group 1 is the token body with `@` and any quotes
    /// stripped.
    private nonisolated static let mentionPattern = "(?:^|\\s)@((?:\"[^\"]*\")|[^\\s]+)"

    /// One extracted mention token: its body and whether it was quoted.
    /// Quoted tokens are always literal paths -- quotes are the escape hatch
    /// that names a file colliding with a reserved word (`@"diff"`), so they
    /// never parse as a scheme.
    struct MentionToken: Equatable {
        let text: String
        let quoted: Bool
    }

    /// Cheap pre-check so the regex runs only on drafts that could hold one.
    nonisolated static func draftContainsMentions(_ draft: String) -> Bool {
        draft.contains("@") && !mentionTokens(in: draft).isEmpty
    }

    /// Mention tokens in draft order, `@` and quotes stripped, `@@` treated
    /// as an escaped literal and skipped, deduplicated.
    nonisolated static func mentionTokens(in draft: String) -> [String] {
        mentionTokensDetailed(in: draft).map(\.text)
    }

    /// The same tokens with their quoted flag, which scheme parsing needs.
    nonisolated static func mentionTokensDetailed(in draft: String) -> [MentionToken] {
        guard let regex = try? NSRegularExpression(pattern: mentionPattern) else {
            return []
        }
        let range = NSRange(draft.startIndex..., in: draft)
        var tokens: [MentionToken] = []
        var seen = Set<String>()
        for match in regex.matches(in: draft, range: range) {
            guard let full = Range(match.range(at: 1), in: draft) else { continue }
            var token = String(draft[full])
            var quoted = false
            if token.hasPrefix("\""), token.count >= 2, token.hasSuffix("\"") {
                token = String(token.dropFirst().dropLast())
                quoted = true
            }
            if token.isEmpty || token.hasPrefix("@") { continue }
            if seen.insert(token).inserted {
                tokens.append(MentionToken(text: token, quoted: quoted))
            }
        }
        return tokens
    }

    /// The canonical URL a typed token points at when it remains inside the
    /// project. The attachment picker is the explicit route for files outside
    /// the project; mentions must not turn repository-controlled text into an
    /// unapproved read through an absolute path, traversal, or symlink.
    nonisolated static func resolveURL(for token: String, projectRoot: URL?) -> URL? {
        guard let projectRoot else { return nil }
        let candidate = token.hasPrefix("/")
            ? URL(fileURLWithPath: token)
            : projectRoot.appendingPathComponent(token)
        return PathContainment.resolvedIfContained(candidate, in: projectRoot)
    }

    /// Resolves every token and appends what exists: a directory expands
    /// through the bounded folder walk, a file imports directly, and the
    /// typed schemes pull git or web content. Returns one failure line per
    /// existing path that failed to import (missing paths are not failures
    /// -- see the header). The caller may discard it: bare-path resolution
    /// is deliberately silent in production.
    @discardableResult
    static func resolveMentions(
        in draft: String,
        projectRoot: URL?,
        chatID: UUID?,
        into model: AppModel
    ) async -> [String] {
        var failures: [String] = []
        var seenPaths = Set<String>()
        for token in mentionTokensDetailed(in: draft) {
            guard let parsed = ContextReferenceParser.parse(token: token.text, quoted: token.quoted)
            else { continue }
            switch parsed.reference {
            case .path(let text, let lines):
                let outcome = await resolvePathReference(
                    text: text,
                    unslicedPath: parsed.unslicedPath,
                    lines: lines,
                    isExplicit: parsed.isExplicit,
                    projectRoot: projectRoot,
                    chatID: chatID,
                    seenPaths: &seenPaths,
                    into: model)
                failures.append(contentsOf: outcome)
            case .diff, .staged, .git:
                await resolveGitReference(parsed.reference, projectRoot: projectRoot, chatID: chatID, into: model)
            case .url(let url):
                await resolveWebReference(url, chatID: chatID, into: model)
            }
        }
        return failures
    }

    /// One path reference: the sensitive-path guard, project containment, the
    /// line-range fallback to the raw path, and the import. Inouts cannot
    /// cross an await, so `seenPaths` lives in the caller's loop and this
    /// returns its import failures like the folder/file arms always did.
    private static func resolvePathReference(
        text: String,
        unslicedPath: String?,
        lines: ClosedRange<Int>?,
        isExplicit: Bool,
        projectRoot: URL?,
        chatID: UUID?,
        seenPaths: inout Set<String>,
        into model: AppModel
    ) async -> [String] {
        // Defense in depth beside project containment: a project rooted at
        // the home directory would otherwise let `@.env` or `@.ssh/id_rsa`
        // through the containment check. The tool read path applies the same
        // classifier.
        if sensitiveRefusal(text: text, resolved: nil, isExplicit: isExplicit, into: model) {
            return []
        }
        guard let url = resolveURL(for: text, projectRoot: projectRoot) else { return [] }
        if sensitiveRefusal(text: text, resolved: url.path, isExplicit: isExplicit, into: model) {
            return []
        }

        var isDir: ObjCBool = false
        var effectiveURL = url
        var effectiveLines = lines
        if !FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
            let unsliced = unslicedPath,
            let raw = resolveURL(for: unsliced, projectRoot: projectRoot),
            FileManager.default.fileExists(atPath: raw.path, isDirectory: &isDir)
        {
            // The range-shaped suffix was part of the real file's name.
            effectiveURL = raw
            effectiveLines = nil
        }
        guard FileManager.default.fileExists(atPath: effectiveURL.path, isDirectory: &isDir) else {
            return []
        }
        guard seenPaths.insert(effectiveURL.path).inserted else { return [] }

        if isDir.boolValue {
            // A folder has no lines to slice; the range is dropped.
            return await AttachmentImporter.importFolder(
                effectiveURL, into: model, chatID: chatID, allowDuringSubmission: true
            ).failures
        }
        let outcome = await AttachmentImporter.importDocuments(
            [effectiveURL], into: model, chatID: chatID, allowDuringSubmission: true)
        if let lines = effectiveLines {
            sliceImportedAttachments(ids: outcome.importedIDs, to: lines, chatID: chatID, into: model)
        }
        return outcome.failures
    }

    /// True when the path hits the same sensitive-file classifier the read
    /// tool uses. Explicit scheme refs toast; bare mentions keep the
    /// resolver's silence (the missing chip is the feedback).
    private static func sensitiveRefusal(
        text: String, resolved: String?, isExplicit: Bool, into model: AppModel
    ) -> Bool {
        let hit = ToolRiskClassifier.isSensitivePath(text)
            || (resolved.map { ToolRiskClassifier.isSensitivePath($0) } ?? false)
        guard hit else { return false }
        if isExplicit {
            model.showToast(
                "@file: \(text) looks like a credential file; it was not read.", style: .warning)
        }
        return true
    }

    /// Slices freshly imported attachments to a 1-indexed inclusive line
    /// range of their extracted text. A range starting past the last line is
    /// invalid and leaves the full file (Hermes rule); a range that runs
    /// past the end is clamped and flags truncation.
    static func sliceImportedAttachments(
        ids: [UUID], to lines: ClosedRange<Int>, chatID: UUID?, into model: AppModel
    ) {
        guard !ids.isEmpty else { return }
        let targetID = chatID ?? model.selectedChatID
        guard let index = model.chats.firstIndex(where: { $0.id == targetID }) else { return }
        for id in ids {
            guard let at = model.chats[index].draftAttachments.firstIndex(where: { $0.id == id })
            else { continue }
            let all = model.chats[index].draftAttachments[at].extractedText
                .split(separator: "\n", omittingEmptySubsequences: false)
            let lower = max(lines.lowerBound, 1)
            guard lower <= all.count else { continue }
            let upper = min(lines.upperBound, all.count)
            let sliced = all[(lower - 1)..<upper].joined(separator: "\n")
            model.chats[index].draftAttachments[at].extractedText = sliced
            if upper < lines.upperBound {
                model.chats[index].draftAttachments[at].wasTruncatedDuringExtraction = true
            }
        }
        model.persistChats()
        model.updateTokenEstimate()
    }
}
