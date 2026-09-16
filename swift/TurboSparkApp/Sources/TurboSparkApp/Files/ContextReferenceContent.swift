import Foundation

/// Content references that are not a project path: the git-derived
/// `@diff` / `@staged` / `@git:N` family and the web `@url:` fetch.
///
/// Both build an ordinary prompt attachment through the same unguarded
/// append the path mentions use, so the content rides the existing
/// pipeline: inlined into the outgoing message as `--- Attachment:` blocks,
/// counted by the context ring, summarized by compaction. Every failure is
/// an explicit scheme failure, so each one toasts -- these are the routes a
/// user asks for by name, unlike a bare `@word` in prose.
extension MentionResolver {
    static func resolveGitReference(
        _ reference: ContextReference, projectRoot: URL?, chatID: UUID?, into model: AppModel
    ) async {
        switch reference {
        case .diff:
            await appendGitAttachment(
                label: "diff", fileName: "Working tree diff", formatLabel: "diff",
                arguments: ["diff", "--no-ext-diff", "--no-textconv"],
                emptyMessage: "@diff: no unstaged changes to attach.",
                projectRoot: projectRoot, chatID: chatID, into: model)
        case .staged:
            await appendGitAttachment(
                label: "staged", fileName: "Staged diff", formatLabel: "diff",
                arguments: ["diff", "--no-ext-diff", "--no-textconv", "--cached"],
                emptyMessage: "@staged: nothing is staged to attach.",
                projectRoot: projectRoot, chatID: chatID, into: model)
        case .git(let count):
            await appendGitAttachment(
                label: "git:\(count)", fileName: "Last \(count) commits", formatLabel: "git",
                arguments: ["log", "--no-ext-diff", "--no-textconv", "-p", "-n", String(count)],
                emptyMessage: "@git: the repository has no commits yet.",
                projectRoot: projectRoot, chatID: chatID, into: model)
        case .path, .url:
            break  // not this file's routes
        }
    }

    /// One bounded git invocation turned into an attachment. No `rev-parse`
    /// probe first: a non-repository fails the command itself with a
    /// `fatal:` on stderr, and that stderr is what the toast shows.
    private static func appendGitAttachment(
        label: String, fileName: String, formatLabel: String, arguments: [String],
        emptyMessage: String, projectRoot: URL?, chatID: UUID?, into model: AppModel
    ) async {
        guard let root = projectRoot?.path, !root.isEmpty else {
            model.showToast("@\(label) needs a project with a git repository.", style: .warning)
            return
        }
        let result = await AppModel.runProcess(
            executable: "/usr/bin/git", arguments: arguments, workingDirectory: root)
        guard result.exitCode == 0 else {
            let detail = result.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            model.showToast(
                "@\(label) failed: \(detail.isEmpty ? "git exited with status \(result.exitCode)." : detail)",
                style: .warning)
            return
        }
        guard !result.stdout.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            model.showToast(emptyMessage, style: .info)
            return
        }
        // The same double bound the /diff sheet uses: ProcessExecutor caps
        // captured bytes, this caps rendered lines.
        let capped = AppModel.limitDiffLines(result.stdout)
        model.appendPromptAttachmentDuringSubmission(
            AppPromptAttachment(
                fileName: fileName,
                formatLabel: formatLabel,
                extractedText: capped,
                wasTruncatedDuringExtraction: capped != result.stdout,
                sourceByteSize: result.stdout.utf8.count),
            toChatID: chatID)
    }

    static func resolveWebReference(_ url: URL, chatID: UUID?, into model: AppModel) async {
        // The same toggle gates the model's own web tools; a fetch the user
        // typed by hand does not step around it.
        guard model.webSearchEnabled else {
            model.showToast("@url: is unavailable while web tools are disabled for this chat.", style: .warning)
            return
        }
        do {
            let content = try await WebFetchExecutor.fetch(url: url.absoluteString, format: "markdown")
            guard !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                model.showToast("@url: no content could be extracted from \(url.host ?? "that page").", style: .warning)
                return
            }
            // The fetch caps bytes; this caps the characters that reach the
            // context, matching what document extraction allows one file.
            let cap = DocumentTextExtractor.maximumExtractedCharacters
            let truncated = content.count > cap
            let text = truncated ? String(content.prefix(cap)) : content
            model.appendPromptAttachmentDuringSubmission(
                AppPromptAttachment(
                    fileName: url.host ?? "web page",
                    formatLabel: "Web",
                    extractedText: text,
                    wasTruncatedDuringExtraction: truncated),
                toChatID: chatID)
        } catch {
            model.showToast("@url: fetch failed: \(error.localizedDescription)", style: .warning)
        }
    }
}
