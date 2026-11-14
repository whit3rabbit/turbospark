import Foundation

/// What a post-edit step sees: the edit tool that just succeeded and the
/// project-relative paths it declared.
struct PostEditContext: Sendable {
    var toolName: String
    var relativePaths: [String]
    var rootURL: URL
    var chatID: UUID?
}

/// One step run after a successful edit tool call. A step returns text to
/// append to the tool result, or nil to add nothing.
protocol PostEditStep: Sendable {
    var identifier: String { get }
    func run(_ context: PostEditContext) async -> String?
}

/// OC-10: the one seam that runs after an edit tool succeeds, called from
/// `AppModel.applyPostToolUse` (the same place `PostToolUse` hook feedback is
/// folded into a result). **Order is part of the contract:** the OC-11
/// formatter registers BEFORE diagnostics, so the language server checks the
/// formatted file and the model never sees diagnostics for text the formatter
/// is about to rewrite. Steps run sequentially, never concurrently.
enum PostEditSeam {
    /// The edit tools whose result carries file paths a server can check.
    /// `document_write` is binary and `notebook_edit` is JSON: neither is a
    /// source file a language server reads.
    static let editToolNames: Set<String> = AppToolRegistry.writeFileAliases
        .union(AppToolRegistry.editFileAliases)
        .union(["apply_patch", "applypatch", "multiedit", "multi_edit"])

    /// Registered steps, in run order. OC-11 inserts its formatter ahead of
    /// `LSPDiagnosticsStep`.
    static var steps: [any PostEditStep] = [LSPDiagnosticsStep()]

    static let maximumPaths = 5

    static func isEditTool(_ name: String) -> Bool {
        editToolNames.contains(name.lowercased())
    }

    /// The text to append to an edit result, or nil. A failed edit, a call
    /// without a project, or a non-edit tool adds nothing.
    static func note(
        call: AppToolCall, isError: Bool, project: AppProject?, chatID: UUID?,
        steps override: [any PostEditStep]? = nil
    ) async -> String? {
        guard !isError, isEditTool(call.name), let root = project?.rootDirectoryURL else { return nil }
        let paths = Array(ToolActionFusion.targetPaths(for: call).prefix(maximumPaths))
        guard !paths.isEmpty else { return nil }
        let context = PostEditContext(toolName: call.name, relativePaths: paths, rootURL: root, chatID: chatID)
        var notes: [String] = []
        for step in override ?? steps {
            if let text = await step.run(context), !text.isEmpty { notes.append(text) }
        }
        return notes.isEmpty ? nil : notes.joined(separator: "\n\n")
    }
}

/// The OC-10 step: language server diagnostics for the files an edit wrote.
/// Off unless the user turned language servers on AND a server they enabled
/// covers a written file; otherwise it adds nothing and waits for nothing.
struct LSPDiagnosticsStep: PostEditStep {
    /// The bounded wait. A cold first edit usually misses it (the server is
    /// still starting); the late result is kept and `project_diagnostics`
    /// returns it.
    static let defaultBudgetSeconds: TimeInterval = 2

    let identifier = "lsp-diagnostics"
    var budgetSeconds: TimeInterval = LSPDiagnosticsStep.defaultBudgetSeconds
    var isEnabled: @Sendable () -> Bool = { LSPSettings.isEnabled() }
    var managerProvider: @Sendable (URL) async -> LanguageServerManager = { root in
        await LanguageServerRegistry.shared.manager(for: root)
    }

    func run(_ context: PostEditContext) async -> String? {
        guard isEnabled() else { return nil }
        let manager = await managerProvider(context.rootURL)
        guard let reports = await manager.postEdit(paths: context.relativePaths, budget: budgetSeconds) else { return nil }
        let body = LSPOutputFormatting.diagnosticsSummary(
            reports, itemLimit: LSPOutputFormatting.postEditItemLimit, budgetSeconds: budgetSeconds)
        guard !body.isEmpty else { return nil }
        return "<lsp_diagnostics>\n\(body)\n</lsp_diagnostics>"
    }
}
