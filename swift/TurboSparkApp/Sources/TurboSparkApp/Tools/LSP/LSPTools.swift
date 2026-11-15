import Foundation

/// OC-10: schemas for the three language server tools. `all` is empty while
/// the master switch is off, the gate pattern `CodemodeToolDefinitions` and
/// `MemoryToolDefinitions` use, so the catalog never advertises a tool the
/// executor refuses.
///
/// **Ladder position:** category `.fileRead`, workspace-rooted. A project
/// denies them with its File read permission; they are read-only and never
/// edit a file. They are NOT a new category: a language server is the user's
/// own toolchain, opted in by the master switch and a per-server row, and what
/// the tools return is bounded text about files the file tools could already
/// read. Sensitive paths are refused by the same hard gate as `read_file`
/// (`ToolRiskClassifier`), and results are filtered to the project root.
enum LSPToolDefinitions {
    static let names: Set<String> = ["project_diagnostics", "document_symbols", "workspace_symbols"]

    static var all: [OpenAITool] {
        guard LSPSettings.isEnabled() else { return [] }
        return [projectDiagnostics, documentSymbols, workspaceSymbols]
    }

    static let projectDiagnostics = OpenAITool.function(
        name: "project_diagnostics",
        description: """
            Errors and warnings from the language server the user enabled, for files it has \
            checked. With 'file', re-checks that file and waits a few seconds. Without it, \
            returns what the server last reported for files edited or opened this session, \
            plus each server's state. A file with no enabled server is reported as such, never \
            as clean.
            """,
        parameters: .object(
            properties: [
                "file": .string(description: "Optional project-relative path. Omit to list diagnostics for every checked file."),
            ],
            required: []
        )
    )

    static let documentSymbols = OpenAITool.function(
        name: "document_symbols",
        description: """
            The symbol tree of one source file (types, functions, properties) from the language \
            server, as indented lines with 1-based line:column. Use it instead of reading a \
            whole file to find where something is declared.
            """,
        parameters: .object(
            properties: [
                "file": .string(description: "Project-relative path of the source file."),
            ],
            required: ["file"]
        )
    )

    static let workspaceSymbols = OpenAITool.function(
        name: "workspace_symbols",
        description: """
            Search symbols by name across the project through the language server. Returns \
            bounded matches as path:line:column with kind and container. Only locations inside \
            the project are returned.
            """,
        parameters: .object(
            properties: [
                "query": .string(description: "Part of a symbol name, at most 200 characters."),
                "limit": .string(description: "Optional maximum results, 1 to 100. Default 30."),
            ],
            required: ["query"]
        )
    )
}

enum LSPToolExecutor {
    /// The explicit-call wait for `project_diagnostics` with a file. Longer
    /// than the post-edit budget because the model asked and is waiting.
    static let diagnosticsBudgetSeconds: TimeInterval = 5

    private static func error(_ code: Int, _ message: String) -> NSError {
        NSError(domain: "TurboSparkLSPTool", code: code, userInfo: [NSLocalizedDescriptionKey: message])
    }

    /// One path from any accepted spelling. Two spellings that disagree are a
    /// conflict, refused here exactly as the file tools refuse them, so the
    /// risk gate and this executor never judge different files.
    private static func fileArgument(_ arguments: [String: String]) throws -> String? {
        let raw = try ToolArgumentResolver.resolve(arguments, keys: ToolArgumentResolver.filePathKeys, field: "file")
        let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines)
        return (trimmed?.isEmpty ?? true) ? nil : trimmed
    }

    static func execute(
        name: String, arguments: [String: String], rootURL: URL,
        isEnabled: Bool = LSPSettings.isEnabled(),
        projectMode: ProjectLSPMode = .inherit,
        manager: LanguageServerManager? = nil
    ) async throws -> String {
        guard isEnabled else {
            throw error(1, "Language servers are turned off. The user can enable them in Settings > General > Language servers.")
        }
        // The project can only narrow the app-wide switch, never widen it, so
        // this comes after the check above.
        guard projectMode != .off else { throw error(9, LSPManagerError.offForProjectMessage) }
        let lsp: LanguageServerManager
        if let manager { lsp = manager } else { lsp = await LanguageServerRegistry.shared.manager(for: rootURL) }
        await lsp.setProjectMode(projectMode)
        do {
            switch name {
            case "project_diagnostics": return try await diagnostics(arguments, manager: lsp)
            case "document_symbols": return try await documentSymbols(arguments, manager: lsp)
            case "workspace_symbols": return try await workspaceSymbols(arguments, manager: lsp)
            default: throw error(2, "Unknown language server tool '\(name)'.")
            }
        } catch let failure as LSPManagerError {
            throw error(3, failure.localizedDescription)
        }
    }

    private static func diagnostics(_ arguments: [String: String], manager: LanguageServerManager) async throws -> String {
        let file = try fileArgument(arguments)
        if let file, ToolRiskClassifier.isSensitivePath(file) {
            throw error(4, "'\(file)' is a sensitive path and is never sent to a language server.")
        }
        let result = try await manager.diagnostics(file: file, budget: diagnosticsBudgetSeconds)
        var lines: [String] = []
        if result.reports.isEmpty {
            lines.append("No documents have been checked yet. Edit a file or pass 'file' to check one.")
        } else {
            lines.append(LSPOutputFormatting.diagnosticsSummary(
                result.reports, itemLimit: LSPOutputFormatting.toolItemLimit, budgetSeconds: diagnosticsBudgetSeconds))
        }
        lines.append("Servers: " + (result.statuses.isEmpty
            ? "none enabled"
            : result.statuses.map { "\(LSPOutputFormatting.sanitize($0.name, limit: 60)) \(LSPOutputFormatting.sanitize($0.state, limit: 160))" }
                .joined(separator: "; ")))
        return lines.joined(separator: "\n")
    }

    private static func documentSymbols(_ arguments: [String: String], manager: LanguageServerManager) async throws -> String {
        guard let file = try fileArgument(arguments) else { throw error(5, "document_symbols needs a 'file' argument.") }
        if ToolRiskClassifier.isSensitivePath(file) {
            throw error(4, "'\(file)' is a sensitive path and is never sent to a language server.")
        }
        let result: (server: String, relativePath: String, symbols: [LSPDocumentSymbol])
        do { result = try await manager.documentSymbols(file: file) }
        catch let failure as LSPClientError { throw error(6, failure.localizedDescription) }
        guard !result.symbols.isEmpty else {
            return "\(LSPOutputFormatting.sanitize(result.relativePath, limit: 200)): no symbols reported by \(LSPOutputFormatting.sanitize(result.server, limit: 60))."
        }
        return "\(LSPOutputFormatting.sanitize(result.relativePath, limit: 200)) (\(LSPOutputFormatting.sanitize(result.server, limit: 60))):\n"
            + LSPOutputFormatting.symbolTree(result.symbols)
    }

    private static func workspaceSymbols(_ arguments: [String: String], manager: LanguageServerManager) async throws -> String {
        let query = (arguments["query"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { throw error(7, "workspace_symbols needs a non-empty 'query'.") }
        guard query.count <= 200 else { throw error(8, "The query is over 200 characters.") }
        let requested = Int(arguments["limit"] ?? "") ?? LSPOutputFormatting.workspaceSymbolDefaultLimit
        let limit = min(max(requested, 1), LSPOutputFormatting.workspaceSymbolMaximumLimit)
        let result = await manager.workspaceSymbols(query: query)
        var lines: [String] = []
        for (server, symbol) in result.symbols.prefix(limit) {
            guard let path = await manager.relativePath(forURI: symbol.uri) else { continue }
            var line = "\(LSPOutputFormatting.sanitize(path, limit: 200))"
            if let range = symbol.range { line += ":\(range.start.line + 1):\(range.start.character + 1)" }
            line += " \(LSPParse.kindName(symbol.kind)) \(LSPOutputFormatting.sanitize(symbol.name, limit: 120))"
            if let container = symbol.container, !container.isEmpty { line += " in \(LSPOutputFormatting.sanitize(container, limit: 80))" }
            line += " (\(LSPOutputFormatting.sanitize(server, limit: 60)))"
            lines.append(line)
        }
        if result.symbols.count > limit { lines.append("...\(result.symbols.count - limit) more matches not shown. Narrow the query.") }
        if lines.isEmpty { lines.append("No symbols matched '\(LSPOutputFormatting.sanitize(query, limit: 80))' inside the project.") }
        for note in result.notes { lines.append("Note: \(LSPOutputFormatting.sanitize(note, limit: 200))") }
        return lines.joined(separator: "\n")
    }
}
