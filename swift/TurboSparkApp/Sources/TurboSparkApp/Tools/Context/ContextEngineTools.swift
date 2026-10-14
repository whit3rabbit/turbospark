import Foundation

/// The context engine's three read-only tools (`swift/docs/SWIFT_CONTEXT_ENGINE.md`):
/// search again, read whole lines of a source, and outline what the sources
/// hold, over the project store the turn's evidence came from.
///
/// The definitions mirror `tool_definitions()` in syntext's
/// `context/src/tools/mod.rs` at the vendored revision, written with this
/// app's schema builders rather than decoded from the engine's JSON (whose
/// `minimum`/`maximum` keywords the schema type does not carry).
/// `ContextEngineToolTests` compares names and required arguments with the
/// engine's own list, so a syntext bump that changes them fails there.
///
/// Advertised only for the project whose context engine is active
/// (`AppToolRegistry.contextEngineRoot`), like `MemoryToolDefinitions`
/// gating at advertising time rather than refusing a call.
public enum ContextEngineToolDefinitions {
    public static let searchName = "context_search"
    public static let readName = "context_read"
    public static let outlineName = "context_outline"
    public static let names: Set<String> = [searchName, readName, outlineName]

    public static let search = OpenAITool.function(
        name: searchName,
        description: "Search the project's files, documents, and earlier conversation. Returns the passages that best match the query as <evidence> blocks, each with an id to cite, its source, and its line range. Call it when the evidence you have does not answer the question. Search for the names, identifiers, or phrases the answer would contain, not for the question itself.",
        parameters: .object(
            properties: [
                "query": .string(description: "Words, names, or identifiers to look for."),
                "max_results": .integer(description: "Most passages to return (1 to 10)."),
            ],
            required: ["query"]))

    public static let read = OpenAITool.function(
        name: readName,
        description: "Read lines of one source: a file path or document title, as an evidence block's source attribute shows it. Returns whole lines. A result that stops early says where to continue.",
        parameters: .object(
            properties: [
                "source": .string(description: "File path or document title, as shown in evidence."),
                "start_line": .integer(description: "First line to read (default 1)."),
                "end_line": .integer(description: "Last line to read (default: the end)."),
            ],
            required: ["source"]))

    public static let outline = OpenAITool.function(
        name: outlineName,
        description: "Show what the sources contain. Without a source: the documents and directories available. With a directory such as \"src/\": the documents in it. With a file path or document title: its sections and definitions with line ranges, to read with context_read.",
        parameters: .object(
            properties: [
                "source": .string(description: "Optional directory, file path, or document title.")
            ]))

    public static let all: [OpenAITool] = [search, read, outline]

    /// The tools when `projectURL` is the active context-engine project.
    public static func tools(for projectURL: URL?) -> [OpenAITool] {
        AppToolRegistry.contextToolsAvailable(projectURL: projectURL) ? all : []
    }
}

extension AppToolRegistry {
    /// Root of the project whose context engine is active, or nil. Set by
    /// `AppModel.refreshContextEngineActivation`, read when tools are
    /// advertised and run (the same static-flag pattern as
    /// `syntextIndexingEnabled`).
    public static var contextEngineRoot: URL?

    static func contextToolsAvailable(projectURL: URL?) -> Bool {
        guard let projectURL, let root = contextEngineRoot else { return false }
        return projectURL.standardizedFileURL.resolvingSymlinksInPath()
            == root.standardizedFileURL.resolvingSymlinksInPath()
    }

    /// Runs one context tool call. The model's arguments say only what to
    /// look for; scopes and the result budget come from the turn
    /// (`ContextEngineManager.callTool`). A malformed call comes back as an
    /// error the model can correct.
    static func executeContextTool(
        name: String, arguments: [String: String], project: AppProject?, chatID: UUID?
    ) async throws -> (output: String, isError: Bool) {
        guard let project, contextToolsAvailable(projectURL: project.rootDirectoryURL) else {
            return ("Error: '\(name)' is available only in a project with the context engine on.", true)
        }
        let data = try JSONSerialization.data(withJSONObject: arguments, options: [.sortedKeys])
        let result = try await ContextEngineManager.shared.callTool(
            name: name, arguments: String(decoding: data, as: UTF8.self),
            chatID: chatID, projectID: project.id)
        return (result.content, result.isError)
    }
}
