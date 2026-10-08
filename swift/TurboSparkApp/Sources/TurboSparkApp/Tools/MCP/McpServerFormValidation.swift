import Foundation

/// Why the MCP server editor refuses to save, or nil when it does not.
///
/// A VALUE rather than inline view code, for the reason `ServerStatusRows` is
/// one (swift/CLAUDE.md Gotcha 26): a decision living in a `View` body needs a
/// running app to exercise, so in practice nothing exercises it. Every rule
/// here is reachable from a test with no window, no model and no subprocess.
public struct McpServerFormValidation: Equatable, Sendable {

    /// Parses the editor's arguments box. A multi-line box is one argument per
    /// line, which is how `argsText(for:)` fills it, so an argument that holds
    /// a space ("/Users/me/Application Support/notes") survives a round trip.
    /// Only a single-line box is split on spaces (the quick "-y pkg" entry).
    public static func parseArguments(_ text: String) -> [String] {
        let lines = text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        if lines.count == 1 {
            return lines[0].components(separatedBy: " ").filter { !$0.isEmpty }
        }
        return lines
    }

    /// The text the editor shows for existing arguments.
    public static func argsText(for args: [String]) -> String {
        args.joined(separator: "\n")
    }

    /// The transport arm the form is currently on.
    public enum Transport: String, Equatable, Sendable {
        case stdio
        case sse
    }

    public var name: String
    public var transport: Transport
    public var command: String
    public var endpointURLText: String
    /// Names already taken at the target scope, with the row being edited
    /// excluded by the caller so an edit can keep its own name.
    public var existingNames: [String]

    public init(
        name: String,
        transport: Transport,
        command: String,
        endpointURLText: String,
        existingNames: [String]
    ) {
        self.name = name
        self.transport = transport
        self.command = command
        self.endpointURLText = endpointURLText
        self.existingNames = existingNames
    }

    /// The endpoint URL, or nil when the field does not parse.
    ///
    /// This used to be `URL(string: text) ?? URL(string: "http://localhost:8000/sse")!`
    /// inside the sheet's `buildConfig`, so an unparseable URL was accepted and
    /// SAVED pointing at somewhere the user never typed. A field that cannot be
    /// read blocks the save; it does not get a value invented for it.
    public var parsedEndpointURL: URL? {
        let trimmed = endpointURLText.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, let url = URL(string: trimmed), url.scheme != nil else {
            return nil
        }
        return url
    }

    /// Why Save is refused, or nil when it is allowed.
    public var message: String? {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedName.isEmpty {
            return "A server name is required."
        }
        // Name is the identity key in `AppToolRegistry.executeMcpCall` and in
        // `AppToolPermissionEngine.evaluate`, both of which resolve with
        // `first(where:)`. A duplicate is not cosmetic: the second server can
        // never be dialled, and the approval card cannot tell them apart.
        if McpServerConfig.nameIsTaken(trimmedName, among: existingNames) {
            return "A server named '\(trimmedName)' already exists."
        }
        switch transport {
        case .stdio:
            if command.trimmingCharacters(in: .whitespaces).isEmpty {
                return "A command is required."
            }
        case .sse:
            if parsedEndpointURL == nil {
                return "A valid server URL is required."
            }
        }
        return nil
    }

    public var isValid: Bool { message == nil }
}
