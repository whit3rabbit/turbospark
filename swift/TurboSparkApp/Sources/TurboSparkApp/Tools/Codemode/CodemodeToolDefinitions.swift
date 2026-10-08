import Foundation

/// The opt-in feature flag for the codemode tool. Default OFF: the tool
/// spawns a worker process and batches MCP calls, and it ships disabled
/// until proven, like the rest of the app's risky machinery.
public enum CodemodeSettings {
    public static let storageKey = "codemode.enabled"

    public static var isEnabled: Bool {
        UserDefaults.standard.bool(forKey: storageKey)
    }

    public static func setEnabled(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: storageKey)
    }
}

/// The `codemode` tool definition. `all` is empty while the flag is off,
/// the same gate pattern `MemoryToolDefinitions.all` uses, so the catalog
/// filter never advertises a tool the executor refuses.
public enum CodemodeToolDefinitions {
    public static let toolName = "codemode"

    public static var all: [OpenAITool] {
        guard CodemodeSettings.isEnabled else { return [] }
        return [codemode]
    }

    public static let codemode = OpenAITool.function(
        name: "codemode",
        description: "Run ONE JavaScript script that batches many deferred MCP tool calls. "
            + "Inside the script, call a tool with `await tools.<name>({...arguments})` (bindings "
            + "listed in the Codemode section), print with `text(...)` or `console.log(...)`, and "
            + "`return` the useful summary. Top-level await and return work. Only your printed "
            + "output and return value come back, not the individual tool results. The sandbox "
            + "has no network, filesystem, timers, or modules; every inner call still runs the "
            + "tool's own permission and hook gates.",
        parameters: .object(
            properties: [
                "code": .string(
                    description: "The JavaScript to run as an async function body. Optional first "
                        + "line: // @options: {\"timeout_ms\": 120000, \"max_output_chars\": 32000}.")
            ],
            required: ["code"]
        )
    )
}

/// Builds the Codemode declarations section appended to the system prompt:
/// one binding line per granted deferred MCP tool, budgeted like the
/// deferred listing (8,000 characters, or 5% of context when known).
public enum CodemodeCatalog {
    public static let listingCharacterLimit = 8_000
    private static let descriptionCharacterLimit = 120

    public static func declarationsSection(
        descriptors: [DeferredToolDescriptor], contextTokens: Int?
    ) -> String {
        guard CodemodeSettings.isEnabled else { return "" }
        let entries = CodemodeIdentifier.entries(for: descriptors)
        guard !entries.isEmpty else { return "" }
        var lines = [
            "",
            "## Codemode",
            "The `codemode` tool runs one JavaScript script in a sandbox. Inside it, "
                + "`await tools.<binding>({...})` calls the deferred MCP tool of that name: hooks "
                + "and permissions still apply per call, and a call that needs interactive "
                + "approval fails inside the script (make that one directly instead). Each call "
                + "resolves to the tool's text result as a string. Only `text()`/console output "
                + "and the script's return value reach you. Use `tool_describe` for full schemas.",
            "Bindings:"
        ]
        let contextLimit = contextTokens.map { max(600, $0 / 20) } ?? listingCharacterLimit
        let limit = min(listingCharacterLimit, contextLimit)
        for entry in entries {
            let line = "- `tools.\(entry.jsName)(args)`: Promise<text result> // "
                + shortDescription(entry.description)
            let candidate = (lines + [line]).joined(separator: "\n")
            guard candidate.count <= limit else { break }
            lines.append(line)
        }
        return lines.joined(separator: "\n")
    }

    private static func shortDescription(_ raw: String) -> String {
        let clean = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard clean.count > descriptionCharacterLimit else {
            return clean.isEmpty ? "No description." : clean
        }
        return String(clean.prefix(descriptionCharacterLimit)) + "..."
    }
}
