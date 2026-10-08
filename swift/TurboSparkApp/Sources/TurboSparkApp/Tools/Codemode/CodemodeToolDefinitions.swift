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
            + "tool's own permission and hook gates, so a tool that needs interactive approval "
            + "fails inside the script and must be called directly instead. A script may make at "
            + "most 200 tool calls, and a tool result over 2 MiB is rejected rather than clipped.",
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

/// Builds the system-prompt listing of deferred MCP tools.
///
/// With codemode off this is exactly `ToolSearchCatalog.promptListing`. With
/// it on, there is ONE listing under the `## Deferred MCP Tools` heading
/// rather than that listing plus a second `## Codemode` one naming the same
/// tools: each line carries the tool's typed `tools.<name>(args)` signature,
/// so a script can be written correctly without a `tool_describe` round trip.
public enum CodemodeCatalog {
    /// Larger than the plain listing's 8,000 because a typed line is longer,
    /// and still under the 16,000 the two separate listings could add up to.
    public static let listingCharacterLimit = 12_000
    private static let descriptionCharacterLimit = 120
    /// Room kept for the closing "N more tools" line when the listing is cut.
    private static let overflowLineReserve = 100

    public static func promptListing(
        descriptors: [DeferredToolDescriptor],
        contextTokens: Int? = nil,
        codemodeOffered: Bool = true,
        enabled: Bool = CodemodeSettings.isEnabled
    ) -> String {
        let entries = CodemodeIdentifier.entries(for: descriptors)
        // `codemodeOffered` is false for an agent whose allow-list does not
        // include the tool: bindings for a tool it cannot call are noise.
        guard enabled, codemodeOffered, !entries.isEmpty else {
            return ToolSearchCatalog.promptListing(
                descriptors: descriptors, contextTokens: contextTokens)
        }

        var schemas: [String: String] = [:]
        for descriptor in descriptors where schemas[descriptor.name] == nil {
            schemas[descriptor.name] = descriptor.inputSchemaJSON
        }

        var lines = ToolSearchCatalog.promptHeaderLines()
        lines.append(
            "Inside the `codemode` tool, call these as `await tools.<name>(args)` instead: one "
                + "script can chain, loop and filter many calls, and only the script's printed "
                + "output and return value come back. Every call resolves to the tool's text "
                + "result as a string (use JSON.parse when it returns JSON). Hooks and "
                + "permissions still apply per call.")

        let contextLimit = contextTokens.map { max(600, $0 / 20) } ?? listingCharacterLimit
        let limit = min(listingCharacterLimit, contextLimit)
        var used = lines.joined(separator: "\n").count
        var listed = 0
        for entry in entries {
            let typed = typedLine(entry, schemaJSON: schemas[entry.name] ?? "{}")
            let compact = compactLine(entry)
            // Typed when it fits; otherwise the plain name-and-description
            // line (what the listing showed before signatures existed), so a
            // long catalog degrades per tool instead of dropping tools early.
            if used + 1 + typed.count <= limit - overflowLineReserve {
                lines.append(typed)
                used += 1 + typed.count
            } else if used + 1 + compact.count <= limit - overflowLineReserve {
                lines.append(compact)
                used += 1 + compact.count
            } else {
                break
            }
            listed += 1
        }
        if listed < entries.count {
            lines.append("- \(entries.count - listed) more tools are not listed; find them with `tool_search`.")
        }
        return lines.joined(separator: "\n")
    }

    /// `- `tools.name(args: { ... }): Promise<string>`: description`
    static func typedLine(_ entry: CodemodeToolEntry, schemaJSON: String) -> String {
        let arguments = CodemodeSchemaRenderer.argumentsDeclaration(schemaJSON: schemaJSON)
        return "- `tools.\(entry.jsName)(\(arguments)): Promise<string>`: "
            + shortDescription(entry.description) + aliasNote(entry)
    }

    /// `- `tools.name(args)`: description`
    static func compactLine(_ entry: CodemodeToolEntry) -> String {
        "- `tools.\(entry.jsName)(args)`: " + shortDescription(entry.description) + aliasNote(entry)
    }

    /// `tool_describe` and `tool_call` take the advertised name; the script
    /// binding is its sanitized form (and a suffixed one when two collide).
    private static func aliasNote(_ entry: CodemodeToolEntry) -> String {
        entry.jsName == entry.name ? "" : " (tool name: `\(entry.name)`)"
    }

    private static func shortDescription(_ raw: String) -> String {
        let clean = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard clean.count > descriptionCharacterLimit else {
            return clean.isEmpty ? "No description." : clean
        }
        return String(clean.prefix(descriptionCharacterLimit)) + "..."
    }
}
