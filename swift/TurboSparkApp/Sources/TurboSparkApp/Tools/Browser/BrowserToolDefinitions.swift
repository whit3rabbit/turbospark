import Foundation

/// The command vocabulary the model may see for the active browser backend.
/// A missing or stale manifest fails closed, including for cached turn catalogs.
public struct BrowserToolAvailability: Equatable, Sendable {
    public let isEnabled: Bool
    public let manifest: BrowserBackendManifest?
    public let mediaCapability: AppToolMediaCapability

    public var supportsScreenshotMedia: Bool {
        mediaCapability.supportsImageBearingToolResults
    }

    public init(
        isEnabled: Bool,
        manifest: BrowserBackendManifest?,
        mediaCapability: AppToolMediaCapability = .textOnly
    ) {
        self.isEnabled = isEnabled
        self.manifest = manifest
        self.mediaCapability = mediaCapability
    }

    /// Source-compatible initializer for call sites written before media capabilities were typed.
    public init(isEnabled: Bool, manifest: BrowserBackendManifest?, supportsScreenshotMedia: Bool) {
        self.init(
            isEnabled: isEnabled,
            manifest: manifest,
            mediaCapability: supportsScreenshotMedia
                ? .imageBearingToolResults()
                : .textOnly)
    }

    public static let disabled = BrowserToolAvailability(isEnabled: false, manifest: nil)

    public func supports(_ command: BrowserControlCommandKind) -> Bool {
        guard isEnabled,
              let manifest,
              manifest.commandSurfaceVersion == BrowserControlProtocol.currentVersion,
              manifest.support(for: command) == .supported
        else {
            return false
        }
        return command != .screenshot || supportsScreenshotMedia
    }
}

/// Schemas and command mapping for the browser tools exposed to model calls.
public enum BrowserToolDefinitions {
    public static let commandKindsByToolName: [String: BrowserControlCommandKind] = [
        "browser_navigate": .navigate,
        "browser_click": .click,
        "browser_type": .type,
        "browser_press_key": .pressKey,
        "browser_scroll": .scroll,
        "browser_screenshot": .screenshot,
        "browser_snapshot": .readState,
        "browser_wait": .waitFor,
    ]

    public static let all: [OpenAITool] = [
        OpenAITool.function(
            name: "browser_navigate",
            description: "Navigate the controlled browser tab to an HTTP or HTTPS URL.",
            parameters: object(
                [
                    "url": .string(description: "Destination URL."),
                    "wait_until": .string(
                        description: "Load state to wait for.",
                        enumValues: ["started", "committed", "finished"],
                        defaultVal: BrowserLoadState.finished.rawValue),
                    "timeout_seconds": .number(description: "Action deadline, from 0 to 60 seconds."),
                ],
                required: ["url"])),
        OpenAITool.function(
            name: "browser_click",
            description: "Click a live element reference from the current browser snapshot.",
            parameters: object(
                [
                    "reference": .string(description: "Live element reference from browser_snapshot."),
                    "timeout_seconds": .number(description: "Action deadline, from 0 to 60 seconds."),
                ],
                required: ["reference"])),
        OpenAITool.function(
            name: "browser_type",
            description: "Type text into a live element reference without echoing the value in the result.",
            parameters: object(
                [
                    "reference": .string(description: "Live element reference from browser_snapshot."),
                    "text": .string(description: "Text to enter."),
                    "submit": .boolean(description: "Submit after entering the text."),
                    "timeout_seconds": .number(description: "Action deadline, from 0 to 60 seconds."),
                ],
                required: ["reference", "text"])),
        OpenAITool.function(
            name: "browser_press_key",
            description: "Press a key in the controlled tab, optionally targeting a live element reference.",
            parameters: object(
                [
                    "reference": .string(description: "Optional live element reference."),
                    "key": .string(description: "Key name to press."),
                    "timeout_seconds": .number(description: "Action deadline, from 0 to 60 seconds."),
                ],
                required: ["key"])),
        OpenAITool.function(
            name: "browser_scroll",
            description: "Scroll the controlled page or a live element reference.",
            parameters: object(
                [
                    "reference": .string(description: "Optional live element reference."),
                    "direction": .string(
                        description: "Scroll direction.",
                        enumValues: ["up", "down", "left", "right"]),
                    "amount": .integer(description: "Scroll distance in points.", defaultVal: "600"),
                ],
                required: ["direction"])),
        OpenAITool.function(
            name: "browser_screenshot",
            // Viewport only: the WebKit backend rejects full-page capture as
            // unsupported, which read to the model as "screenshots do not work".
            description: "Capture a screenshot of the visible viewport of the controlled browser tab.",
            parameters: object([:])),
        OpenAITool.function(
            name: "browser_snapshot",
            description: "Read a bounded, value-masked snapshot of the controlled browser tab.",
            parameters: object([
                "scope": .string(
                    description: "Snapshot detail to return.",
                    enumValues: ["interactiveElements", "visibleText", "pageStructure"],
                    defaultVal: BrowserSnapshotScope.interactiveElements.rawValue),
            ])),
        OpenAITool.function(
            name: "browser_wait",
            description: "Wait for a load state or element condition in the controlled browser tab.",
            parameters: object([
                "load_state": .string(
                    description: "Wait for this load state, or provide reference and condition.",
                    enumValues: ["started", "committed", "finished"]),
                "reference": .string(description: "Live element reference to inspect."),
                "condition": .string(
                    description: "Element condition, used with reference.",
                    enumValues: ["exists", "visible", "hidden", "enabled"]),
                "timeout_seconds": .number(description: "Wait deadline, from 0 to 60 seconds."),
            ])),
    ]

    public static func tools(availableFor availability: BrowserToolAvailability) -> [OpenAITool] {
        all.filter { tool in
            guard let kind = commandKindsByToolName[tool.function.name] else { return false }
            return availability.supports(kind)
        }
    }

    public static func commandKind(for toolName: String) -> BrowserControlCommandKind? {
        commandKindsByToolName[toolName.lowercased()]
    }

    private static func object(
        _ properties: [String: JSONSchemaProperty],
        required: [String] = []
    ) -> JSONSchema {
        .object(properties: properties, required: required, additionalProperties: false)
    }
}
