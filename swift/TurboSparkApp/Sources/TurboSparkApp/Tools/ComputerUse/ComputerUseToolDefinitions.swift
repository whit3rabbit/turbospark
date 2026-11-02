import Foundation
import ComputerUseKit

/// Whether computer-use tools are offered this turn.
///
/// All three must hold, or the model sees none of the tools: the user turned
/// the feature on, the helper reports the current command surface, and the
/// loaded model takes images. The browser hides only its screenshot tool for
/// a text-only model; here the whole family goes, because driving a desktop
/// without seeing it is guessing.
public struct ComputerUseToolAvailability: Equatable, Sendable {
    public let isEnabled: Bool
    public let manifest: ComputerUseBackendManifest?
    public let mediaCapability: AppToolMediaCapability

    public init(
        isEnabled: Bool,
        manifest: ComputerUseBackendManifest?,
        mediaCapability: AppToolMediaCapability
    ) {
        self.isEnabled = isEnabled
        self.manifest = manifest
        self.mediaCapability = mediaCapability
    }

    public static let disabled = ComputerUseToolAvailability(
        isEnabled: false, manifest: nil, mediaCapability: .textOnly)

    public var isOffered: Bool {
        isEnabled
            && mediaCapability.supportsImageBearingToolResults
            && (manifest?.isCurrent ?? false)
    }

    public func supports(toolName: String) -> Bool {
        guard isOffered, let manifest else { return false }
        let name = toolName.lowercased()
        if name == ComputerUseToolDefinitions.requestAccessToolName { return true }
        guard let kind = ComputerUseToolDefinitions.commandKindsByToolName[name] else { return false }
        return manifest.supports(kind)
    }
}

public enum ComputerUseToolDefinitions {
    public static let requestAccessToolName = "computer_request_access"

    public static let commandKindsByToolName: [String: ComputerUseCommandKind] = [
        "computer_list_apps": .listApps,
        "computer_snapshot": .snapshot,
        "computer_screenshot": .screenshot,
        "computer_click": .click,
        "computer_type": .type,
        "computer_key": .key,
        "computer_scroll": .scroll,
        "computer_open_app": .openApp,
    ]

    public static var toolNames: Set<String> {
        Set(commandKindsByToolName.keys).union([requestAccessToolName])
    }

    private static let appParameter = JSONSchemaProperty.string(
        description: "Bundle identifier of a granted app, for example com.apple.TextEdit.")

    public static let all: [OpenAITool] = [
        OpenAITool.function(
            name: "computer_list_apps",
            description: "List running macOS apps with their bundle identifiers.",
            parameters: object([:])),
        OpenAITool.function(
            name: requestAccessToolName,
            description: "Ask the user to let you control macOS apps. Required before any other computer tool touches an app.",
            parameters: object(
                [
                    "apps": .string(description: "Comma-separated bundle identifiers."),
                    "reason": .string(description: "One sentence the user sees: what you will do in these apps."),
                ],
                required: ["apps", "reason"])),
        OpenAITool.function(
            name: "computer_snapshot",
            description: "Read the accessibility tree of an app's front window. Returns element references for computer_click. Prefer this to screenshots for finding controls.",
            parameters: object(["app": appParameter], required: ["app"])),
        OpenAITool.function(
            name: "computer_screenshot",
            description: "Capture an image of an app's front window. Returns a screenshot_id for pixel clicks.",
            parameters: object(["app": appParameter], required: ["app"])),
        OpenAITool.function(
            name: "computer_click",
            description: "Click in an app. Give reference from computer_snapshot (preferred), or x and y in pixels of the screenshot named by screenshot_id.",
            parameters: object(
                [
                    "app": appParameter,
                    "reference": .string(description: "Element reference from computer_snapshot."),
                    "x": .number(description: "Horizontal pixel in the screenshot."),
                    "y": .number(description: "Vertical pixel in the screenshot."),
                    "screenshot_id": .integer(description: "The screenshot the x and y refer to."),
                    "button": .string(description: "Mouse button.", enumValues: ["left", "right"], defaultVal: "left"),
                    "count": .integer(description: "1 for a click, 2 for a double click.", defaultVal: "1"),
                ],
                required: ["app"])),
        OpenAITool.function(
            name: "computer_type",
            description: "Type text into the focused field of an app.",
            parameters: object(
                ["app": appParameter, "text": .string(description: "Text to type.")],
                required: ["app", "text"])),
        OpenAITool.function(
            name: "computer_key",
            description: "Press a key or shortcut in an app, for example return, escape, cmd+s.",
            parameters: object(
                ["app": appParameter, "key": .string(description: "Key or modifier+key combination.")],
                required: ["app", "key"])),
        OpenAITool.function(
            name: "computer_scroll",
            description: "Scroll an app's front window, or the element at a reference.",
            parameters: object(
                [
                    "app": appParameter,
                    "direction": .string(description: "Scroll direction.", enumValues: ["up", "down", "left", "right"]),
                    "amount": .integer(description: "Scroll distance in lines.", defaultVal: "5"),
                    "reference": .string(description: "Optional element reference from computer_snapshot."),
                ],
                required: ["app", "direction"])),
        OpenAITool.function(
            name: "computer_open_app",
            description: "Launch or bring a granted app to the front.",
            parameters: object(["app": appParameter], required: ["app"])),
    ]

    public static func tools(availableFor availability: ComputerUseToolAvailability) -> [OpenAITool] {
        all.filter { availability.supports(toolName: $0.function.name) }
    }

    private static func object(
        _ properties: [String: JSONSchemaProperty],
        required: [String] = []
    ) -> JSONSchema {
        .object(properties: properties, required: required, additionalProperties: false)
    }
}
