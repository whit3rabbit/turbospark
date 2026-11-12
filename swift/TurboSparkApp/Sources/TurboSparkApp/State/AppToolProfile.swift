import Foundation

/// A named set of tools an interactive turn is offered.
///
/// **A PROFILE ONLY REMOVES CAPABILITY.** It narrows what `AppToolCatalog`
/// captures for a turn, and that one capture feeds the prompt, the native
/// lane, the token estimate and `TurnAvailableTools.validate`. A call to a tool
/// outside the profile is therefore refused at execution too, not merely
/// hidden from the prompt. Permissions, approvals and the sandbox are
/// untouched: a profile can never grant what the permission engine denies.
///
/// `allowedTools` holds lowercase canonical keys from
/// `AppToolCatalog.profileKey(for:)`, so `read` matches `read_file`,
/// `FileRead` and whatever the file reader is named next. `nil` means every
/// tool, which is what lets Default pick up a new tool without a migration.
public struct AppToolProfile: Identifiable, Codable, Equatable, Sendable {
    public var id: UUID
    public var name: String
    public var allowedTools: [String]?
    /// MCP servers, the progressive-disclosure tools that front them, the
    /// codemode script tool and user/project custom tools, as one switch.
    /// They are discovered at runtime, so a per-name allowlist cannot list them.
    public var includesMcpAndCustom: Bool

    public init(
        id: UUID = UUID(), name: String, allowedTools: [String]?,
        includesMcpAndCustom: Bool = true
    ) {
        self.id = id
        self.name = name
        self.allowedTools = allowedTools.map(Self.normalized)
        self.includesMcpAndCustom = includesMcpAndCustom
    }

    /// One malformed row must not discard the user's other profiles. A row
    /// whose allowlist is present but unreadable is dropped by the caller's
    /// element-wise decode rather than defaulted: defaulting to "every tool"
    /// would silently widen a profile the user narrowed on purpose.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = container.decodeLenient(UUID.self, forKey: .id, fallback: UUID())
        name = container.decodeLenient(String.self, forKey: .name, fallback: "Tool Profile")
        allowedTools = try container.decodeIfPresent([String].self, forKey: .allowedTools)
            .map(Self.normalized)
        includesMcpAndCustom = container.decodeLenient(
            Bool.self, forKey: .includesMcpAndCustom, fallback: true)
    }

    static func normalized(_ keys: [String]) -> [String] {
        Array(Set(keys.map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() }
            .filter { !$0.isEmpty })).sorted()
    }

    // MARK: Built-ins

    public static let defaultID = UUID(uuidString: "7A0C4E52-3B1D-4F6A-9C85-1D2E3F4A5B01")!
    public static let minimalID = UUID(uuidString: "7A0C4E52-3B1D-4F6A-9C85-1D2E3F4A5B02")!
    public static let readOnlyID = UUID(uuidString: "7A0C4E52-3B1D-4F6A-9C85-1D2E3F4A5B03")!

    /// Today's complete tool set. Selected on a fresh install and whenever the
    /// stored selection no longer exists.
    public static let `default` = AppToolProfile(
        id: defaultID, name: "Default", allowedTools: nil, includesMcpAndCustom: true)

    /// The Pi coding agent's four tools. `bashoutput` and `killshell` ride
    /// along because `Bash` can start a background shell, and a model that
    /// cannot read that shell's output would be handed a dead feature.
    public static let minimal = AppToolProfile(
        id: minimalID, name: "Minimal",
        allowedTools: ["read", "write", "edit", "bash", "bashoutput", "killshell"],
        includesMcpAndCustom: false)

    /// Inspection only: nothing that writes, runs a command or leaves the app
    /// beyond a page fetch.
    public static let readOnly = AppToolProfile(
        id: readOnlyID, name: "Read-only",
        allowedTools: ["read", "glob", "grep", "grep_search", "webfetch"],
        includesMcpAndCustom: false)

    /// Code-defined and never persisted, so an app update can change them and
    /// the user cannot lose or corrupt one. "Duplicate" makes an editable copy.
    public static let builtIns: [AppToolProfile] = [.default, .minimal, .readOnly]

    public var isBuiltIn: Bool { Self.builtIns.contains { $0.id == id } }

    /// The name to show. A built-in's stored name is its English key; a custom
    /// profile's name is the user's own words and is shown as written.
    public var displayName: String {
        switch id {
        case Self.defaultID: return AppStrings.localized("Default")
        case Self.minimalID: return AppStrings.localized("Minimal")
        case Self.readOnlyID: return AppStrings.localized("Read-only")
        default: return name
        }
    }

    /// True when nothing is filtered, so the capture can skip the filter and
    /// stay byte-identical to the pre-profile behavior.
    public var allowsEverything: Bool { allowedTools == nil && includesMcpAndCustom }
}
