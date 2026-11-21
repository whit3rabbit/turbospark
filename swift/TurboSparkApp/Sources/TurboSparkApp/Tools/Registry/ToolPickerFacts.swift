import Foundation

/// A rough prompt cost for one tool definition, for the composer picker.
///
/// An estimate, labelled as one in the UI: the exact cost depends on the
/// checkpoint's chat template (`nativeToolTokenCost` measures that with the
/// engine, but only for a loaded session and a whole set). The picker needs
/// a per-tool figure with no model loaded, so it uses the same four
/// characters per token the skills budget assumes (`SkillManager.charsPerToken`).
enum ToolPromptCostEstimate {
    static let charsPerToken = SkillManager.charsPerToken

    /// Tokens for `count` characters, rounded up. Nil for a negative count.
    static func tokens(forCharacterCount count: Int) -> Int? {
        guard count >= 0 else { return nil }
        return (count + charsPerToken - 1) / charsPerToken
    }

    /// The estimate for `tool` as the JSON the native lane renders. Nil when
    /// it does not encode.
    static func tokens(for tool: OpenAITool) -> Int? {
        guard let data = try? JSONEncoder().encode(tool) else { return nil }
        return tokens(forCharacterCount: data.count)
    }
}

/// What the permission settings typically do with one tool's calls, for the
/// picker's label. Display only: the gate is `AppToolPermissionEngine`.
///
/// It follows the engine's ladder for an ordinary, low-risk call with no
/// session grant (`ToolPickerFactsTests` checks it against the engine). Where
/// the answer depends on the call itself (the domain a web tool reaches, a
/// browser origin), it reports the stricter answer, so the label can read
/// tighter than a given call but never looser.
enum ToolPermissionSummary {
    static func typicalDecision(
        toolName: String,
        permissions: AppProjectPermissions,
        networkPolicy: AppNetworkPolicy
    ) -> AppToolPermission {
        let name = toolName.lowercased()
        let category = AppToolCatalog.builtInCategory(for: name) ?? .automation
        let base = categoryPermission(category, toolName: name, permissions: permissions)

        // Browser automation needs a grant per exact origin before any mode
        // applies, so the best an agent call can count on is a question.
        if category == .browser { return base == .deny ? .deny : .ask }

        var decision: AppToolPermission
        if permissions.mode == .readOnly {
            decision = category == .fileRead ? .allow : .deny
        } else if base == .deny {
            decision = .deny
        } else if permissions.mode == .fullAccess || permissions.mode == .permissive {
            decision = .allow
        } else if permissions.mode == .ask || base == .ask {
            decision = category == .fileRead ? .allow : .ask
        } else {
            decision = .allow
        }

        // The network policy is a ceiling over every mode. Unlisted domains
        // ask (or are refused in Custom), and a web tool can reach any domain.
        if category == .web {
            switch networkPolicy.mode {
            case .blockAll: decision = .deny
            case .ask: decision = stricter(decision, .ask)
            case .custom: decision = stricter(decision, stricter(networkPolicy.customUnlisted, .ask))
            case .allowAll: break
            }
        }
        return decision
    }

    /// The category row a call answers to, as `evaluateProjectPolicy` picks it.
    private static func categoryPermission(
        _ category: AppToolCategory, toolName: String, permissions: AppProjectPermissions
    ) -> AppToolPermission {
        switch category {
        case .fileRead: return permissions.fileRead
        case .fileWrite: return permissions.fileWrite
        case .terminal: return permissions.terminal
        case .web:
            // image_fetch writes a project file, so file writes gate it too.
            return toolName == "image_fetch"
                ? stricter(permissions.web, permissions.fileWrite) : permissions.web
        case .browser: return permissions.browser
        case .computerUse: return permissions.computerUse
        case .mcp: return permissions.mcp
        case .automation: return permissions.automation
        }
    }

    private static func rank(_ permission: AppToolPermission) -> Int {
        switch permission {
        case .allow: return 0
        case .ask: return 1
        case .deny: return 2
        }
    }

    static func stricter(_ a: AppToolPermission, _ b: AppToolPermission) -> AppToolPermission {
        rank(a) >= rank(b) ? a : b
    }
}
