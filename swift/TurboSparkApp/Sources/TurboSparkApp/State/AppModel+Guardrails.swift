import Foundation
import TurboSpark

/// Forge tool-call guardrails, and the question they default from.
///
/// **THE ARGUMENT FORM IS THE ONE THE GENERATION PATH USES** (state#30). A
/// turn's project is a property of its chat, and the selection can move while
/// the turn is in flight -- a call can sit at an approval card for minutes
/// with `generating` false, which is exactly when a user is free to switch.
/// `effectiveForgeGuardrailsEnabled` is the UI's read of the same rule and
/// takes the selection deliberately.
///
/// Not to be confused with the MEMORY guardrails in `AppRuntimeOptions`
/// (`swift/CLAUDE.md` Gotcha 24): two unrelated things in this app are called
/// guardrails, they share no code, no settings key and no UI surface.
extension AppModel {
    /// Whether the active model supports tool calling and structured function invocation.
    ///
    /// **NO `installed.first` FALLBACK** (state#97). With nothing selected
    /// this answered for whatever row happened to sort first, which is a
    /// statement about a DIFFERENT model -- and it feeds
    /// `forgeGuardrailsEnabled`, so the guardrails default was decided by an
    /// unrelated install. With no model there is no answer, and the honest
    /// default is the permissive one the empty case already returned. The
    /// family list also lagged `crates/model-io/src/arch_config/family.rs`
    /// by three entries.
    public var isToolCallingSupported: Bool {
        guard let selectedModel = selected else {
            return true
        }
        let family = selectedModel.family.lowercased()
        let dialect = info?.dialect.lowercased() ?? ""
        let toolFamilies: Set<String> = [
            "gemma4", "qwen36", "qwen3moe", "qwen35", "gptoss", "llama",
            "museglimmer", "qwen4exp", "deepseekv4flash",
        ]
        if toolFamilies.contains(family) {
            return true
        }
        if dialect.contains("chatml") || dialect.contains("harmony") || dialect.contains("llama") || dialect.contains("gemma") || dialect.contains("mistral") {
            return true
        }
        return false
    }

    /// Forge Guardrails resolved for a NAMED project rather than for the
    /// selection (state#30).
    ///
    /// A turn's project is a property of its chat, and the selection can move
    /// while the turn is in flight -- a call can sit at an approval card for
    /// minutes with `generating` false, which is exactly when the user is
    /// free to switch. Every read on the generation path takes the argument
    /// form; `effectiveForgeGuardrailsEnabled` below is the UI's read of the
    /// same rule.
    public func forgeGuardrailsEnabled(for project: AppProject?) -> Bool {
        switch guardrailsMode {
        case .alwaysOn:
            return true
        case .alwaysOff:
            return false
        case .select:
            if let projectOverride = project?.forgeGuardrailsEnabled {
                return projectOverride
            }
            if let composerOverride = composerGuardrailsOverride {
                return composerOverride
            }
            return isToolCallingSupported
        }
    }

    /// Effective resolution of whether Forge Guardrails is active for the current context.
    public var effectiveForgeGuardrailsEnabled: Bool {
        forgeGuardrailsEnabled(for: selectedProject)
    }

    /// The agent profile of a NAMED project. See `forgeGuardrailsEnabled(for:)`
    /// for why the generation path takes the argument form (state#30).
    public func agentType(for project: AppProject?) -> AppAgentType {
        project?.agentType ?? .coder
    }

    /// Toggles or sets the Forge Guardrails active state for the current project or draft.
    public func setForgeGuardrailsEnabled(_ enabled: Bool) {
        if var proj = selectedProject {
            proj.forgeGuardrailsEnabled = enabled
            proj.updatedAt = Date()
            updateProject(proj)
        } else {
            composerGuardrailsOverride = enabled
        }
    }
}
