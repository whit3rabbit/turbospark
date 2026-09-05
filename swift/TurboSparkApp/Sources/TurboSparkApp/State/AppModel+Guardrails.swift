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
    /// Whether this checkpoint's OWN markup frames tool calls the engine
    /// parses.
    ///
    /// **READ OFF THE ENGINE, NOT GUESSED FROM THE FAMILY** (root Gotcha 62:
    /// read a property off the artifact, never off a note about it). This
    /// used to match a hardcoded nine-entry family set and then fall through
    /// to dialect substrings, and both halves were wrong in ways only the
    /// engine can settle. `museglimmer` was in that set and is NOT native:
    /// its checkpoint frames calls as an `<atem:function_calls>` block this
    /// engine has no parser for, so `StructuredAssistantDecoder` reports them
    /// as REASONING and no call is ever handed over. The list also carried
    /// `deepseekv4flash`, whose decode flow does not exist.
    ///
    /// **THIS IS NOT "CAN THIS MODEL USE TOOLS", AND GATING A CONTROL ON IT
    /// WOULD BE BACKWARDS.** A model on a dialect with no tool markup can
    /// still be prompted into emitting a call as ordinary prose, and
    /// recovering exactly that is what Forge Guardrails' rescue is for -- so
    /// `false` marks the case a rescue helps MOST. It picks the DEFAULT and
    /// feeds a tooltip; `guardrailsInertReason` is what actually gates.
    public var isToolCallingSupported: Bool {
        // With no session there is no answer. The permissive default is
        // state#97's: the old `installed.first` fallback answered for a
        // DIFFERENT model, which is worse than answering nothing.
        info?.toolCalling.native ?? true
    }

    /// Why this checkpoint hands no framed call over, or nil when it does.
    /// Shown beside the control rather than used to hide it.
    public var toolCallingNativeReason: String? { info?.toolCalling.reason }

    /// Why the Forge Guardrails control would do nothing right now, or nil
    /// when it is live.
    ///
    /// **THE HONEST INERT CONDITION IS "NO TOOLS WERE OFFERED", NOT "THE
    /// MODEL LACKS MARKUP".** `ForgeGuardrailsEngine.inspect`'s first branch
    /// accepts unconditionally when the request carried no tools, exactly as
    /// `docs/FORGE_GUARDRAILS.md` section 2 describes -- and this app offers
    /// none in conversational Chat mode or under a chat whose project is gone
    /// (`extractToolCalls`'s own guard, state#82). Those are the cases where
    /// the toggle genuinely changes nothing.
    ///
    /// Takes the TURN's project rather than the selection, for the reason
    /// every other read on this path does (state#30).
    public func guardrailsInertReason(for project: AppProject?) -> String? {
        if interactionMode != .projects {
            return "This chat sends no tools, so guardrails has nothing to check. "
                + "Switch to Projects mode to use tools."
        }
        if project == nil {
            return "This chat has no project, so no tools are offered and guardrails has "
                + "nothing to check."
        }
        return nil
    }

    /// The UI's read of the same rule, taking the selection deliberately.
    public var effectiveGuardrailsInertReason: String? {
        guardrailsInertReason(for: selectedProject)
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
