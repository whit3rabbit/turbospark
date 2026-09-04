import Foundation
import TurboSpark

/// Which reasoning levels this checkpoint can express, what to call them, and
/// what to do with a preference that does not carry over.
///
/// **THE ACCEPTED SET BELONGS TO THE CHECKPOINT'S OWN TEMPLATE**, probed at
/// open and reported as `SessionInfo.reasoningEfforts` -- not to the family,
/// and not to `Reasoning.allCases` (`swift/CLAUDE.md` Gotcha 9). Every
/// decision below is delegated to `ReasoningLevelPolicy`, which is pure and
/// therefore testable without a model, a Metal device and a 13 GB install;
/// what lives here is the binding of arguments to it.
extension AppModel {
    // `reasoningAvailable` was here and is gone: it had no callers, and once
    // `isReasoningSupported` stopped guessing from the family the two bodies
    // were byte-identical. Two names for one fact is a drift hazard, and the
    // one that drifts is the one nothing exercises.

    /// Whether the LOADED checkpoint's template can express a reasoning
    /// level at all.
    ///
    /// **There is deliberately no answer for an unloaded model.** This used
    /// to fall back to a hardcoded `reasoningFamilies` set when `info` was
    /// nil, which is the per-family table root Gotcha 56 exists to refuse:
    /// it guessed for a checkpoint whose template nobody had read, and every
    /// family it named has releases on both sides of the question. The
    /// levels arrive with the session, so the control waits for the session
    /// (see `reasoningPickerEnabled`).
    public var isReasoningSupported: Bool {
        info?.reasoningSupport != SessionInfo.ReasoningSupport.none
    }

    /// Whether the reasoning picker should accept input.
    ///
    /// One accessor rather than three view-local spellings, the same reason
    /// `activeLoadGuard` and `visionIsActive` are one each: a view assembling
    /// its own would drift, and the one that drifted would offer a level the
    /// turn then refuses.
    public var reasoningPickerEnabled: Bool {
        session != nil && isReasoningSupported
    }

    /// The reasoning levels worth offering for the loaded checkpoint.
    ///
    /// **THE SET COMES OFF THE CHECKPOINT'S OWN TEMPLATE**, probed at open
    /// and reported as `SessionInfo.reasoningEfforts`. It is not derivable
    /// from the family: Qwen 3.8 answers `[.off, .low, .medium, .xhigh]` and
    /// RAISES on `.high`, where gpt-oss and Muse Glimmer answer
    /// `[.off, .low, .medium, .high]`. This returned `allCases` for every
    /// `.level` checkpoint until 2026-08-31, so the menu carried an entry
    /// that failed the turn with a template error.
    ///
    /// `.toggleOnly` is the one case still decided here rather than by the
    /// engine. Its on-levels all render the same prompt, so the engine
    /// collapses them to one and reports `.low` BY POSITION; `.medium` is
    /// the friendlier middle-of-the-road spelling for a control that is
    /// really an on/off switch, and the views label it accordingly.
    /// The decision itself lives in `ReasoningLevelPolicy`, which is pure and
    /// therefore testable without a model, a Metal device and a 13 GB
    /// install. This is the binding of arguments to it and nothing else.
    public var availableReasoningLevels: [GenerateOptions.Reasoning] {
        ReasoningLevelPolicy.offered(
            support: info?.reasoningSupport,
            efforts: info?.reasoningEfforts ?? []
        )
    }

    /// What to call a level in this checkpoint's picker. See
    /// `ReasoningLevelPolicy.label(for:support:)`.
    public func reasoningLabel(for level: GenerateOptions.Reasoning) -> String {
        ReasoningLevelPolicy.label(for: level, support: info?.reasoningSupport)
    }

    /// The one-line description under a level in this checkpoint's picker.
    public func reasoningDescription(for level: GenerateOptions.Reasoning) -> String {
        ReasoningLevelPolicy.description(for: level, support: info?.reasoningSupport)
    }

    /// The offered level closest to `wanted`, used when a preference restored
    /// from another model does not carry over. See
    /// `ReasoningLevelPolicy.nearest(to:in:)`.
    public func nearestAvailableReasoning(
        to wanted: GenerateOptions.Reasoning
    ) -> GenerateOptions.Reasoning {
        ReasoningLevelPolicy.nearest(to: wanted, in: availableReasoningLevels)
    }

    /// Updates the current reasoning effort level and saves it as the preferred default for the active model.
    /// **KEYED ON `path`, NOT ON `alias`** (state#96, which is state#14's
    /// rule arriving in a third place). `??` takes the alias whenever there
    /// is one, so the path arm was dead code -- and an alias is not unique
    /// once a scanned LM Studio or custom-folder row exists, so two installs
    /// sharing a name shared one remembered reasoning level.
    /// `restoreReasoningPreference` still READS the alias as a legacy
    /// fallback, so nobody's existing preference is forgotten, and
    /// `deleteModel` prunes the entry.
    public func setReasoning(_ level: GenerateOptions.Reasoning) {
        self.reasoning = level
        if let path = selected?.path, !path.isEmpty {
            modelReasoningDefaults[path] = level.rawValue
        } else if let alias = selected?.alias {
            modelReasoningDefaults[alias] = level.rawValue
        }
        persistSettings()
        updateTokenEstimate()
    }
}
