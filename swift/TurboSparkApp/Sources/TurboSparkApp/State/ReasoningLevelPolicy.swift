import Foundation
import TurboSpark

/// Which reasoning levels to OFFER, given what a checkpoint says it accepts.
///
/// **A PURE TYPE SO IT CAN BE TESTED AT ALL.** `AppModel.info` is computed
/// from `session`, so every accessor built directly on it needs a real model,
/// a Metal device and a multi-gigabyte install before a single case can run --
/// which is why the level-offering logic shipped a bug for as long as it did.
/// Same move as `ServerStatusRows` and `AppModel.serverAPIKey(from:)`
/// (swift/CLAUDE.md Gotcha 26): the decision is a function of its arguments,
/// so it lives somewhere the arguments can be supplied.
///
/// The engine has already done the hard half. `SessionInfo.reasoningEfforts`
/// is probed off the checkpoint's own chat template at open, and levels that
/// render the same prompt are collapsed there. What is left here is
/// presentation and the one clamp a model SWITCH needs.
public enum ReasoningLevelPolicy {
    /// The levels a picker should list.
    ///
    /// `support` is nil when no model is loaded. There is deliberately no
    /// guess for that case: the accepted set belongs to the checkpoint's
    /// template and cannot be derived from its family (root Gotcha 56), so
    /// the honest answer before anyone has read the template is `[.off]`.
    public static func offered(
        support: SessionInfo.ReasoningSupport?,
        efforts: [GenerateOptions.Reasoning]
    ) -> [GenerateOptions.Reasoning] {
        switch support {
        // `nil` is no session and `.some(.none)` is a loaded checkpoint
        // whose template names no reasoning key. Same answer for different
        // reasons: nothing has been read, or there is nothing to read.
        case nil, .some(.none):
            return [.off]
        case .some(.toggleOnly):
            // Its template reads `enable_thinking` and no effort key, so every
            // on-level renders the same prompt. The engine collapses them to
            // one and reports `.low` by position; `.medium` is the spelling
            // this app puts on a control that is really a switch, and
            // `label(for:support:)` prints it as "On".
            return [.off, .medium]
        case .some(.level):
            // Never empty and always opening at `.off` on the engine's side,
            // but this arrives as a decoded wire value: an empty menu would
            // strand a user with no way to change the setting at all.
            return efforts.isEmpty ? [.off] : efforts
        }
    }

    /// The offered level closest to `wanted`, for a preference carried over
    /// from a DIFFERENT checkpoint.
    ///
    /// Falling straight to `.off` is the wrong answer for the common case.
    /// `.high` and `.xhigh` are both "the top setting", spelled differently by
    /// different checkpoints -- Qwen 3.8 accepts `xhigh` and RAISES on `high`,
    /// gpt-oss and Muse Glimmer the other way round -- so a user who set one
    /// and switched models meant the other. Dropping to `.off` silently turns
    /// thinking off for someone who explicitly turned it on, which is the
    /// silent no-op this whole feature exists to avoid.
    ///
    /// **THAT TOP-SPELLING CASE NEEDS NO ARM OF ITS OWN, AND IT TOOK A
    /// SURVIVING MUTATION TO NOTICE.** This shipped with an explicit
    /// `high <-> xhigh` swap ahead of the general rule, which read as the
    /// load-bearing line; deleting it changed no test and no behaviour,
    /// because `xhigh` is the TOP of `allCases`. Whenever one of the two is
    /// missing, the other is by definition the highest on-level offered, so
    /// the general rule already returns it. Two rules where one suffices, and
    /// the redundant one was the one carrying the comment.
    public static func nearest(
        to wanted: GenerateOptions.Reasoning,
        in offered: [GenerateOptions.Reasoning]
    ) -> GenerateOptions.Reasoning {
        if offered.contains(wanted) { return wanted }
        // Off is never clamped UPWARDS. This rescues a request to think and
        // must not manufacture one.
        if wanted == .off { return .off }

        // The highest on-level available, so an explicit request to think
        // still thinks. `allCases` is ascending, so the last match is the
        // highest -- which is also what carries the top-spelling case above.
        return GenerateOptions.Reasoning.allCases
            .filter { $0 != .off && offered.contains($0) }
            .last ?? .off
    }

    /// What to call a level in this checkpoint's picker.
    ///
    /// A `toggleOnly` template drops the level, so any effort word printed
    /// beside its on-state is a promise the template cannot keep.
    public static func label(
        for level: GenerateOptions.Reasoning,
        support: SessionInfo.ReasoningSupport?
    ) -> String {
        guard support == .toggleOnly, level != .off else { return level.label }
        return "On"
    }

    /// The one-line description under a level, for the same reason.
    public static func description(
        for level: GenerateOptions.Reasoning,
        support: SessionInfo.ReasoningSupport?
    ) -> String {
        guard support == .toggleOnly, level != .off else { return level.descriptionText }
        return "Thinking on (this model has no effort levels)"
    }
}
