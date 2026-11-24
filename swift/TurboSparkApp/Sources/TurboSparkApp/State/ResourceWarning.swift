import Foundation

/// The resource strip above the composer (`ResourceWarningStripView`): which
/// local resource, if any, is past its warning line right now, and whether the
/// user already dismissed that notice. Pure so the thresholds, the priority
/// between resources and the re-show rule are testable without a model.
///
/// The readings are the ones the status bar already shows
/// (`model.telemetry` and the context meter's fraction); nothing here polls.
enum ResourceWarning {
    /// Declaration order is the tie-break when two resources share a level:
    /// memory pressure can end the process, a full window stops the reply,
    /// and heat only slows it.
    enum Kind: Int, CaseIterable, Comparable {
        case memory, context, thermal

        static func < (lhs: Kind, rhs: Kind) -> Bool { lhs.rawValue < rhs.rawValue }
    }

    enum Level: Int, Comparable {
        case warning = 1
        case critical = 2

        static func < (lhs: Level, rhs: Level) -> Bool { lhs.rawValue < rhs.rawValue }
    }

    struct Notice: Equatable {
        let kind: Kind
        let level: Level
    }

    /// The context meter's tiers. 85 percent leaves room for one ordinary
    /// reply; 95 is where the next turn is likely to stop for space. Above
    /// auto-compact's trigger (4/5 of the usable window) on purpose, so a
    /// chat that compacts on its own never shows this.
    static let contextWarningFraction = 0.85
    static let contextCriticalFraction = 0.95

    /// The engine's `MemoryPressure` case names (`normal`, `warn`,
    /// `critical`). An unknown word is nil: never warn on a guess.
    static func memoryLevel(_ reading: String?) -> Level? {
        switch reading?.lowercased() {
        case "warn", "warning": return .warning
        case "critical": return .critical
        default: return nil
        }
    }

    /// `ProcessInfo.ThermalState` names. `fair` is ordinary load on a laptop
    /// and stays in the status bar only; `serious` is where macOS throttles.
    static func thermalLevel(_ reading: String?) -> Level? {
        switch reading?.lowercased() {
        case "serious": return .warning
        case "critical": return .critical
        default: return nil
        }
    }

    /// Nil while compacting: the summarizer is already the fix, and the
    /// status bar and ring both say it is running.
    static func contextLevel(fraction: Double?, isCompacting: Bool) -> Level? {
        guard let fraction, !isCompacting, fraction.isFinite else { return nil }
        if fraction >= contextCriticalFraction { return .critical }
        if fraction >= contextWarningFraction { return .warning }
        return nil
    }

    /// Every resource's level, nil for a fine one. Empty without a loaded
    /// local model: no strip for a feature the user is not using.
    static func levels(
        hasSession: Bool, memoryPressure: String?, thermalLevel: String?,
        contextFraction: Double?, isCompacting: Bool
    ) -> [Kind: Level] {
        guard hasSession else { return [:] }
        var out: [Kind: Level] = [:]
        out[.memory] = memoryLevel(memoryPressure)
        out[.thermal] = self.thermalLevel(thermalLevel)
        out[.context] = contextLevel(fraction: contextFraction, isCompacting: isCompacting)
        return out
    }

    /// The one notice to show, or nil.
    ///
    /// - `blockedByOtherStrip`: a model-load failure, a stall notice or an
    ///   error banner is up. Those describe an event that needs the user now,
    ///   so they win and this strip waits instead of stacking under them.
    /// - A dismissed resource stays hidden until its level rises above the
    ///   level that was dismissed.
    /// - Otherwise the highest level wins, then `Kind` order.
    static func notice(
        levels: [Kind: Level], dismissed: [Kind: Level], blockedByOtherStrip: Bool
    ) -> Notice? {
        guard !blockedByOtherStrip else { return nil }
        let candidates = levels.compactMap { kind, level -> Notice? in
            if let held = dismissed[kind], level <= held { return nil }
            return Notice(kind: kind, level: level)
        }
        return candidates.min { a, b in
            a.level != b.level ? a.level > b.level : a.kind < b.kind
        }
    }

    /// The dismissals that still hold after a new reading. A resource that
    /// went back to fine forgets its dismissal, so crossing the line again
    /// later shows the strip again. A resource that eased from critical to
    /// warning keeps the critical hold: the user already saw worse.
    static func prunedDismissals(_ dismissed: [Kind: Level], levels: [Kind: Level]) -> [Kind: Level] {
        dismissed.filter { kind, _ in levels[kind] != nil }
    }

    /// The dismissals after the user closes `notice`.
    static func dismissing(_ notice: Notice, from dismissed: [Kind: Level]) -> [Kind: Level] {
        var out = dismissed
        out[notice.kind] = max(notice.level, dismissed[notice.kind] ?? notice.level)
        return out
    }
}
