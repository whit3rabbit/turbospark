import Foundation
import TurboSpark

/// What should happen when someone asks to install a model.
///
/// A value rather than a `.disabled(...)` expression so the branches can be
/// tested at all, which is `ServerStatusRows`' argument (swift Gotcha 26) and
/// matters more here: this decides whether a user is allowed to spend twenty
/// minutes streaming a checkpoint that cannot open.
enum ModelInstallDecision: Equatable {
    /// Nothing known against it.
    case allowed
    /// Runs, but the user should see why first. Never silently blocked: a
    /// tight fit and a full-ish disk are both the user's call.
    case confirm(String)
    /// Will not work here. The reason is the engine's own words.
    case blocked(String)

    var isBlocked: Bool { if case .blocked = self { return true }; return false }
    var reason: String? {
        switch self {
        case .allowed: return nil
        case .confirm(let r), .blocked(let r): return r
        }
    }
}

/// The one place install permission is decided.
///
/// **THE GATE IS NOT THE BUTTON.** `AppModel.installModel` and
/// `installRepo` call this too, because a default spelled at the button and
/// nowhere else is exactly how every project in this app came to run shell
/// commands unprompted (swift Gotcha 28): the conservative spelling was
/// stated three times and the one users reached stated something else.
enum ModelInstallGate {
    /// Leave this much free after an install rather than filling the volume.
    /// APFS needs room to breathe and a machine at zero free bytes is not a
    /// working machine.
    static let diskHeadroomBytes: UInt64 = 8 * 1024 * 1024 * 1024

    /// - Parameters:
    ///   - probeRunnable: `nil` when nothing has probed this repository.
    ///     Absence is not a refusal: the curated rows are curated, and
    ///     requiring a probe before every install would cost a round trip on
    ///     the common path.
    ///   - refusedBecause: the probe's own wording, including the bring-up
    ///     clause for an architecture with no decode flow here.
    ///   - verdict: `nil` or `.unknown` when nothing has been sized.
    static func decide(
        probeRunnable: Bool?,
        refusedBecause: String?,
        verdict: ModelRecommendation.FitVerdict?,
        installBytes: UInt64?,
        downloadBytes: UInt64? = nil,
        freeDiskBytes: UInt64?
    ) -> ModelInstallDecision {
        // **A REFUSAL UPSTREAM OF THE ARITHMETIC OUTRANKS THE ARITHMETIC.**
        // A model with no decode flow does not "fit" whatever its footprint
        // would be, and the architecture message is the one that tells a user
        // something actionable. `catalog::recommend::discover` orders these
        // the same way, and `ProbeReport::refuse` keeps the FIRST refusal for
        // the same reason (catalog Gotcha 4).
        if probeRunnable == false {
            return .blocked(
                refusedBecause ?? "This port has no decode flow for this checkpoint.")
        }

        if verdict == .refused {
            return .blocked(
                "What the engine allocates for this model does not fit in memory. "
                    + "The KV cache and the expert streamer both allocate up front, so it "
                    + "would fail to load rather than run slowly.")
        }

        // Disk before memory: it is the one that wastes the download.
        let temporaryBytes = installBytes.map { install in
            let sum = install.addingReportingOverflow(downloadBytes ?? 0)
            return sum.overflow ? UInt64.max : sum.partialValue
        }
        if let need = temporaryBytes, let free = freeDiskBytes,
            need > free || (free - need) < diskHeadroomBytes
        {
            return .confirm(
                "This needs up to \(MetricFormat.storage(need)) temporarily and there is "
                    + "\(MetricFormat.storage(free)) free. Verified download ranges are kept "
                    + "until the install passes verification, then removed.")
        }

        if verdict == .tight {
            return .confirm(
                "This fits with under ten percent to spare. It will run, and anything "
                    + "else on this machine competes with it for memory.")
        }

        return .allowed
    }

    /// Free space on the volume holding `url`, or `nil` when the system
    /// declines to say.
    ///
    /// `nil` is NOT zero. A failed query must not read as a full disk, or the
    /// gate refuses every install on a machine whose volume it cannot inspect.
    static func freeSpace(at url: URL) -> UInt64? {
        guard
            let values = try? url.resourceValues(forKeys: [
                .volumeAvailableCapacityForImportantUsageKey
            ]),
            let available = values.volumeAvailableCapacityForImportantUsage
        else { return nil }
        return available >= 0 ? UInt64(available) : nil
    }
}
