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

    /// `decide`, for the destination the user chose: its own free space, its name
    /// in the sentences, and the connect statement when it is not there.
    static func decide(
        probeRunnable: Bool?,
        refusedBecause: String?,
        verdict: ModelRecommendation.FitVerdict?,
        installBytes: UInt64?,
        downloadBytes: UInt64? = nil,
        to destination: InstallGateDestination
    ) -> ModelInstallDecision {
        decide(
            probeRunnable: probeRunnable, refusedBecause: refusedBecause, verdict: verdict,
            installBytes: installBytes, downloadBytes: downloadBytes,
            freeDiskBytes: destination.freeBytes, destinationName: destination.libraryName,
            unconnectedDestination: destination.notConnectedName)
    }

    /// - Parameters:
    ///   - probeRunnable: `nil` when nothing has probed this repository.
    ///     Absence is not a refusal: the curated rows are curated, and
    ///     requiring a probe before every install would cost a round trip on
    ///     the common path.
    ///   - refusedBecause: the probe's own wording, including the bring-up
    ///     clause for an architecture with no decode flow here.
    ///   - verdict: `nil` or `.unknown` when nothing has been sized.
    ///   - freeDiskBytes: the free space of the CHOSEN destination, never of
    ///     another disk (requirement 2.3).
    ///   - destinationName: the library the model is going to, named in the
    ///     low-space sentence. `nil` is This Mac, whose sentence is unchanged.
    ///   - unconnectedDestination: the library to connect when the chosen one is
    ///     not there. An install to it cannot start, so this blocks.
    static func decide(
        probeRunnable: Bool?,
        refusedBecause: String?,
        verdict: ModelRecommendation.FitVerdict?,
        installBytes: UInt64?,
        downloadBytes: UInt64? = nil,
        freeDiskBytes: UInt64?,
        destinationName: String? = nil,
        unconnectedDestination: String? = nil
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

        // An install into a library that is not there cannot start, and the
        // free space below would be some other disk's. Connecting the drive is
        // the fix, so say that (requirement 2.4).
        if let name = unconnectedDestination {
            return .blocked(notConnectedSentence(name: name))
        }

        // Disk before memory: it is the one that wastes the download.
        let temporaryBytes = installBytes.map { install in
            let sum = install.addingReportingOverflow(downloadBytes ?? 0)
            return sum.overflow ? UInt64.max : sum.partialValue
        }
        if let need = temporaryBytes, let free = freeDiskBytes,
            need > free || (free - need) < diskHeadroomBytes
        {
            // A library is named and counted with its own free space. This Mac
            // keeps the sentence it always had.
            if let name = destinationName {
                return .confirm(
                    lowSpaceSentence(
                        need: MetricFormat.storage(need), free: MetricFormat.storage(free), name: name))
            }
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

    /// The statement for a destination that is not there (requirement 2.4):
    /// the library to connect, escaped, and what to do about it.
    static func notConnectedSentence(name: String, language: AppLanguage? = nil) -> String {
        let name = UntrustedText.display(name, limit: InstallDestinationModel.nameLimit)
        return AppStrings.localized(
            "\(name) is not connected. Connect it or choose another library.",
            language: language ?? AppStrings.currentLanguage)
    }

    /// The confirmation for a library with too little room (requirement 2.3):
    /// what the install needs, the library by name, and that library's free
    /// space. The sizes are formatted strings, never a Double in a key.
    static func lowSpaceSentence(
        need: String, free: String, name: String, language: AppLanguage? = nil
    ) -> String {
        let name = UntrustedText.display(name, limit: InstallDestinationModel.nameLimit)
        return AppStrings.localized(
            "This needs up to \(need) temporarily, but \(name) has only \(free) free. Verified download ranges are kept until the install passes verification, then removed.",
            language: language ?? AppStrings.currentLanguage)
    }

    /// Free space on This Mac, which is the home folder's volume: the engine's
    /// store root is always the home, whatever libraries are listed. A library's
    /// room comes from `LibraryState.freeBytes`, never from here.
    ///
    /// The models folder may not exist yet, so this walks up to the nearest
    /// existing ancestor before asking. `nil` is NOT zero: a failed query must
    /// not read as a full disk, or the gate refuses every install on a machine
    /// whose volume it cannot inspect.
    static func freeSpaceOnThisMac() -> UInt64? {
        var url = URL(fileURLWithPath: ModelStorageManager.defaultTurboSparkStoreRoot)
        while !FileManager.default.fileExists(atPath: url.path), url.path != "/" {
            url = url.deletingLastPathComponent()
        }
        return freeSpace(at: url)
    }

    /// Free space on the volume holding `url`, or `nil` when the system
    /// declines to say.
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
