import Foundation
import TurboSpark

/// Where an install goes, and what the app does when that destination is not
/// there or the drive is lost (task 6.9, requirements 2.1 to 2.6).
///
/// **THE `library:` ARGUMENT IS "THIS MAC" WHEN IT IS NIL.** Every install entry
/// point takes `library: String? = nil`, where nil and `internal` both mean This
/// Mac, so a caller with no menu (Humanize's download, a restored queue row)
/// installs where installs always went. A surface with a menu resolves the user's
/// choice to an explicit id first (`InstallDestinationModel.libraryForInstall`),
/// because for the menu a choice never made means "follow the default".
extension AppModel {
    /// The destinations an install may go to now, with each one's free space. This
    /// Mac's is read from the disk on each call.
    var installDestinations: InstallDestinationModel {
        InstallDestinationModel(state: libraryState, thisMacFreeBytes: thisMacFreeSpaceReader())
    }

    /// The library id to hand the engine: nil for This Mac (nil, empty or
    /// `internal`), otherwise the id as given.
    nonisolated static func destinationLibrary(_ library: String?) -> String? {
        guard let id = InstallDestinationModel.normalized(library),
            id != InstallDestinationModel.thisMacId
        else { return nil }
        return id
    }

    // MARK: Gates

    /// The install gate for a destination (requirement 2.3): the chosen library's
    /// own free space and name, and the connect statement when it is not there.
    func installGateDecision(
        verdict: ModelRecommendation.FitVerdict?, installBytes: UInt64?, downloadBytes: UInt64?,
        library: String?
    ) -> ModelInstallDecision {
        ModelInstallGate.decide(
            probeRunnable: nil, refusedBecause: nil, verdict: verdict, installBytes: installBytes,
            downloadBytes: downloadBytes, to: installDestinations.gateDestination(library: library))
    }

    /// The same for an audio profile, sized by its download (requirement 2.3 and
    /// task 6.10). Audio had no free-space check before.
    func audioInstallDecision(for profile: AudioProfile, library: String?) -> ModelInstallDecision {
        installGateDecision(
            verdict: nil, installBytes: profile.downloadBytes > 0 ? profile.downloadBytes : nil,
            downloadBytes: nil, library: library)
    }

    /// The sentence for a destination that cannot take an install now, or nil when
    /// it can. Every modality refuses an away drive here, before it queues.
    func installDestinationBlockReason(library: String?) -> String? {
        installDestinations.gateDestination(library: library).notConnectedName.map {
            ModelInstallGate.notConnectedSentence(name: $0)
        }
    }

    // MARK: Failures (requirement 2.4)

    /// The destination library's name for a sentence, raw: This Mac in the UI
    /// language, the listed library's name, or nil for one no longer listed.
    func installLibraryName(_ library: String?) -> String? {
        guard let id = Self.destinationLibrary(library) else {
            return AppStrings.localized("This Mac")
        }
        return libraryState.libraries.first { $0.id == id }?.name
    }

    /// The sentence for a failed install when the engine sent one of the install
    /// codes, or nil for any other failure (which keeps its old surface).
    func libraryInstallFailureSentence(
        _ error: Error, alias: String, library: String?, modality: String = "text"
    ) -> String? {
        let holder = libraryState.allInstalled.first { $0.alias == alias && $0.modality == modality }
        return InstallFailureText.sentence(
            for: error, alias: alias, libraryName: installLibraryName(library),
            place: InstallFailureText.place(of: holder))
    }

    /// Shows a failed install's sentence the way an install failure is shown: on
    /// the download's row and as a warning toast. It is never `error`, and never an
    /// availability banner: an unplugged drive is not the app failing.
    func presentLibraryInstallFailure(_ sentence: String) {
        setModelDownloadStatus(.failed, failure: sentence)
        showToast(sentence, style: .warning, duration: 8.0)
    }

    /// After any install that did not finish: read the installed list and, in the
    /// background, the library state again, so a model that the engine did not
    /// keep is not on screen (requirement 2.4: no partial model).
    func refreshAfterInstallFailure() {
        refreshModels()
        refreshLibraryStateInBackground()
    }

    /// After an install that finished: the model is listed under the library it
    /// went into (requirement 2.6). The engine recorded its `library` and
    /// `installId`; this makes the installed list and the library state read them.
    func refreshAfterInstallSuccess() {
        refreshModels()
        refreshLibraryStateInBackground()
    }

    /// Reads the library state again WITHOUT making the install wait for it.
    ///
    /// **THE INSTALL NEVER WAITS ON A DRIVE.** A listing probes drives and the
    /// engine can stall on one that was just pulled, which is the moment a failed
    /// install asks for it. If the install's tail awaited it, `isInstallingModel`
    /// would stay set and every later install would be refused until the drive
    /// answered. `LibraryState` folds overlapping requests into one more run, so
    /// firing this is always safe.
    func refreshLibraryStateInBackground() {
        let state = libraryState
        Task { await state.refresh() }
    }
}
