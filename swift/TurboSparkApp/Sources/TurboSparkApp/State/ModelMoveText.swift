import Foundation
import TurboSpark

/// What the app knows about a move that ended without moving, to word it.
/// Every name is raw (it comes from a drive); the text escapes each one.
struct MoveFailureContext: Equatable, Sendable {
    let alias: String
    /// The library the model was in. Display text already (the plan made it), and
    /// `UntrustedText.display` leaves its own output as it is.
    let sourceName: String
    let destinationName: String
    /// The model's size, and the destination's free space as the app measured it
    /// before the engine was called. The engine's `insufficient_space` sentence
    /// holds neither figure in a form to parse, so these are the app's own.
    let requiredBytes: UInt64?
    let availableBytes: UInt64?
    /// The library, source or destination, that is not connected now, when the
    /// library state could say exactly one.
    let awayDriveName: String?
}

/// How a move that did not finish ended, in words.
struct MoveFailureText: Equatable, Sendable {
    enum Kind: Equatable, Sendable {
        /// The engine refused or lost the move. The sentence ends with the
        /// reassurance that the original was kept.
        case failed
        /// The user cancelled. An info line, not a failure.
        case cancelled
    }

    let kind: Kind
    let text: String
}

/// How a move that put the install in its destination ended, in words.
struct MoveSuccessText: Equatable, Sendable {
    /// "Moved qwen to T7."
    let headline: String
    /// One more plain line when the original could not be fully removed, saying
    /// the model works from its new place. Nil for a clean move.
    let detail: String?

    /// What VoiceOver and the toast say: the headline and the detail.
    var spoken: String {
        [headline, detail].compactMap { $0 }.joined(separator: " ")
    }
}

/// The words of a move (requirements 7.3, 7.4 and 7.5), in the language the UI is
/// showing.
///
/// **ONE SENTENCE PER ENGINE CODE, KEYED ON THE PREFIX.** The engine answers a
/// refusal or a failure as `"<code>: <sentence>"` and a host localizes by the
/// code, never by the sentence (`LibraryFault`). A move can answer `not_connected`,
/// `library_read_only`, `home_only`, `same_library`, `unknown_install`,
/// `unknown_library`, `destination_taken`, `insufficient_space`, `verify_failed`,
/// `free_space_unknown`, `io`, `busy` and `cancelled`; it never answers
/// `library_disconnected` (the catalog folds a drive lost mid-move into
/// `not_connected`).
///
/// **EVERY FAILURE ENDS IN "THE ORIGINAL WAS KEPT".** A move that did not finish
/// leaves the original where it was, and the sentence says so as a separate,
/// shared key, so a test can hold every code in every language to it. A cancel
/// is the exception: it is not a failure, and its line says the original was not
/// changed.
///
/// **THE ONLY ENGINE TEXT SHOWN IS A NAME WE ALREADY HOLD.** Library names and the
/// alias come from a drive, so each goes through `UntrustedText.display`; the
/// engine's own sentence is never shown. A code this build does not know, and an
/// error that is not the engine's, fall to one generic sentence.
enum ModelMoveText {
    /// The longest alias a sentence carries, after escaping.
    static let aliasLimit = ModelDeleteText.nameLimit
    /// The longest library name, after escaping.
    static let nameLimit = InstallDestinationModel.nameLimit

    // MARK: - Before copying

    static func refusal(_ refusal: MoveRefusal, language: AppLanguage? = nil) -> String {
        let language = language ?? AppStrings.currentLanguage
        switch refusal {
        case .moveRunning, .installRunning:
            return LibraryActionText.busy(language: language)
        case .loadedInChat:
            return AppStrings.localized("This model is loaded in Chat. Unload it, then move it.", language: language)
        case .loadedInImages:
            return AppStrings.localized(
                "This model is loaded in Image Generation. Unload it, then move it.", language: language)
        case .attachedToServer:
            return AppStrings.localized(
                "This model is attached to the server. Detach it there, then move it.", language: language)
        case .usedByHumanize:
            return AppStrings.localized(
                "Humanize is using this model. Unload it there, then move it.", language: language)
        case .residentInAudio:
            return AppStrings.localized(
                "This model is loaded by Audio. Try again after Audio releases it.", language: language)
        case .turnRunning:
            return AppStrings.localized(
                "A reply is being generated. Try again when it finishes.", language: language)
        case .modelOpening:
            return AppStrings.localized("A model is loading. Try again when it finishes.", language: language)
        case .noDestination:
            return AppStrings.localized(
                "There is no other library connected to move this model to.", language: language)
        case .sameLibrary:
            return AppStrings.localized("The model is already in that library.", language: language)
        case .destinationNotConnected(let name):
            return ModelInstallGate.notConnectedSentence(
                name: name ?? AppStrings.localized("Unknown library", language: language), language: language)
        case .sizeUnavailable:
            return AppStrings.localized(
                "TurboSpark could not work out how big this model is.", language: language)
        case .freeSpaceUnknown(let name):
            return freeSpaceUnknown(name: name, language: language)
        case .insufficientSpace(let required, let available, let name):
            return insufficientSpace(required: required, available: available, name: name, language: language)
        }
    }

    // MARK: - A load asked for while the model moves

    /// The answer to a load of a model whose move is running. Its own sentence, not
    /// the busy one: the busy sentence says a model is installing or moving, and this
    /// one is about THIS model.
    static func modelBeingMoved(language: AppLanguage? = nil) -> String {
        AppStrings.localized(
            "This model is being moved. Try again when the move finishes.",
            language: language ?? AppStrings.currentLanguage)
    }

    // MARK: - After a move that did not finish

    /// The reassurance every failure ends with.
    static func originalKept(language: AppLanguage? = nil) -> String {
        AppStrings.localized("The original was kept.", language: language ?? AppStrings.currentLanguage)
    }

    static func failure(
        for error: Error, context: MoveFailureContext, language: AppLanguage? = nil
    ) -> MoveFailureText {
        let language = language ?? AppStrings.currentLanguage
        let engineError = error as? TurboSparkError
        if error is CancellationError || engineError?.code == .cancelled || engineError?.libraryFault?.code == .cancelled {
            return MoveFailureText(
                kind: .cancelled,
                text: AppStrings.localized("Move cancelled. The original was not changed.", language: language))
        }
        let destination = display(context.destinationName, limit: nameLimit)
        let reason: String
        switch engineError?.libraryFault?.code {
        case .notConnected?:
            if let away = context.awayDriveName, !away.isEmpty {
                reason = AppStrings.localized(
                    "\(display(away, limit: nameLimit)) was disconnected during the move.", language: language)
            } else {
                reason = AppStrings.localized("A drive was disconnected during the move.", language: language)
            }
        case .libraryReadOnly?:
            // The engine answers `library_read_only` for a read-only SOURCE as well as a
            // read-only destination (it cannot delete the original from one), and the
            // app holds no read-only flag to tell which. So the sentence names both
            // libraries and claims neither.
            let source = display(context.sourceName, limit: nameLimit)
            reason = AppStrings.localized(
                "\(source) or \(destination) is read-only, so the model cannot be moved.", language: language)
        case .homeOnly?:
            reason = AppStrings.localized("This kind of model can only be stored on This Mac.", language: language)
        case .sameLibrary?:
            reason = AppStrings.localized("The model is already in that library.", language: language)
        case .unknownInstall?:
            let alias = display(context.alias, limit: aliasLimit)
            reason = AppStrings.localized("'\(alias)' is no longer in the model list.", language: language)
        case .unknownLibrary?:
            reason = AppStrings.localized("That library is no longer in the list.", language: language)
        case .destinationTaken?:
            reason = AppStrings.localized("\(destination) already has a model with this name.", language: language)
        case .insufficientSpace?:
            // **THE FIGURES MUST AGREE WITH THE SENTENCE.** The engine measured the
            // model itself and found it too big for the free space the app passed; the
            // sizes here are the app's own, from before the call. A recorded size that
            // is SMALLER than the free space (a listing that records less than is on
            // disk) would read "has only 3 GB free, but this model needs 2 GB". Figures
            // are quoted only when they say what the sentence says; otherwise the
            // sentence is the plain one, which is true whatever the numbers are.
            if let required = context.requiredBytes, let available = context.availableBytes, required > available {
                reason = insufficientSpace(
                    required: required, available: available, name: context.destinationName, language: language)
            } else {
                reason = notEnoughSpace(name: context.destinationName, language: language)
            }
        case .verifyFailed?:
            reason = AppStrings.localized(
                "The copied files did not match the original, so the move was stopped.", language: language)
        case .freeSpaceUnknown?:
            reason = freeSpaceUnknown(name: context.destinationName, language: language)
        case .io?:
            reason = AppStrings.localized("The move stopped because of a file system error.", language: language)
        case .busy?:
            reason = LibraryActionText.busy(language: language)
        default:
            // A code this build does not know, or an error that is not the
            // engine's (the audio device's busy status has no library prefix).
            reason = generic(language: language)
        }
        return MoveFailureText(kind: .failed, text: join(reason, originalKept(language: language), language: language))
    }

    // MARK: - After a move that put the install in its destination

    /// All three outcomes mean the install is in the destination, so none of them
    /// is worded as a failure. `destinationName` is raw.
    static func success(
        _ outcome: MoveOutcome, alias: String, destinationName: String, language: AppLanguage? = nil
    ) -> MoveSuccessText {
        let language = language ?? AppStrings.currentLanguage
        let model = display(alias, limit: aliasLimit)
        let place = display(destinationName, limit: nameLimit)
        let headline = AppStrings.localized("Moved \(model) to \(place).", language: language)
        let detail: String?
        switch outcome {
        case .sourceMarkerStale(_, let stale):
            if let stale, !stale.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                detail = AppStrings.localized(
                    "\(display(stale, limit: nameLimit)) may still list this model. The model works from its new place.",
                    language: language)
            } else {
                detail = AppStrings.localized(
                    "The original library may still list this model. The model works from its new place.",
                    language: language)
            }
        case .sourceCleanupFailed(_, _, let path):
            let shown = display(path, limit: ModelLibraryRowModel.pathLimit)
            detail = AppStrings.localized(
                "The original folder could not be removed. It is left at \(shown). You can delete it yourself. The model works from its new place.",
                language: language)
        case .moved, .unrecognized:
            detail = nil
        }
        return MoveSuccessText(headline: headline, detail: detail)
    }

    // MARK: - Pieces

    private static func freeSpaceUnknown(name: String, language: AppLanguage) -> String {
        AppStrings.localized(
            "TurboSpark could not read how much space \(display(name, limit: nameLimit)) has free.",
            language: language)
    }

    /// "T7 has only 3 GB free, but this model needs 4 GB." The figures are
    /// formatted to strings first: a number never goes into a localized key.
    private static func insufficientSpace(
        required: UInt64, available: UInt64, name: String, language: AppLanguage
    ) -> String {
        let place = display(name, limit: nameLimit)
        let free = MetricFormat.storage(available)
        let need = MetricFormat.storage(required)
        return AppStrings.localized("\(place) has only \(free) free, but this model needs \(need).", language: language)
    }

    /// The destination is too small, with no figures to quote.
    private static func notEnoughSpace(name: String, language: AppLanguage) -> String {
        AppStrings.localized(
            "\(display(name, limit: nameLimit)) does not have enough free space for this model.",
            language: language)
    }

    private static func generic(language: AppLanguage) -> String {
        AppStrings.localized("TurboSpark could not move the model.", language: language)
    }

    private static func display(_ raw: String, limit: Int) -> String {
        UntrustedText.display(raw, limit: limit)
    }

    /// The reason and the reassurance as one sentence pair. Japanese and Chinese
    /// close a sentence with its own full stop and write no space after it.
    static func join(_ reason: String, _ kept: String, language: AppLanguage) -> String {
        switch language {
        case .japanese, .simplifiedChinese, .traditionalChinese:
            return reason + kept
        default:
            return reason + " " + kept
        }
    }
}
