import Foundation
import TurboSpark

/// Whether an installed model may be loaded now, and on what terms (requirements
/// 3.2, 3.7, 5.4 and 5.5, design.md "Library UI set").
///
/// **IT IS PURE.** Every input is passed in, so a table test pins each rule
/// without a view or a drive. The action bar, the confirmation before a load and
/// the one choke point every interactive load goes through (`AppModel.requestLoad`)
/// all read this one answer, so they cannot disagree about a model.
///
/// **THE FOUR STATES**
/// - `open`: load as before.
/// - `notConnected`: the model's drive is away. Load is replaced by a statement
///   naming the drive to connect.
/// - `missing`: the drive is there and the model's folder is gone. Load is
///   replaced by a statement, and the model can be removed from the list.
/// - `warn`: the model can be loaded, and the drive's speed will hurt. Advice
///   never prevents a load (requirement 5.5): the load action becomes "Load
///   anyway" and the warning stays on screen beside it.
///
/// **A MODEL THAT CANNOT BE LOADED IS NEVER `warn`.** The connect or missing
/// statement wins, because a speed warning about a drive that is not there would
/// offer a button that cannot work.
///
/// **THE SPEED RULES ARE NOT RE-DERIVED HERE.** `DriveSpeedAdvisor.model` owns
/// every threshold. Its `warnBeforeLoad` is true for two different reasons and
/// the user needs two different sentences: `streamsFromDrive` false is a dense
/// model on a slow link (loading is slow, and the estimate is known), true is a
/// mixture-of-experts model that likely does not fit in memory (generation speed
/// is limited by the drive for as long as the model is loaded).
enum LoadGate: Equatable, Sendable {
    case open
    /// `libraryName` is the drive to connect, raw and untrusted: show it through
    /// `UntrustedText.display`. Nil when nothing names it.
    case notConnected(libraryName: String?)
    case missing
    case warn(Warning)

    enum Warning: Equatable, Sendable {
        /// A dense model on a slow link. The estimate is the advisor's, in
        /// seconds, or nil for a model of unknown size.
        case slowDrive(estimateSeconds: Int?)
        /// A mixture-of-experts model that likely streams its experts from the
        /// drive while it generates.
        case streamsFromDrive
    }

    /// Whether the model can be loaded at all (the warning does not prevent it).
    var canLoad: Bool {
        switch self {
        case .open, .warn: return true
        case .notConnected, .missing: return false
        }
    }

    /// The model, the library it belongs to as `LibraryState` lists it, what the
    /// system said about that library's link, and the measured speed.
    ///
    /// - Parameter isMoE: only asked for a model that could be warned about.
    ///   Answering it reads a manifest from the drive, so the argument is an
    ///   autoclosure: This Mac, a scanned folder and a model that is not
    ///   available never trigger the read.
    static func evaluate(
        model: InstalledModel,
        library: ModelLibrary?,
        connection: DriveConnection?,
        measuredMBps: Double?,
        isMoE: @autoclosure () -> Bool,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> LoadGate {
        switch model.availability {
        case .notConnected:
            // The drive to connect: the library, or for a custom folder the
            // volume. An empty name is no name.
            let name = [model.connectName, model.libraryName, library?.name]
                .compactMap { $0 }.first { !$0.isEmpty }
            return .notConnected(libraryName: name)
        case .missing:
            return .missing
        case .available:
            break
        }

        // Speed advice is about a library on an external drive. This Mac, a
        // folder another app owns (a scan's placeholder row) and a custom folder
        // get none (requirement 5.7), whatever link a caller passes. An unknown
        // link has nothing to warn about.
        guard !ModelIdentityKey.isPlaceholder(model),
            model.locationKind == .library,
            let library, library.available, !library.isInternal,
            let connection
        else { return .open }

        let libraryAdvice = DriveSpeedAdvisor.library(connection, measuredMBps: measuredMBps)
        let advice = DriveSpeedAdvisor.model(
            installBytes: Int64(clamping: model.installBytes),
            isMoE: isMoE(),
            physicalMemoryBytes: physicalMemoryBytes,
            library: libraryAdvice,
            connection: connection)
        guard advice.warnBeforeLoad else { return .open }
        return advice.streamsFromDrive
            ? .warn(.streamsFromDrive)
            : .warn(.slowDrive(estimateSeconds: advice.estimatedLoadSeconds))
    }
}

/// What a request to load a model comes to, given its gate (requirements 5.4 and
/// 5.5). Pure, so the table is a test and the one choke point that acts on it
/// (`AppModel.requestLoad`) cannot decide differently from it.
///
/// - A model that cannot be loaded is refused, whatever the user confirmed:
///   "Load anyway" overrides advice, never a missing drive.
/// - A warned model asks first, unless the user already said so or the model is
///   already in memory (nothing is read from the drive again).
enum LoadDecision: Equatable {
    case load
    case confirm(LoadGate.Warning)
    case refuse(LoadGate)

    static func decide(gate: LoadGate, isResident: Bool, confirmed: Bool) -> LoadDecision {
        switch gate {
        case .open:
            return .load
        case .notConnected, .missing:
            return .refuse(gate)
        case .warn(let warning):
            return (isResident || confirmed) ? .load : .confirm(warning)
        }
    }
}

/// The sentences a gate says. Pure, so a test renders each one in every
/// language; the views and the confirmation draw these strings and nothing else.
///
/// A drive name comes from a marker on the drive and goes through
/// `UntrustedText.display`. A duration is formatted to a String first: a number
/// never goes into a localized key.
enum LoadGateText {
    /// The glyph of the speed warnings. A third shape beside the minus (not
    /// connected) and the triangle (missing), so the three survive color
    /// blindness and Differentiate Without Color.
    static let warningGlyph = "speedometer"

    /// "Connect T7 to load", or "The files for this model are missing.": what
    /// replaces Load. Nil for a gate that offers Load.
    static func statement(for gate: LoadGate, language: AppLanguage = AppStrings.currentLanguage) -> String? {
        switch gate {
        case .notConnected(let libraryName):
            let name = libraryName.flatMap { $0.isEmpty ? nil : UntrustedText.display($0) }
                ?? AppStrings.localized("Unknown library", language: language)
            return AppStrings.localized("Connect \(name) to load", language: language)
        case .missing:
            return AppStrings.localized("The files for this model are missing.", language: language)
        case .open, .warn:
            return nil
        }
    }

    /// The warning before a load that will be slow or limited by the drive. Nil
    /// for every other gate.
    static func warning(for gate: LoadGate, language: AppLanguage = AppStrings.currentLanguage) -> String? {
        guard case .warn(let warning) = gate else { return nil }
        return text(for: warning, language: language)
    }

    static func text(for warning: LoadGate.Warning, language: AppLanguage = AppStrings.currentLanguage) -> String {
        switch warning {
        case .slowDrive(let seconds?):
            let duration = InstalledModelRowModel.durationText(seconds: seconds, language: language)
            let sentence = AppStrings.localized(
                "This drive's connection is slow, so loading will take about \(duration).", language: language)
            // The duration ends the sentence, and a language that abbreviates its
            // minutes with a period ("3 Min.") meets the sentence's own full stop:
            // "etwa 3 Min..". One period is enough.
            return sentence.hasSuffix("..") ? String(sentence.dropLast()) : sentence
        case .slowDrive(nil):
            return AppStrings.localized(
                "This drive's connection is slow, so loading will take a long time.", language: language)
        case .streamsFromDrive:
            return AppStrings.localized(
                "This model likely streams from the drive, so generation speed will be limited by the drive.",
                language: language)
        }
    }

    /// The title of the confirmation before a warned load. The alias is an
    /// identifier the user did not necessarily type (a scan names it), so it is
    /// escaped.
    static func confirmationTitle(alias: String, language: AppLanguage = AppStrings.currentLanguage) -> String {
        let name = UntrustedText.display(alias, limit: ModelDeleteText.nameLimit)
        return AppStrings.localized("Load \(name) anyway?", language: language)
    }

    /// The label of the action that loads in spite of the warning.
    static func loadAnywayTitle(language: AppLanguage = AppStrings.currentLanguage) -> String {
        AppStrings.localized("Load anyway", language: language)
    }
}
