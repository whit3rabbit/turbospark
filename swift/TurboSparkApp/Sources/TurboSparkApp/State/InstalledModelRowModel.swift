import Foundation
import TurboSpark

/// Whether an installed model can be opened now, when it cannot. Rendered as a
/// glyph AND a word, never as a color alone (requirements 3.1 and 3.7,
/// DESIGN.md section 4).
///
/// An available row has no status. The titles are catalog keys;
/// `LocalizationParityTests` holds each one to a translation in every language.
enum InstalledRowStatus: String, CaseIterable, Sendable {
    /// The model's library is not connected (requirement 3.1).
    case notConnected = "Not connected"
    /// The library is connected and the model's folder is gone (requirement 3.7).
    case missing = "Missing"

    var title: String { rawValue }

    /// A minus and a triangle are different shapes, so the two states survive
    /// color blindness and Differentiate Without Color, and tell the user apart
    /// "plug the drive in" from "the files are gone".
    var glyph: String {
        switch self {
        case .notConnected: return "minus.circle.fill"
        case .missing: return "exclamationmark.triangle.fill"
        }
    }

    init?(_ availability: LibraryAvailability) {
        switch availability {
        case .available: return nil
        case .notConnected: self = .notConnected
        case .missing: self = .missing
        }
    }
}

/// Everything one Installed Models row says about where a model lives and
/// whether it can be opened (requirements 3.1, 3.7, 5.3 and 5.7, design.md
/// "Library UI set").
///
/// **IT IS PURE.** Every input is passed in, so a table test pins each wording,
/// glyph, rounding and order without a view. The row, the hero header and the
/// action bar draw these strings and nothing else, and VoiceOver reads
/// `accessibilityParts`, built from the same strings in the order the row reads.
///
/// **THE LIBRARY NAME COMES FROM A DRIVE.** A marker can name its library
/// anything, so it goes through `UntrustedText.display` (control and direction
/// characters become visible escapes, the length is cut). This Mac is told by
/// `locationKind`, never by its name: a marker can name a library "This Mac".
///
/// **A ROW THE APP SCANNED HAS NO BADGE.** Rows from LM Studio or a custom scan
/// folder have no library and a placeholder id; the source badge already says
/// where they came from, and calling them This Mac would claim a registry
/// install that does not exist.
///
/// **NO SPEED ADVICE FOR THIS MAC.** The estimate exists only for an available
/// model in an external library whose link the system described with a rate
/// (requirement 5.7). An unknown link is never guessed.
struct InstalledModelRowModel: Equatable, Sendable {
    enum LocationKind: Equatable, Sendable {
        case thisMac
        case library
        case custom
        case unknown
        /// Another app's folder, found by a scan.
        case scanned
    }

    let installId: String
    let locationKind: LocationKind
    /// The badge text, nil for a scanned row. Already escaped and localized.
    let locationText: String?
    let locationGlyph: String?
    let status: InstalledRowStatus?
    let statusText: String?
    /// Offline and missing rows are drawn with secondary text and a faded logo.
    let isDimmed: Bool
    /// Seconds to read the model at the link's rate, rounded up. Nil for This
    /// Mac, an unavailable row and a link with no known rate.
    let estimatedLoadSeconds: Int?
    let estimateText: String?
    /// Why Load is not offered, nil when it is. Under it, the action bar shows
    /// this as a statement in place of the Load button (task 6.7).
    let loadBlockedReason: String?
    /// Whether, and on what terms, the model may be loaded now (task 6.7).
    let gate: LoadGate
    /// The speed warning that stands beside "Load anyway", nil when the gate does
    /// not warn. Never set for a model that cannot be loaded: the statement wins.
    let warningText: String?
    /// Delete removes the model's files, so it is offered only for a model whose
    /// files can be reached. A model on a drive that is away is never offered it
    /// (the engine would refuse, and the recorded path may be another drive now),
    /// and a model whose folder is gone has nothing left to delete.
    var offersDelete: Bool { status == nil }
    /// "Remove from list" stops listing the model and touches no file: the one
    /// removal a model whose folder is gone offers (requirement 3.7). A model on
    /// a drive that is away offers neither removal: the engine forgets a row
    /// only with its library connected, because the library's marker would keep
    /// listing it, so the button would always fail. The row says to connect the
    /// drive instead (`loadBlockedReason`).
    var offersRemoveFromList: Bool { status == .missing }
    let sizeText: String
    /// The parts VoiceOver reads, in order: name, library, status, size, estimate.
    let accessibilityParts: [String]

    var statusGlyph: String? { status?.glyph }

    var accessibilityLabel: String {
        accessibilityParts.map { $0.hasSuffix(".") ? String($0.dropLast()) : $0 }.joined(separator: ", ")
    }

    /// - Parameters:
    ///   - library: the library the row belongs to as `LibraryState` lists it,
    ///     nil when it is not listed or the row has none.
    ///   - connection: what the system said about the library's drive.
    ///   - measuredMBps: the speed test's result for the library, if any.
    ///   - isMoE: only feeds the advisor; the estimate does not depend on it.
    init(
        model: InstalledModel,
        library: ModelLibrary?,
        connection: DriveConnection?,
        measuredMBps: Double?,
        isMoE: Bool = false,
        physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory,
        language: AppLanguage = AppStrings.currentLanguage
    ) {
        let placeholder = ModelIdentityKey.isPlaceholder(model)
        let kind: LocationKind
        let text: String?
        let glyph: String?
        if placeholder {
            (kind, text, glyph) = (.scanned, nil, nil)
        } else {
            switch model.locationKind {
            case .thisMac:
                kind = .thisMac
                text = AppStrings.localized("This Mac", language: language)
                glyph = LibraryRowGlyph.thisMac
            case .library:
                kind = .library
                let name = (library?.name).flatMap { $0.isEmpty ? nil : $0 }
                    ?? model.libraryName.flatMap { $0.isEmpty ? nil : $0 }
                text = name.map { UntrustedText.display($0) }
                    ?? AppStrings.localized("Unknown library", language: language)
                glyph = LibraryRowGlyph.drive
            case .custom:
                kind = .custom
                text = AppStrings.localized("Custom Folder", language: language)
                glyph = "folder"
            case .unknown:
                kind = .unknown
                text = AppStrings.localized("Unknown library", language: language)
                glyph = "questionmark.circle"
            }
        }

        let status = InstalledRowStatus(model.availability)
        let statusText = status.map {
            AppStrings.localized(String.LocalizationValue(stringLiteral: $0.title), language: language)
        }

        // The estimate: an available model in an external library whose link
        // the system described with a rate.
        var seconds: Int?
        if kind == .library, status == nil, let library, library.available, !library.isInternal,
            let connection
        {
            let advice = DriveSpeedAdvisor.library(connection, measuredMBps: measuredMBps)
            seconds = DriveSpeedAdvisor.model(
                installBytes: Int64(clamping: model.installBytes), isMoE: isMoE,
                physicalMemoryBytes: physicalMemoryBytes, library: advice, connection: connection
            ).estimatedLoadSeconds
        }
        let estimateText = seconds.map {
            AppStrings.localized(
                "About \(Self.durationText(seconds: $0, language: language)) to load", language: language)
        }

        // The gate decides what replaces Load and what warns beside it. Its
        // sentences are `LoadGateText`'s, the same ones the confirmation uses.
        let gate = LoadGate.evaluate(
            model: model, library: library, connection: connection, measuredMBps: measuredMBps,
            isMoE: isMoE, physicalMemoryBytes: physicalMemoryBytes)
        let blocked = LoadGateText.statement(for: gate, language: language)

        let size = MetricFormat.storage(model.installBytes)

        self.installId = model.installId
        self.locationKind = kind
        self.locationText = text
        self.locationGlyph = glyph
        self.status = status
        self.statusText = statusText
        self.isDimmed = status != nil
        self.estimatedLoadSeconds = seconds
        self.estimateText = estimateText
        self.loadBlockedReason = blocked
        self.gate = gate
        self.warningText = LoadGateText.warning(for: gate, language: language)
        self.sizeText = size
        self.accessibilityParts = [model.alias, text, statusText, size, estimateText].compactMap { $0 }
    }

    // MARK: - Durations

    /// What the estimate shows: rounded UP, so a wait is never understated, and
    /// coarser as it grows, so "About 47 sec" is not claimed to the second.
    /// Under 10 seconds as is, under a minute to 5 seconds, under an hour to a
    /// whole minute, then to 5 minutes.
    static func displaySeconds(_ seconds: Int) -> Int {
        let seconds = max(seconds, 0)
        func up(_ value: Int, to step: Int) -> Int { ((value + step - 1) / step) * step }
        switch seconds {
        case ..<11: return seconds
        case ..<60: return up(seconds, to: 5)
        case ..<3_600: return up(seconds, to: 60)
        default: return up(seconds, to: 300)
        }
    }

    /// "20 sec", "2 min", "1 hr, 5 min", in the language's own units. A
    /// duration is formatted to a String first: a number never goes into a
    /// localized key.
    static func durationText(seconds: Int, language: AppLanguage) -> String {
        let rounded = displaySeconds(seconds)
        let style = Duration.UnitsFormatStyle(
            allowedUnits: [.hours, .minutes, .seconds], width: .abbreviated, maximumUnitCount: 2
        ).locale(language.locale)
        return Duration.seconds(rounded).formatted(style)
    }

    // MARK: - From the library state

    /// The row for `model` with everything read from `state`: its library, that
    /// library's link and the measured speed.
    @MainActor
    static func make(
        for model: InstalledModel, in state: LibraryState, isMoE: Bool = false,
        language: AppLanguage = AppStrings.currentLanguage
    ) -> InstalledModelRowModel {
        let library = state.library(for: model)
        return InstalledModelRowModel(
            model: model, library: library,
            connection: library.flatMap { state.connections[$0.id] },
            measuredMBps: library.flatMap { state.measuredSpeeds[$0.id] },
            isMoE: isMoE, language: language)
    }
}

/// The words of the confirmation before a model leaves the list or the disk
/// (requirements 3.7 and 5.3).
///
/// Three different promises, and each says which one it makes:
/// - `.thisMacFiles`: the files are removed from disk, at this path.
/// - `.libraryFiles`: the files are removed from the drive of this library.
/// - `.listOnly`: TurboSpark stops listing the model and no file is deleted
///   (files that are already gone).
///
/// The library name comes from a drive and is escaped. The strings are built
/// here, not in the view, so a test renders each in every language.
struct ModelDeleteDialog: Equatable, Sendable {
    enum Kind: Equatable, Sendable {
        case thisMacFiles
        case libraryFiles
        case listOnly
    }

    let kind: Kind
    let title: String
    let message: String
    let confirmTitle: String

    private init(kind: Kind, title: String, message: String, confirmTitle: String) {
        self.kind = kind
        self.title = title
        self.message = message
        self.confirmTitle = confirmTitle
    }

    /// The confirmation of "Remove from list": no file is deleted, whatever the
    /// row's state.
    init(removingFromList model: InstalledModel, language: AppLanguage = AppStrings.currentLanguage) {
        let alias = UntrustedText.display(model.alias, limit: ModelDeleteText.nameLimit)
        self.init(
            kind: .listOnly,
            title: AppStrings.localized("Remove \(alias) from the list?", language: language),
            message: AppStrings.localized(
                "TurboSpark stops listing this model. Its files are not deleted.", language: language),
            confirmTitle: AppStrings.localized("Remove from list", language: language))
    }

    /// The confirmation of Delete. A row that is not available has no file to
    /// delete, so it gets the list-only wording.
    init(model: InstalledModel, libraryName: String?, language: AppLanguage = AppStrings.currentLanguage) {
        let alias = UntrustedText.display(model.alias, limit: ModelDeleteText.nameLimit)
        let size = MetricFormat.storage(model.installBytes)
        let title = AppStrings.localized("Delete \(alias)?", language: language)
        let confirm = AppStrings.localized("Delete Model", language: language)
        if model.availability != .available {
            self.init(removingFromList: model, language: language)
        } else if model.locationKind == .library {
            let library = (libraryName?.isEmpty == false ? libraryName : model.libraryName)
                .map { UntrustedText.display($0) }
                ?? AppStrings.localized("Unknown library", language: language)
            self.init(
                kind: .libraryFiles, title: title,
                message: AppStrings.localized(
                    "This permanently deletes the model files (\(size)) from the drive of the library \(library).",
                    language: language),
                confirmTitle: confirm)
        } else {
            let path = UntrustedText.display(model.path, limit: ModelLibraryRowModel.pathLimit)
            self.init(
                kind: .thisMacFiles, title: title,
                message: AppStrings.localized(
                    "This will permanently remove the model files (\(size)) from disk at:\n\(path)",
                    language: language),
                confirmTitle: confirm)
        }
    }
}
