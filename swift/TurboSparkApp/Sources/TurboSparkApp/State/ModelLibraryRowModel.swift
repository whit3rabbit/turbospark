import Foundation
import TurboSpark

/// Whether a library can be read now. Rendered as a glyph AND a sentence, never
/// as a color alone (requirement 3.1 and DESIGN.md section 4).
///
/// The titles are catalog keys; `LocalizationParityTests` holds each to a
/// translation in every language.
enum LibraryRowStatus: String, CaseIterable, Sendable {
    case connected = "Connected"
    case notConnected = "Not connected"

    var title: String { rawValue }

    /// A check mark and a minus are different shapes, so the two states survive
    /// color blindness and Differentiate Without Color.
    var glyph: String {
        switch self {
        case .connected: return "checkmark.circle.fill"
        case .notConnected: return "minus.circle.fill"
        }
    }
}

/// What the user can do to one library row. The titles are catalog keys.
enum LibraryRowAction: String, CaseIterable, Sendable {
    case setDefault = "Set as default"
    case testSpeed = "Test speed"
    case reveal = "Reveal in Finder"
    case remove = "Remove"
    /// The moved-store placeholder's only action: forget the conversion.
    case abandonConversion = "Stop waiting for this drive"

    var title: String { rawValue }

    var systemImage: String {
        switch self {
        case .setDefault: return "star"
        case .testSpeed: return "speedometer"
        case .reveal: return "arrow.up.right.square"
        case .remove: return "trash"
        case .abandonConversion: return "xmark.circle"
        }
    }

    /// Goes through a destructive confirmation and `Button(role: .destructive)`.
    var isDestructive: Bool {
        self == .remove || self == .abandonConversion
    }
}

/// The symbols a row draws besides its status and action glyphs.
enum LibraryRowGlyph {
    static let warning = "exclamationmark.triangle.fill"
    static let defaultBadge = "star.fill"
    static let info = "checkmark.circle.fill"
    static let thisMac = "laptopcomputer"
    static let drive = "externaldrive"
}

/// Whether one action can run on one row now, and if not, why.
struct LibraryActionAvailability: Equatable, Sendable {
    let isEnabled: Bool
    /// The sentence for the tooltip of a disabled control (DESIGN.md: a
    /// disabled control says why). Nil when enabled.
    let disabledReason: String?
}

/// Everything one library row says, as a value (requirements 1.3, 3.8, 5.1,
/// 5.2 and 5.7, design.md "Library UI set").
///
/// **IT IS PURE.** Every input is passed in, so a table test pins each field,
/// each wording and each language without a view. The view draws these strings
/// and nothing else, and VoiceOver reads `accessibilityLabel`, which is built
/// from the same strings in the order the row reads.
///
/// **NAMES, PATHS AND DATES COME FROM A DRIVE.** A library name is whatever its
/// marker says, a path is wherever the drive is mounted and a last-seen date is
/// a string in a file. Each goes through `UntrustedText.display`, which turns
/// control and direction characters into visible escapes and cuts the length.
///
/// **THIS MAC IS TOLD BY ITS FLAG, NOT ITS NAME.** The engine names it with the
/// English literal "This Mac", and a marker on a drive can name its library the
/// same. The title is localized when `isInternal` and shown as given otherwise.
struct ModelLibraryRowModel: Equatable, Identifiable, Sendable {
    enum Kind: Equatable, Sendable {
        case thisMac
        case library
        /// The not-connected placeholder for a store Move Store moved to a drive
        /// that is away (requirement 9.3). It is a row, not a library.
        case pendingConversion
    }

    let id: String
    let kind: Kind
    let title: String
    /// The root, escaped and cut. Nil while the drive is away.
    let location: String?
    let countText: String?
    let freeSpaceText: String?
    /// "Connection: USB 10 Gb/s", or "Connection: Unknown". Nil for This Mac, a
    /// library that is away and the placeholder.
    let connectionText: String?
    let status: LibraryRowStatus
    let statusText: String
    let slowLinkWarning: String?
    let measuredSpeedText: String?
    /// The placeholder's sentence: it converts when its drive is connected.
    let explanation: String?
    let isDefault: Bool
    let iconName: String
    /// The root to show in Finder, nil unless the drive is connected.
    let revealPath: String?
    /// The actions this row has, in the order it lists them.
    let offeredActions: [LibraryRowAction]
    let accessibilityLabel: String
    /// The language the strings above are in, which the disabled reasons share.
    let language: AppLanguage

    var statusGlyph: String { status.glyph }
    var isConnected: Bool { status == .connected }

    /// - Parameters:
    ///   - isPendingConversion: `library` is the placeholder
    ///     (`LibraryState.isPendingMovedStore`).
    ///   - pendingRoot: the placeholder's recorded root, its location.
    ///   - connection: what the system said about the library's drive, nil when
    ///     it has not been probed.
    ///   - freeBytes: nil when the capacity could not be read.
    ///   - measuredMBps: the speed test's result for this library, if any.
    init(
        library: ModelLibrary,
        isPendingConversion: Bool = false,
        pendingRoot: String? = nil,
        connection: DriveConnection? = nil,
        freeBytes: Int64? = nil,
        measuredMBps: Double? = nil,
        language: AppLanguage = AppStrings.currentLanguage
    ) {
        let kind: Kind =
            isPendingConversion ? .pendingConversion : (library.isInternal ? .thisMac : .library)
        let connected = library.available && !isPendingConversion
        let status: LibraryRowStatus = connected ? .connected : .notConnected

        let title = kind == .thisMac
            ? AppStrings.localized("This Mac", language: language)
            : UntrustedText.display(library.name)

        let location: String?
        if isPendingConversion {
            location = pendingRoot.map { UntrustedText.display($0, limit: Self.pathLimit) }
        } else if connected {
            location = library.root.map { UntrustedText.display($0, limit: Self.pathLimit) }
        } else {
            location = nil
        }

        let countText = isPendingConversion
            ? nil : AppStrings.localized("\(library.modelCount) model(s)", language: language)

        let freeSpaceText: String? = {
            guard connected, let freeBytes else { return nil }
            return Self.freeSpaceText(freeBytes, language: language)
        }()

        // A link means nothing for This Mac (requirement 5.7), for a library
        // that is away and for the placeholder.
        let showsLink = kind == .library && connected
        let connectionText = showsLink
            ? AppStrings.localized(
                "Connection: \(Self.connectionDescription(connection, language: language))", language: language)
            : nil

        let slowLinkWarning: String? = {
            guard showsLink, let connection,
                DriveSpeedAdvisor.library(connection, measuredMBps: measuredMBps).slowLink
            else { return nil }
            return AppStrings.localized(
                "Slow connection. Use a USB-C cable rated for 10 Gb/s or faster.", language: language)
        }()

        let measuredSpeedText = showsLink
            ? measuredMBps.flatMap { Self.measuredSpeedText($0, language: language) } : nil

        let statusText: String = {
            guard !connected else { return AppStrings.localized(String.LocalizationValue(stringLiteral: status.title), language: language) }
            guard !isPendingConversion, let seen = library.lastSeen else {
                return AppStrings.localized(String.LocalizationValue(stringLiteral: status.title), language: language)
            }
            let date = Self.lastSeenDescription(seen, language: language)
            return AppStrings.localized("Not connected, last seen \(date)", language: language)
        }()

        let explanation = isPendingConversion
            ? AppStrings.localized(
                "Your earlier model store converts to a library when its drive is connected.", language: language)
            : nil

        let isDefault = library.isDefault && !isPendingConversion
        let defaultText = isDefault
            ? AppStrings.localized("Default library for new downloads", language: language) : nil

        var actions: [LibraryRowAction] = []
        switch kind {
        case .pendingConversion:
            actions = [.abandonConversion]
        case .thisMac:
            if !isDefault { actions.append(.setDefault) }
            actions.append(.reveal)
        case .library:
            if !isDefault { actions.append(.setDefault) }
            actions += [.testSpeed, .reveal, .remove]
        }

        self.id = library.id
        self.kind = kind
        self.title = title
        self.location = location
        self.countText = countText
        self.freeSpaceText = freeSpaceText
        self.connectionText = connectionText
        self.status = status
        self.statusText = statusText
        self.slowLinkWarning = slowLinkWarning
        self.measuredSpeedText = measuredSpeedText
        self.explanation = explanation
        self.isDefault = isDefault
        self.iconName = kind == .thisMac ? LibraryRowGlyph.thisMac : LibraryRowGlyph.drive
        self.revealPath = connected ? library.root : nil
        self.offeredActions = actions
        self.language = language

        // Name, status, location, count, free space, then the warnings, then
        // whether it is the default: the order VoiceOver reads a row in.
        let parts = [
            title, statusText, location, countText, freeSpaceText, connectionText, slowLinkWarning,
            measuredSpeedText, explanation, defaultText,
        ].compactMap { $0 }
        self.accessibilityLabel = parts
            .map { $0.hasSuffix(".") ? String($0.dropLast()) : $0 }
            .joined(separator: ". ")
    }

    /// The longest path a row carries after escaping. The view truncates it in
    /// the middle and keeps all of this as the tooltip, so it is generous; it
    /// exists because a path from a drive has no other bound.
    static let pathLimit = 1024

    // MARK: - Actions

    /// Whether `action` can run now. Nil when the row does not offer it at all
    /// (This Mac has no Remove, the placeholder has only its own action).
    ///
    /// A change to the library list is disabled, not hidden, while an install or
    /// a move runs (the engine would answer `busy`). Stopping the wait for a
    /// moved store changes no library and is never blocked.
    func availability(of action: LibraryRowAction, changesBlocked: Bool) -> LibraryActionAvailability? {
        guard offeredActions.contains(action) else { return nil }
        switch action {
        case .setDefault, .remove:
            if changesBlocked {
                return LibraryActionAvailability(
                    isEnabled: false,
                    disabledReason: AppStrings.localized(
                        "Not available while a model is installing or moving.", language: language))
            }
            return LibraryActionAvailability(isEnabled: true, disabledReason: nil)
        case .testSpeed, .reveal:
            if !isConnected {
                return LibraryActionAvailability(
                    isEnabled: false,
                    disabledReason: AppStrings.localized("Connect the drive to use this.", language: language))
            }
            return LibraryActionAvailability(isEnabled: true, disabledReason: nil)
        case .abandonConversion:
            return LibraryActionAvailability(isEnabled: true, disabledReason: nil)
        }
    }

    // MARK: - Pieces

    /// "USB 10 Gb/s", "Thunderbolt", "Unknown": what the system said about the
    /// link, from `DriveConnection`. A missing probe and a probe that learned
    /// nothing both read "Unknown" (requirement 5.1).
    static func connectionDescription(_ connection: DriveConnection?, language: AppLanguage) -> String {
        guard let connection else { return AppStrings.localized("Unknown", language: language) }
        var parts: [String] = []
        switch connection.kind {
        case .usb: parts.append("USB")
        case .thunderbolt: parts.append("Thunderbolt")
        case .internalDrive: parts.append(AppStrings.localized("Internal drive", language: language))
        case .other(let name): parts.append(UntrustedText.display(name, limit: 40))
        case .unknown: break
        }
        if let bits = connection.bitsPerSecond, bits > 0 {
            parts.append(linkRate(bits, language: language))
        }
        return parts.isEmpty ? AppStrings.localized("Unknown", language: language) : parts.joined(separator: " ")
    }

    /// "900 GB free": the one reading of a drive's free space, so the library
    /// row, the install destination menu and the move sheet say it the same way.
    static func freeSpaceText(_ freeBytes: Int64, language: AppLanguage) -> String {
        let amount = freeBytes.formatted(
            ByteCountFormatStyle(
                style: .file, allowedUnits: .all, spellsOutZero: false, includesActualByteCount: false,
                locale: language.locale))
        return AppStrings.localized("\(amount) free", language: language)
    }

    /// "480 Mb/s", "10 Gb/s": the unit is a unit, not a word to translate.
    static func linkRate(_ bitsPerSecond: UInt64, language: AppLanguage) -> String {
        let gigabit = 1_000_000_000.0
        let megabit = 1_000_000.0
        let bits = Double(bitsPerSecond)
        let (value, unit) = bits >= gigabit ? (bits / gigabit, "Gb/s") : (bits / megabit, "Mb/s")
        let number = value.formatted(
            .number.precision(.fractionLength(0...1)).locale(language.locale))
        return "\(number) \(unit)"
    }

    /// "Measured read speed: 412 MB/s", or nil for a figure that is not a
    /// positive number.
    static func measuredSpeedText(_ mbps: Double, language: AppLanguage) -> String? {
        guard mbps.isFinite, mbps > 0 else { return nil }
        let digits = mbps < 10 ? 1 : 0
        let number = mbps.formatted(
            .number.precision(.fractionLength(digits)).locale(language.locale))
        let speed = "\(number) MB/s"
        return AppStrings.localized("Measured read speed: \(speed)", language: language)
    }

    /// The engine's `YYYY-MM-DD` as a date in the language's own format. The
    /// day is a calendar day, not an instant, so it is read and written in one
    /// fixed time zone and cannot slip across midnight. Anything else the file
    /// holds is shown escaped rather than trusted.
    static func lastSeenDescription(_ raw: String, language: AppLanguage) -> String {
        let fields = raw.split(separator: "-", omittingEmptySubsequences: false)
        if raw.utf8.count == 10, fields.count == 3,
            let year = Int(fields[0]), let month = Int(fields[1]), let day = Int(fields[2])
        {
            var calendar = Calendar(identifier: .gregorian)
            calendar.timeZone = TimeZone(secondsFromGMT: 0) ?? .gmt
            // A month 13 or a day 40 would roll over into a real date; only a
            // date that reads back as written is a date.
            if let date = calendar.date(from: DateComponents(year: year, month: month, day: day)),
                case let back = calendar.dateComponents([.year, .month, .day], from: date),
                back.year == year, back.month == month, back.day == day
            {
                return date.formatted(
                    Date.FormatStyle(
                        date: .abbreviated, time: .omitted, locale: language.locale, calendar: calendar,
                        timeZone: calendar.timeZone))
            }
        }
        return UntrustedText.display(raw, limit: 40)
    }

    // MARK: - The list

    /// The rows the section shows: This Mac first, the other libraries in the
    /// order the state lists them, then the moved-store placeholder.
    @MainActor
    static func rows(
        from state: LibraryState, language: AppLanguage = AppStrings.currentLanguage
    ) -> [ModelLibraryRowModel] {
        let listed = state.displayLibraries
        let placeholders = listed.filter { LibraryState.isPendingMovedStore($0) }
        let real = listed.filter { !LibraryState.isPendingMovedStore($0) }
        let ordered = real.filter(\.isInternal) + real.filter { !$0.isInternal } + placeholders
        return ordered.map { library in
            let pending = LibraryState.isPendingMovedStore(library)
            return ModelLibraryRowModel(
                library: library,
                isPendingConversion: pending,
                pendingRoot: pending ? state.pendingMovedStore?.root : nil,
                connection: state.connections[library.id],
                freeBytes: state.freeBytes[library.id],
                measuredMBps: state.measuredSpeeds[library.id],
                language: language)
        }
    }
}
