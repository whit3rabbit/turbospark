import Foundation
import TurboSpark

/// One library a model may be installed into (requirements 2.1 and 2.3).
struct InstallDestinationChoice: Identifiable, Equatable, Sendable {
    /// `ModelLibrary.id`; `internal` for This Mac.
    let id: String
    let isThisMac: Bool
    /// "This Mac" in the UI language, or the library's name made safe to show.
    let name: String
    /// Nil when the system could not read the capacity. Never a zero.
    let freeBytes: UInt64?
    /// "900 GB free", nil with `freeBytes`.
    let freeText: String?
    /// False for a row the menu keeps only because it is what was chosen.
    let isConnected: Bool
    /// What the menu row says: "T7, 900 GB free", "T7", or "T7, Not connected".
    let menuTitle: String
}

/// Where a choice stands now.
enum InstallDestinationResolution: Equatable {
    case connected(InstallDestinationChoice)
    /// A listed library whose drive is away. `name` is already safe to show.
    case notConnected(name: String)
    /// A library the list no longer holds.
    case unknown

    var choice: InstallDestinationChoice? {
        if case .connected(let choice) = self { return choice }
        return nil
    }
}

/// What the install gate needs to know about a destination.
struct InstallGateDestination: Equatable {
    /// The library to name in a sentence. Nil for This Mac, whose sentences are
    /// the ones the gate always had.
    let libraryName: String?
    /// Set when the destination cannot take an install now: the name to connect.
    let notConnectedName: String?
    /// The destination's own free space. Nil when it could not be read.
    let freeBytes: UInt64?
}

/// The libraries an install may go to, the default among them, and the free space
/// of each (requirements 2.1 to 2.3). Everything the destination picker and the
/// install gates say comes from here.
///
/// **IT IS PURE.** It takes the library list, the free space the state read, the
/// default destination and This Mac's free space, so each rule is a table test
/// with no view and no engine. The picker draws it, `ModelInstallGate` reads its
/// `gateDestination`, and the move sheet (task 6.11) takes the same menu.
///
/// **THE LIBRARY NAME IS FROM A DRIVE.** It is escaped and cut (`UntrustedText`),
/// and This Mac is labelled by the library's `isInternal` flag, never by a name:
/// a marker can call its library "This Mac".
///
/// **A CHOICE THAT WENT STALE IS NOT REPLACED.** A drive picked and then pulled
/// stays the choice, and the install is refused naming it. Quietly sending the
/// model to This Mac instead would put gigabytes on the disk the user chose a
/// drive to spare.
struct InstallDestinationModel: Equatable {
    static let thisMacId = "internal"
    /// The longest library name a menu row or a sentence carries, after escaping.
    static let nameLimit = 40

    /// Where an install may go now: This Mac first, then every connected library
    /// in list order. A library that is away is not a destination.
    let choices: [InstallDestinationChoice]
    /// The preselected destination: the default library when it is connected,
    /// otherwise This Mac.
    let defaultId: String
    /// Set when the default library is away, so the install falls back to This
    /// Mac and says so (requirement 2.2). Shown even when no menu is.
    let fallbackNote: String?
    /// The libraries that are listed but away, by id, kept to name a stale choice.
    private let away: [String: InstallDestinationChoice]
    private let unknownName: String
    private let notConnectedWord: String

    /// The menu is drawn only when there is something to choose between
    /// (requirement 2.1). With This Mac alone the install goes there, as before.
    var isShown: Bool { choices.count > 1 }

    init(
        libraries: [ModelLibrary], freeBytes: [String: Int64],
        destinationDefault: DefaultInstallDestination, thisMacFreeBytes: UInt64?,
        language: AppLanguage = AppStrings.currentLanguage
    ) {
        let unknown = AppStrings.localized("Unknown library", language: language)
        let notConnected = AppStrings.localized("Not connected", language: language)

        func choice(
            id: String, name: String, isThisMac: Bool, free: UInt64?, connected: Bool
        ) -> InstallDestinationChoice {
            let freeText = free.map {
                ModelLibraryRowModel.freeSpaceText(Int64(clamping: $0), language: language)
            }
            let title: String
            if !connected {
                title = "\(name), \(notConnected)"
            } else if let freeText {
                title = "\(name), \(freeText)"
            } else {
                title = name
            }
            return InstallDestinationChoice(
                id: id, isThisMac: isThisMac, name: name, freeBytes: free, freeText: freeText,
                isConnected: connected, menuTitle: title)
        }

        func safeName(_ raw: String) -> String {
            let shown = UntrustedText.display(raw, limit: Self.nameLimit)
            return shown.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? unknown : shown
        }

        var built = [
            choice(
                id: Self.thisMacId, name: AppStrings.localized("This Mac", language: language),
                isThisMac: true, free: thisMacFreeBytes, connected: true)
        ]
        var awayChoices: [String: InstallDestinationChoice] = [:]
        for library in libraries where !library.isInternal {
            if library.available {
                let free = freeBytes[library.id].flatMap { $0 >= 0 ? UInt64($0) : nil }
                built.append(
                    choice(
                        id: library.id, name: safeName(library.name), isThisMac: false, free: free,
                        connected: true))
            } else {
                awayChoices[library.id] = choice(
                    id: library.id, name: safeName(library.name), isThisMac: false, free: nil,
                    connected: false)
            }
        }

        self.choices = built
        self.away = awayChoices
        self.unknownName = unknown
        self.notConnectedWord = notConnected
        self.defaultId = built.contains { $0.id == destinationDefault.libraryId }
            ? destinationDefault.libraryId : Self.thisMacId
        if destinationDefault.fallbackBecauseNotConnected {
            let name = safeName(destinationDefault.notConnectedLibraryName ?? "")
            self.fallbackNote = AppStrings.localized(
                "\(name) is the default library but is not connected, so this goes to This Mac.",
                language: language)
        } else {
            self.fallbackNote = nil
        }
    }

    // MARK: - Resolving a choice

    /// Where `library` stands now, with nil and `internal` both meaning This Mac.
    /// This is the engine-facing reading, where nil is This Mac.
    func resolve(library: String?) -> InstallDestinationResolution {
        resolve(id: Self.normalized(library) ?? Self.thisMacId)
    }

    /// Where the user's choice stands now, with nil meaning "follow the default".
    /// This is the picker-facing reading: a picker that was never touched has no
    /// selection and installs to the preselected destination.
    func resolve(selection: String?) -> InstallDestinationResolution {
        resolve(id: Self.normalized(selection) ?? defaultId)
    }

    private func resolve(id: String) -> InstallDestinationResolution {
        if let choice = choices.first(where: { $0.id == id }) { return .connected(choice) }
        if let stale = away[id] { return .notConnected(name: stale.name) }
        return .unknown
    }

    /// The `library:` argument an install takes for the user's choice: nil for
    /// This Mac, otherwise the library id. A stale or unknown id is passed on as
    /// it is, so the install is refused naming the drive and never rerouted.
    func libraryForInstall(selection: String?) -> String? {
        let id = Self.normalized(selection) ?? defaultId
        return id == Self.thisMacId ? nil : id
    }

    /// The menu's rows for the user's choice: the connected destinations, plus the
    /// chosen library when it left after it was chosen, so the menu keeps showing
    /// what is selected.
    func menuChoices(for selection: String?) -> [InstallDestinationChoice] {
        guard let id = Self.normalized(selection), let stale = away[id] else { return choices }
        return choices + [stale]
    }

    // MARK: - For the install gate (requirement 2.3)

    /// What the gate asks of `library` (nil is This Mac): its name, whether it can
    /// take an install now, and its OWN free space. A library never borrows This
    /// Mac's figure.
    func gateDestination(library: String?) -> InstallGateDestination {
        switch resolve(library: library) {
        case .connected(let choice):
            return InstallGateDestination(
                libraryName: choice.isThisMac ? nil : choice.name, notConnectedName: nil,
                freeBytes: choice.freeBytes)
        case .notConnected(let name):
            return InstallGateDestination(libraryName: name, notConnectedName: name, freeBytes: nil)
        case .unknown:
            return InstallGateDestination(
                libraryName: unknownName, notConnectedName: unknownName, freeBytes: nil)
        }
    }

    /// A selection with nothing in it is no selection.
    static func normalized(_ id: String?) -> String? {
        guard let id, !id.isEmpty else { return nil }
        return id
    }
}

extension InstallDestinationModel {
    /// The destinations for the library state the app holds now. This Mac's free
    /// space is read from the disk at each call, as the gate always did, so a
    /// download that just finished is already out of it.
    @MainActor
    init(
        state: LibraryState, thisMacFreeBytes: UInt64?, language: AppLanguage = AppStrings.currentLanguage
    ) {
        self.init(
            libraries: state.libraries, freeBytes: state.freeBytes,
            destinationDefault: state.destinationDefault,
            thisMacFreeBytes: thisMacFreeBytes ?? state.freeBytes[Self.thisMacId].flatMap {
                $0 >= 0 ? UInt64($0) : nil
            }, language: language)
    }
}
