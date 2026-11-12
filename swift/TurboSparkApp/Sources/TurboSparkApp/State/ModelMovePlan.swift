import Foundation
import TurboSpark

// The pure decisions behind "Move to..." (task 6.11, requirements 7.1 to 7.5).
//
// Nothing here reads the app, the engine or the disk. The sheet and the driver
// (`AppModel+Moves.swift`) gather the inputs and draw or act on the answer, so
// every rule is a table test.

/// What a move is about, built from a text, an image or an audio row.
///
/// **A SUBJECT EXISTS ONLY FOR A REAL INSTALL THAT CAN BE MOVED NOW.** A row the
/// app scanned from another app's folder, a custom folder, a library that is no
/// longer listed, and a model on a drive that is away or whose folder is gone all
/// produce no subject, so the action bar offers no "Move to..." for them. What a
/// subject cannot say is whether the app is busy: that is `MoveAppState`.
struct MoveSubject: Equatable, Sendable {
    enum Kind: Equatable, Sendable {
        case text
        case image
        case audio
    }

    let kind: Kind
    /// The engine's id, `<library>:<modality>/<alias>`, as it is BEFORE the move.
    let installId: String
    let alias: String
    /// Where the model is before the move: the legacy metadata key and the
    /// selection are spelled with it.
    let path: String
    /// The library it is in: `internal` for This Mac, otherwise the library's id.
    let libraryId: String
    /// The library's name, raw (it comes from a drive). Nil for This Mac.
    let libraryName: String?
    let isThisMac: Bool
    /// The bytes the engine will copy, when the listing records them. Nil is "not
    /// recorded": an image listing carries no size, and a zero would pass any
    /// free-space check, so the sheet measures the folder instead.
    let knownBytes: UInt64?
    /// The text row as it was listed before the move, for the per-model state that
    /// is keyed by it. Nil for an image or an audio model.
    let row: InstalledModel?

    // MARK: Builders

    static func make(text row: InstalledModel) -> MoveSubject? {
        guard !ModelIdentityKey.isPlaceholder(row), row.availability == .available else { return nil }
        switch row.locationKind {
        case .thisMac, .library: break
        case .custom, .unknown: return nil
        }
        guard let library = row.library, !library.isEmpty else { return nil }
        let isThisMac = row.locationKind == .thisMac
        return MoveSubject(
            kind: .text, installId: row.installId, alias: row.alias, path: row.path,
            libraryId: isThisMac ? InstallDestinationModel.thisMacId : library,
            libraryName: isThisMac ? nil : row.libraryName, isThisMac: isThisMac,
            knownBytes: row.installBytes > 0 ? row.installBytes : nil, row: row)
    }

    /// `isManaged`: the image is in the store the engine lists. A folder a scan
    /// found has neither an install id nor anything for the engine to move.
    static func make(image: ImageInstalledModel, isManaged: Bool) -> MoveSubject? {
        guard isManaged, let installId = image.installId, !installId.isEmpty,
            let library = image.library, !library.isEmpty
        else { return nil }
        let isThisMac = library == InstallDestinationModel.thisMacId
        return MoveSubject(
            kind: .image, installId: installId, alias: image.alias, path: image.path, libraryId: library,
            libraryName: isThisMac ? nil : image.libraryName, isThisMac: isThisMac, knownBytes: nil, row: nil)
    }

    /// An installed audio model with a receipt (one the report lists as installed).
    /// An install without a receipt is adopted first; the engine refuses to move it.
    static func make(audio record: AudioInstalledRecord) -> MoveSubject? {
        let library = (record.library?.isEmpty == false ? record.library : nil) ?? InstallDestinationModel.thisMacId
        let isThisMac = library == InstallDestinationModel.thisMacId
        let installId = (record.installId?.isEmpty == false ? record.installId : nil)
            ?? "\(library):audio/\(record.alias)"
        return MoveSubject(
            kind: .audio, installId: installId, alias: record.alias, path: record.path, libraryId: library,
            libraryName: isThisMac ? nil : record.libraryName, isThisMac: isThisMac,
            knownBytes: record.installBytes > 0 ? record.installBytes : nil, row: nil)
    }

    /// This subject with its library's name read from `libraries`, for a row whose
    /// listing names the library by id only (an audio report row). A subject that
    /// already has a name, or is on This Mac, or whose library is not listed, is
    /// returned as it is.
    func resolvingLibraryName(in libraries: [ModelLibrary]) -> MoveSubject {
        guard !isThisMac, libraryName?.isEmpty != false,
            let library = libraries.first(where: { $0.id == libraryId })
        else { return self }
        return MoveSubject(
            kind: kind, installId: installId, alias: alias, path: path, libraryId: libraryId,
            libraryName: library.name, isThisMac: isThisMac, knownBytes: knownBytes, row: row)
    }

    /// The library the model is in, as a name to show.
    func sourceName(language: AppLanguage = AppStrings.currentLanguage) -> String {
        if isThisMac { return AppStrings.localized("This Mac", language: language) }
        guard let libraryName, !libraryName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return AppStrings.localized("Unknown library", language: language)
        }
        return UntrustedText.display(libraryName, limit: InstallDestinationModel.nameLimit)
    }
}

/// How big the model is, as far as the move knows.
enum MoveSize: Equatable, Sendable {
    case known(UInt64)
    /// The folder is being measured off the main actor.
    case measuring
    /// It could not be measured. The free-space check cannot run, so the move is
    /// refused rather than started on a guess.
    case unavailable
}

/// What the rest of the app is doing that bears on a move (requirement 7.5). The
/// driver reads each flag from the model; the planner only orders them.
struct MoveAppState: Equatable, Sendable {
    /// Another move is running. The engine allows one at a time.
    var moveRunning = false
    /// A text, image or audio install is running. The engine refuses a move then.
    var installRunning = false
    /// The model is loaded in Chat, or is the selected model with a live session.
    var loadedInChat = false
    /// The image model is loaded in the Images workspace.
    var loadedInImages = false
    /// The model is attached to the running server.
    var attachedToServer = false
    var usedByHumanize = false
    /// The audio model is resident in the Audio workspace.
    var residentInAudio = false
    /// A reply is being generated or submitted.
    var turnRunning = false
    /// A model is being opened.
    var modelOpening = false

    init(
        moveRunning: Bool = false, installRunning: Bool = false, loadedInChat: Bool = false,
        loadedInImages: Bool = false, attachedToServer: Bool = false, usedByHumanize: Bool = false,
        residentInAudio: Bool = false, turnRunning: Bool = false, modelOpening: Bool = false
    ) {
        self.moveRunning = moveRunning
        self.installRunning = installRunning
        self.loadedInChat = loadedInChat
        self.loadedInImages = loadedInImages
        self.attachedToServer = attachedToServer
        self.usedByHumanize = usedByHumanize
        self.residentInAudio = residentInAudio
        self.turnRunning = turnRunning
        self.modelOpening = modelOpening
    }
}

/// Why a move is refused BEFORE a byte is copied (requirements 7.4 and 7.5). Each
/// case has one localized sentence (`ModelMoveText.refusal`). The names are already
/// safe to show.
enum MoveRefusal: Equatable, Sendable {
    case moveRunning
    case installRunning
    case loadedInChat
    case loadedInImages
    case attachedToServer
    case usedByHumanize
    case residentInAudio
    case turnRunning
    case modelOpening
    /// No other library is connected, so there is nowhere to move it.
    case noDestination
    case sameLibrary
    /// A name of nil is a library the list no longer holds.
    case destinationNotConnected(name: String?)
    case sizeUnavailable
    /// The destination is connected but its capacity cannot be read. The engine
    /// would answer `free_space_unknown`; the app says so without asking it.
    case freeSpaceUnknown(name: String)
    case insufficientSpace(required: UInt64, available: UInt64, name: String)
}

/// Everything the move sheet shows and decides, for one subject, one choice of
/// destination and one state of the app.
struct ModelMovePlan: Equatable {
    /// The other places the model can go: every connected destination but its own
    /// library, plus a chosen library that has since left (so the menu still shows
    /// what is selected).
    let choices: [InstallDestinationChoice]
    let selectedId: String?
    let destinationId: String?
    let destinationName: String?
    let sourceName: String
    /// The model's size as text; nil while it is measured or if it cannot be.
    let requiredText: String?
    /// The destination's free space as text, or "Unknown".
    let availableText: String
    let isMeasuring: Bool
    let refusal: MoveRefusal?
    let refusalText: String?
    let canMove: Bool
    /// The free-space figure the engine is handed. Set ONLY for a plan that can
    /// move, and only from a connected destination's measured capacity: an
    /// unplugged or unmeasured destination is never passed as a number.
    let freeBytesForEngine: Int64?

    // MARK: The decision

    /// The first reason the move cannot start, or nil. **The order is the
    /// contract:** a move or an install anywhere stops every move; then what holds
    /// THIS model (Chat, Images, the server, Humanize, Audio); then what is busy
    /// around it (a turn, a load); then the destination (same library, away, size,
    /// unreadable free space, too little free space).
    static func refusal(
        subject: MoveSubject, size: MoveSize, destination: InstallDestinationResolution?, app: MoveAppState
    ) -> MoveRefusal? {
        if app.moveRunning { return .moveRunning }
        if app.installRunning { return .installRunning }
        if app.loadedInChat { return .loadedInChat }
        if app.loadedInImages { return .loadedInImages }
        if app.attachedToServer { return .attachedToServer }
        if app.usedByHumanize { return .usedByHumanize }
        if app.residentInAudio { return .residentInAudio }
        if app.turnRunning { return .turnRunning }
        if app.modelOpening { return .modelOpening }
        guard let destination else { return .noDestination }
        switch destination {
        case .notConnected(let name):
            return .destinationNotConnected(name: name)
        case .unknown:
            return .destinationNotConnected(name: nil)
        case .connected(let choice):
            if choice.id == subject.libraryId { return .sameLibrary }
            if size == .unavailable { return .sizeUnavailable }
            guard let free = choice.freeBytes else { return .freeSpaceUnknown(name: choice.name) }
            if case .known(let required) = size, free < required {
                return .insufficientSpace(required: required, available: free, name: choice.name)
            }
            return nil
        }
    }

    /// The plan for `subject` going to `selection` (nil follows the default: the
    /// default library unless it is the model's own, otherwise the first other
    /// destination).
    static func make(
        subject: MoveSubject, size: MoveSize, destinations: InstallDestinationModel, selection: String?,
        app: MoveAppState, language: AppLanguage = AppStrings.currentLanguage
    ) -> ModelMovePlan {
        let choices = destinations.menuChoices(for: selection).filter { $0.id != subject.libraryId }
        let explicit = InstallDestinationModel.normalized(selection)
        let preselected = choices.first { $0.id == destinations.defaultId }?.id
            ?? choices.first { $0.isConnected }?.id
        let targetId = explicit ?? preselected
        let resolution = targetId.map { destinations.resolve(selection: $0) }

        let refusal = Self.refusal(subject: subject, size: size, destination: resolution, app: app)
        let connected = resolution?.choice
        let isMeasuring: Bool
        let requiredText: String?
        switch size {
        case .known(let bytes):
            (isMeasuring, requiredText) = (false, MetricFormat.storage(bytes))
        case .measuring:
            (isMeasuring, requiredText) = (true, nil)
        case .unavailable:
            (isMeasuring, requiredText) = (false, nil)
        }
        let canMove = refusal == nil && !isMeasuring
        let availableText = connected?.freeBytes.map { MetricFormat.storage($0) }
            ?? AppStrings.localized("Unknown", language: language)
        var destinationName = connected?.name
        if destinationName == nil, case .notConnected(let name)? = resolution { destinationName = name }

        return ModelMovePlan(
            choices: choices, selectedId: targetId, destinationId: targetId, destinationName: destinationName,
            sourceName: subject.sourceName(language: language), requiredText: requiredText,
            availableText: availableText, isMeasuring: isMeasuring, refusal: refusal,
            refusalText: refusal.map { ModelMoveText.refusal($0, language: language) }, canMove: canMove,
            freeBytesForEngine: canMove ? connected?.freeBytes.map { Int64(clamping: $0) } : nil)
    }
}
