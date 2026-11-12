import Combine
import Foundation

/// What the move sheet is doing and what it decides (task 6.11, requirements 7.1
/// to 7.5). The view draws this and nothing else.
///
/// **THE SHEET HAS FOUR STAGES.** It starts `choosing` (the destination menu, the
/// space, and the reason a move cannot start). Move starts the engine and the
/// stage is `moving` until the engine answers; then it is `finished` (the install
/// is in the destination) or `failed` (the original was kept; the sheet stays open
/// to retry or close). A cancel is a `failed` stage of kind `cancelled`.
///
/// **IT CANNOT BE DISMISSED WHILE A MOVE RUNS.** `canDismiss` is false from the
/// moment Move is pressed until the engine answers, and the view disables Escape,
/// the close gesture and the Close button on it. Cancel is the way out; it asks the
/// engine to stop and the sheet waits for the answer. Escape does not cancel
/// either: an accidental key press must not abandon twenty minutes of copying.
///
/// **THE PLAN IS RECOMPUTED, NOT STORED.** Every figure on the sheet comes from
/// `plan(language:)` over the live state, so a drive unplugged while the sheet is
/// open refuses the move at once, and the sheet never shows a stale "fits".
@MainActor
final class ModelMoveSheetModel: ObservableObject, Identifiable {
    enum Stage: Equatable {
        case choosing
        case moving
        case finished(MoveSuccessText)
        /// The move did not finish. `kind` tells a failure from a cancel.
        case failed(MoveFailureText)
    }

    let id = UUID()
    let subject: MoveSubject
    /// The user's choice of destination (a library id, `internal` for This Mac),
    /// or nil to follow the default.
    @Published var selection: String?
    @Published private(set) var stage: Stage = .choosing
    @Published private(set) var size: MoveSize
    /// The folder-size read, when the listing carries no size. A test seam.
    private(set) var measuringTask: Task<Void, Never>?
    private var runTask: Task<Void, Never>?
    private weak var app: AppModel?

    init(subject: MoveSubject, app: AppModel) {
        self.subject = subject
        self.app = app
        if let bytes = subject.knownBytes {
            size = .known(bytes)
        } else {
            size = .measuring
            measure(with: app.modelSizeReader)
        }
    }

    // MARK: - What the view draws

    /// The progress of THIS model's move, while it runs.
    var progress: MoveProgress? {
        guard let move = app?.activeMove, move.installId == subject.installId else { return nil }
        return move
    }

    func plan(language: AppLanguage = AppStrings.currentLanguage) -> ModelMovePlan {
        guard let app else {
            // The app that owned the sheet is gone: there is nowhere to move to.
            return ModelMovePlan.make(
                subject: subject, size: size,
                destinations: InstallDestinationModel(
                    libraries: [], freeBytes: [:],
                    destinationDefault: DefaultInstallDestination(
                        libraryId: InstallDestinationModel.thisMacId, library: nil,
                        fallbackBecauseNotConnected: false, notConnectedLibraryName: nil),
                    thisMacFreeBytes: nil, language: language),
                selection: nil, app: MoveAppState(), language: language)
        }
        return ModelMovePlan.make(
            subject: subject, size: size, destinations: app.installDestinations, selection: selection,
            app: app.moveAppState(for: subject), language: language)
    }

    // MARK: - Dismissal

    /// False while a move runs.
    var canDismiss: Bool { stage != .moving }

    /// Whether Escape closes the sheet. It never cancels a move.
    var escapeCloses: Bool { canDismiss }

    /// Closes the sheet through the app, which owns it. False, and nothing
    /// happens, while a move runs.
    @discardableResult
    func close() -> Bool {
        guard canDismiss else { return false }
        guard let app, app.moveSheet === self else { return true }
        return app.dismissMoveSheet()
    }

    /// Stops what the sheet started on its own (the size read). Called when the
    /// sheet leaves.
    func discard() {
        measuringTask?.cancel()
    }

    // MARK: - Actions

    /// Starts the move to the chosen destination. Nothing happens, and nil is
    /// returned, while the plan is refused or the size is still being measured:
    /// the sheet shows the reason. The move is claimed before this returns, so a
    /// Cancel pressed at once finds it.
    @discardableResult
    func move() -> Task<Void, Never>? {
        guard stage == .choosing, let app else { return nil }
        let destination = plan().destinationId ?? ""
        switch app.beginMove(subject: subject, destinationId: destination, size: size) {
        case .refused:
            return nil
        case .started(let ticket):
            stage = .moving
            let task = Task { [weak self] in
                let result = await app.performMove(ticket)
                self?.finish(result)
            }
            runTask = task
            return task
        }
    }

    /// Asks the engine to stop. The sheet stays on "Cancelling..." until the
    /// engine answers.
    func cancel() {
        guard stage == .moving else { return }
        app?.cancelActiveMove()
    }

    /// Back to the destination menu after a failure or a cancel.
    func retry() {
        guard case .failed = stage else { return }
        stage = .choosing
    }

    private func finish(_ result: MoveRunResult) {
        switch result {
        case .moved(let text):
            stage = .finished(text)
        case .failed(let text):
            stage = .failed(text)
        case .refused, .abandoned:
            // Nothing happened, or the app that asked is gone: the plan says why.
            stage = .choosing
        }
    }

    // MARK: - Sizes the listing does not record

    /// An image listing carries no size, and a row can record none. The folder is
    /// measured off the main actor; a folder that measures as nothing is a size we
    /// do not have, not a free move.
    private func measure(with reader: @escaping @Sendable (String) -> UInt64) {
        let path = subject.path
        measuringTask = Task { [weak self] in
            let bytes = await Task.detached(priority: .userInitiated) { reader(path) }.value
            guard !Task.isCancelled, let self else { return }
            self.size = bytes > 0 ? .known(bytes) : .unavailable
        }
    }
}
