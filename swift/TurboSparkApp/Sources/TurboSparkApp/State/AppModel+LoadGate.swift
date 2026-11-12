import Foundation
import TurboSpark

/// What a request to load is for. The gate and the confirmation are the same for
/// every purpose; only what happens when the load starts differs.
enum LoadPurpose: Equatable {
    /// Open the model as the chat model. `switchToChat` also moves to the Chat
    /// pane once the load has started.
    case chat(switchToChat: Bool)
    /// Open the model in its own session and attach it to the running server.
    case serverAttach
    /// Open the folder the user chose in the folder picker.
    case chosenFolder
}

enum LoadRequestOutcome: Equatable {
    /// The load began.
    case started
    /// The model is the one already in memory (chat requests only).
    case alreadyLoaded
    /// A warned model: the confirmation is up, nothing was loaded.
    case needsConfirmation
    /// The model is not there (its drive is away, or its folder is gone). Nothing
    /// was loaded, and a toast said why.
    case refused(LoadGate)
    /// A turn or another load is running, or the server cannot take a model now.
    case busy
}

/// A load that is waiting for the user to say "Load anyway" (requirement 5.5).
/// One at a time: a newer request replaces it.
struct PendingLoadConfirmation: Identifiable, Equatable {
    let id = UUID()
    let model: InstalledModel
    let warning: LoadGate.Warning
    let purpose: LoadPurpose

    func title(language: AppLanguage = AppStrings.currentLanguage) -> String {
        LoadGateText.confirmationTitle(alias: model.alias, language: language)
    }

    func message(language: AppLanguage = AppStrings.currentLanguage) -> String {
        LoadGateText.text(for: warning, language: language)
    }

    func confirmTitle(language: AppLanguage = AppStrings.currentLanguage) -> String {
        LoadGateText.loadAnywayTitle(language: language)
    }
}

/// The one choke point every INTERACTIVE load goes through (task 6.7).
///
/// **WHICH PATH IS WHICH**
/// - Interactive, gated here: the chat loader, the composer control, the
///   inspector picker, the server menu, the Load button of the loader and the
///   Model menu, the installed-model detail pane (Load, Load anyway, Start Chat),
///   the Model Hub's Load and Chat, the `/model` command, a folder chosen in the
///   open panel, and the server pane's Attach.
/// - Cannot ask, and load as they did (minus a model that is not available or is
///   being moved, which `open(_:)` refuses): the load that follows an install, a
///   reload of the model in memory, Humanize's own engine, and a server favorite's
///   auto-start. Launch restoration only SELECTS a model; it never loads one.
/// - A model that is BEING MOVED answers every one of them with a refusal
///   (`isMoving`): the engine removes its folder when the move ends. Humanize, the
///   image session and the audio session ask it themselves.
extension AppModel {
    /// The gate of `model` as the libraries stand now: its live availability, its
    /// library, that library's link and measured speed.
    func loadGate(for model: InstalledModel) -> LoadGate {
        let live = LoadableModelList.liveRow(for: model, in: libraryState.allInstalled)
        let library = libraryState.library(for: live)
        return LoadGate.evaluate(
            model: live,
            library: library,
            connection: library.flatMap { libraryState.connection(for: $0) },
            measuredMBps: library.flatMap { libraryState.measuredSpeed(for: $0.id) },
            isMoE: ModelFeatureDescriptor.isMoE(installedModel: live))
    }

    /// Asks to load `model`. Loads at once when nothing stands in the way, asks
    /// first when the drive's speed will hurt, and refuses when the model is not
    /// there.
    ///
    /// - Parameters:
    ///   - confirmed: the user already said "Load anyway" (the action bar's button
    ///     or the confirmation). It overrides advice only.
    ///   - bringWindowForward: the request came from the menu bar, which cannot
    ///     host the confirmation: the main window comes forward to ask.
    @discardableResult
    func requestLoad(
        _ model: InstalledModel,
        purpose: LoadPurpose = .chat(switchToChat: false),
        confirmed: Bool = false,
        bringWindowForward: Bool = false
    ) -> LoadRequestOutcome {
        // **A MODEL THAT IS BEING MOVED IS NEVER LOADED** (task 6.11). The engine
        // removes the original folder once the copy is verified, so a load that began
        // now would map files that are about to go (the state#43 class). First of all
        // the answers: nothing about a moving model is resident, warned or confirmed.
        if isMoving(model) {
            announceModelBeingMoved()
            return .busy
        }
        let resident = loadedModelPath == model.path
        if resident, case .chat(let switchToChat) = purpose {
            // Already in memory: nothing is read, nothing is warned about.
            if switchToChat { activeSection = .chat }
            return .alreadyLoaded
        }
        guard canStartLoad(for: purpose) else {
            if case .chat(true) = purpose { activeSection = .chat }
            return .busy
        }

        let gate = loadGate(for: model)
        switch LoadDecision.decide(gate: gate, isResident: resident, confirmed: confirmed) {
        case .refuse(let gate):
            announceCannotLoad(gate)
            return .refused(gate)
        case .confirm(let warning):
            pendingLoadConfirmation = PendingLoadConfirmation(model: model, warning: warning, purpose: purpose)
            if bringWindowForward { showMainWindow() }
            return .needsConfirmation
        case .load:
            // A newer request that loads replaces a question nobody answered.
            pendingLoadConfirmation = nil
            return performLoad(purpose, model)
        }
    }

    /// The user's "Load anyway" in the action bar: the same load the Load button
    /// runs, minus the question (the warning is on screen beside the button).
    func loadAnyway(_ model: InstalledModel) {
        requestLoad(model, purpose: .chat(switchToChat: false), confirmed: true)
    }

    /// The user answered the confirmation with "Load anyway". The availability is
    /// read again: the drive may have left while the question was on screen.
    func confirmPendingLoad() {
        guard let pending = pendingLoadConfirmation else { return }
        confirmLoad(pending)
    }

    func confirmLoad(_ pending: PendingLoadConfirmation) {
        if pendingLoadConfirmation?.id == pending.id { pendingLoadConfirmation = nil }
        requestLoad(pending.model, purpose: pending.purpose, confirmed: true)
    }

    func cancelPendingLoad() {
        pendingLoadConfirmation = nil
    }

    // MARK: - Not available

    /// Says why a model cannot be loaded: a toast with the statement, never the
    /// app error (an unplugged drive is not an availability banner, requirement
    /// 3.4). The answer to a click, so it is not "a notification at launch".
    func announceCannotLoad(_ gate: LoadGate) {
        guard let statement = LoadGateText.statement(for: gate) else { return }
        showToast(statement, style: .info)
    }

    /// Whether the model can be opened now. The net under every load path: the
    /// engine open refuses a model whose drive is away or whose folder is gone,
    /// even from a path that cannot ask (a load after an install, a favorite).
    func canOpenNow(_ model: InstalledModel) -> Bool {
        // The same refusal for the paths that cannot ask: the load after an install, a
        // reload, a server attach and a favorite's auto-start all end here.
        if isMoving(model) {
            announceModelBeingMoved()
            return false
        }
        let gate = loadGate(for: model)
        if !gate.canLoad { announceCannotLoad(gate) }
        return gate.canLoad
    }

    // MARK: - Starting

    private func canStartLoad(for purpose: LoadPurpose) -> Bool {
        switch purpose {
        case .chat, .chosenFolder:
            return !generating && !opening
        case .serverAttach:
            return server != nil && !serverBusy
        }
    }

    private func performLoad(_ purpose: LoadPurpose, _ model: InstalledModel) -> LoadRequestOutcome {
        switch purpose {
        case .chat(let switchToChat):
            beginChatLoad(model)
            if switchToChat { activeSection = .chat }
        case .serverAttach:
            beginServerAttach(model)
        case .chosenFolder:
            beginOpeningFolder(model)
        }
        return .started
    }
}
