import Foundation
import SwiftUI
import TurboSpark

/// How a move ended, as the sheet needs it.
enum MoveRunResult: Equatable, Sendable {
    /// Refused before the engine was called. Nothing changed.
    case refused(MoveRefusal)
    /// The install is in the destination. All three engine outcomes land here.
    case moved(MoveSuccessText)
    /// The engine refused, failed or was cancelled. The original is where it was.
    case failed(MoveFailureText)
    /// The model was shut down (a quit, a vault lock, a profile switch) while the
    /// move ran. Nothing more is said or written.
    case abandoned
}

/// What holds a model, read from the app. Live by default; a test replaces the
/// readers it cannot build. The pure decision is `ModelMovePlan.refusal`; this
/// only gathers its inputs.
@MainActor
struct MoveHolderProbe {
    /// The path of the model loaded for chat.
    var loadedChatPath: @MainActor (AppModel) -> String? = { $0.loadedModelPath }
    var hasChatSession: @MainActor (AppModel) -> Bool = { $0.session != nil }
    /// The ids the server serves models under.
    var servedIDs: @MainActor (AppModel) -> Set<String> = {
        Set($0.serverAttachedSessions.keys).union($0.serverInfo?.models ?? [])
    }
    var humanizeUses: @MainActor (AppModel, String) -> Bool = { $0.humanizeWorkspace.usesModel(at: $1) }
    /// The path of the image model resident in the Images workspace.
    var loadedImagePath: @MainActor (AppModel) -> String? = { $0.imageSession == nil ? nil : $0.imageSessionPath }
    /// The path of the audio model resident in the Audio workspace. Asked only if
    /// the workspace was built: reading the lazy property builds it.
    var audioResidentPath: @MainActor (AppModel) -> String? = {
        $0.audioWorkspaceCreated ? $0.audioWorkspace.residentModelPath : nil
    }
}

/// State a move keeps between its start and its end that no view reads.
final class MoveDriverState {
    /// Bumped at each start and by a shutdown. A move whose generation is no longer
    /// current finishes into nothing: its model was shut down meanwhile.
    var generation = 0
    /// Cancel was pressed but the engine said no move was running yet (the window
    /// between the click and the engine registering the move). The first progress
    /// figure proves it is running, and the cancel is sent again.
    var cancelNeedsRepeat = false
    /// The same for the cancel a SHUTDOWN sends. The shutdown bumps `generation`, so
    /// the first figure of the move it cancelled arrives under a generation that is
    /// no longer current and `applyMoveProgress` would drop it; this flag is what
    /// lets that figure send the cancel again, so the copy does not run to its end
    /// after the app was told to stop it.
    var shutdownCancelNeedsRepeat = false
    /// The generation of the engine call that is awaiting the engine now, nil
    /// between moves. A repeated shutdown cancel is sent only while this still names
    /// the call the figure belongs to, because the engine runs one move at a time
    /// for the whole process: a late figure of a move that has already ended must
    /// not cancel whichever move started after it.
    var engineCallGeneration: Int?
}

/// The model move (task 6.11, requirements 7.1 to 7.5 and 8.1).
///
/// **THE APP ASKS AND THE ENGINE MOVES.** The engine copies into the destination's
/// staging folder, compares total bytes and file count, publishes, rewrites the
/// registry row and both markers, and only then deletes the original. The app never
/// moves, copies, renames or removes a file (`ModelMoveStringsTests` scans for it);
/// it refuses what must not start, hands the engine a measured free-space figure,
/// shows progress, and rewrites ITS OWN state around the engine's answer.
///
/// **WHAT IT REFUSES FIRST** is `ModelMovePlan.refusal`, one pure function over
/// the subject, the destination and `moveAppState`. The engine refuses too
/// (`busy`, `insufficient_space`), and that stays the net.
///
/// **THE ENGINE CALL IS OFF THE MAIN ACTOR** (the seam's async `move` runs it on a
/// dedicated thread) and Cancel is made from the main actor, a different thread
/// from the move's.
///
/// **ALL THREE SUCCESS OUTCOMES MEAN THE INSTALL IS IN THE DESTINATION.** A stale
/// marker or a leftover folder is one plain extra line, never a failure.
extension AppModel {
    // MARK: - What holds the model

    /// The app's state as it bears on moving `subject` (requirement 7.5).
    func moveAppState(for subject: MoveSubject) -> MoveAppState {
        var state = MoveAppState()
        state.moveRunning = activeMove != nil
        state.installRunning = isInstallRunning
        state.turnRunning = generating || submitting
        state.modelOpening = opening
        let probe = moveHolderProbe
        switch subject.kind {
        case .text:
            let selectedWithSession = selected?.installId == subject.installId && probe.hasChatSession(self)
            state.loadedInChat = Self.samePath(probe.loadedChatPath(self), subject.path) || selectedWithSession
            if let row = subject.row {
                state.attachedToServer = Self.isAttachedToServer(model: row, servedIDs: probe.servedIDs(self))
            }
            state.usedByHumanize = probe.humanizeUses(self, subject.path)
        case .image:
            state.loadedInImages = Self.samePath(probe.loadedImagePath(self), subject.path)
            state.attachedToServer = Self.samePath(serverImageAttachedModel?.path, subject.path)
        case .audio:
            state.residentInAudio = Self.samePath(probe.audioResidentPath(self), subject.path)
            state.attachedToServer = serverSpeechAttachedModel?.alias == subject.alias
                || serverAudioModelPaths.values.contains { Self.samePath($0, subject.path) }
        }
        return state
    }

    private static func samePath(_ candidate: String?, _ path: String) -> Bool {
        guard let candidate, !candidate.isEmpty else { return false }
        return normalizedModelPath(candidate) == normalizedModelPath(path)
    }

    // MARK: - A model that is being moved is never loaded

    /// **THE LOAD SIDE OF THE MOVE'S HAZARD** (the state#43 class). The engine removes
    /// the original folder once the copy is verified, so a model that is loaded while
    /// it moves is mapped from a folder that is about to go. `moveAppState` refuses a
    /// move for a model that is loaded when the move STARTS; these answer for the
    /// time after, from `activeMove`, which is set at the claim (before the engine is
    /// called) and cleared when the move ends in any way.
    ///
    /// Every load goes through one of them: `requestLoad` (every interactive load),
    /// `canOpenNow` (the net under `open`, a reload, a server attach and a favorite's
    /// auto-start), `canLoadModel` (the Model menu), and the loaders that open a
    /// model by path (Humanize, the image session, the audio session and the server's
    /// image and audio attaches).
    func isMoving(_ model: InstalledModel) -> Bool {
        isMoving(installId: model.installId) || isMoving(path: model.path)
    }

    func isMoving(installId: String) -> Bool {
        guard let move = activeMove, !installId.isEmpty else { return false }
        return move.installId == installId
    }

    /// By path, for a loader that is not handed a row. A chosen folder, an image or
    /// audio record and Humanize's engine all name a model by its folder.
    func isMoving(path: String) -> Bool {
        guard let move = activeMove, !move.path.isEmpty else { return false }
        return Self.samePath(path, move.path)
    }

    /// The answer to a click on a load that was refused because of a move: a toast,
    /// never `error`, and never silence.
    func announceModelBeingMoved() {
        showToast(ModelMoveText.modelBeingMoved(), style: .warning)
    }

    // MARK: - The sheet

    /// The sheet for `subject`, made and kept in `moveSheet`. The same model asked
    /// again gets the sheet it already has; a sheet that is mid-move is never
    /// replaced.
    @discardableResult
    func presentMoveSheet(for subject: MoveSubject) -> ModelMoveSheetModel? {
        if let existing = moveSheet {
            if existing.subject.installId == subject.installId { return existing }
            guard existing.canDismiss else { return nil }
            existing.discard()
        }
        // An audio report row names its library by id only: the name comes from the list.
        let sheet = ModelMoveSheetModel(
            subject: subject.resolvingLibraryName(in: libraryState.libraries), app: self)
        moveSheet = sheet
        return sheet
    }

    /// Closes the sheet. False, and the sheet stays, while a move runs: Cancel is
    /// the way out of that (requirement 7.2).
    @discardableResult
    func dismissMoveSheet() -> Bool {
        guard let sheet = moveSheet else { return true }
        guard sheet.canDismiss else { return false }
        sheet.discard()
        moveSheet = nil
        return true
    }

    // MARK: - Running a move

    /// Everything a started move needs to finish, captured at the start: the
    /// words and the selection are about the model as it was BEFORE the engine
    /// moved it.
    struct MoveTicket {
        let subject: MoveSubject
        let destinationId: String
        let destinationName: String
        let sourceName: String
        let freeBytes: Int64
        let requiredBytes: UInt64
        let generation: Int
        let wasSelected: Bool
        let imageWasSelected: Bool
        let language: AppLanguage
    }

    enum MoveStart {
        case refused(MoveRefusal)
        case started(MoveTicket)
    }

    /// Checks everything that can be checked, then claims the move: from the
    /// moment this returns `.started`, `activeMove` is set, the row is in
    /// `modelsBeingChanged`, and every install and library change is refused. The
    /// engine has not been called yet. A cancel that arrives before it is sees the
    /// move and is repeated once the engine reports its first figure.
    ///
    /// Refuses first, from the live state (`ModelMovePlan.refusal`). Nothing is
    /// claimed for a refusal.
    func beginMove(subject: MoveSubject, destinationId: String, size: MoveSize) -> MoveStart {
        let language = AppStrings.currentLanguage
        let plan = ModelMovePlan.make(
            subject: subject, size: size, destinations: installDestinations, selection: destinationId,
            app: moveAppState(for: subject), language: language)
        if let refusal = plan.refusal { return .refused(refusal) }
        // A size still being measured has no refusal and no figure to check.
        // `freeBytesForEngine` is set only from a connected destination's measured
        // capacity, so there is never a negative number to pass.
        guard plan.canMove, let freeBytes = plan.freeBytesForEngine, let destination = plan.destinationId,
            case .known(let requiredBytes) = size
        else { return .refused(.sizeUnavailable) }

        moveDriver.generation += 1
        moveDriver.cancelNeedsRepeat = false
        moveDriver.shutdownCancelNeedsRepeat = false
        let destinationName = plan.destinationName ?? ""
        activeMove = MoveProgress(
            installId: subject.installId, alias: UntrustedText.display(subject.alias, limit: ModelMoveText.aliasLimit),
            sourceName: plan.sourceName, destinationName: destinationName, path: subject.path)
        modelsBeingChanged.insert(subject.installId)
        return .started(
            MoveTicket(
                subject: subject, destinationId: destination, destinationName: destinationName,
                sourceName: plan.sourceName, freeBytes: freeBytes, requiredBytes: requiredBytes,
                generation: moveDriver.generation, wasSelected: selected?.installId == subject.installId,
                imageWasSelected: subject.kind == .image && Self.samePath(imageModelPath, subject.path),
                language: language))
    }

    /// `beginMove`, then `performMove`: the whole move for a caller that does not
    /// need to act between the two.
    func runMove(subject: MoveSubject, destinationId: String, size: MoveSize) async -> MoveRunResult {
        switch beginMove(subject: subject, destinationId: destinationId, size: size) {
        case .refused(let refusal): return .refused(refusal)
        case .started(let ticket): return await performMove(ticket)
        }
    }

    /// Calls the engine off the main actor, then, for each way it can end, brings
    /// the app's own state in line. The result carries the words, ready to show.
    func performMove(_ ticket: MoveTicket) async -> MoveRunResult {
        let subject = ticket.subject
        let generation = ticket.generation
        // A shutdown between the claim and this call: the move was never started,
        // and a cancel made now would find nothing to stop, so the engine must not
        // be asked at all.
        guard moveDriver.generation == generation else {
            modelsBeingChanged.remove(subject.installId)
            return .abandoned
        }
        let service = libraryState.service
        let throttle = MoveProgressThrottle(minimumInterval: moveProgressInterval)
        let progress: @Sendable (UInt64, UInt64) -> Void = { [weak self] done, total in
            guard throttle.shouldForward(done: done, total: total) else { return }
            Task { @MainActor in self?.applyMoveProgress(done: done, total: total, generation: generation) }
        }
        let outcome: Result<MoveOutcome, Error>
        moveDriver.engineCallGeneration = generation
        do {
            outcome = .success(
                try await service.move(
                    installId: subject.installId, to: ticket.destinationId, freeBytes: ticket.freeBytes,
                    progress: progress))
        } catch {
            outcome = .failure(error)
        }
        // Only while this call is still the one in flight: a shutdown and a new
        // profile's move may already have set it again.
        if moveDriver.engineCallGeneration == generation { moveDriver.engineCallGeneration = nil }

        modelsBeingChanged.remove(subject.installId)
        // A shutdown while the engine worked: the app that asked is gone (a quit,
        // a vault lock, a profile switch). Nothing it holds is written, and
        // nothing is said, so the next profile's state is not touched by it.
        guard moveDriver.generation == generation else { return .abandoned }
        activeMove = nil

        switch outcome {
        case .success(let moved):
            let text = ModelMoveText.success(
                moved, alias: subject.alias, destinationName: ticket.destinationName, language: ticket.language)
            followMovedModel(
                subject, to: moved.installId, wasSelected: ticket.wasSelected,
                imageWasSelected: ticket.imageWasSelected)
            showToast(text.spoken, style: .info)
            startNextModelDownloadIfPossible()
            await refreshLibraryStateAfterMove()
            return .moved(text)
        case .failure(let error):
            // The listing first: a drive that was lost shows as away in it, which
            // is how the sentence can name it.
            await refreshLibraryStateAfterMove()
            let context = MoveFailureContext(
                alias: subject.alias, sourceName: ticket.sourceName, destinationName: ticket.destinationName,
                requiredBytes: ticket.requiredBytes, availableBytes: UInt64(max(ticket.freeBytes, 0)),
                awayDriveName: awayDriveName(among: [subject.libraryId, ticket.destinationId]))
            let text = ModelMoveText.failure(for: error, context: context, language: ticket.language)
            _ = AccessibilityNotification.Announcement.post(.init(text.text))
            startNextModelDownloadIfPossible()
            return .failed(text)
        }
    }

    /// Brings the app's own state in line with a move the engine finished: the
    /// per-model state follows the model by its new install id (requirement 8.1),
    /// the lists are read again, and a selection that was on the model is on it
    /// again at its new place.
    private func followMovedModel(
        _ subject: MoveSubject, to newInstallId: String, wasSelected: Bool, imageWasSelected: Bool
    ) {
        // The reasoning level of a model nobody has opened since an older build
        // still sits in the settings file, keyed by the path it had. Reading it
        // through the store folds it under the OLD id, so the re-key below carries
        // it; once the path is gone nothing could find it.
        if let row = subject.row {
            let before = modelReasoningDefaults
            _ = ModelOrganizationStore.shared.reasoningDefault(for: row, migratingLegacyFrom: &modelReasoningDefaults)
            if modelReasoningDefaults != before { persistSettingsDebounced() }
        }
        // The path BEFORE the move: an entry no view has read since an older build
        // wrote it still sits under `alias::oldPath`, and with any other path the
        // store would not find it.
        metadataRekey(subject.installId, newInstallId, subject.path)

        let favorites = serverFavorites.map {
            $0.followingMove(from: subject.installId, oldPath: subject.path, to: newInstallId) ?? $0
        }
        if favorites != serverFavorites {
            serverFavorites = favorites
            persistSettingsDebounced()
        }

        refreshModels()
        var newPath: String?
        switch subject.kind {
        case .text:
            // `refreshModels` moves the selection to the first loadable row when
            // the one it held is gone; the moved model is selected again.
            if let moved = loadableModels.first(where: { $0.installId == newInstallId }) {
                newPath = moved.path
                if wasSelected {
                    selected = moved
                    modelPathText = moved.path
                }
            }
        case .image:
            if let moved = managedImageModels.first(where: { $0.installId == newInstallId }) {
                newPath = moved.path
                if imageWasSelected { imageModelPathText = moved.path }
            }
        case .audio:
            // The audio picker reads the report, not `installed`.
            if audioWorkspaceCreated { audioWorkspace.refreshModels() }
        }
        lastMovedModel = MovedModelEvent(
            oldInstallId: subject.installId, newInstallId: newInstallId, oldPath: subject.path, newPath: newPath)
    }

    /// The one library of `ids` that is not connected now, by name, when the
    /// listing can say exactly one. Raw: the sentence escapes it.
    private func awayDriveName(among ids: [String]) -> String? {
        let away = libraryState.libraries.filter {
            ids.contains($0.id) && !$0.isInternal && !$0.available
        }
        return away.count == 1 ? away[0].name : nil
    }

    // MARK: - Progress and Cancel

    func applyMoveProgress(done: UInt64, total: UInt64, generation: Int) {
        guard moveDriver.generation == generation else {
            // A figure of a move whose model was shut down. The shutdown's cancel may
            // have reached the engine before it had registered the move (it answered
            // false); this figure proves the move is running, so the cancel goes out
            // again. Nothing else of the figure is used.
            if moveDriver.shutdownCancelNeedsRepeat, moveDriver.engineCallGeneration == generation {
                moveDriver.shutdownCancelNeedsRepeat = !libraryState.service.cancelMove()
            }
            return
        }
        guard var move = activeMove else { return }
        move.apply(done: done, total: total)
        activeMove = move
        // The first figure proves the engine is running: a cancel that found no
        // move to stop (it was pressed before the engine registered one) goes out
        // again, or the move would run to the end under a screen saying Cancelling.
        if moveDriver.cancelNeedsRepeat, move.cancelRequested {
            moveDriver.cancelNeedsRepeat = !libraryState.service.cancelMove()
        }
    }

    /// Asks the engine to stop the move in flight, from the main actor: not the
    /// thread the move runs on. The sheet stays on "Cancelling..." until the
    /// engine answers, with `cancelled`, or with a success if the copy had been
    /// published already.
    func cancelActiveMove() {
        guard var move = activeMove, !move.cancelRequested else { return }
        move.cancelRequested = true
        activeMove = move
        moveDriver.cancelNeedsRepeat = !libraryState.service.cancelMove()
    }

    /// A quit, a vault lock or a profile switch: stop the move and leave nothing
    /// set. The engine leaves the original intact and may leave a
    /// `.staging/move-*` folder on the destination (there is no sweep yet). A move
    /// that finishes anyway lands in `.abandoned` and writes nothing.
    func cancelActiveMoveForShutdown() {
        guard let move = activeMove else { return }
        moveDriver.generation += 1
        // False is "no move is running yet": the engine has not registered the call.
        // The first figure it reports sends the cancel again (`applyMoveProgress`),
        // or the copy would run to its end after the app was told to stop it.
        moveDriver.shutdownCancelNeedsRepeat = !libraryState.service.cancelMove()
        modelsBeingChanged.remove(move.installId)
        activeMove = nil
    }

    // MARK: - After a move

    /// Reads the library state again, but waits only `moveRefreshPatience` for it.
    /// A listing that hangs on a drive that was just pulled must not hold the
    /// answer back, and `LibraryState` folds the late result in when it lands.
    private func refreshLibraryStateAfterMove() async {
        let state = libraryState
        let patience = moveRefreshPatience
        let gate = MoveRaceGate()
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let sleeper = Task { @MainActor in
                try? await Task.sleep(for: patience)
                if gate.claim() { continuation.resume() }
            }
            Task { @MainActor in
                await state.refresh()
                sleeper.cancel()
                if gate.claim() { continuation.resume() }
            }
        }
    }
}

/// Lets exactly one of two racing tasks resume a continuation.
private final class MoveRaceGate: @unchecked Sendable {
    private let lock = NSLock()
    private var claimed = false

    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if claimed { return false }
        claimed = true
        return true
    }
}
