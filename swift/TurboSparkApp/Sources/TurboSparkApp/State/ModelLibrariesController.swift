import Combine
import Foundation
import SwiftUI
import TurboSpark

/// A sentence the libraries section keeps on screen until it is dismissed or the
/// next library action replaces it (requirements 1.4, 1.5, 1.7, 1.10).
struct LibraryMessage: Equatable, Identifiable {
    enum Kind: Equatable {
        case error
        case info
    }

    let id: UUID
    let kind: Kind
    let text: String
}

/// What a row's speed test is doing, or why it did not produce a number. A
/// finished test is not a state here: its figure is `LibraryState.measuredSpeeds`.
enum LibrarySpeedTestState: Equatable {
    case running
    case failed(String)
}

extension AppModel {
    /// A text, image or audio install is running.
    ///
    /// The audio workspace is asked only if it was built: reading the lazy
    /// property builds it, and a settings pane must not load the audio library
    /// to answer a question about installs.
    var isInstallRunning: Bool {
        isInstallingModel || isInstallingImageModel || (audioWorkspaceCreated && audioWorkspace.isInstalling)
    }

    /// An install or a model move is running. The engine refuses a change to the
    /// library list then (`busy`), so the section, the offer banner and every other
    /// library change disable themselves instead of letting the user find out by
    /// trying. A move counts from the moment it starts (`activeMove`) until it ends
    /// in any way (task 6.11).
    var isInstallOrMoveRunning: Bool {
        isInstallRunning || activeMove != nil
    }
}

/// The actions of the Model libraries section: add, set as default, remove, test
/// speed, reveal and stop waiting for a moved store (tasks 6.1 and 6.2).
///
/// **IT NEVER WRITES THE APP ERROR OR POSTS A TOAST.** A library that is away
/// is state (requirement 3.4), and a refusal of something the user just asked
/// for is an answer to that question: it is a message in the section, with a
/// glyph and text, that stays until the user dismisses it or asks again. The
/// controller holds the model only to ask whether an install is running.
///
/// **EVERY ENGINE CALL IS OFF THE MAIN ACTOR**, through `LibraryState`, which
/// owns the detached tasks and the refresh that follows each change.
///
/// **THIS MAC IS NEVER REMOVED, AND THE ENGINE IS NOT ASKED.** The row does not
/// offer it, and the controller refuses it again before the service, so a
/// caller that skips the row cannot reach the engine's removal for the
/// internal library.
@MainActor
final class ModelLibrariesController: ObservableObject {
    typealias SpeedTestRunner = @Sendable (URL, Bool) async throws -> DriveSpeedResult

    // MARK: Published state

    /// The refusal or outcome on screen. Replaced by the next library action.
    @Published private(set) var message: LibraryMessage?
    /// Speed tests in flight and the last failure, by library id.
    @Published private(set) var speedTests: [String: LibrarySpeedTestState] = [:]
    /// A change to the library list is running, so a second one is not started.
    @Published private(set) var isChanging = false

    // MARK: Collaborators

    let libraryState: LibraryState
    private weak var model: AppModel?
    private let migration: @MainActor () -> MoveStoreMigration?
    private let speedTest: SpeedTestRunner
    private let reveal: @MainActor (String) -> Void
    private let announce: @MainActor (String) -> Void
    private let language: @MainActor () -> AppLanguage
    private var speedTasks: [String: Task<Void, Never>] = [:]

    init(
        libraryState: LibraryState,
        model: AppModel? = nil,
        migration: @escaping @MainActor () -> MoveStoreMigration? = { nil },
        speedTest: @escaping SpeedTestRunner = ModelLibrariesController.liveSpeedTest,
        reveal: @escaping @MainActor (String) -> Void = { ModelStorageManager.revealInFinder(path: $0) },
        announce: @escaping @MainActor (String) -> Void = { _ = AccessibilityNotification.Announcement.post(.init($0)) },
        language: @escaping @MainActor () -> AppLanguage = { AppStrings.currentLanguage }
    ) {
        self.libraryState = libraryState
        self.model = model
        self.migration = migration
        self.speedTest = speedTest
        self.reveal = reveal
        self.announce = announce
        self.language = language
    }

    /// The real test: reads from the drive on its own thread.
    nonisolated static let liveSpeedTest: SpeedTestRunner = { root, isInternal in
        try await DriveSpeedTest.measureReadSpeed(libraryRoot: root, isInternal: isInternal)
    }

    /// An install or a move is running, or this controller is already changing
    /// the list. Every control that changes a library is disabled meanwhile.
    var isChangeBlocked: Bool {
        isChanging || (model?.isInstallOrMoveRunning ?? false)
    }

    // MARK: - Messages

    func dismissMessage() {
        message = nil
    }

    private func post(_ kind: LibraryMessage.Kind, _ text: String) {
        message = LibraryMessage(id: UUID(), kind: kind, text: text)
        // Completion and failure are announced: the sentence appears in a place
        // a VoiceOver user is not looking.
        announce(text)
    }

    /// Starts a change: drops the sentence of the one before, and refuses at
    /// once, in the busy sentence, while an install or a move runs.
    private func begin() -> Bool {
        message = nil
        guard !isChanging else { return false }
        if model?.isInstallOrMoveRunning == true {
            post(.error, LibraryActionText.busy(language: language()))
            return false
        }
        return true
    }

    private func displayName(of library: ModelLibrary?, fallback id: String) -> String {
        guard let library else { return UntrustedText.display(id) }
        if library.isInternal { return AppStrings.localized("This Mac", language: language()) }
        return UntrustedText.display(library.name)
    }

    // MARK: - Add, default, remove

    /// Adds the library at `path` (the folder the user chose) and says so, or
    /// shows the one sentence for the engine's refusal.
    func add(path: String) async {
        guard begin() else { return }
        isChanging = true
        defer { isChanging = false }
        do {
            let added = try await libraryState.addLibrary(path: path, name: nil)
            post(.info, LibraryActionText.added(name: added.library.name, language: language()))
        } catch {
            post(.error, LibraryActionText.sentence(for: error, language: language()))
        }
    }

    func setDefault(id: String) async {
        guard begin() else { return }
        isChanging = true
        defer { isChanging = false }
        let name = displayName(of: libraryState.libraries.first { $0.id == id }, fallback: id)
        do {
            try await libraryState.setDefaultLibrary(id: id)
            post(.info, LibraryActionText.nowDefault(name: name, language: language()))
        } catch {
            post(.error, LibraryActionText.sentence(for: error, language: language()))
        }
    }

    /// Stops listing the library and its models. No file on the drive is
    /// touched (requirement 1.8); the caller has already asked the user.
    func remove(id: String) async {
        message = nil
        // The row does not offer it; this keeps the engine out of it for a
        // caller that did not look.
        if id == LibraryState.internalLibraryId {
            post(.error, LibraryActionText.internalLibrary(language: language()))
            return
        }
        guard begin() else { return }
        isChanging = true
        defer { isChanging = false }
        let name = displayName(of: libraryState.libraries.first { $0.id == id }, fallback: id)
        do {
            try await libraryState.removeLibrary(id: id)
            post(.info, LibraryActionText.removed(name: name, language: language()))
        } catch {
            post(.error, LibraryActionText.sentence(for: error, language: language()))
        }
    }

    // MARK: - Reveal

    /// Shows a connected library's folder in Finder. A library that is away has
    /// no folder to show, and the row disables the action.
    func revealInFinder(libraryId: String) {
        guard let library = libraryState.libraries.first(where: { $0.id == libraryId }),
            library.available, let root = library.root
        else { return }
        reveal(root)
    }

    // MARK: - Test speed (requirement 5.6)

    /// Reads from the library's drive with the cache out of the way and stores
    /// the figure for its estimates until the drive is disconnected. Runs on its
    /// own thread; `cancelSpeedTest` stops it at the next chunk.
    func testSpeed(libraryId: String) {
        guard speedTests[libraryId] != .running else { return }
        guard let library = libraryState.libraries.first(where: { $0.id == libraryId }) else { return }
        if library.isInternal {
            // The row does not offer it. A caller that does gets a sentence
            // instead of a number measured against the internal disk.
            speedTests[libraryId] = .failed(
                LibraryActionText.speedTestSentence(for: DriveSpeedTestError.thisMac, language: language()))
            return
        }
        guard library.available, let root = library.root else { return }
        speedTests[libraryId] = .running
        let runner = speedTest
        let url = URL(fileURLWithPath: root, isDirectory: true)
        let isInternal = library.isInternal
        speedTasks[libraryId] = Task { [weak self] in
            let outcome: Result<DriveSpeedResult, Error>
            do {
                outcome = .success(try await runner(url, isInternal))
            } catch {
                outcome = .failure(error)
            }
            self?.finishSpeedTest(libraryId, outcome)
        }
    }

    private func finishSpeedTest(_ id: String, _ outcome: Result<DriveSpeedResult, Error>) {
        speedTasks[id] = nil
        switch outcome {
        case .success(let result):
            speedTests[id] = nil
            libraryState.recordMeasuredSpeed(result.megabytesPerSecond, for: id)
            // A drive that left while the test ran keeps no figure, and says
            // nothing about one.
            if libraryState.measuredSpeed(for: id) != nil,
                let text = ModelLibraryRowModel.measuredSpeedText(result.megabytesPerSecond, language: language())
            {
                announce(text)
            }
        case .failure(let error):
            if error is CancellationError {
                speedTests[id] = nil
                return
            }
            let text = LibraryActionText.speedTestSentence(for: error, language: language())
            speedTests[id] = .failed(text)
            announce(text)
        }
    }

    func cancelSpeedTest(libraryId: String) {
        speedTasks[libraryId]?.cancel()
    }

    /// The section left the screen: no test keeps reading a drive unseen.
    func cancelAllSpeedTests() {
        for task in speedTasks.values { task.cancel() }
    }

    /// Clears a failure the user has read.
    func dismissSpeedTestFailure(libraryId: String) {
        if case .failed? = speedTests[libraryId] { speedTests[libraryId] = nil }
    }

    /// Waits for the test on `libraryId` to finish. A test seam.
    func waitForSpeedTest(libraryId: String) async {
        await speedTasks[libraryId]?.value
    }

    // MARK: - Stop waiting for a moved store (requirement 9.3)

    /// Forgets the conversion of a store Move Store moved. Not a change to the
    /// library list, so an install does not block it. Nothing on the drive is
    /// touched; if the record cannot be cleared the placeholder stays and the
    /// user is told.
    func abandonConversion() {
        message = nil
        guard let migration = migration() else { return }
        if !migration.abandon() {
            post(.error, LibraryActionText.stopWaitingFailed(language: language()))
        }
    }
}
