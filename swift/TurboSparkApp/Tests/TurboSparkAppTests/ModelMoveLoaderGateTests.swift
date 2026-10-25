import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// A Humanize engine that does nothing: opening one is what the cases watch for.
private final class IdleHumanizeEngine: HumanizeEngine, @unchecked Sendable {
    func humanize(
        _ draft: String, options: HumanizeOptions
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
    func cancel() {}
}

/// Task 6.11, review round 1: the loaders that open a model by PATH refuse one that
/// is being moved.
///
/// `ModelMoveLoadGateTests` holds the chat doors. These are the other ones: the
/// Humanize engine, the image session (the Images screen and the API worker share
/// one), the audio session and the Qwen speech job, and the server's image and audio
/// attaches. Each is handed a folder, not a row, so each asks `isMoving(path:)`.
///
/// `activeMove` is set directly: what these cases hold is what each loader does when
/// a move is on, not how a move runs (`ModelMoveFlowTests`).
@MainActor
final class ModelMoveLoaderGateTests: LibraryStateTestCase {
    private let movingSentence = "This model is being moved. Try again when the move finishes."
    private let movingPath = "/Users/test/.turbospark/models/text/lg-moving.gturbo"

    private func moving(_ path: String? = nil) -> MoveProgress {
        MoveProgress(
            installId: "internal:text/lg-moving", alias: "lg-moving", sourceName: "This Mac",
            destinationName: "T7", path: path ?? movingPath)
    }

    private func makeBareModel() async -> AppModel {
        let model = await makeModel(using: makeState())
        model.enableLMStudioDetection = false
        model.customModelDirectories = []
        return model
    }

    // MARK: isMoving

    func testIsMovingAnswersByInstallIdAndByNormalizedPath() async throws {
        let model = await makeBareModel()
        let row = InstalledModel(
            alias: "lg-moving", repo: "owner/lg-moving", path: movingPath, family: "qwen", installBytes: 1,
            library: "internal", libraryName: "This Mac", installId: "internal:text/lg-moving",
            availability: .available, locationKind: .thisMac)
        XCTAssertFalse(model.isMoving(row), "nothing moves")
        XCTAssertFalse(model.isMoving(path: movingPath))
        XCTAssertFalse(model.isMoving(installId: row.installId))

        model.activeMove = moving()

        XCTAssertTrue(model.isMoving(row))
        XCTAssertTrue(model.isMoving(installId: "internal:text/lg-moving"))
        XCTAssertTrue(model.isMoving(path: movingPath))
        XCTAssertTrue(model.isMoving(path: movingPath + "/"), "a trailing slash is the same folder")
        XCTAssertFalse(model.isMoving(path: "/Users/test/.turbospark/models/text/other.gturbo"))
        XCTAssertFalse(model.isMoving(installId: "internal:text/other"))
        XCTAssertFalse(model.isMoving(installId: ""), "a row with no id is never the moving one")
        XCTAssertFalse(model.isMoving(path: ""))
        let sameAliasElsewhere = InstalledModel(
            alias: "lg-moving", repo: "owner/lg-moving", path: "/Volumes/Other/TurboSpark/text/lg-moving.gturbo",
            family: "qwen", installBytes: 1, library: "o", libraryName: "Other", installId: "o:text/lg-moving",
            availability: .available, locationKind: .library)
        XCTAssertFalse(model.isMoving(sameAliasElsewhere), "the alias alone is not the model: another library's copy loads")
    }

    func testAProgressMadeWithoutAPathMatchesNoFolder() async throws {
        let model = await makeBareModel()
        model.activeMove = MoveProgress(installId: "i", alias: "a", sourceName: "s", destinationName: "d")

        XCTAssertFalse(model.isMoving(path: ""))
        XCTAssertFalse(model.isMoving(path: "/anything"))
        XCTAssertTrue(model.isMoving(installId: "i"))
    }

    // MARK: Humanize

    /// How many times the planner asked the model's tokenizer to count.
    final class TokenCounts: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0
        func add() { lock.lock(); value += 1; lock.unlock() }
        var count: Int { lock.lock(); defer { lock.unlock() }; return value }
    }

    private func humanize(
        movingNow: @escaping () -> Bool
    ) -> (HumanizeController, opens: () -> [String], counts: TokenCounts) {
        let controller = HumanizeController(defaults: UserDefaults(suiteName: "lg.humanize.\(UUID().uuidString)")!)
        var opened: [String] = []
        let counts = TokenCounts()
        controller.countDraftTokens = { text, _ in
            counts.add()
            return text.count
        }
        controller.installedPath = { "/models/humanizer-12b.gturbo" }
        controller.modelMoving = { _ in movingNow() }
        controller.openEngine = { path in
            opened.append(path)
            return IdleHumanizeEngine()
        }
        return (controller, { opened }, counts)
    }

    private func settle(_ controller: HumanizeController) async {
        for _ in 0..<200 where controller.isBusy || controller.phase != .idle {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    func testHumanizeNeverOpensItsEngineFromAModelThatIsMoving() async throws {
        var isMoving = true
        let (controller, opens, counts) = humanize(movingNow: { isMoving })
        controller.draft = "Greetings."

        controller.run()
        await settle(controller)

        XCTAssertEqual(opens(), [], "the engine was not opened")
        XCTAssertEqual(counts.count, 0, "and the model's tokenizer was not read either")
        XCTAssertEqual(controller.errorMessage, movingSentence, "the run says why")
        XCTAssertFalse(controller.engineLoaded)

        isMoving = false
        controller.run()
        await settle(controller)
        XCTAssertEqual(opens(), ["/models/humanizer-12b.gturbo"], "when the move is over it opens again")
        XCTAssertGreaterThan(counts.count, 0)
        XCTAssertNil(controller.errorMessage)
    }

    func testHumanizeLoadButtonSaysWhyInsteadOfDoingNothing() async throws {
        let (controller, opens, _) = humanize(movingNow: { true })

        controller.load()

        // Straight away, before any task could have run: a button that began a load and
        // failed a moment later would flash "loading" for a model it must not open.
        XCTAssertEqual(controller.phase, .idle, "nothing is shown as loading")
        XCTAssertFalse(controller.isBusy)
        XCTAssertEqual(controller.errorMessage, movingSentence)
        await settle(controller)
        XCTAssertEqual(opens(), [])
    }

    func testAMoveThatBeginsWhileHumanizeCountsTokensStillStopsTheOpen() async throws {
        // The net under the open itself: the move starts after the run was accepted
        // and the first check passed, during the token count.
        var isMoving = false
        let (controller, opens, _) = humanize(movingNow: { isMoving })
        controller.countDraftTokens = { text, _ in
            isMoving = true
            return text.count
        }
        controller.draft = "Greetings."

        controller.run()
        await settle(controller)

        XCTAssertEqual(opens(), [], "ensureEngine looked again")
        XCTAssertEqual(controller.errorMessage, movingSentence)
    }

    func testTheAppWiresHumanizeToTheMove() async throws {
        let model = await makeBareModel()
        model.activeMove = moving()

        XCTAssertTrue(model.humanizeWorkspace.modelMoving(movingPath))
        XCTAssertFalse(model.humanizeWorkspace.modelMoving("/Users/test/.turbospark/models/text/other.gturbo"))
    }

    // MARK: Images

    private func image(path: String) -> ImageInstalledModel {
        ImageInstalledModel(
            alias: "lg-image", modelID: "andrevp/Z-Image-Turbo-MLX-4bit", revision: "r", path: path, width: 1024,
            height: 1024, schedulerSteps: 9, library: "internal", libraryName: "This Mac",
            installId: "internal:image/lg-image")
    }

    func testTheImageSessionIsNeverOpenedFromAModelThatIsMoving() async throws {
        let model = await makeBareModel()
        let target = image(path: "/Users/test/.turbospark/models/image/lg-image")
        model.activeMove = moving(target.path)

        XCTAssertThrowsError(try model.sharedImageSession(for: target)) { error in
            XCTAssertTrue("\(error)".contains(movingSentence), "\(error)")
        }
        XCTAssertNil(model.imageSession, "no session was made")
        XCTAssertNil(model.imageSessionPath)
    }

    func testGenerateImageForAMovingModelToastsAndWritesNothingToTheChat() async throws {
        let model = await makeBareModel()
        let target = image(path: "/Users/test/.turbospark/models/image/lg-image")
        model.imageModels = [target]
        model.imageModelPathText = target.path
        model.promptText = "a lighthouse at dusk"
        model.activeMove = moving(target.path)
        let messagesBefore = model.chats.map { $0.messages.count }

        model.generateImage()

        XCTAssertEqual(model.activeToast?.message, movingSentence)
        XCTAssertNil(model.imageGenerationTask, "no job began")
        XCTAssertNil(model.imageJob)
        XCTAssertEqual(
            model.chats.map { $0.messages.count }, messagesBefore,
            "the prompt was not written into a chat that never answers")
        XCTAssertEqual(model.promptText, "a lighthouse at dusk", "and the draft is still there")
    }

    func testTheServerDoesNotTakeAMovingImageModel() async throws {
        let model = await makeBareModel()
        let target = image(path: "/Users/test/.turbospark/models/image/lg-image")
        model.activeMove = moving(target.path)

        model.attachImageModelToServer(target)

        // No server runs here: only the move's refusal toasts.
        XCTAssertEqual(model.activeToast?.message, movingSentence)
        XCTAssertNil(model.serverImageAttachedModel)
    }

    // MARK: Audio

    func testTheServerDoesNotTakeAMovingAudioModel() async throws {
        let model = await makeBareModel()
        model.activeMove = moving("/Users/test/.turbospark/models/audio/lg-audio")

        model.attachAudioModelToServer(path: "/Users/test/.turbospark/models/audio/lg-audio", task: "text_to_speech")

        XCTAssertEqual(model.activeToast?.message, movingSentence)
        XCTAssertTrue(model.serverAudioModelPaths.isEmpty)
    }

    func testTheServerDoesNotTakeAMovingSpeechModel() async throws {
        let model = await makeBareModel()
        let path = "/Users/test/.turbospark/models/audio/lg-speech"
        model.activeMove = moving(path)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let record = try decoder.decode(
            AudioInstalledRecord.self,
            from: Data(
                """
                {"alias":"lg-speech","path":"\(path)","family":"qwen3_tts","status":"runs","install_bytes":300000000}
                """.utf8))

        model.attachAudioSpeechModelToServer(record)

        XCTAssertEqual(model.activeToast?.message, movingSentence)
        XCTAssertNil(model.serverSpeechAttachedModel)
    }

    func testTheAudioSessionIsNeverOpenedFromAModelThatIsMoving() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "lg-audio" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let empty = try decoder.decode(
            AudioInstalledReport.self,
            from: Data(
                #"{"installed":[],"needs_adoption":[],"incompatible":[],"other_audio":[],"not_connected":[]}"#.utf8))
        let controller = AudioWorkspaceController(
            library: AudioLibraryStore(repository: ProfileRepository(store: vault)),
            assets: ManagedAssetStore(vault: vault),
            catalogReader: { AudioCatalogSnapshot(profiles: [], installed: empty, capabilities: []) })
        let profile = AudioWorkspaceController.localQwenProfile
        let path = "/Users/test/.turbospark/models/audio/local-qwen3-asr"
        controller.installedPaths[profile.identity.alias] = path
        controller.modelMoving = { $0 == path }

        do {
            _ = try await controller.residentSession(for: profile)
            XCTFail("a session was opened from a model that is moving")
        } catch let AudioWorkspaceError.message(text) {
            XCTAssertEqual(text, movingSentence)
        }
        XCTAssertNil(controller.session)
        XCTAssertNil(controller.opening, "the open never began")
    }
}
