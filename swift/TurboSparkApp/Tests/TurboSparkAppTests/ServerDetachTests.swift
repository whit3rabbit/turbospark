import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// A scripted engine: the three calls a detach attempt makes, with refusals the
/// way the engine refuses a model that has a request in flight.
@MainActor
private final class ScriptedServer {
    struct Refused: Error, LocalizedError {
        var errorDescription: String? { "the model is busy" }
    }

    /// Ids the engine lists as attached audio models.
    var held: Set<String>
    /// Detach calls refused before one works. `Int.max` never works.
    var refusals = 0
    /// The engine's read throws (the server cannot answer).
    var infoUnreadable = false
    private(set) var detachAttempts = 0

    init(held: Set<String>) { self.held = held }

    var engine: ServerDetachEngine {
        ServerDetachEngine(
            detach: { [self] in try detach($0) },
            detachAudio: { [self] in try detach($0) },
            info: { [self] in
                if infoUnreadable { throw Refused() }
                return try ScriptedServer.makeInfo(audio: held.sorted())
            })
    }

    private func detach(_ id: String) throws {
        detachAttempts += 1
        if detachAttempts <= refusals { throw Refused() }
        // Nothing attached under that id: the engine says so by throwing.
        guard held.remove(id) != nil else { throw Refused() }
    }

    static func makeInfo(audio: [String]) throws -> ServerInfo {
        let names = String(decoding: try JSONEncoder().encode(audio), as: UTF8.self)
        let json = """
            {"port":1,"host":"127.0.0.1","modelId":"","models":[],"imageModels":[],\
            "audioModels":\(names),"authEnabled":false}
            """
        return try JSONDecoder().decode(ServerInfo.self, from: Data(json.utf8))
    }
}

/// Second fix pass, finding 2 of tasks 5.11 and 5.12: the LIVE detach paths
/// (`attemptServerDetach`, `detachServerModels`) are tested, not copies of them.
///
/// A served text model cannot be built without a checkpoint, so these use
/// audio ids (`serverAudioModelPaths`), which go through the same
/// `attemptServerDetach` decision and the same `releaseServerAttachment` that
/// drops both the text and the audio record.
final class ServerDetachTests: LibraryStateTestCase {
    private typealias Fx = EjectFixtures
    private let path = EjectFixtures.server.path

    private func makeServerModel(ids: [String]) async -> AppModel {
        let model = await makeModel(using: makeState())
        for id in ids { model.serverAudioModelPaths[id] = path }
        return model
    }

    // MARK: - attemptServerDetach, with the engine injected

    func testADetachThatWorksReleasesTheRecord() async {
        let model = await makeServerModel(ids: ["tts"])
        let script = ScriptedServer(held: ["tts"])

        let outcome = model.attemptServerDetach(id: "tts", engine: script.engine)
        XCTAssertEqual(outcome, .detached)
        XCTAssertNil(model.serverAudioModelPaths["tts"])
        XCTAssertTrue(script.held.isEmpty)
    }

    func testAnEngineThatNoLongerListsTheModelCountsAsGoneAndReleasesTheRecord() async {
        let model = await makeServerModel(ids: ["ghost"])
        let script = ScriptedServer(held: [])

        let outcome = model.attemptServerDetach(id: "ghost", engine: script.engine)
        guard case .alreadyGone = outcome else { return XCTFail("expected alreadyGone, got \(outcome)") }
        XCTAssertTrue(outcome.releasedTheModel)
        XCTAssertNil(model.serverAudioModelPaths["ghost"], "state#53: nothing is attached, so the record goes")
    }

    func testARefusedDetachOfAModelTheEngineStillHoldsKeepsTheRecord() async {
        let model = await makeServerModel(ids: ["tts"])
        let script = ScriptedServer(held: ["tts"])
        script.refusals = Int.max

        let outcome = model.attemptServerDetach(id: "tts", engine: script.engine)
        guard case .stillServed(let reason) = outcome else { return XCTFail("expected stillServed, got \(outcome)") }
        XCTAssertEqual(reason, "the model is busy")
        XCTAssertFalse(outcome.releasedTheModel)
        XCTAssertEqual(model.serverAudioModelPaths["tts"], path, "the app's record is the only sign it is resident")
    }

    func testARefusedDetachFollowedByAnUnreadableServerKeepsTheRecord() async {
        let model = await makeServerModel(ids: ["tts"])
        let script = ScriptedServer(held: ["tts"])
        script.refusals = Int.max
        script.infoUnreadable = true

        let outcome = model.attemptServerDetach(id: "tts", engine: script.engine)
        guard case .stillServed = outcome else { return XCTFail("expected stillServed, got \(outcome)") }
        XCTAssertEqual(
            model.serverAudioModelPaths["tts"], path,
            "a read that failed is not proof the engine let go")
    }

    func testWithNoServerAtAllTheRecordIsReleased() async {
        let model = await makeServerModel(ids: ["tts"])
        XCTAssertNil(model.server)
        XCTAssertEqual(model.attemptServerDetach(id: "tts"), .detached, "stopping the server released everything")
        XCTAssertNil(model.serverAudioModelPaths["tts"])
    }

    func testTheEngineStillHoldsAModelItListsAsTextOrAudio() throws {
        let reading = try ScriptedServer.makeInfo(audio: ["tts"])
        XCTAssertTrue(AppModel.engineStillHolds("tts", info: reading))
        XCTAssertFalse(AppModel.engineStillHolds("other", info: reading))
        XCTAssertTrue(AppModel.engineStillHolds("x", info: nil), "no reading is not proof of absence")
    }

    // MARK: - The real, empty server

    func testARealEmptyServerSaysAGhostIsAlreadyGone() async throws {
        let server = try TurboSparkServer.start(options: ServerOptions(port: 0))
        defer { server.stop() }
        let model = await makeServerModel(ids: ["ghost"])
        model.server = server
        defer { model.server = nil }

        let started = Date()
        let outcome = model.attemptServerDetach(id: "ghost")
        guard case .alreadyGone = outcome else { return XCTFail("expected alreadyGone, got \(outcome)") }
        XCTAssertNil(model.serverAudioModelPaths["ghost"])
        XCTAssertLessThan(Date().timeIntervalSince(started), 2)
    }

    func testTheLiveDetachOfAGhostOnARealServerReleasesItsRecordWithoutWaiting() async throws {
        let server = try TurboSparkServer.start(options: ServerOptions(port: 0))
        defer { server.stop() }
        let model = await makeServerModel(ids: ["ghost"])
        model.server = server
        defer { model.server = nil }

        let started = Date()
        await model.detachServerModels(atPaths: [AppModel.normalizedModelPath(path)], patience: .seconds(3))
        XCTAssertNil(model.serverAudioModelPaths["ghost"])
        XCTAssertLessThan(Date().timeIntervalSince(started), 2, "an engine that answers 'not attached' ends the wait")
    }

    // MARK: - The live retry (detachServerModels)

    func testTheLiveRetryDetachesAModelThatIsRefusedTwice() async {
        let model = await makeServerModel(ids: ["tts"])
        let script = ScriptedServer(held: ["tts"])
        script.refusals = 2

        await model.detachServerModels(
            atPaths: [AppModel.normalizedModelPath(path)], patience: .seconds(3), engine: script.engine)
        XCTAssertEqual(script.detachAttempts, 3, "two refusals, then the detach that worked")
        XCTAssertNil(model.serverAudioModelPaths["tts"])
    }

    func testTheLiveRetryIsBoundedAndKeepsAModelThatNeverDetaches() async {
        let model = await makeServerModel(ids: ["tts"])
        let script = ScriptedServer(held: ["tts"])
        script.refusals = Int.max

        let started = Date()
        await model.detachServerModels(
            atPaths: [AppModel.normalizedModelPath(path)], patience: .milliseconds(350), engine: script.engine)
        XCTAssertGreaterThan(script.detachAttempts, 1, "it retried")
        XCTAssertLessThan(script.detachAttempts, 30, "and the retry is bounded")
        XCTAssertLessThan(Date().timeIntervalSince(started), 3)
        XCTAssertEqual(model.serverAudioModelPaths["tts"], path, "still tracked")
    }

    func testTheLiveRetryLeavesModelsOnOtherPathsAlone() async {
        let model = await makeServerModel(ids: ["tts"])
        model.serverAudioModelPaths["other"] = Fx.onSpare.path
        let script = ScriptedServer(held: ["tts", "other"])

        await model.detachServerModels(
            atPaths: [AppModel.normalizedModelPath(path)], patience: .seconds(1), engine: script.engine)
        XCTAssertNil(model.serverAudioModelPaths["tts"])
        XCTAssertEqual(model.serverAudioModelPaths["other"], Fx.onSpare.path)
        XCTAssertEqual(script.held, ["other"])
    }

    // MARK: - Through the coordinator, over the live source

    private func makeCoordinator(
        _ model: AppModel, script: ScriptedServer, patience: Duration
    ) async -> (LibraryEjectCoordinator, FakeEjector, NoticeRecorder, AppModelLoadedModels) {
        let log = EjectLog()
        let source = AppModelLoadedModels(
            model: model, patience: patience, seams: .init(serverEngine: script.engine))
        let ejector = FakeEjector(log: log)
        let notices = NoticeRecorder(log: log)
        let coordinator = LibraryEjectCoordinator(
            source: source, ejector: ejector, locator: FakeVolumeLocator(), present: { notices.add($0) })
        await coordinator.refreshSnapshot()
        return (coordinator, ejector, notices, source)
    }

    func testAServerModelRefusedTwiceIsUnloadedByTheLiveRetryAndEjectedOnce() async {
        let model = await makeServerModel(ids: ["tts"])
        let script = ScriptedServer(held: ["tts"])
        script.refusals = 2
        let (coordinator, ejector, notices, _) = await makeCoordinator(model, script: script, patience: .seconds(3))
        XCTAssertEqual(coordinator.decisions.snapshot.map(\.model.path), [path])

        XCTAssertNotNil(coordinator.unmountRequested(for: Fx.t7Disk))
        await coordinator.waitForIdle(timeout: 10)

        XCTAssertEqual(script.detachAttempts, 3)
        XCTAssertTrue(model.serverAudioModelPaths.isEmpty)
        XCTAssertEqual(ejector.calls, [Fx.t7Disk], "one eject")
        XCTAssertEqual(notices.all, [.ejected(models: ["gemma"])])
    }

    func testAServerModelThatNeverDetachesIsNotUnloadedAndTheDriveIsNotEjected() async {
        let model = await makeServerModel(ids: ["tts"])
        let script = ScriptedServer(held: ["tts"])
        script.refusals = Int.max
        let (coordinator, ejector, notices, source) = await makeCoordinator(
            model, script: script, patience: .milliseconds(350))

        XCTAssertNotNil(coordinator.unmountRequested(for: Fx.t7Disk))
        await coordinator.waitForIdle(timeout: 10)

        XCTAssertGreaterThan(script.detachAttempts, 1)
        XCTAssertTrue(ejector.calls.isEmpty, "no eject while the engine still holds the model")
        XCTAssertEqual(notices.all, [.unloadBlocked(models: ["gemma"])])
        XCTAssertEqual(coordinator.decisions.snapshot.map(\.model.path), [path], "still in the snapshot")
        XCTAssertEqual(source.loadedModels().map(\.path), [path], "and still listed as loaded")
    }
}
