import Foundation
import Metal
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// A shutdown (profile lock, quit) drains the in-flight load's
/// `AudioModelOpening` on the main thread. A Qwen3-TTS load never calls
/// `open`, so these hold it in each slow step and require the drain to return
/// rather than wait for an open that never comes.
@MainActor
final class AudioQwenLoadShutdownTests: XCTestCase {
    private let alias = "qwen3-base"
    /// Long enough that shutdown runs inside the held step. The verdict does
    /// not depend on it: the opening stays set until the load ends either way.
    private let hold = 0.3

    private var profile: AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(
                task: .textToSpeech, alias: alias, repository: "Qwen/Qwen3-TTS-12Hz-1.7B-Base", revision: "main", assetFingerprint: "f"),
            family: "qwen3_tts", displayName: "Qwen3-TTS Base",
            capabilities: AudioCapabilities(
                operations: ["synthesize"], backend: "metal", cancellation: "token", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 24_000, channels: 1), readiness: "qualified")
    }

    private func fixture() throws -> (AudioWorkspaceController, model: URL, root: URL) {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "the load's memory budget needs a Metal device")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root.appendingPathComponent("vault") }, profileIDProvider: { "qwen-shutdown" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let model = root.appendingPathComponent("model", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        // Some weight bytes, so the memory budget check before the load passes.
        try Data(count: 16).write(to: model.appendingPathComponent("model.safetensors"))
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        return (controller, model, root)
    }

    /// Runs `controller.shutdown()` and reports whether its drain hung. A hung
    /// drain is let go from here, so a regression fails the test instead of
    /// hanging the suite.
    private func shutdownHangs(_ controller: AudioWorkspaceController) throws -> Bool {
        let opening = try XCTUnwrap(controller.opening, "the load must still be in flight")
        let returned = DispatchSemaphore(value: 0)
        let hung = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            guard returned.wait(timeout: .now() + 5) == .timedOut else { return }
            hung.signal()
            // `open` ends the barrier the drain waits on; the stop already
            // requested makes it throw at once.
            _ = try? opening.open(
                path: "/missing/audio/model", task: .textToSpeech, portable: false,
                experimentalMetal: false, expectedFamily: "qwen3_tts")
        }
        controller.shutdown()
        returned.signal()
        return hung.wait(timeout: .now()) == .success
    }

    func testShutdownDuringVerificationReturns() async throws {
        let (controller, model, root) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let verifying = expectation(description: "verification started")
        let hold = hold
        controller.qwenLoadDependencies.verify = { _, _, _ in
            verifying.fulfill()
            // A real verification ends within a chunk once the drain cancels its token.
            Thread.sleep(forTimeInterval: hold)
            throw CancellationError()
        }
        controller.qwenLoadDependencies.load = { _ in
            XCTFail("a cancelled verification must not load")
            throw CancellationError()
        }
        let profile = profile, alias = alias
        let load = Task {
            controller.installedPaths[alias] = model.path
            return try await controller.residentQwenSession(for: profile)
        }
        await fulfillment(of: [verifying], timeout: 5)
        XCTAssertFalse(try shutdownHangs(controller), "shutdown waited for an open that never comes")
        if case .success = await load.result { XCTFail("a load stopped by shutdown must not succeed") }
        XCTAssertNil(controller.qwenModel)
    }

    func testShutdownWhileLoadingAVerifiedInstallReturns() async throws {
        let (controller, model, root) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let loading = expectation(description: "load started")
        let hold = hold
        controller.qwenLoadDependencies.verify = { _, _, _ in model.path }
        controller.qwenLoadDependencies.load = { _ in
            loading.fulfill()
            try await Task.sleep(for: .seconds(hold))
            throw CancellationError()
        }
        let profile = profile, alias = alias
        let load = Task {
            controller.installedPaths[alias] = model.path
            return try await controller.residentQwenSession(for: profile)
        }
        await fulfillment(of: [loading], timeout: 5)
        XCTAssertFalse(try shutdownHangs(controller), "shutdown waited for an open that never comes")
        if case .success = await load.result { XCTFail("a load stopped by shutdown must not succeed") }
        XCTAssertNil(controller.qwenModel)
    }

    /// A chosen folder has no catalog install to verify, which once left the
    /// opening's barrier with nothing at all to end it.
    func testShutdownWhileLoadingALocalFolderReturns() async throws {
        let (controller, model, root) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let loading = expectation(description: "load started")
        let hold = hold
        controller.qwenLoadDependencies.verify = { _, _, _ in
            XCTFail("a chosen folder has no catalog install to verify")
            throw CancellationError()
        }
        controller.qwenLoadDependencies.load = { _ in
            loading.fulfill()
            try await Task.sleep(for: .seconds(hold))
            throw CancellationError()
        }
        controller.preferences.localModels[alias] = try model.bookmarkData(
            options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
        let profile = profile
        let load = Task { try await controller.residentQwenSession(for: profile) }
        await fulfillment(of: [loading], timeout: 5)
        XCTAssertFalse(try shutdownHangs(controller), "shutdown waited for an open that never comes")
        if case .success = await load.result { XCTFail("a load stopped by shutdown must not succeed") }
        XCTAssertNil(controller.qwenModel)
    }
}
