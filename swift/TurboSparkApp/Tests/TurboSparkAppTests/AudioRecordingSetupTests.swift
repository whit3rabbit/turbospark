import Foundation
import XCTest
@testable import TurboSparkApp

final class AudioRecordingSetupTests: XCTestCase {
    @MainActor private func makeController() throws -> (AudioWorkspaceController, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "recording-setup" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        controller.liveTranscription = false
        return (controller, root)
    }

    @MainActor func testPermissionSetupRejectsDuplicateChecksAndIgnoresLatePermission() async throws {
        let (controller, root) = try makeController()
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        let requested = expectation(description: "permission requested")
        var permission: CheckedContinuation<Bool, Never>?
        var permissionRequests = 0
        controller.recordingSetupDependencies.requestMicrophonePermission = {
            permissionRequests += 1
            return await withCheckedContinuation {
                permission = $0; requested.fulfill()
            }
        }
        controller.recordingSetupDependencies.microphones = { XCTFail("Cancelled setup must not enumerate devices"); return [] }
        controller.prepareRecording()
        let setup = try XCTUnwrap(controller.recordingSetupTask)
        controller.prepareRecording()
        controller.startRecording()
        await fulfillment(of: [requested], timeout: 2)
        XCTAssertEqual(permissionRequests, 1)
        XCTAssertTrue(controller.hasRecordingActivity)
        XCTAssertNotNil(controller.recordingSetupStatus)
        XCTAssertFalse(controller.isRecording)
        XCTAssertTrue(controller.items.isEmpty)
        controller.cancelRecordingSetup()
        XCTAssertFalse(controller.hasRecordingActivity)
        permission?.resume(returning: true)
        await setup.value
        XCTAssertTrue(controller.microphones.isEmpty)
        XCTAssertTrue(controller.items.isEmpty)
        XCTAssertNil(controller.recordingSetupStatus)
    }

    @MainActor func testStopWhilePermissionPendingKeepsInterruptedDraftWithoutStartingCapture() async throws {
        let (controller, root) = try makeController()
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        let requested = expectation(description: "permission requested")
        var permission: CheckedContinuation<Bool, Never>?
        controller.recordingSetupDependencies.requestMicrophonePermission = {
            await withCheckedContinuation { permission = $0; requested.fulfill() }
        }
        controller.recordingSetupDependencies.start = { _, _ in XCTFail("Late permission must not start capture") }
        controller.startRecording()
        let setup = try XCTUnwrap(controller.recordingSetupTask)
        let id = try XCTUnwrap(controller.recordingID)
        await fulfillment(of: [requested], timeout: 2)
        XCTAssertFalse(controller.isRecording)
        XCTAssertEqual(try controller.library.item(id)?.status, .draft)
        controller.pauseRecording(); controller.addMarker()
        XCTAssertFalse(controller.isPaused)
        XCTAssertTrue(try XCTUnwrap(controller.library.item(id)).markers.isEmpty)
        controller.stopRecording()
        let interrupted = try XCTUnwrap(controller.library.item(id))
        XCTAssertEqual(interrupted.status, .interrupted)
        XCTAssertNotNil(interrupted.failure)
        XCTAssertTrue(interrupted.clips.isEmpty)
        XCTAssertNil(controller.recordingID)
        XCTAssertFalse(controller.hasRecordingActivity)
        permission?.resume(returning: true)
        await setup.value
        XCTAssertEqual(try controller.library.item(id)?.status, .interrupted)
        XCTAssertFalse(controller.isRecording)
    }

    @MainActor func testCancellingNativeStartupRejectsLateActivation() async throws {
        let (controller, root) = try makeController()
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        let starting = expectation(description: "native capture starting")
        var completion: CheckedContinuation<Void, Never>?
        controller.recordingSetupDependencies.requestMicrophonePermission = { true }
        controller.recordingSetupDependencies.start = { _, _ in
            await withCheckedContinuation { completion = $0; starting.fulfill() }
        }
        controller.startRecording()
        let setup = try XCTUnwrap(controller.recordingSetupTask)
        let id = try XCTUnwrap(controller.recordingID)
        await fulfillment(of: [starting], timeout: 2)
        XCTAssertFalse(controller.isRecording)
        XCTAssertTrue(controller.hasRecordingActivity)
        controller.cancelRecordingSetup()
        completion?.resume()
        await setup.value
        XCTAssertFalse(controller.hasRecordingActivity)
        XCTAssertNil(controller.capture)
        XCTAssertEqual(try controller.library.item(id)?.status, .interrupted)
        XCTAssertTrue(controller.pendingRefinementIDs.isEmpty)
    }

    @MainActor func testRecordingBeginsOnlyAfterStartupCompletesAndEmptyStopIsInterrupted() async throws {
        let (controller, root) = try makeController()
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        let starting = expectation(description: "native capture starting")
        var completion: CheckedContinuation<Void, Never>?
        controller.recordingSetupDependencies.requestMicrophonePermission = { true }
        controller.recordingSetupDependencies.start = { _, _ in
            await withCheckedContinuation { completion = $0; starting.fulfill() }
        }
        controller.startRecording()
        let setup = try XCTUnwrap(controller.recordingSetupTask)
        let id = try XCTUnwrap(controller.recordingID)
        await fulfillment(of: [starting], timeout: 2)
        XCTAssertFalse(controller.isRecording)
        completion?.resume()
        await setup.value
        XCTAssertTrue(controller.isRecording)
        XCTAssertNil(controller.recordingSetupStatus)
        XCTAssertEqual(try controller.library.item(id)?.status, .recording)
        controller.stopRecording()
        XCTAssertFalse(controller.hasRecordingActivity)
        XCTAssertEqual(try controller.library.item(id)?.status, .interrupted)
        XCTAssertNotNil(try controller.library.item(id)?.failure)
    }

    @MainActor func testDeniedPermissionLeavesRecoverableItemAndVisibleError() async throws {
        let (controller, root) = try makeController()
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        controller.recordingSetupDependencies.requestMicrophonePermission = { false }
        controller.recordingSetupDependencies.start = { _, _ in XCTFail("Denied permission must not start capture") }
        controller.startRecording()
        let setup = try XCTUnwrap(controller.recordingSetupTask)
        let id = try XCTUnwrap(controller.recordingID)
        await setup.value
        XCTAssertFalse(controller.hasRecordingActivity)
        XCTAssertNotNil(controller.error)
        XCTAssertEqual(try controller.library.item(id)?.status, .interrupted)
        XCTAssertNil(controller.capture)
    }
}
