import XCTest
@testable import TurboSparkApp

/// The menu bar panel's status light, Loaded list and memory bar
/// (`MenuBarServerState.swift`). The phase order matters: a running server
/// outranks a stale start failure, a start in flight outranks it too, and a
/// stopped server is gray rather than red.
final class MenuBarServerStateTests: XCTestCase {
    private func phase(
        running: Bool = false, busy: Bool = false, served: Int = 0, infoFailing: Bool = false,
        failure: String? = nil, daemon: String? = nil
    ) -> MenuBarServerPhase {
        MenuBarServerPhase.resolve(
            serverRunning: running, serverBusy: busy, servedModelCount: served,
            infoReadFailing: infoFailing, startFailure: failure, daemonEndpoint: daemon)
    }

    func testStoppedIsOffNotAnError() {
        XCTAssertEqual(phase(), .stopped)
        XCTAssertEqual(phase().level, .off)
    }

    func testRunningWithAModelIsGreen() {
        XCTAssertEqual(phase(running: true, served: 2), .running)
        XCTAssertEqual(phase(running: true, served: 2).level, .ok)
    }

    func testRunningWithNothingServedIsAWarning() {
        XCTAssertEqual(phase(running: true, served: 0), .idle)
        XCTAssertEqual(phase(running: true, served: 0).level, .warning)
    }

    func testBusyWhileRunningIsWorkingNotStopping() {
        // An embedding or audio attach holds serverBusy on a live server.
        XCTAssertEqual(phase(running: true, busy: true, served: 1), .working)
    }

    func testFailingInfoReadIsDegraded() {
        XCTAssertEqual(phase(running: true, served: 1, infoFailing: true), .degraded)
        XCTAssertEqual(phase(running: true, served: 1, infoFailing: true).level, .warning)
    }

    func testStartFailureIsRedUntilTheNextStart() {
        XCTAssertEqual(phase(failure: "port in use"), .failed("port in use"))
        XCTAssertEqual(phase(failure: "port in use").level, .error)
        // A retry in flight replaces the stale failure.
        XCTAssertEqual(phase(busy: true, failure: "port in use"), .starting)
        // And a running server is never shown as failed.
        XCTAssertEqual(phase(running: true, served: 1, failure: "port in use"), .running)
    }

    func testDaemonCountsAsRunningOnlyWithoutAnInAppServer() {
        let endpoint = "http://127.0.0.1:8080/v1"
        XCTAssertEqual(phase(daemon: endpoint), .daemon(endpoint: endpoint))
        XCTAssertEqual(phase(daemon: endpoint).level, .ok)
        XCTAssertEqual(phase(running: true, served: 1, daemon: endpoint), .running)
        XCTAssertEqual(phase(failure: "x", daemon: endpoint), .failed("x"))
    }

    // MARK: - Loaded list

    func testChatSessionIsListedOnceEvenWhenServed() {
        var input = MenuBarSlotInput()
        input.chatSessionName = "gemma4"
        input.chatServedIDs = ["gemma4.gturbo"]
        input.servedText = [
            MenuBarServedModel(id: "gemma4.gturbo", name: "gemma4"),
            MenuBarServedModel(id: "ornith9b.gturbo", name: "ornith9b"),
        ]
        let slots = MenuBarSlots.build(input)
        XCTAssertEqual(slots.map(\.name), ["gemma4", "ornith9b"])
        XCTAssertEqual(slots.map(\.unload), [.chatSession, .detach(id: "ornith9b.gturbo")])
        XCTAssertEqual(Set(slots.map(\.kind)), [.chat])
    }

    func testEveryKindMapsToItsOwnReleasePath() {
        var input = MenuBarSlotInput()
        input.embeddingIDs = ["embeddinggemma-2-coreml"]
        input.audioWorkspaceAlias = "parakeet-unified-en-coreml"
        input.audioIDs = ["whisper-base"]
        input.speechAlias = "kokoro-82m-bf16"
        input.imageSessionName = "z-image-turbo"
        input.typeSafeRunning = true
        let slots = MenuBarSlots.build(input)
        XCTAssertEqual(slots.map(\.kind), [.embedding, .audio, .audio, .audio, .image, .decision])
        XCTAssertEqual(slots.map(\.unload), [
            .detach(id: "embeddinggemma-2-coreml"), .audioWorkspace, .detach(id: "whisper-base"),
            .speechModel, .imageSession, .typeSafe,
        ])
        XCTAssertEqual(slots.last?.name, MenuBarSlots.typeSafeProcessName)
    }

    func testOpenImagePipelineWinsOverItsServedID() {
        var input = MenuBarSlotInput()
        input.imageServedIDs = ["z-image-turbo"]
        XCTAssertEqual(MenuBarSlots.build(input).map(\.unload), [.servedImage])
        input.imageSessionName = "z-image-turbo"
        XCTAssertEqual(MenuBarSlots.build(input).map(\.unload), [.imageSession])
    }

    func testSpeechModelAlreadyInAudioIDsIsNotListedTwice() {
        var input = MenuBarSlotInput()
        input.audioIDs = ["kokoro-82m-bf16"]
        input.speechAlias = "kokoro-82m-bf16"
        XCTAssertEqual(MenuBarSlots.build(input).count, 1)
    }

    func testNothingLoadedIsEmpty() {
        XCTAssertTrue(MenuBarSlots.build(MenuBarSlotInput()).isEmpty)
    }

    func testSlotIDsAreUniqueAcrossKindsWithTheSameName() {
        var input = MenuBarSlotInput()
        input.embeddingIDs = ["shared"]
        input.audioIDs = ["shared"]
        let ids = MenuBarSlots.build(input).map(\.id)
        XCTAssertEqual(Set(ids).count, ids.count)
    }

    // MARK: - Memory

    func testMemoryScalesAgainstPhysicalRAMWithoutABudget() {
        XCTAssertNil(MenuBarMemoryReading.budgetBytes)
        let reading = MenuBarMemoryReading(usedBytes: 9, physicalBytes: 36)
        XCTAssertEqual(reading.capacityBytes, 36)
        XCTAssertEqual(reading.fraction, 0.25, accuracy: 1e-9)
    }

    func testABudgetReplacesPhysicalRAMAndTheBarClamps() {
        let reading = MenuBarMemoryReading(usedBytes: 30, physicalBytes: 36, budgetBytes: 24)
        XCTAssertEqual(reading.capacityBytes, 24)
        XCTAssertEqual(reading.fraction, 1)
        XCTAssertEqual(MenuBarMemoryReading(usedBytes: 1, physicalBytes: 0).fraction, 0)
    }

    func testUptimeIsUnitOnly() {
        XCTAssertEqual(MenuBarUptime.text(42), "42s")
        XCTAssertEqual(MenuBarUptime.text(600), "10m")
        XCTAssertEqual(MenuBarUptime.text(2 * 3600 + 14 * 60), "2h 14m")
    }
}
