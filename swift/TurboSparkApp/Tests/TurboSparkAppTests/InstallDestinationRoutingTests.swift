import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The engine's answer for a refused or failed library install: the code first,
/// then a sentence the app must never show.
private func engineError(_ code: String, _ sentence: String = "engine sentence") -> TurboSparkError {
    TurboSparkError(code: .invalidArgument, message: "\(code): \(sentence)")
}

/// Task 6.9 (requirements 2.1 to 2.6): an install goes where the user chose, is
/// checked against that library's room, reports a drive lost mid-install in one
/// sentence, and lists the model under the library it went into.
///
/// **EVERY INSTALL GOES THROUGH THE SEAM.** The app suites link the release
/// engine, which cannot be given a library, so these drive `FakeModelLibraryService`
/// and read back the `library:` it was asked for.
@MainActor
final class InstallDestinationRoutingTests: LibraryStateTestCase {
    private let gb: UInt64 = 1_000_000_000

    // MARK: Fixtures

    private func row(
        _ alias: String, library: ModelLibrary? = nil, availability: LibraryAvailability = .available
    ) -> InstalledModel {
        LibraryFixtures.row(alias, library: library, availability: availability)
    }

    private func finishing(_ installed: InstalledModel) -> @Sendable (FakeModelLibraryService.InstallCall)
        -> AsyncThrowingStream<InstallEvent, Error>
    {
        { _ in
            AsyncThrowingStream { continuation in
                continuation.yield(.stage("[phase] transfer"))
                continuation.yield(.bytes(done: 1, total: 2))
                continuation.yield(.finished(installed))
                continuation.finish()
            }
        }
    }

    private func failing(_ error: Error) -> @Sendable (FakeModelLibraryService.InstallCall)
        -> AsyncThrowingStream<InstallEvent, Error>
    {
        { _ in AsyncThrowingStream { $0.finish(throwing: error) } }
    }

    /// A model on the fake engine with This Mac and T7 connected, its listing
    /// landed, and the installed list reading from the fake the way the engine
    /// reads from its registry. The install alias is not in the catalog, so the
    /// fit arithmetic has nothing to say and only the destination decides.
    private func makeInstallModel(
        thisMacFree: UInt64? = 900 * 1_000_000_000, extraLibraries: [ModelLibrary] = []
    ) async -> (AppModel, LibraryState) {
        service.with { $0.libraries.append(contentsOf: extraLibraries) }
        // (the scheduler an AppModel starts is stopped with the test)
        let state = makeState(freeSpace: { url in
            switch url.path {
            case "/Volumes/T7/TurboSpark": return 800_000_000_000
            default: return nil
            }
        })
        await state.refresh()
        let model = await makeModel(using: state)
        // The queue tests do not share the signed-in profile's download history
        // or installed image models: a machine with a real install of a fixture
        // alias would take the already-installed path, and a save would write a
        // test's rows into the real history.
        model.downloadHistoryWritable = false
        model.modelDownloads = []
        model.activeModelDownloadID = nil
        model.imageModels = []
        model.thisMacFreeSpaceReader = { thisMacFree }
        addTeardownBlock { @MainActor in model.stopCronScheduler() }
        let fake = service!
        model.catalogInstalledReader = {
            try fake.installedWithAvailability().filter { $0.availability == .available }
        }
        return (model, state)
    }

    private func runText(_ model: AppModel, alias: String, library: String?) async {
        model.installModel(alias: alias, library: library)
        await model.installTask?.value
    }

    private var imageSource: ImageCatalogEntry {
        get throws {
            try JSONDecoder().decode(
                ImageCatalogEntry.self,
                from: Data(
                    #"{"alias":"z-image-turbo-mlx-8bit","modelID":"andrevp/Z-Image-Turbo-MLX-8bit","revision":"pinned","quantization":"8-bit"}"#
                        .utf8))
        }
    }

    private func imageRow(library: ModelLibrary?) -> ImageInstalledModel {
        ImageInstalledModel(
            alias: "z-image-turbo-mlx-8bit", modelID: "andrevp/Z-Image-Turbo-MLX-8bit", revision: "pinned",
            path: "/Volumes/T7/TurboSpark/image/z-image-turbo-mlx-8bit.gturbo", width: 1024, height: 1024,
            schedulerSteps: 9, quantization: "8-bit", library: library?.id ?? "internal",
            libraryName: library?.name ?? "This Mac",
            installId: "\(library?.id ?? "internal"):image/z-image-turbo-mlx-8bit")
    }

    private func audioProfile(alias: String = "kokoro-fixture", bytes: UInt64 = 2_000_000_000) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(
                task: .textToSpeech, alias: alias, repository: "owner/audio", revision: "pinned",
                assetFingerprint: "assets"),
            family: "kokoro", displayName: alias,
            capabilities: AudioCapabilities(operations: ["synthesize"], backend: "metal", cancellation: "step", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 24_000, channels: 1), downloadBytes: bytes,
            readiness: "qualified")
    }

    private func audioRecord(alias: String, library: ModelLibrary?) throws -> AudioInstalledRecord {
        let json = """
            {"alias":"\(alias)","path":"/Volumes/T7/TurboSpark/audio/\(alias).gturbo","family":"kokoro",
             "status":"runs","installBytes":10,"library":"\(library?.id ?? "internal")",
             "libraryName":"\(library?.name ?? "This Mac")","installId":"\(library?.id ?? "internal"):audio/\(alias)"}
            """
        return try JSONDecoder().decode(AudioInstalledRecord.self, from: Data(json.utf8))
    }

    // MARK: The install targets the chosen library (requirement 2.1)

    func testACatalogInstallGoesToTheChosenLibrary() async {
        let (model, _) = await makeInstallModel()
        service.with { $0.textInstallScript = finishing(row("fixture-a", library: t7)) }

        await runText(model, alias: "fixture-a", library: t7.id)

        XCTAssertEqual(service.installCalls.map(\.kind), [.text])
        XCTAssertEqual(service.installCalls.map(\.alias), ["fixture-a"])
        XCTAssertEqual(service.installCalls.map(\.library), ["t7-id"], "the chosen library reaches the engine")
        XCTAssertEqual(model.modelDownloads.first?.library, "t7-id", "the queue row remembers the destination")
        XCTAssertEqual(model.modelDownloads.first?.status, .completed)
    }

    func testAThisMacInstallPassesNoLibrary() async {
        let (model, _) = await makeInstallModel()
        service.with { $0.textInstallScript = finishing(row("fixture-b")) }

        await runText(model, alias: "fixture-b", library: nil)
        XCTAssertEqual(service.installCalls.map(\.library), [nil], "nil is This Mac")

        service.with { $0.installCalls.removeAll() }
        await runText(model, alias: "fixture-c", library: "internal")
        XCTAssertEqual(
            service.installCalls.map(\.library), [nil],
            "internal is This Mac and reaches the engine as the absence of a destination")
        XCTAssertNil(model.modelDownloads.first?.library)
    }

    func testARepositoryInstallGoesToTheChosenLibrary() async {
        let (model, _) = await makeInstallModel()
        service.with { $0.textInstallScript = finishing(row("my-model", library: t7)) }

        model.installRepo(repo: "owner/model", alias: "my-model", file: "w.gguf", sidecarRepo: nil, library: t7.id)
        await model.installTask?.value

        XCTAssertEqual(service.installCalls.map(\.kind), [.repo])
        XCTAssertEqual(service.installCalls.map(\.repo), ["owner/model"])
        XCTAssertEqual(service.installCalls.map(\.library), ["t7-id"])
        XCTAssertEqual(model.modelDownloads.first?.library, "t7-id")
    }

    func testAnImageInstallGoesToTheChosenLibrary() async throws {
        let (model, _) = await makeInstallModel()
        let installed = imageRow(library: t7)
        service.with {
            $0.imageInstallScript = { _ in
                AsyncThrowingStream { continuation in
                    continuation.yield(.bytes(done: 1, total: 2))
                    continuation.yield(.finished(installed))
                    continuation.finish()
                }
            }
        }

        model.installImageModel(try imageSource, library: t7.id)
        await model.imageInstallTask?.value

        XCTAssertEqual(service.installCalls.map(\.kind), [.image])
        XCTAssertEqual(service.installCalls.map(\.library), ["t7-id"])
        XCTAssertEqual(model.modelDownloads.first?.library, "t7-id")
    }

    func testAnAudioInstallGoesToTheChosenLibraryAndKeepsItsAcknowledgements() async throws {
        let (model, _) = await makeInstallModel()
        let record = try audioRecord(alias: "kokoro-fixture", library: t7)
        service.with { $0.audioInstallScript = { _ in record } }
        defer { model.audioWorkspace.shutdown() }

        XCTAssertTrue(
            model.enqueueAudioDownload(profile: audioProfile(), acknowledged: ["terms-v1"], library: t7.id))
        await model.installTask?.value

        XCTAssertEqual(service.installCalls.map(\.kind), [.audio])
        XCTAssertEqual(service.installCalls.map(\.library), ["t7-id"])
        XCTAssertEqual(service.installCalls.first?.acknowledged, ["terms-v1"], "the acknowledgement flow is untouched")
        XCTAssertEqual(service.installCalls.first?.onMainThread, false)
        XCTAssertEqual(model.modelDownloads.first?.library, "t7-id")
    }

    func testRetryingAnInstallKeepsItsDestination() async {
        let (model, _) = await makeInstallModel()
        service.with { $0.textInstallScript = failing(FakeServiceError.scripted("network down")) }
        await runText(model, alias: "fixture-r", library: t7.id)
        let failed = try? XCTUnwrap(model.modelDownloads.first)
        XCTAssertEqual(failed?.status, .failed)
        XCTAssertEqual(failed?.library, "t7-id")

        service.with {
            $0.installCalls.removeAll()
            $0.textInstallScript = finishing(row("fixture-r", library: t7))
        }
        if let failed { model.retryModelDownload(failed) }
        await model.installTask?.value

        XCTAssertEqual(service.installCalls.map(\.library), ["t7-id"], "a retry does not change where it goes")
    }

    // MARK: Embedding and OCR never choose (requirement 2.5)

    func testAnEmbeddingDownloadNeverCarriesALibrary() async {
        let (model, _) = await makeInstallModel()
        var started: [ModelDownload.Request] = []
        model.modelDownloadStartOverride = { started.append($0) }

        XCTAssertTrue(model.enqueueModelDownload(.embedding(alias: "embedder"), library: t7.id))

        XCTAssertEqual(started, [.embedding(alias: "embedder")])
        XCTAssertNil(model.modelDownloads.first?.library, "embedding and OCR models install only into This Mac")
        XCTAssertEqual(service.installCalls, [], "and the embedding path never touches the library seam")
    }

    func testTheEmbeddingInstallSourceNeverNamesALibrary() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
        for file in ["State/AppModel+Embeddings.swift", "Components/RecommendedEmbeddingModelRow.swift"] {
            let code = MoveStoreRetirementTests.codeWithoutComments(
                try String(contentsOf: sources.appendingPathComponent(file), encoding: .utf8))
            XCTAssertFalse(code.contains("library:"), "\(file) must not pass a destination")
            XCTAssertFalse(code.contains("InstallDestinationPicker"), "\(file) must show no picker")
            XCTAssertFalse(code.contains("installDestinations"), "\(file) must not read the destinations")
        }
    }

    // MARK: Free space of the chosen library (requirement 2.3)

    func testTheLowSpaceDecisionUsesTheChosenLibrarysRoom() async throws {
        // This Mac has plenty, T7 has 800 GB.
        let (model, _) = await makeInstallModel(thisMacFree: 900 * gb)
        let toThisMac = model.installGateDecision(
            verdict: .streams, installBytes: 795 * gb, downloadBytes: nil, library: nil)
        let toDrive = model.installGateDecision(
            verdict: .streams, installBytes: 795 * gb, downloadBytes: nil, library: t7.id)
        XCTAssertEqual(toThisMac, .allowed)
        let reason = try XCTUnwrap(toDrive.reason, "795 GB into 800 GB leaves no headroom on the drive")
        XCTAssertTrue(reason.contains("T7"), reason)
        XCTAssertTrue(reason.contains(MetricFormat.storage(800 * gb)), reason)
    }

    func testTheLowSpaceDecisionForThisMacIgnoresTheDrivesRoom() async {
        // Now the Mac is the small one and the drive is big.
        let (model, _) = await makeInstallModel(thisMacFree: 20 * gb)
        let toThisMac = model.installGateDecision(
            verdict: .streams, installBytes: 18 * gb, downloadBytes: nil, library: nil)
        let toDrive = model.installGateDecision(
            verdict: .streams, installBytes: 18 * gb, downloadBytes: nil, library: t7.id)
        XCTAssertNotNil(toThisMac.reason)
        XCTAssertFalse(toThisMac.isBlocked)
        XCTAssertEqual(toDrive, .allowed, "the boot disk is not the drive")
    }

    func testAnAwayDestinationBlocksTheInstallAndQueuesNothing() async {
        let away = LibraryFixtures.external("away-id", name: "Spare", available: false)
        let (model, _) = await makeInstallModel(extraLibraries: [away])
        service.with { $0.textInstallScript = finishing(row("fixture-x")) }

        model.installModel(alias: "fixture-x", library: away.id)

        XCTAssertEqual(service.installCalls, [], "no request leaves the app")
        XCTAssertTrue(model.modelDownloads.isEmpty, "nothing is queued for a drive that is not there")
        XCTAssertEqual(model.activeToast?.style, .warning)
        XCTAssertTrue(model.activeToast?.message.contains("Spare") ?? false, model.activeToast?.message ?? "")
        XCTAssertNil(model.error, "an unplugged drive is never the app's error")
    }

    func testEveryOtherInstallPathRefusesAnAwayDestinationToo() async throws {
        let away = LibraryFixtures.external("away-id", name: "Spare", available: false)
        let (model, _) = await makeInstallModel(extraLibraries: [away])
        defer { model.audioWorkspace.shutdown() }

        model.installRepo(repo: "o/r", alias: "away-repo", file: nil, sidecarRepo: nil, library: away.id)
        model.installImageModel(try imageSource, library: away.id)
        XCTAssertFalse(model.enqueueAudioDownload(profile: audioProfile(), acknowledged: [], library: away.id))

        XCTAssertEqual(service.installCalls, [])
        XCTAssertTrue(model.modelDownloads.isEmpty)
        XCTAssertNil(model.error)
    }

    func testAudioIsGatedOnTheChosenLibrarysRoomFromTheProfileSize() async throws {
        let (model, _) = await makeInstallModel(thisMacFree: 900 * gb)
        let big = audioProfile(bytes: 795 * gb)
        XCTAssertEqual(model.audioInstallDecision(for: big, library: nil), .allowed)
        let toDrive = model.audioInstallDecision(for: big, library: t7.id)
        XCTAssertNotNil(toDrive.reason)
        XCTAssertFalse(toDrive.isBlocked)
        XCTAssertTrue(toDrive.reason?.contains("T7") ?? false)
        XCTAssertTrue(model.audioInstallDecision(for: big, library: "gone-id").isBlocked)
    }

    // MARK: A drive lost mid-install (requirement 2.4)

    func testADriveLostMidInstallSaysSoAndLeavesNoPartialModel() async throws {
        let (model, state) = await makeInstallModel()
        service.with {
            $0.textInstallScript = failing(
                engineError("library_disconnected", "the drive of the library \"T7\" was disconnected"))
        }
        let listsBefore = service.listCalls
        let installedBefore = service.installedCalls

        await runText(model, alias: "fixture-d", library: t7.id)

        let expected = InstallFailureText.sentence(
            for: engineError("library_disconnected"), alias: "fixture-d", libraryName: "T7", place: nil,
            language: AppStrings.currentLanguage)
        XCTAssertEqual(model.activeToast?.message, expected)
        XCTAssertEqual(model.activeToast?.style, .warning, "a warning toast, not an availability banner")
        XCTAssertTrue(expected?.contains("T7") ?? false)
        XCTAssertNil(model.error, "an unplugged drive never becomes the app's error")
        XCTAssertEqual(model.modelDownloads.first?.status, .failed)
        XCTAssertEqual(model.modelDownloads.first?.failure, expected)
        XCTAssertGreaterThan(service.installedCalls, installedBefore, "the installed list is read again")
        XCTAssertFalse(model.installed.contains { $0.alias == "fixture-d" }, "no partial model on screen")
        XCTAssertFalse(model.isInstallingModel, "the install does not wait for the drive's listing")
        // The library state is read in the background, so the install is over
        // before it lands.
        let listed = await eventually { self.service.listCalls > listsBefore }
        XCTAssertTrue(listed, "the library list is read again")
        let settled = await eventually { !state.allInstalled.contains { $0.alias == "fixture-d" } }
        XCTAssertTrue(settled, "no partial row in the library state either")
    }

    func testAFailedInstallNeverWaitsOnTheLibraryListing() async throws {
        // A drive that was just pulled can hold the engine's listing. The install
        // ends with its failure and frees the install slot while that read is
        // still parked: a later install is not refused because a drive is slow.
        let (model, _) = await makeInstallModel()
        let gate = DispatchSemaphore(value: 0)
        let entered = DispatchSemaphore(value: 0)
        service.with {
            $0.textInstallScript = failing(engineError("library_disconnected"))
            $0.listGate = gate
            $0.listEntered = entered
        }

        model.installModel(alias: "fixture-slow", library: t7.id)
        let task = model.installTask
        // Polled with a deadline rather than awaited, so an install that DOES wait
        // on the listing fails this test instead of hanging it.
        let finished = await eventually(timeout: 3) { !model.isInstallingModel }
        // Let the parked listing finish, whatever happened, so nothing outlives
        // the test. The gate is dropped first: a later listing must not park on a
        // semaphore nobody will signal.
        service.with { $0.listGate = nil }
        gate.signal()
        await task?.value

        XCTAssertTrue(finished, "the install ended while the listing was parked")
        XCTAssertFalse(model.isInstallingModel)
        XCTAssertNil(AppModel.modelInstallOwner, "the install slot is free")
        XCTAssertEqual(model.modelDownloads.first?.status, .failed)
        XCTAssertNil(model.error)
        let arrived = await eventually { entered.wait(timeout: .now()) == .success }
        XCTAssertTrue(arrived, "the library state was read in the background")
    }

    func testEachInstallCodeHasItsOwnSentenceAndNoneIsTheAppError() async throws {
        let codes = [
            "library_disconnected", "not_connected", "library_read_only", "unknown_library",
            "alias_elsewhere", "busy",
        ]
        var seen: Set<String> = []
        for code in codes {
            let (model, _) = await makeInstallModel()
            service.with {
                $0.installCalls.removeAll()
                $0.textInstallScript = failing(engineError(code))
            }
            await runText(model, alias: "fixture-\(code)", library: t7.id)

            let message = try XCTUnwrap(model.activeToast?.message, code)
            XCTAssertEqual(model.activeToast?.style, .warning, code)
            XCTAssertNil(model.error, "\(code): never model.error")
            XCTAssertFalse(message.contains("engine sentence"), "\(code): the engine's English is never shown")
            XCTAssertFalse(message.contains(code), "\(code): the code is not the message")
            seen.insert(message)
        }
        XCTAssertEqual(seen.count, codes.count, "each code reads differently")
    }

    func testAnOrdinaryFailureKeepsItsOldSurface() async {
        let (model, _) = await makeInstallModel()
        service.with { $0.textInstallScript = failing(FakeServiceError.scripted("network down")) }

        await runText(model, alias: "fixture-n", library: nil)

        XCTAssertEqual(model.modelDownloads.first?.status, .failed)
        XCTAssertNotNil(model.error, "only library faults moved off the app error")
        XCTAssertEqual(model.activeToast?.style, .error)
    }

    func testARepositoryInstallReportsADriveLostTheSameWay() async {
        let (model, _) = await makeInstallModel()
        service.with { $0.textInstallScript = failing(engineError("library_disconnected")) }

        model.installRepo(repo: "o/r", alias: "repo-d", file: nil, sidecarRepo: nil, library: t7.id)
        await model.installTask?.value

        XCTAssertEqual(model.activeToast?.style, .warning)
        XCTAssertTrue(model.activeToast?.message.contains("T7") ?? false)
        XCTAssertNil(model.error)
        XCTAssertEqual(model.modelDownloads.first?.status, .failed)
    }

    func testAnImageInstallReportsADriveLostTheSameWay() async throws {
        let (model, _) = await makeInstallModel()
        service.with { $0.imageInstallScript = { _ in
            AsyncThrowingStream { $0.finish(throwing: engineError("library_disconnected")) }
        } }

        model.installImageModel(try imageSource, library: t7.id)
        await model.imageInstallTask?.value

        XCTAssertEqual(model.activeToast?.style, .warning)
        XCTAssertTrue(model.activeToast?.message.contains("T7") ?? false)
        XCTAssertNil(model.error)
        XCTAssertEqual(model.modelDownloads.first?.status, .failed)
    }

    func testAnAudioInstallReportsADriveLostTheSameWay() async {
        let (model, _) = await makeInstallModel()
        service.with { $0.audioInstallScript = { _ in throw engineError("library_disconnected") } }
        defer { model.audioWorkspace.shutdown() }

        XCTAssertTrue(model.enqueueAudioDownload(profile: audioProfile(), acknowledged: [], library: t7.id))
        await model.installTask?.value

        XCTAssertEqual(model.activeToast?.style, .warning)
        XCTAssertTrue(model.activeToast?.message.contains("T7") ?? false)
        XCTAssertNil(model.error)
        XCTAssertNil(model.audioWorkspace.error, "the audio pane's error is for real failures too")
        XCTAssertEqual(model.modelDownloads.first?.status, .failed)
    }

    // MARK: Listed under the library it went into (requirement 2.6)

    func testACompletedInstallIsListedUnderItsLibrary() async throws {
        let (model, state) = await makeInstallModel()
        model.startInstalledListFollowing()
        let installed = row("fixture-ok", library: t7)
        let fake = service!
        service.with {
            $0.textInstallScript = { _ in
                AsyncThrowingStream { continuation in
                    // The engine records the row before it reports the end.
                    fake.with { $0.installed.append(installed) }
                    continuation.yield(.finished(installed))
                    continuation.finish()
                }
            }
        }
        let listsBefore = service.listCalls

        await runText(model, alias: "fixture-ok", library: t7.id)

        let read = await eventually { self.service.listCalls > listsBefore }
        XCTAssertTrue(read, "the library state is read again")
        let landed = await eventually { state.allInstalled.contains { $0.alias == "fixture-ok" } }
        XCTAssertTrue(landed)
        let listed = try XCTUnwrap(state.allInstalled.first { $0.alias == "fixture-ok" })
        XCTAssertEqual(listed.library, "t7-id")
        XCTAssertEqual(listed.libraryName, "T7")
        XCTAssertEqual(listed.installId, "t7-id:text/fixture-ok")
        let appeared = await eventually {
            model.installed.contains { $0.alias == "fixture-ok" && $0.library == "t7-id" }
        }
        XCTAssertTrue(appeared, "the installed list shows it under the library")
    }

    func testACompletedImageInstallRefreshesTheLibraryState() async throws {
        let (model, _) = await makeInstallModel()
        let installed = imageRow(library: t7)
        service.with {
            $0.imageInstallScript = { _ in
                AsyncThrowingStream { continuation in
                    continuation.yield(.finished(installed))
                    continuation.finish()
                }
            }
        }
        let listsBefore = service.listCalls

        model.installImageModel(try imageSource, library: t7.id)
        await model.imageInstallTask?.value

        let read = await eventually { self.service.listCalls > listsBefore }
        XCTAssertTrue(read, "the library state is read again")
        XCTAssertEqual(model.modelDownloads.first?.status, .completed)
    }

    // MARK: No picker work in the entry points that have no choice

    func testTheIntegrationHasNoDirectEngineInstallCallsLeftInTheApp() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
        var offenders: [String] = []
        let walker = FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil)
        while let url = walker?.nextObject() as? URL {
            guard url.pathExtension == "swift" else { continue }
            let relative = String(url.path.dropFirst(sources.path.count + 1))
            // The seam is the one door, and the embedding install is home-only.
            if relative == "State/ModelLibraryService.swift" || relative == "State/AppModel+Embeddings.swift" {
                continue
            }
            let code = MoveStoreRetirementTests.codeWithoutComments(
                try String(contentsOf: url, encoding: .utf8))
            for call in ["TurboSparkCatalog.install(", "TurboSparkCatalog.installImage(", "AudioCatalog.install("] {
                if code.contains(call) { offenders.append("\(relative): \(call)") }
            }
            if code.contains("stream: TurboSparkCatalog.install") {
                offenders.append("\(relative): stream: TurboSparkCatalog.install")
            }
        }
        XCTAssertEqual(offenders, [], "an install that skips the seam cannot be given a destination")
    }
}
