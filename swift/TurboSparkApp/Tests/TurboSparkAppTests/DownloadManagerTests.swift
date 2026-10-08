import TurboSpark
import XCTest
@testable import TurboSparkApp

@MainActor
final class DownloadManagerTests: XCTestCase {
    private func makeModel() -> AppModel {
        let model = AppModel()
        // Queue unit tests do not share the signed-in profile's download
        // history or installed image models; a developer machine with a
        // real install of the fixture alias would otherwise take the
        // already-installed path instead of the queue path. Persistence
        // behavior has its own temporary-vault suite.
        model.downloadHistoryWritable = false
        model.modelDownloads = []
        model.activeModelDownloadID = nil
        model.imageModels = []
        return model
    }

    func testPauseAndResumeRequireEngineAcknowledgementAndKeepProgress() {
        let model = makeModel()
        defer { model.stopCronScheduler() }
        model.beginModelDownload(.catalog(alias: "fixture"))
        model.isInstallingModel = true
        model.installProgressFraction = 0.4
        model.installDownloadedBytes = 400
        let id = model.activeModelDownloadID

        model.pauseModelDownload(signal: { false })
        XCTAssertFalse(model.isInstallPaused)
        XCTAssertEqual(model.modelDownloads.first?.status, .running)

        model.pauseModelDownload(signal: { true })
        XCTAssertTrue(model.isInstallPaused)
        XCTAssertTrue(model.isInstallingModel, "pause must continue to own the install slot")
        XCTAssertEqual(model.modelDownloads.first?.status, .paused)
        XCTAssertTrue(
            model.canInstall(alias: "another-model"),
            "a different model may join the queue while this download is paused")

        model.resumeModelDownload(signal: { false })
        XCTAssertTrue(model.isInstallPaused)
        model.resumeModelDownload(signal: { true })
        XCTAssertFalse(model.isInstallPaused)
        XCTAssertEqual(model.activeModelDownloadID, id)
        XCTAssertEqual(model.installDownloadedBytes, 400)
        XCTAssertEqual(model.installProgressFraction, 0.4)
        XCTAssertEqual(model.modelDownloads.first?.status, .running)
    }

    func testLoadingAndFinishedDownloadsCannotPauseAnotherInstall() {
        let model = makeModel()
        defer { model.stopCronScheduler() }
        model.beginModelDownload(.catalog(alias: "fixture"))
        model.isInstallingModel = true
        model.setModelDownloadStatus(.loading)
        model.pauseModelDownload(signal: { XCTFail("loading must not signal the downloader"); return true })
        model.setModelDownloadStatus(.completed)
        model.pauseModelDownload(signal: { XCTFail("completion must not signal another downloader"); return true })
        XCTAssertNil(model.activeModelDownloadID)
        XCTAssertEqual(model.modelDownloads.first?.status, .completed)
    }

    func testNativeStagesExposeDownloadPackingAndVerification() {
        let model = makeModel()
        defer { model.stopCronScheduler() }
        model.beginModelDownload(.image(alias: "fixture"))
        model.isInstallingImageModel = true

        model.recordModelDownloadStage("downloading transformer/model.safetensors")
        XCTAssertEqual(model.modelDownloads.first?.status, .running)
        model.recordModelDownloadStage("packing transformer")
        XCTAssertEqual(model.modelDownloads.first?.status, .packing)
        XCTAssertFalse(model.canPauseModelDownload)
        model.recordModelDownloadStage("[phase] transfer")
        XCTAssertEqual(model.modelDownloads.first?.status, .running)
        model.recordModelDownloadStage("packing transformer")
        model.recordModelDownloadStage("verifying image install")
        XCTAssertEqual(model.modelDownloads.first?.status, .verifying)
    }

    func testCancellationKeepsRetryHistoryAndRetryCanJoinTheQueue() throws {
        let model = makeModel()
        defer { model.stopCronScheduler() }
        model.beginModelDownload(.repository(
            repo: "owner/model@revision", alias: "fixture", file: "weights.gguf", sidecarRepo: "owner/tokenizer"))
        model.installingAlias = "fixture"
        model.isInstallingModel = true
        model.pauseModelDownload(signal: { true })
        model.cancelInstall()

        XCTAssertFalse(model.isInstallPaused)
        XCTAssertTrue(model.isInstallingModel)
        XCTAssertNotNil(model.activeModelDownloadID)
        let stoppingRow = try XCTUnwrap(model.modelDownloads.first)
        XCTAssertEqual(stoppingRow.status, .cancelling)
        XCTAssertFalse(model.canRetryModelDownload(stoppingRow))
        model.finishModelInstallCancellation()
        let row = try XCTUnwrap(model.modelDownloads.first)
        XCTAssertEqual(row.status, .cancelled)
        XCTAssertTrue(model.canRetryModelDownload(row))
        XCTAssertEqual(row.request, .repository(
            repo: "owner/model@revision", alias: "fixture", file: "weights.gguf", sidecarRepo: "owner/tokenizer"))
        model.isInstallingModel = false
        XCTAssertTrue(model.canRetryModelDownload(row))
        model.isInstallingModel = true
        XCTAssertTrue(model.canRetryModelDownload(row))
    }

    func testClearPreservesPausedWorkAndFailureRemainsInspectable() {
        let model = makeModel()
        defer { model.stopCronScheduler() }
        model.beginModelDownload(.catalog(alias: "failed"))
        model.setModelDownloadStatus(.failed, failure: "connection interrupted")
        XCTAssertEqual(model.modelDownloads.first?.failure, "connection interrupted")
        model.beginModelDownload(.catalog(alias: "active"))
        model.isInstallingModel = true
        model.pauseModelDownload(signal: { true })
        let activeID = model.activeModelDownloadID
        model.clearFinishedDownloads()
        XCTAssertEqual(model.modelDownloads.count, 1)
        XCTAssertEqual(model.modelDownloads.first?.id, activeID)
        XCTAssertEqual(model.modelDownloads.first?.status, .paused)
    }

    func testImageDownloadQueuesBehindTextDownloadAndRepeatedClicksAreDeduplicated() throws {
        let model = makeModel()
        defer { model.stopCronScheduler() }
        model.beginModelDownload(.catalog(alias: "qwen36"))
        model.isInstallingModel = true
        let source = try JSONDecoder().decode(ImageCatalogEntry.self, from: Data(
            #"{"alias":"z-image-turbo-mlx-8bit","modelID":"andrevp/Z-Image-Turbo-MLX-8bit","revision":"pinned","quantization":"8-bit"}"#.utf8))

        model.installImageModel(source)
        model.installImageModel(source)

        XCTAssertEqual(model.modelDownloads.count, 2)
        XCTAssertEqual(model.modelDownloads.filter { $0.request.alias == source.alias }.count, 1)
        XCTAssertEqual(model.modelDownloads.first?.request, .image(alias: source.alias))
        XCTAssertEqual(model.modelDownloads.first?.status, .queued)
        XCTAssertEqual(model.modelDownloads.last?.request, .catalog(alias: "qwen36"))
        XCTAssertEqual(model.modelDownloads.last?.status, .running)
        XCTAssertFalse(model.canInstallImageModel(alias: source.alias))
        XCTAssertTrue(model.canInstall(alias: "another-text-model"))
    }

    func testQueueUsesFIFOOrderAcrossModelFamilies() {
        let model = makeModel()
        defer { model.stopCronScheduler() }
        model.beginModelDownload(.catalog(alias: "active"))
        model.isInstallingModel = true

        XCTAssertTrue(model.enqueueModelDownload(.catalog(alias: "first")))
        XCTAssertTrue(model.enqueueModelDownload(.image(alias: "second")))
        XCTAssertFalse(model.enqueueModelDownload(.catalog(alias: "FIRST")))

        let queued = model.modelDownloads.filter { $0.status == .queued }
        XCTAssertEqual(queued.map(\.request.alias), ["second", "first"])
        model.setModelDownloadStatus(.completed)
        model.isInstallingModel = false
        var started: [ModelDownload.Request] = []
        model.startNextModelDownloadIfPossible { started.append($0) }
        XCTAssertEqual(started, [.catalog(alias: "first")])
        XCTAssertEqual(
            model.modelDownloads.first(where: { $0.id == model.activeModelDownloadID })?.status,
            .running)

        model.setModelDownloadStatus(.completed)
        model.startNextModelDownloadIfPossible { started.append($0) }
        XCTAssertEqual(started, [.catalog(alias: "first"), .image(alias: "second")])
    }

    func testQueuedDownloadStartsWhenGenerationEnds() {
        let model = makeModel()
        defer { model.stopCronScheduler() }
        var started: [ModelDownload.Request] = []
        model.modelDownloadStartOverride = { started.append($0) }
        model.generating = true
        model.modelDownloads.insert(
            ModelDownload(request: .catalog(alias: "waiting-for-reply"), status: .queued), at: 0)
        model.startNextModelDownloadIfPossible { _ in XCTFail("must not start while generating") }
        XCTAssertEqual(model.modelDownloads.first?.status, .queued)

        // The reply ending is the only remaining trigger.
        model.generating = false
        XCTAssertEqual(started, [.catalog(alias: "waiting-for-reply")])
    }

    func testRetryIsRefusedOnceTheAliasIsInstalled() {
        let model = makeModel()
        defer { model.stopCronScheduler() }
        var failed = ModelDownload(request: .catalog(alias: "stale-alias"), status: .failed)
        failed.failure = "boom"
        XCTAssertTrue(model.canRetryModelDownload(failed))

        model.installed = [
            InstalledModel(
                alias: "stale-alias", repo: "r/x", path: "/tmp/stale-alias.gturbo", family: "gemma4")
        ]
        XCTAssertFalse(model.canRetryModelDownload(failed))
        model.retryModelDownload(failed)
        XCTAssertTrue(model.modelDownloads.isEmpty)
    }

    func testAudioQueueIdentityIncludesTaskAndEveryPinnedField() throws {
        func request(task: AudioTask = .speechToText, repository: String = "owner/audio",
                     revision: String = "revision-a", fingerprint: String = "assets-a") -> ModelDownload.Request {
            .audio(identity: AudioProfileIdentity(task: task, alias: "fixture", repository: repository,
                                                  revision: revision, assetFingerprint: fingerprint))
        }
        let original = request()
        XCTAssertEqual(original.alias, "fixture")
        let identities = [original, request(task: .alignment), request(repository: "other/audio"),
                          request(revision: "revision-b"), request(fingerprint: "assets-b"),
                          .catalog(alias: "fixture"), .image(alias: "fixture")]
        XCTAssertEqual(Set(identities.map(\.queueKey)).count, identities.count)
        let decoded = try JSONDecoder().decode(ModelDownload.Request.self, from: JSONEncoder().encode(original))
        XCTAssertEqual(decoded, original)
        XCTAssertEqual(decoded.queueKey, original.queueKey)
    }

    func testAudioCancellationKeepsSharedWriterUntilNativeInstallDrains() async throws {
        let model = makeModel()
        defer { model.stopCronScheduler(); model.audioWorkspace.shutdown() }
        let identity = AudioProfileIdentity(task: .speechToText, alias: "audio-fixture",
                                            repository: "owner/audio", revision: "pinned", assetFingerprint: "assets")
        let entered = expectation(description: "native audio writer entered")
        let release = DispatchSemaphore(value: 0)
        var next: [ModelDownload.Request] = []
        model.modelDownloadStartOverride = { next.append($0) }
        model.beginModelDownload(.audio(identity: identity))
        model.startAudioModelInstall(identity: identity) { _, _ in
            entered.fulfill()
            release.wait()
            throw NSError(domain: "cancelled-fixture", code: 1)
        }
        let task = model.installTask
        await fulfillment(of: [entered], timeout: 3)
        model.cancelInstall()
        XCTAssertTrue(model.isInstallingModel)
        XCTAssertTrue(model.audioWorkspace.isInstalling)
        XCTAssertTrue(AppModel.modelInstallOwner === model)
        XCTAssertEqual(model.modelDownloads.first?.status, .cancelling)
        XCTAssertTrue(model.enqueueModelDownload(.image(alias: "next-image")))
        XCTAssertTrue(next.isEmpty, "an image install must wait for the cancelled audio writer to exit")
        release.signal()
        await task?.value
        XCTAssertFalse(model.audioWorkspace.isInstalling)
        XCTAssertFalse(model.isInstallingModel)
        XCTAssertNil(AppModel.modelInstallOwner)
        XCTAssertEqual(model.modelDownloads.first(where: { $0.request == .audio(identity: identity) })?.status, .cancelled)
        XCTAssertEqual(next, [.image(alias: "next-image")])
    }
}
