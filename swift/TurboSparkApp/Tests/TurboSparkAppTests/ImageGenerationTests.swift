import Foundation
import XCTest

import TurboSpark

@testable import TurboSparkApp

@MainActor
final class ImageGenerationTests: XCTestCase {
    func testImageRequestRetainsTheCamelCaseGenerationEnvelope() throws {
        let options = ImageGenerateOptions(
            prompt: "a red kite", seed: 42, width: 1024, height: 1024, steps: 9)
        let encoded = try JSONEncoder().encode(options)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])

        XCTAssertEqual(object["prompt"] as? String, "a red kite")
        XCTAssertEqual(object["seed"] as? UInt64, 42)
        XCTAssertEqual(object["width"] as? UInt32, 1024)
        XCTAssertEqual(object["height"] as? UInt32, 1024)
        XCTAssertEqual(object["steps"] as? UInt32, 9)
    }

    func testSavedImageArtifactRetainsTheRequestForRegeneration() throws {
        let chatID = UUID()
        let request = AppImageRequest(
            options: ImageGenerateOptions(
                prompt: "a red kite", seed: 42, width: 1024, height: 1024, steps: 9))
        let artifact = AppArtifact(
            chatID: chatID,
            path: "/tmp/red-kite.png",
            title: "Generated image",
            origin: .imageGeneration,
            imageRequest: request)

        let roundTrip = try JSONDecoder().decode(
            AppArtifact.self,
            from: JSONEncoder().encode(artifact))

        XCTAssertEqual(roundTrip.imageRequest, request)
        XCTAssertEqual(roundTrip.imageRequest?.options.prompt, "a red kite")
        XCTAssertEqual(roundTrip.imageRequest?.options.seed, 42)
    }

    func testImageSizeComesFromTheSelectedInstall() {
        let model = AppModel()
        model.imageModels = [ImageInstalledModel(
            alias: "z-image-turbo",
            modelID: "Tongyi-MAI/Z-Image-Turbo",
            revision: String(repeating: "a", count: 40),
            path: "/models/z-image-turbo.image.gturbo",
            width: 1024,
            height: 1024,
            schedulerSteps: 9,
            quantization: "mlx-affine-linear-weights-group-64-bits-4")]
        model.selectImageModel(model.imageModels[0])

        XCTAssertEqual(model.imageSupportedSize?.width, 1024)
        XCTAssertEqual(model.imageSupportedSize?.height, 1024)
        XCTAssertEqual(model.imageSizeLabel, "1024 x 1024")
        XCTAssertEqual(model.imageSchedulerSteps, 9)
        XCTAssertEqual(
            model.selectedImageModel?.quantization,
            "mlx-affine-linear-weights-group-64-bits-4")
    }

    func testImageInstallDecodesTheObservedMlxWidth() throws {
        let data = Data(
            "{\"alias\":\"z-image-turbo-mlx-6bit\",\"modelID\":\"andrevp/Z-Image-Turbo-MLX-6bit\",\"revision\":\"rev\",\"path\":\"/models/z.image.gturbo\",\"width\":1024,\"height\":1024,\"schedulerSteps\":9,\"quantization\":\"mlx-affine-linear-weights-group-64-bits-6\"}"
                .utf8)
        let model = try JSONDecoder().decode(ImageInstalledModel.self, from: data)
        XCTAssertEqual(model.quantization, "mlx-affine-linear-weights-group-64-bits-6")
    }

    func testZImageRecommendationsFollowPhysicalMemoryAndPreferTestedMlxRows() {
        let eightGB = AppModel.recommendedZImageAliases(
            physicalMemoryBytes: 8 * 1024 * 1024 * 1024)
        let sixteenGB = AppModel.recommendedZImageAliases(
            physicalMemoryBytes: 16 * 1024 * 1024 * 1024)
        let thirtyTwoGB = AppModel.recommendedZImageAliases(
            physicalMemoryBytes: 32 * 1024 * 1024 * 1024)

        XCTAssertEqual(eightGB, ["z-image-turbo-mlx-2bit"])
        XCTAssertEqual(
            sixteenGB,
            ["z-image-turbo-mlx-4bit", "z-image-turbo-mlx-2bit"])
        XCTAssertEqual(
            thirtyTwoGB,
            [
                "z-image-turbo-mlx-8bit",
                "z-image-turbo-mlx-4bit",
                "z-image-turbo-mlx-2bit",
            ])
        XCTAssertTrue(
            sixteenGB.allSatisfy { AppModel.testedZImageAliases.contains($0) })
    }

    func testInstalledZImageDetectionUsesTheValidatedImageIdentity() {
        let model = AppModel()
        model.imageModels = [ImageInstalledModel(
            alias: "custom-image",
            modelID: "andrevp/Z-Image-Turbo-MLX-4bit",
            revision: "rev",
            path: "/models/custom.image.gturbo",
            width: 1024,
            height: 1024,
            schedulerSteps: 9)]

        XCTAssertTrue(model.hasInstalledZImageModel)
    }

    func testImageJobStartsWaitingAndCarriesTheRequest() {
        let options = ImageGenerateOptions(prompt: "a red kite", seed: 42)
        let chatID = UUID()
        let job = AppImageJob(chatID: chatID, options: options)

        XCTAssertEqual(job.status, .waiting)
        XCTAssertEqual(job.chatID, chatID)
        XCTAssertEqual(job.options, options)
        XCTAssertNil(job.result)
        XCTAssertNil(job.savedPath)
    }

    func testSelectingAnInstalledImageModelUsesItsNativePath() {
        let model = AppModel()
        let image = ImageInstalledModel(
            alias: "z-image-turbo",
            modelID: "Tongyi-MAI/Z-Image-Turbo",
            revision: String(repeating: "a", count: 40),
            path: "/models/z-image-turbo.image.gturbo",
            width: 1024,
            height: 1024,
            schedulerSteps: 9)

        model.selectImageModel(image)

        XCTAssertEqual(model.imageModelPath, image.path)
    }

    func testImageGenerationDoesNotBypassTheBusyAdmissionGate() {
        let chatID = UUID()
        let model = AppModel()
        model.chats = [AppChat(id: chatID, title: "Image")]
        model.selectedChatID = chatID
        model.imageModelPathText = "/tmp/image.image.gturbo"
        model.promptText = "a red kite"
        model.generating = true

        model.generateImage()

        XCTAssertNil(model.imageJob)
        XCTAssertTrue(model.chats[0].messages.isEmpty)
    }

    func testImageRegenerationDoesNotBypassTheBusyAdmissionGate() {
        let chatID = UUID()
        let model = AppModel()
        model.chats = [AppChat(id: chatID, title: "Image")]
        model.selectedChatID = chatID
        model.imageModelPathText = "/tmp/image.image.gturbo"
        model.imageJob = AppImageJob(
            chatID: chatID,
            options: ImageGenerateOptions(prompt: "a red kite", seed: 42),
            status: .completed)
        model.generating = true

        model.regenerateImage()

        XCTAssertNil(model.imageGenerationTask)
    }

    func testImageArtifactIsAnAutoRegisteredCandidate() {
        let file = ArtifactRegistrar.ProducedFile(
            url: URL(fileURLWithPath: "/tmp/generated.png"),
            toolCallID: UUID(),
            toolName: "image_generation",
            origin: .imageGeneration)

        let candidates = ArtifactRegistrar.artifactCandidates(from: [file])

        XCTAssertEqual(candidates, [file])
    }

    func testCancelledImageWaiterDoesNotConsumeTheNextSlot() async {
        let coordinator = ImageJobCoordinator()
        let acquired = await coordinator.acquire()
        XCTAssertTrue(acquired)

        let waiter = Task { await coordinator.acquire() }
        await Task.yield()
        waiter.cancel()
        let cancelled = await waiter.value
        XCTAssertFalse(cancelled)

        await coordinator.release()
        let next = await coordinator.acquire()
        XCTAssertTrue(next)
        await coordinator.release()
    }

    func testSavingAnImageRegistersAProfileScopedArtifactAndTranscriptPath() throws {
        let chatID = UUID()
        let model = AppModel()
        model.chats = [AppChat(id: chatID, title: "Image")]
        model.selectedChatID = chatID
        let metadata = try JSONDecoder().decode(
            GeneratedImageMetadata.self,
            from: Data(
                """
                {"prompt":"a red kite","seed":42,"width":1024,"height":1024,"batch":1,"schedulerSteps":9,"transformerForwards":270,"guidanceScale":0,"modelID":"z-image","modelRevision":"test","componentRevisions":{},"quantization":"int4","scheduler":{"numTrainTimesteps":1000,"shift":3,"timesteps":[],"sigmas":[],"evaluationCount":9,"guidancePolicy":"zero"},"noiseProvenance":"test","engineRevision":"test"}
                """.utf8))
        XCTAssertEqual(metadata.scheduler.evaluationCount, 9)
        let options = ImageGenerateOptions(prompt: "a red kite", seed: 42)
        model.imageJob = AppImageJob(
            chatID: chatID,
            options: options,
            status: .completed,
            result: ImageGenerationResult(
                png: Data([137, 80, 78, 71, 13, 10, 26, 10]), metadata: metadata))

        model.saveImage()

        let reference = try XCTUnwrap(model.imageJob?.savedPath)
        XCTAssertNotNil(ManagedAssetStore.assetID(from: reference))
        let materializedPath = AppStorageRoot.resolveStoredPath(reference)
        XCTAssertTrue(FileManager.default.fileExists(atPath: materializedPath))
        XCTAssertEqual(model.selectedChat.artifacts.last?.origin, .imageGeneration)
        XCTAssertEqual(model.selectedChat.artifacts.last?.path, reference)
        XCTAssertEqual(
            model.selectedChat.messages.last?.imagePaths,
            [reference])
        XCTAssertEqual(AppStorageRoot.resolveStoredPath(
            model.selectedChat.messages.last!.imagePaths[0]), materializedPath)

        let reloaded = AppModel()
        let persisted = try XCTUnwrap(reloaded.chats.first { $0.id == chatID })
        XCTAssertEqual(persisted.messages.last?.imagePaths, model.selectedChat.messages.last?.imagePaths)
        XCTAssertEqual(persisted.artifacts.last?.origin, .imageGeneration)
    }

    func testProfileSwitchWaitsForAnUnsavedImageResult() throws {
        let metadata = try JSONDecoder().decode(
            GeneratedImageMetadata.self,
            from: Data(
                """
                {"prompt":"a red kite","seed":42,"width":1024,"height":1024,"batch":1,"schedulerSteps":9,"transformerForwards":270,"guidanceScale":0,"modelID":"z-image","modelRevision":"test","componentRevisions":{},"quantization":"int4","scheduler":{"numTrainTimesteps":1000,"shift":3,"timesteps":[],"sigmas":[],"evaluationCount":9,"guidancePolicy":"zero"},"noiseProvenance":"test","engineRevision":"test"}
                """.utf8))
        let chatID = UUID()
        let model = AppModel()
        model.imageJob = AppImageJob(
            chatID: chatID,
            options: ImageGenerateOptions(prompt: "a red kite", seed: 42),
            status: .completed,
            result: ImageGenerationResult(png: Data([1]), metadata: metadata))

        XCTAssertFalse(model.canSwitchProfile)

        var savedJob = try XCTUnwrap(model.imageJob)
        savedJob.savedPath = "/tmp/image.png"
        model.imageJob = savedJob
        XCTAssertTrue(model.canSwitchProfile)
    }

    func testDeletingTheImageChatWaitsForAnUnsavedImageResult() throws {
        let metadata = try JSONDecoder().decode(
            GeneratedImageMetadata.self,
            from: Data(
                """
                {"prompt":"a red kite","seed":42,"width":1024,"height":1024,"batch":1,"schedulerSteps":9,"transformerForwards":270,"guidanceScale":0,"modelID":"z-image","modelRevision":"test","componentRevisions":{},"quantization":"int4","scheduler":{"numTrainTimesteps":1000,"shift":3,"timesteps":[],"sigmas":[],"evaluationCount":9,"guidancePolicy":"zero"},"noiseProvenance":"test","engineRevision":"test"}
                """.utf8))
        let chatID = UUID()
        let model = AppModel()
        model.chats = [AppChat(id: chatID, title: "Image")]
        model.selectedChatID = chatID
        model.imageJob = AppImageJob(
            chatID: chatID,
            options: ImageGenerateOptions(prompt: "a red kite", seed: 42),
            status: .completed,
            result: ImageGenerationResult(png: Data([1]), metadata: metadata))

        model.deleteChat(id: chatID)

        XCTAssertTrue(model.chats.contains { $0.id == chatID })
    }

    func testImageModelPresentationFamilyResolvesZImageAliases() {
        XCTAssertEqual(ImageModelPresentation.family("Tongyi-MAI/Z-Image-Turbo"), "Z-Image Turbo")
        XCTAssertEqual(ImageModelPresentation.family("andrevp/z-image-mlx"), "Z-Image Turbo")
        XCTAssertEqual(ImageModelPresentation.family("custom-model"), "custom-model")
    }

    func testReconcileImageSelectionAutoSelectsInstalledModelWhenEmpty() {
        let model = AppModel()
        let image = ImageInstalledModel(
            alias: "z-image-turbo-mlx-4bit",
            modelID: "Tongyi-MAI/Z-Image-Turbo",
            revision: "rev",
            path: "/models/z-image-turbo-mlx-4bit.image.gturbo",
            width: 1024,
            height: 1024,
            schedulerSteps: 9,
            quantization: "4-bit")
        model.imageModels = [image]
        model.imageModelPathText = ""

        model.reconcileImageSelection()

        XCTAssertEqual(model.selectedImageModel?.alias, "z-image-turbo-mlx-4bit")
        XCTAssertEqual(model.imageSupportedSize?.width, 1024)
    }
}
