import Foundation
import XCTest
import TurboSpark
@testable import TurboSparkApp

@MainActor
final class ImageWorkspaceTests: XCTestCase {
    func testSequenceDefaultsAndSeedsRetainTheNativeSize() {
        let model = AppModel()
        XCTAssertEqual(model.imageCount, 1)
        let options = ImageGenerateOptions(prompt: "kite", seed: UInt64.max, width: 768, height: 512, steps: 7)
        let requests = ImageGenerationSequence.requests(options: options, count: 4)
        XCTAssertEqual(requests.map(\.seed), [UInt64.max, 0, 1, 2])
        XCTAssertEqual(requests.map(\.width), [768, 768, 768, 768])
        XCTAssertEqual(requests.map(\.height), [512, 512, 512, 512])
        XCTAssertEqual(requests.map(\.steps), [7, 7, 7, 7])
        XCTAssertEqual(requests.map(\.prompt), ["kite", "kite", "kite", "kite"])
        XCTAssertEqual(ImageGenerationSequence.requests(options: options, count: 0), [options])
        XCTAssertEqual(ImageGenerationSequence.requests(options: options, count: 99).count, 4)
    }

    func testSequenceDoesNotStartAnotherImageAfterFailure() async {
        enum SaveError: Error { case diskFull }
        let requests = ImageGenerationSequence.requests(options: .init(prompt: "kite", seed: 42), count: 3)
        var completed: [UInt64] = []
        do {
            try await ImageGenerationSequence.run(requests) { index, request in
                if index == 1 { throw SaveError.diskFull }
                completed.append(request.seed)
            }
            XCTFail("Save failure must stop the sequence")
        } catch { XCTAssertTrue(error is SaveError) }
        XCTAssertEqual(completed, [42])
    }

    func testSequenceChecksCancellationBetweenCompletedImages() async {
        let requests = ImageGenerationSequence.requests(options: .init(prompt: "kite", seed: 42), count: 3)
        var completed: [UInt64] = []
        let task = Task {
            try await ImageGenerationSequence.run(requests) { _, request in
                completed.append(request.seed)
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        do { try await task.value; XCTFail("Cancellation must stop the sequence") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(completed, [42])
    }

    func testTrashRemovesOnlySelectedImageAndItsTranscriptReference() throws {
        let model = AppModel()
        let chatID = UUID()
        let first = try makeArtifact(chatID: chatID)
        let second = try makeArtifact(chatID: chatID)
        defer { try? FileManager.default.removeItem(atPath: second.path!) }
        var chat = AppChat(id: chatID, title: "Images")
        chat.artifacts = [first, second]
        chat.messages = [AppChatMessage(role: .assistant, content: "images", imagePaths: [
            "image-artifacts/" + URL(fileURLWithPath: first.path!).lastPathComponent,
            second.path!,
        ])]
        model.chats = [chat]
        model.selectedChatID = UUID() // Removal must follow ownership, not the selected chat.
        model.openArtifact(id: first.id)
        var moved: [String] = []
        let removed = model.trashGeneratedImages(ids: [first.id]) {
            moved.append($0.path)
            try FileManager.default.removeItem(at: $0)
        }
        XCTAssertEqual(removed, [first.id])
        XCTAssertEqual(moved, [first.path!])
        XCTAssertEqual(model.chats[0].artifacts.map(\.id), [second.id])
        XCTAssertEqual(model.chats[0].messages[0].imagePaths, [second.path!])
        XCTAssertNil(model.openArtifactID)
        let reloaded = AppModel()
        XCTAssertEqual(reloaded.chats.first { $0.id == chatID }?.artifacts.map(\.id), [second.id])
    }

    func testFailedTrashPreservesImageAndHistory() throws {
        let model = AppModel()
        let image = try makeArtifact(chatID: UUID())
        defer { try? FileManager.default.removeItem(atPath: image.path!) }
        var chat = AppChat(id: image.chatID, title: "Images")
        chat.artifacts = [image]
        model.chats = [chat]
        let removed = model.trashGeneratedImages(ids: [image.id]) { _ in
            throw CocoaError(.fileWriteNoPermission)
        }
        XCTAssertTrue(removed.isEmpty)
        XCTAssertEqual(model.chats[0].artifacts, [image])
        XCTAssertTrue(FileManager.default.fileExists(atPath: image.path!))
    }

    func testMissingImagesStayVisibleAndCanBeRemoved() {
        let model = AppModel()
        let chatID = UUID()
        let image = AppArtifact(chatID: chatID,
            path: AppStorageRoot.subdirectory("image-artifacts").appendingPathComponent("missing-\(UUID()).png").path,
            title: "Missing", origin: .imageGeneration)
        var chat = AppChat(id: chatID, title: "Images")
        chat.artifacts = [image]
        model.chats = [chat]
        XCTAssertEqual(model.savedImageArtifacts, [image])
        XCTAssertEqual(model.trashGeneratedImages(ids: [image.id]) { _ in XCTFail("No file to trash") }, [image.id])
        XCTAssertTrue(model.savedImageArtifacts.isEmpty)
    }

    func testTrashRefusesExternalFilesAndOtherArtifactOrigins() throws {
        let model = AppModel()
        let chatID = UUID()
        var owned = try makeArtifact(chatID: chatID)
        defer { try? FileManager.default.removeItem(atPath: owned.path!) }
        owned.origin = .fileWrite
        let external = AppArtifact(chatID: chatID, path: "/tmp/external.png", title: "External", origin: .imageGeneration)
        var chat = AppChat(id: chatID, title: "Images")
        chat.artifacts = [owned, external]
        model.chats = [chat]
        XCTAssertTrue(model.trashGeneratedImages(ids: [owned.id, external.id]) { _ in
            XCTFail("Must not trash an external or non-image-generation file")
        }.isEmpty)
        XCTAssertEqual(model.chats[0].artifacts, [owned, external])
    }

    func testModelGroupingPreservesFutureModelIdentities() {
        XCTAssertEqual(ImageModelPresentation.family("Tongyi-MAI/Z-Image-Turbo"), "Z-Image Turbo")
        XCTAssertEqual(ImageModelPresentation.family("andrevp/Z-Image-Turbo-MLX-4bit"), "Z-Image Turbo")
        XCTAssertEqual(ImageModelPresentation.family("studio/another-model"), "studio/another-model")
        XCTAssertEqual(ImageModelPresentation.quantization("mlx-affine-linear-weights-group-64-bits-8"), "8-bit")
        XCTAssertEqual(ImageModelPresentation.quantization("custom-format"), "custom-format")
    }

    private func makeArtifact(chatID: UUID) throws -> AppArtifact {
        let file = AppStorageRoot.subdirectory("image-artifacts").appendingPathComponent("\(UUID()).png")
        try Data([1, 2, 3]).write(to: file)
        return AppArtifact(chatID: chatID, path: file.path, title: "Image", origin: .imageGeneration)
    }
}
