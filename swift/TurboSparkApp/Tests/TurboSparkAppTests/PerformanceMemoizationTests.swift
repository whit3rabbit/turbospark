import Foundation
import Testing

@testable import TurboSparkApp

/// Pins the behaviour of the per-render caches added for the review's
/// Performance findings: the pure parse, and invalidation. No timing.
@Suite(.serialized)
struct ManifestFactsCacheTests {
    private func makeInstall(_ json: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mf-cache-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try json.write(to: dir.appendingPathComponent("manifest.json"), atomically: true, encoding: .utf8)
        return dir
    }

    @Test func parseReadsArchFields() {
        let data = Data(#"{"arch":{"trainedContext":4096,"numLayers":12,"hiddenSize":512,"vocabSize":100,"numExperts":8,"topKExperts":2}}"#.utf8)
        let facts = ManifestFacts.parse(data)
        #expect(facts.context == 4096)
        #expect(facts.layers == 12)
        #expect(facts.hidden == 512)
        #expect(facts.vocab == 100)
        #expect(facts.experts == 8)
        #expect(facts.topK == 2)
        #expect(!facts.supportsKvQuant)
    }

    @Test func parseGarbageIsAllNil() {
        #expect(ManifestFacts.parse(Data("not json".utf8)) == ManifestFacts())
        #expect(ManifestFacts.parse(Data("{}".utf8)) == ManifestFacts())
    }

    @Test func cacheReturnsSameValueAndPicksUpRewrite() throws {
        ManifestFactsCache.reset()
        let dir = try makeInstall(#"{"arch":{"numLayers":4}}"#)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ManifestFactsCache.facts(forInstallPath: dir.path).layers == 4)
        #expect(ManifestFactsCache.cachedCount == 1)
        #expect(ManifestFactsCache.facts(forInstallPath: dir.path).layers == 4)
        #expect(ManifestFactsCache.cachedCount == 1)
        // A rewrite with a different size must invalidate the entry.
        try #"{"arch":{"numLayers":40,"hiddenSize":64}}"#
            .write(to: dir.appendingPathComponent("manifest.json"), atomically: true, encoding: .utf8)
        let after = ManifestFactsCache.facts(forInstallPath: dir.path)
        #expect(after.layers == 40)
        #expect(after.hidden == 64)
    }

    @Test func missingManifestIsAllNilAndSeesLaterCreation() throws {
        ManifestFactsCache.reset()
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mf-cache-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ManifestFactsCache.facts(forInstallPath: dir.path) == ManifestFacts())
        try #"{"arch":{"numLayers":7}}"#
            .write(to: dir.appendingPathComponent("manifest.json"), atomically: true, encoding: .utf8)
        #expect(ManifestFactsCache.facts(forInstallPath: dir.path).layers == 7)
    }
}

struct ProfileMigrationIntegrityGateTests {
    @Test func skipsSecondIntegrityCheckWhenNothingWasImported() {
        #expect(!ProfileRepository.needsPostMigrationIntegrityCheck(
            importedFiles: 0, memoryRoots: 0, observationRoots: 0))
        #expect(ProfileRepository.needsPostMigrationIntegrityCheck(
            importedFiles: 1, memoryRoots: 0, observationRoots: 0))
        #expect(ProfileRepository.needsPostMigrationIntegrityCheck(
            importedFiles: 0, memoryRoots: 1, observationRoots: 0))
        #expect(ProfileRepository.needsPostMigrationIntegrityCheck(
            importedFiles: 0, memoryRoots: 0, observationRoots: 2))
    }
}


struct LMStudioCountTests {
    @Test func disabledDetectionSkipsTheScanAndReturnsZero() throws {
        // A directory that would count if scanned must not be walked.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("lms-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let model = dir.appendingPathComponent("pub/repo", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: model.appendingPathComponent("m.gguf"))
        let on = ModelsSettingsPaneView.lmStudioModelCount(path: dir.path, detectionEnabled: true)
        #expect(on == ModelStorageManager.scanModels(in: dir.path, sourceTag: "LM Studio").count)
        #expect(ModelsSettingsPaneView.lmStudioModelCount(path: dir.path, detectionEnabled: false) == 0)
    }
}

struct CustomFolderCountsTests {
    @Test func duplicatePathsDoNotTrapAndAreScannedOnce() {
        var calls: [String] = []
        let counts = CustomModelFoldersSectionView.folderCounts(for: ["/a", "/b", "/a"]) { path in
            calls.append(path)
            return path == "/a" ? 3 : 5
        }
        #expect(counts == ["/a": 3, "/b": 5])
        #expect(calls == ["/a", "/b"])
    }
}

@MainActor
struct OutputConversationPredicateTests {
    @Test func cheapPredicateAgreesWithTheBuiltString() {
        let model = AppModel()
        var chat = AppChat(title: "t")
        model.chats = [chat]
        model.selectedChatID = chat.id
        #expect(model.hasOutputConversationText == !model.outputConversationPlainText.isEmpty)
        #expect(!model.hasOutputConversationText)

        model.outputText = "live"
        #expect(model.hasOutputConversationText)
        #expect(model.hasOutputConversationText == !model.outputConversationPlainText.isEmpty)
        model.outputText = ""

        // An empty-content message still yields a labelled line.
        chat.messages = [AppChatMessage(role: .user, content: "")]
        model.chats = [chat]
        #expect(model.hasOutputConversationText)
        #expect(model.hasOutputConversationText == !model.outputConversationPlainText.isEmpty)
    }
}

@MainActor
struct ToolCallSummaryCacheTests {
    @Test func reusesWhileArgumentsAreEqualAndRecomputesOnChange() {
        ToolCallSummaryCache.reset()
        var call = AppToolCall(name: "write_file", arguments: ["path": "a.txt", "content": "one\ntwo"], category: .fileWrite)
        let first = ToolCallSummaryCache.summary(for: call)
        let again = ToolCallSummaryCache.summary(for: call)
        #expect(ToolCallSummaryCache.computeCount == 1)
        #expect(first.action == again.action && first.target == again.target)
        #expect(first.additions == again.additions)

        // Streaming appends content: must be recomputed, and match the formatter.
        call.arguments["content"] = "one\ntwo\nthree"
        let grown = ToolCallSummaryCache.summary(for: call)
        #expect(ToolCallSummaryCache.computeCount == 2)
        let direct = ToolCallDiffFormatter.summarize(callName: call.name, arguments: call.arguments)
        #expect(grown.additions == direct.additions)
        #expect(grown.target == direct.target)
    }
}
