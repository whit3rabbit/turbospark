import XCTest
import TurboSpark
@testable import TurboSparkApp

final class ModelStorageDetectionTests: XCTestCase {
    var tempDirectoryURL: URL!

    override func setUp() {
        super.setUp()
        tempDirectoryURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: tempDirectoryURL, withIntermediateDirectories: true)
    }

    override func tearDown() {
        if let temp = tempDirectoryURL {
            try? FileManager.default.removeItem(at: temp)
        }
        super.tearDown()
    }

    func testDefaultStoragePaths() {
        let turboPath = ModelStorageManager.defaultTurboSparkModelsDirectory
        XCTAssertTrue(turboPath.contains(".turbospark/models") || turboPath.contains("models"))

        let lmPath = ModelStorageManager.defaultLMStudioModelsDirectory
        XCTAssertTrue(lmPath.contains(".lmstudio/models"))
    }

    func testPathExpansion() {
        let tildePath = "~/models"
        let expanded = ModelStorageManager.expandPath(tildePath)
        XCTAssertFalse(expanded.hasPrefix("~"))
        XCTAssertTrue(expanded.hasSuffix("/models"))

        let empty = ModelStorageManager.expandPath("")
        XCTAssertEqual(empty, "")
    }

    func testLMStudioDirectoryDetection() {
        let mockLMStudioDir = tempDirectoryURL.appendingPathComponent(".lmstudio/models", isDirectory: true)
        XCTAssertFalse(ModelStorageManager.isLMStudioDirectoryPresent(customPath: mockLMStudioDir.path))

        try? FileManager.default.createDirectory(at: mockLMStudioDir, withIntermediateDirectories: true)
        XCTAssertTrue(ModelStorageManager.isLMStudioDirectoryPresent(customPath: mockLMStudioDir.path))
    }

    func testScanModelsWithoutCopying() throws {
        let mockDir = tempDirectoryURL.appendingPathComponent("lmstudio_models", isDirectory: true)
        try FileManager.default.createDirectory(at: mockDir, withIntermediateDirectories: true)

        // 1. Create a mock .gturbo bundle directory
        let gturboDir = mockDir.appendingPathComponent("qwen-30b.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: gturboDir, withIntermediateDirectories: true)
        let manifestData = """
        {
            "version": 1,
            "family": "qwen3moe",
            "arch": {
                "family": "qwen3moe"
            }
        }
        """.data(using: .utf8)!
        try manifestData.write(to: gturboDir.appendingPathComponent("manifest.json"))
        let dummyWeight = Data(repeating: 0x42, count: 1024)
        try dummyWeight.write(to: gturboDir.appendingPathComponent("layer-00.bin"))

        // 2. Create a mock .gguf file
        let ggufFile = mockDir.appendingPathComponent("gemma-4-9b-it.gguf")
        let ggufDummy = Data(repeating: 0x99, count: 2048)
        try ggufDummy.write(to: ggufFile)

        // Scan the directory
        let discovered = ModelStorageManager.scanModels(in: mockDir.path, sourceTag: "LM Studio")
        XCTAssertEqual(discovered.count, 2)

        let gturboModel = discovered.first(where: { $0.alias == "qwen-30b" })
        XCTAssertNotNil(gturboModel)
        XCTAssertEqual(
            URL(fileURLWithPath: gturboModel!.path).resolvingSymlinksInPath().path,
            gturboDir.resolvingSymlinksInPath().path
        )
        XCTAssertEqual(gturboModel?.family, "qwen3moe")
        XCTAssertTrue((gturboModel?.installBytes ?? 0) >= 1024)

        let ggufModel = discovered.first(where: { $0.alias == "gemma-4-9b-it" })
        XCTAssertNotNil(ggufModel)
        XCTAssertEqual(
            URL(fileURLWithPath: ggufModel!.path).resolvingSymlinksInPath().path,
            ggufFile.resolvingSymlinksInPath().path
        )
        XCTAssertEqual(ggufModel?.family, "gemma4")
        XCTAssertEqual(ggufModel?.installBytes, 2048)
    }

    func testFamilyInference() {
        XCTAssertEqual(ModelStorageManager.inferFamilyFromName("Gemma-4-26B-A4B-Q4"), "gemma4")
        XCTAssertEqual(ModelStorageManager.inferFamilyFromName("Qwen3-30B-A3B-Instruct"), "qwen3moe")
        XCTAssertEqual(ModelStorageManager.inferFamilyFromName("Qwen-2.5-7B-Instruct"), "qwen36")
        XCTAssertEqual(ModelStorageManager.inferFamilyFromName("Mistral-Small-24B"), "mistral")
        XCTAssertEqual(ModelStorageManager.inferFamilyFromName("TinyLlama-1.1B"), "llama")
        XCTAssertEqual(ModelStorageManager.inferFamilyFromName("Ornith-1.5-9B"), "ornith")
        XCTAssertEqual(ModelStorageManager.inferFamilyFromName("unknown-model-format"), "custom")
    }

    func testSettingsPersistenceRoundtrip() throws {
        let settings = MacAppSettings(
            modelsDirectory: "/Volumes/FastSSD/turbospark/models",
            enableLMStudioDetection: true,
            lmStudioDirectory: "/Users/testuser/.lmstudio/models",
            customModelDirectories: ["/Volumes/External/models", "/Users/testuser/gguf"]
        )

        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)

        XCTAssertEqual(decoded.modelsDirectory, "/Volumes/FastSSD/turbospark/models")
        XCTAssertTrue(decoded.enableLMStudioDetection)
        XCTAssertEqual(decoded.lmStudioDirectory, "/Users/testuser/.lmstudio/models")
        XCTAssertEqual(decoded.customModelDirectories, ["/Volumes/External/models", "/Users/testuser/gguf"])
    }
}
