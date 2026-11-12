import Foundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

final class MusicMemoryAdvisorTests: XCTestCase {
    func testMusicMemoryAdvisorEvaluateDefaultValues() {
        let profile = MusicMemoryAdvisor.evaluate(for: nil)
        XCTAssertGreaterThan(profile.physicalMemoryBytes, 0)
        XCTAssertGreaterThan(profile.budgetBytes, 0)
        XCTAssertEqual(profile.maxModelContextTokens, 10_240)
        XCTAssertEqual(profile.maxModelFrames, 9_000)
        XCTAssertEqual(profile.frameRate, 25.0)
        XCTAssertGreaterThan(profile.maxSupportedFrames, 0)
        XCTAssertGreaterThan(profile.maxSupportedSeconds, 0.0)
        XCTAssertFalse(profile.tierSummary.isEmpty)
    }

    func testMusicMemoryAdvisorEvaluateWithCustomProfile() throws {
        let json = """
        {
            "identity": {
                "task": "music",
                "alias": "custom-music",
                "repository": "test/music",
                "revision": "main",
                "assetFingerprint": "abc"
            },
            "family": "minimax_music3",
            "displayName": "Custom Music",
            "capabilities": {
                "languages": ["en"],
                "voices": [],
                "operations": ["generate"],
                "backend": "metal",
                "cancellation": "supported",
                "canRun": true,
                "supportsLyrics": true,
                "maxInputSeconds": 180,
                "maxContextTokens": 8192,
                "maxAudioFrames": 4500,
                "frameRateHz": 25
            },
            "pcmFormat": {"sampleRate": 24000, "channels": 2, "interleaved": true},
            "downloadBytes": 1000000,
            "residentMemoryBytes": 8000000000,
            "residentMemoryEvidence": "measured",
            "readiness": "ready",
            "evidence": []
        }
        """
        let audioProfile = try JSONDecoder().decode(AudioProfile.self, from: Data(json.utf8))

        let evaluated = MusicMemoryAdvisor.evaluate(for: audioProfile)
        XCTAssertEqual(evaluated.maxModelContextTokens, 8_192)
        XCTAssertEqual(evaluated.maxModelFrames, 4_500)
        XCTAssertEqual(evaluated.frameRate, 25.0)
        XCTAssertLessThanOrEqual(evaluated.maxSupportedFrames, 4_500)
        XCTAssertLessThanOrEqual(evaluated.maxSupportedSeconds, 180.0)
    }

    func testEstimatePromptTokens() {
        XCTAssertEqual(MusicMemoryAdvisor.estimatePromptTokens(caption: "", lyrics: ""), 0)
        let caption = "A calm acoustic guitar piece with gentle percussion"
        let tokens = MusicMemoryAdvisor.estimatePromptTokens(caption: caption, lyrics: "")
        XCTAssertGreaterThan(tokens, 0)
        XCTAssertEqual(tokens, Int((Double(caption.count) / 3.5).rounded(.up)))
    }

    func testValidateRecipeDurationMustBePositive() {
        var recipe = AudioRecipe()
        recipe.durationSeconds = 0
        let errorZero = MusicMemoryAdvisor.validate(recipe: recipe, profile: nil)
        XCTAssertNotNil(errorZero)
        XCTAssertTrue(errorZero?.contains("greater than 0") == true)

        recipe.durationSeconds = -5
        let errorNegative = MusicMemoryAdvisor.validate(recipe: recipe, profile: nil)
        XCTAssertNotNil(errorNegative)
    }

    func testValidateRecipeEnforcesContextLimit() {
        var recipe = AudioRecipe()
        let mem = MusicMemoryAdvisor.evaluate(for: nil)
        // Set duration well within supported frame limits
        recipe.durationSeconds = 10.0 // 250 frames @ 25 fps
        recipe.music = AudioMusicSettings(contextSize: 300, maxFrames: mem.maxSupportedFrames, frameRate: mem.frameRate)
        // Add prompt tokens that push total (250 frames + tokens) over 300
        recipe.caption = String(repeating: "epic soundtrack ", count: 25) // ~400 chars -> ~115 tokens; 250 + 115 = 365 > 300
        let error = MusicMemoryAdvisor.validate(recipe: recipe, profile: nil)
        XCTAssertNotNil(error)
        XCTAssertTrue(error?.contains("exceeding context size") == true, "Expected context limit error, got: \(String(describing: error))")
    }

    func testValidateRecipeEnforcesMemoryLimitWhenExceedingSupportedFrames() {
        var recipe = AudioRecipe()
        let mem = MusicMemoryAdvisor.evaluate(for: nil)
        recipe.durationSeconds = Double(mem.maxSupportedFrames + 500) / mem.frameRate
        let error = MusicMemoryAdvisor.validate(recipe: recipe, profile: nil)
        XCTAssertNotNil(error)
        XCTAssertTrue(error?.contains("exceeding") == true)
    }

    func testAudioMusicSettingsCodableAndRecipeIntegration() throws {
        let settings = AudioMusicSettings(contextSize: 8_192, maxFrames: 5_000, frameRate: 25.0)
        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(AudioMusicSettings.self, from: data)
        XCTAssertEqual(decoded, settings)

        var recipe = AudioRecipe()
        XCTAssertNil(recipe.music)
        recipe.music = settings
        XCTAssertEqual(recipe.musicSettings.contextSize, 8_192)
        XCTAssertEqual(recipe.musicSettings.maxFrames, 5_000)
        XCTAssertEqual(recipe.musicFrames, 250) // Default 10.0s * 25 fps = 250 frames

        recipe.durationSeconds = 20.0
        XCTAssertEqual(recipe.musicFrames, 500)

        let recipeData = try JSONEncoder().encode(recipe)
        let decodedRecipe = try JSONDecoder().decode(AudioRecipe.self, from: recipeData)
        XCTAssertEqual(decodedRecipe.music, settings)
    }

    @MainActor func testSetMusicContextToMaxIfSupported() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "music-test" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let controller = AudioWorkspaceController(library: library, assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        while controller.isRefreshingModels { try await Task.sleep(for: .milliseconds(10)) }
        controller.profiles = try AudioCatalog.profiles()
        controller.selectPage(.music)

        let advice = controller.musicMemoryAdvice

        // When maximizing context
        controller.setMusicContextToMaxIfSupported()
        if advice.canSupportMaxContext {
            XCTAssertEqual(controller.recipe.durationSeconds, advice.maxSupportedSeconds)
            XCTAssertEqual(controller.recipe.music?.maxFrames, advice.maxSupportedFrames)
        } else {
            // Constrained hardware clamps to max supported
            XCTAssertEqual(controller.recipe.durationSeconds, advice.maxSupportedSeconds)
            XCTAssertEqual(controller.recipe.music?.maxFrames, advice.maxSupportedFrames)
        }
    }
}
