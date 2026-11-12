import TurboSpark
import XCTest
@testable import TurboSparkApp

/// The music engine refuses empty lyrics and requires the explicit
/// `[instrumental]` token for instrumental output. The workspace treats an
/// untouched lyrics field as instrumental by mapping it to that token at
/// request time; these tests pin the mapping that stands between an empty
/// field and the engine's refusal.
@MainActor
final class AudioWorkspaceMusicLyricsTests: XCTestCase {
    private func musicRequest(lyrics: String) -> AudioRequest {
        var recipe = AudioRecipe()
        recipe.caption = "soft piano ballad"
        recipe.lyrics = lyrics
        return AudioWorkspaceController.request(from: recipe, task: .music)
    }

    func testEmptyLyricsRequestInstrumental() {
        XCTAssertEqual(musicRequest(lyrics: "").lyrics, "[instrumental]")
    }

    func testWhitespaceLyricsRequestInstrumental() {
        XCTAssertEqual(musicRequest(lyrics: " \n\t ").lyrics, "[instrumental]")
    }

    func testWrittenLyricsPassThroughUnchanged() {
        XCTAssertEqual(musicRequest(lyrics: "first verse line").lyrics, "first verse line")
    }

    func testExplicitInstrumentalTokenPassesThrough() {
        XCTAssertEqual(musicRequest(lyrics: "[instrumental]").lyrics, "[instrumental]")
    }

    func testNonMusicTasksKeepEmptyLyrics() {
        XCTAssertEqual(AudioWorkspaceController.request(from: AudioRecipe(), task: .textToSpeech).lyrics, "")
    }

    func testMusicRequestUsesCheckpointStepRecommendationUnlessOverridden() {
        var recipe = AudioRecipe()
        recipe.caption = "soft piano ballad"
        XCTAssertNil(AudioWorkspaceController.request(from: recipe, task: .music).steps)

        recipe.musicStepsOverride = 12
        XCTAssertEqual(AudioWorkspaceController.request(from: recipe, task: .music).steps, 12)

        recipe.musicStepsOverride = nil
        recipe.steps = 14
        XCTAssertEqual(AudioWorkspaceController.request(from: recipe, task: .music).steps, 14)
    }

    func testSamplingRecommendationRecognizesTurboMetadataAndRejectsUnmarkedConfig() {
        let data = Data(#"{"extra":{"turbospark_music3_variant":{"name":"turbo_lora_8step_merged","flow_steps":8,"flow_solver":"euler","flow_guidance":"exact"}},"quantization":{"mode":"affine","bits":4,"group_size":64}}"#.utf8)
        XCTAssertEqual(AudioMusicSamplingRecommendation.fromConfig(data), .init(profile: "Turbo", steps: 8))
        XCTAssertEqual(AudioMusicSamplingRecommendation.fromConfig(Data("{}".utf8)), .fast)
    }

    /// The pinned Turbo `config.json`, which the Rust catalog test checks
    /// against the pin and the Rust profile test parses too, so both sides
    /// recommend 8 steps for the same bytes.
    func testSamplingRecommendationOfThePinnedTurboConfigIsEightSteps() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("crates/audio/tests/fixtures/minimax_music3_turbo_config.json")
        let data = try Data(contentsOf: fixture)
        XCTAssertEqual(AudioMusicSamplingRecommendation.fromConfig(data), .init(profile: "Turbo", steps: 8))
        XCTAssertEqual(AudioWorkspaceController.defaultMusicProfileAlias, "minimax-music3-turbo-4bit-8step")
    }
}
