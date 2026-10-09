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
}
