import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// A single-language speech model is named with its language in the picker
/// and becomes the default for a user whose UI is in that language, while
/// multilingual rows and live-only rows keep the catalog's preference order.
@MainActor
final class AudioModelLanguageTests: XCTestCase {
    private func profile(_ alias: String, family: String = "parakeet_tdt", backend: String = "coreml",
                         languages: [String], operations: [String] = ["transcribe"], canRun: Bool = true) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: .speechToText, alias: alias, repository: "o/\(alias)",
                                           revision: String(repeating: "0", count: 40), assetFingerprint: "f"),
            family: family, displayName: alias,
            capabilities: AudioCapabilities(languages: languages, operations: operations, backend: backend,
                                            cancellation: "window", canRun: canRun),
            pcmFormat: AudioPCMFormat(sampleRate: 16_000, channels: 1), readiness: "qualified")
    }

    private var ultra: AudioProfile { profile("parakeet-ultra-coreml", languages: ["auto", "multilingual"]) }
    private var whisper: AudioProfile { profile("whisper-base", family: "whisper", backend: "metal", languages: ["auto", "multilingual"]) }
    private var japanese: AudioProfile {
        profile("parakeet-ja-coreml", family: FluidAudioFamily.parakeetJa.rawValue, backend: "swift_coreml", languages: ["ja"], canRun: false)
    }
    private var mandarin: AudioProfile {
        profile("paraformer-zh-coreml", family: FluidAudioFamily.paraformerZh.rawValue, backend: "swift_coreml", languages: ["zh"], canRun: false)
    }
    private var liveEnglish: AudioProfile {
        profile("parakeet-unified-en-coreml", family: "parakeet_unified", languages: ["en"], operations: ["stream"])
    }

    func testSingleCodeIgnoresCatalogMarkers() {
        XCTAssertEqual(AudioModelLanguage.singleCode(for: japanese), "ja")
        XCTAssertEqual(AudioModelLanguage.singleCode(for: profile("us", languages: ["en-US"])), "en-US")
        XCTAssertNil(AudioModelLanguage.singleCode(for: ultra), "auto and multilingual are markers, not a language")
        XCTAssertNil(AudioModelLanguage.singleCode(for: profile("none", languages: [])))
        XCTAssertNil(AudioModelLanguage.singleCode(for: profile("two", languages: ["en", "fr"])))
        XCTAssertEqual(AudioModelLanguage.singleCode(for: profile("mixed", languages: ["Auto", "ja"])), "ja")
    }

    func testLanguageNameFollowsTheUILocale() {
        XCTAssertEqual(AudioModelLanguage.name(for: "ja", locale: Locale(identifier: "en")), "Japanese")
        XCTAssertEqual(AudioModelLanguage.name(for: "ja", locale: Locale(identifier: "de")), "Japanisch")
        XCTAssertEqual(AudioModelLanguage.name(for: "ja", locale: Locale(identifier: "ja")), "\u{65E5}\u{672C}\u{8A9E}")
        XCTAssertEqual(AudioModelLanguage.name(for: "zh", locale: Locale(identifier: "en")), "Chinese")
    }

    func testDedicatedRowsNeedOneMatchingLanguageAndAFinalPass() {
        XCTAssertTrue(AudioModelLanguage.isDedicated(japanese, to: "ja"))
        XCTAssertTrue(AudioModelLanguage.isDedicated(mandarin, to: "zh-Hans"), "the script subtag does not matter")
        XCTAssertTrue(AudioModelLanguage.isDedicated(mandarin, to: "zh-Hant"))
        XCTAssertFalse(AudioModelLanguage.isDedicated(japanese, to: "en"))
        XCTAssertFalse(AudioModelLanguage.isDedicated(japanese, to: nil))
        XCTAssertFalse(AudioModelLanguage.isDedicated(ultra, to: "ja"), "a multilingual row is nobody's dedicated model")
        XCTAssertFalse(AudioModelLanguage.isDedicated(liveEnglish, to: "en"), "a live-only row cannot run the final pass")
    }

    func testRankingSteersTheUILanguageToItsModel() {
        let catalog = [whisper, liveEnglish, japanese, mandarin, ultra]
        let aliases = { (profiles: [AudioProfile]) in profiles.map(\.identity.alias) }

        // English keeps the preference order: the English-only row streams
        // live text only, so it never becomes the default.
        XCTAssertEqual(aliases(AudioWorkspaceController.preferredSpeechProfiles(catalog, languageCode: "en")),
                       ["parakeet-ultra-coreml", "whisper-base", "parakeet-unified-en-coreml", "parakeet-ja-coreml", "paraformer-zh-coreml"])
        XCTAssertEqual(aliases(AudioWorkspaceController.preferredSpeechProfiles(catalog, languageCode: nil)),
                       aliases(AudioWorkspaceController.preferredSpeechProfiles(catalog, languageCode: "en")))
        XCTAssertEqual(aliases(AudioWorkspaceController.preferredSpeechProfiles(catalog, languageCode: "ja")),
                       ["parakeet-ja-coreml", "parakeet-ultra-coreml", "whisper-base", "parakeet-unified-en-coreml", "paraformer-zh-coreml"])
        XCTAssertEqual(aliases(AudioWorkspaceController.preferredSpeechProfiles(catalog, languageCode: "zh-Hans")).first,
                       "paraformer-zh-coreml")
    }

    func testUnrunnableDedicatedRowStaysBehindRunnableOnes() {
        let broken = profile("parakeet-ja-coreml", languages: ["ja"], canRun: false)
        XCTAssertEqual(AudioWorkspaceController.preferredSpeechProfiles([broken, ultra], languageCode: "ja").map(\.identity.alias),
                       ["parakeet-ultra-coreml", "parakeet-ja-coreml"], "a model this Mac cannot run is never the default")
    }
}
