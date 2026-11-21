import XCTest
@testable import TurboSparkApp

/// The composer sampling pill's pure half: which tone (or saved preset, or
/// Custom) a snapshot reads as, and what applying a tone writes.
final class SamplingToneTests: XCTestCase {
    // MARK: - Tone values

    /// Balanced is the app's own shipped default, so a fresh install reads
    /// "Balanced" rather than "Custom".
    func testFreshDefaultsReadAsBalanced() {
        XCTAssertEqual(SamplingToneMatcher.match(AppSamplingSettings(), savedPresets: []), .tone(.balanced))
    }

    /// Temperature orders the tones, and every value sits inside the range
    /// the Inspector's controls (and `clamped()`) allow, so applying never
    /// gets silently rewritten by the clamp.
    func testToneValuesAreOrderedAndInRange() {
        let t = SamplingTone.allCases.map(\.values.temperature)
        XCTAssertEqual(t, t.sorted())
        XCTAssertEqual(Set(t).count, t.count)
        for tone in SamplingTone.allCases {
            let applied = SamplingToneMatcher.applying(tone, to: AppSamplingSettings())
            XCTAssertEqual(applied, applied.clamped(), "\(tone) is outside the clamp range")
            // Greedy (0) skips repetition penalty and truncation entirely
            // in the sampler, which is why no tone uses it.
            XCTAssertGreaterThan(tone.values.temperature, 0)
        }
    }

    // MARK: - Applying

    func testApplyingSetsOnlyTheToneKnobs() {
        var base = AppSamplingSettings()
        base.maxNewTokens = 4096
        base.repetitionPenaltyEnabled = true
        base.repetitionPenalty = 1.15
        base.seedEnabled = true
        base.seed = 7
        base.stopSequences = "END"
        base.topKEnabled = false
        base.topPEnabled = false

        let applied = SamplingToneMatcher.applying(.creative, to: base)
        XCTAssertEqual(applied.temperature, SamplingTone.creative.values.temperature)
        XCTAssertTrue(applied.topKEnabled)
        XCTAssertEqual(applied.topK, SamplingTone.creative.values.topK)
        XCTAssertTrue(applied.topPEnabled)
        XCTAssertEqual(applied.topP, SamplingTone.creative.values.topP)
        // Untouched: length, penalty, seed, stops belong to the user.
        XCTAssertEqual(applied.maxNewTokens, 4096)
        XCTAssertTrue(applied.repetitionPenaltyEnabled)
        XCTAssertEqual(applied.repetitionPenalty, 1.15)
        XCTAssertTrue(applied.seedEnabled)
        XCTAssertEqual(applied.seed, 7)
        XCTAssertEqual(applied.stopSequences, "END")
    }

    func testEveryToneRoundTripsThroughMatch() {
        for tone in SamplingTone.allCases {
            let applied = SamplingToneMatcher.applying(tone, to: AppSamplingSettings())
            XCTAssertEqual(SamplingToneMatcher.match(applied, savedPresets: []), .tone(tone))
        }
    }

    // MARK: - Matching tolerance

    /// A value that went through a slider (0.05 / 0.01 steps) or JSON can
    /// drift in the last bits; that must still read as the tone.
    func testMatchToleratesFloatDrift() {
        var s = SamplingToneMatcher.applying(.precise, to: AppSamplingSettings())
        s.temperature += 0.000_000_1
        s.topP -= 0.000_000_1
        XCTAssertEqual(SamplingToneMatcher.match(s, savedPresets: []), .tone(.precise))
    }

    /// One slider step away is a real edit, not drift.
    func testOneSliderStepIsCustom() {
        var s = AppSamplingSettings()
        s.temperature += 0.05
        XCTAssertEqual(SamplingToneMatcher.match(s, savedPresets: []), .custom)

        var p = AppSamplingSettings()
        p.topP -= 0.01
        XCTAssertEqual(SamplingToneMatcher.match(p, savedPresets: []), .custom)
    }

    func testDifferentTopKIsCustom() {
        var s = AppSamplingSettings()
        s.topK += 1
        XCTAssertEqual(SamplingToneMatcher.match(s, savedPresets: []), .custom)
    }

    /// With a truncation toggle off the engine skips that knob, so the
    /// snapshot does not behave like the tone even with matching numbers.
    func testDisabledTruncationIsCustom() {
        var k = AppSamplingSettings()
        k.topKEnabled = false
        XCTAssertEqual(SamplingToneMatcher.match(k, savedPresets: []), .custom)

        var p = AppSamplingSettings()
        p.topPEnabled = false
        XCTAssertEqual(SamplingToneMatcher.match(p, savedPresets: []), .custom)
    }

    /// Knobs a tone does not set do not change which tone it reads as.
    func testNonToneKnobsDoNotBreakTheMatch() {
        var s = AppSamplingSettings()
        s.maxNewTokens = 8192
        s.repetitionPenaltyEnabled = true
        s.repetitionPenalty = 1.2
        s.stopSequences = "###"
        XCTAssertEqual(SamplingToneMatcher.match(s, savedPresets: []), .tone(.balanced))
    }

    // MARK: - Saved presets

    func testSavedPresetMatchesWhenNoToneDoes() {
        var custom = AppSamplingSettings()
        custom.temperature = 0.65
        custom.maxNewTokens = 1024
        let other = AppSamplingPreset(name: "Other", settings: AppSamplingSettings(temperature: 1.5))
        let mine = AppSamplingPreset(name: "Mine", settings: custom)
        XCTAssertEqual(
            SamplingToneMatcher.match(custom, savedPresets: [other, mine]), .saved(mine.id))
    }

    /// A saved preset is applied whole, so it matches on every field, not
    /// just the tone knobs.
    func testSavedPresetNeedsEveryFieldToMatch() {
        var saved = AppSamplingSettings()
        saved.temperature = 0.65
        let preset = AppSamplingPreset(name: "Mine", settings: saved)
        var live = saved
        live.maxNewTokens = 512
        XCTAssertEqual(SamplingToneMatcher.match(live, savedPresets: [preset]), .custom)

        var drift = saved
        drift.temperature += 0.000_000_1
        XCTAssertEqual(SamplingToneMatcher.match(drift, savedPresets: [preset]), .saved(preset.id))
    }

    /// The value behind a disabled toggle has no effect, so it is ignored.
    func testSavedPresetIgnoresValueBehindDisabledToggle() {
        var saved = AppSamplingSettings()
        saved.temperature = 0.65
        saved.repetitionPenaltyEnabled = false
        saved.repetitionPenalty = 1.0
        let preset = AppSamplingPreset(name: "Mine", settings: saved)
        var live = saved
        live.repetitionPenalty = 1.4
        XCTAssertEqual(SamplingToneMatcher.match(live, savedPresets: [preset]), .saved(preset.id))
    }

    /// A built-in tone wins over a saved preset holding the same values, so
    /// the pill name does not depend on what happens to be saved.
    func testToneWinsOverIdenticalSavedPreset() {
        let preset = AppSamplingPreset(name: "Copy", settings: AppSamplingSettings())
        XCTAssertEqual(
            SamplingToneMatcher.match(AppSamplingSettings(), savedPresets: [preset]), .tone(.balanced))
    }

    // MARK: - Scope

    func testScopeFollowsTheChatOverride() {
        XCTAssertEqual(SamplingToneMatcher.scope(hasChatOverride: true), .thisChat)
        XCTAssertEqual(SamplingToneMatcher.scope(hasChatOverride: false), .allChats)
    }
}
