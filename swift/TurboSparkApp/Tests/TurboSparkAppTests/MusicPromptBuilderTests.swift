import XCTest
@testable import TurboSparkApp

final class MusicPromptBuilderTests: XCTestCase {
    private func caption(_ configure: (inout MusicPromptSelection) -> Void) -> String {
        var selection = MusicPromptSelection()
        configure(&selection)
        return MusicPromptBuilder.caption(for: selection)
    }

    // MARK: - Caption assembly

    func testEmptySelectionProducesNoText() {
        XCTAssertEqual(MusicPromptBuilder.caption(for: MusicPromptSelection()), "")
    }

    func testFullSelectionFollowsTheGuideOrder() {
        let text = caption {
            $0.genre = "contemporary R&B"
            $0.influence = "trap soul"
            $0.scene = "  a late-night drive. "
            $0.moods = ["lonely", "dreamy"]
            $0.tempo = .slow
            $0.bpm = 80
            $0.groove = "laid-back"
            $0.instruments = ["deep 808 sub-bass", "sparse piano chords", "restrained percussion"]
            $0.voice = "male baritone"
            $0.tone = "warm"
            $0.delivery = "intimate"
            $0.techniques = ["breathy phrasing", "close harmonies"]
            $0.emotion = "detached"
            $0.arc = .sparseToBig
            $0.production = ["clean modern finish", "controlled low end"]
        }
        XCTAssertEqual(
            text,
            "Contemporary R&B with trap soul influences for a late-night drive. "
                + "Lonely and dreamy mood. "
                + "Slow tempo around 80 BPM with a laid-back groove. "
                + "Deep 808 sub-bass, sparse piano chords and restrained percussion. "
                + "Warm male baritone vocals with intimate delivery, breathy phrasing and close harmonies, sounding detached. "
                + "Keep the verses sparse, then open into a wider, bigger chorus. "
                + "Production: clean modern finish and controlled low end.")
    }

    func testGenreAloneIsASentence() {
        XCTAssertEqual(caption { $0.genre = "indie folk" }, "Indie folk.")
    }

    func testInfluenceNeedsAGenreAndMustDiffer() {
        XCTAssertEqual(caption { $0.influence = "jazz" }, "")
        XCTAssertEqual(caption { $0.genre = "pop"; $0.influence = "pop" }, "Pop.")
        XCTAssertEqual(caption { $0.genre = "pop"; $0.influence = "jazz" }, "Pop with jazz influences.")
    }

    func testSceneWithoutGenreStillReads() {
        XCTAssertEqual(caption { $0.scene = "a quiet cafe" }, "Music for a quiet cafe.")
    }

    func testSceneWhitespaceAndTrailingPunctuationAreNormalized() {
        XCTAssertEqual(caption { $0.genre = "pop"; $0.scene = "a\n  road   trip!!" }, "Pop for a road trip.")
        XCTAssertEqual(caption { $0.scene = "  \n " }, "")
    }

    func testSingleMoodReadsNaturally() {
        XCTAssertEqual(caption { $0.moods = ["nostalgic"] }, "Nostalgic mood.")
    }

    // MARK: - Tempo and groove

    func testBPMOutsideTheRangeIsIgnored() {
        XCTAssertEqual(caption { $0.bpm = 10 }, "")
        XCTAssertEqual(caption { $0.bpm = 900 }, "")
        XCTAssertEqual(caption { $0.bpm = 90 }, "Around 90 BPM.")
    }

    func testGrooveAloneAndWithTempo() {
        XCTAssertEqual(caption { $0.groove = "swinging" }, "A swinging groove.")
        XCTAssertEqual(caption { $0.tempo = .mid; $0.groove = "punchy" }, "Mid-tempo with a punchy groove.")
        XCTAssertEqual(caption { $0.tempo = .fast }, "Fast, energetic tempo.")
    }

    func testEveryCatalogGrooveGetsACorrectArticle() {
        for groove in MusicPromptCatalog.grooves {
            let text = caption { $0.groove = groove }
            let article = "aeiou".contains(groove.first!) ? "An" : "A"
            XCTAssertEqual(text, "\(article) \(groove) groove.")
        }
    }

    // MARK: - Instruments and vocals

    func testInstrumentalNamesTheLeadAndSupport() {
        let text = caption {
            $0.vocalMode = .instrumental
            $0.instruments = ["solo cello", "atmospheric synth pads", "lush string section"]
        }
        XCTAssertEqual(
            text,
            "Led by solo cello, supported by atmospheric synth pads and lush string section. Instrumental, no vocals.")
        XCTAssertEqual(
            caption { $0.vocalMode = .instrumental; $0.instruments = ["solo cello"] },
            "Led by solo cello. Instrumental, no vocals.")
    }

    func testSungTrackListsInstrumentsAsAPalette() {
        let text = caption { $0.instruments = ["solo cello", "harp"] }
        XCTAssertEqual(text, "Solo cello and harp.")
        XCTAssertFalse(text.contains("Led by"))
    }

    func testVocalSentenceDegradesGracefullyWithMissingParts() {
        XCTAssertEqual(caption { $0.voice = "female alto" }, "Female alto vocals.")
        XCTAssertEqual(caption { $0.tone = "airy" }, "Airy vocals.")
        XCTAssertEqual(caption { $0.delivery = "powerful" }, "Vocals with powerful delivery.")
        XCTAssertEqual(caption { $0.emotion = "yearning" }, "Vocals, sounding yearning.")
        XCTAssertEqual(caption { $0.techniques = ["subtle vibrato"] }, "Vocals, subtle vibrato.")
    }

    // MARK: - Selection limits

    func testToggledAddsRemovesAndKeepsOrder() {
        var list: [String] = []
        list = MusicPromptSelection.toggled(list, "a", limit: 3)
        list = MusicPromptSelection.toggled(list, "b", limit: 3)
        list = MusicPromptSelection.toggled(list, "c", limit: 3)
        XCTAssertEqual(list, ["a", "b", "c"])
        list = MusicPromptSelection.toggled(list, "b", limit: 3)
        XCTAssertEqual(list, ["a", "c"])
        list = MusicPromptSelection.toggled(list, "b", limit: 3)
        XCTAssertEqual(list, ["a", "c", "b"], "A re-added pick goes to the end, so it no longer leads")
    }

    func testToggledRefusesPastTheLimitButStillRemoves() {
        let full = ["a", "b"]
        XCTAssertEqual(MusicPromptSelection.toggled(full, "c", limit: 2), full)
        XCTAssertEqual(MusicPromptSelection.toggled(full, "a", limit: 2), ["b"])
    }

    // MARK: - Warnings

    func testVocalWarnings() {
        var selection = MusicPromptSelection()
        XCTAssertEqual(
            MusicPromptBuilder.warnings(for: selection, lyricsAreInstrumental: true),
            [.noVoice, .vocalsWithoutLyrics])
        selection.voice = "male baritone"
        XCTAssertEqual(MusicPromptBuilder.warnings(for: selection, lyricsAreInstrumental: false), [])
        XCTAssertEqual(
            MusicPromptBuilder.warnings(for: selection, lyricsAreInstrumental: true), [.vocalsWithoutLyrics])
    }

    func testInstrumentalWarnings() {
        var selection = MusicPromptSelection()
        selection.vocalMode = .instrumental
        XCTAssertEqual(
            MusicPromptBuilder.warnings(for: selection, lyricsAreInstrumental: false),
            [.instrumentalNeedsLead, .instrumentalWithLyrics])
        selection.instruments = ["solo cello"]
        XCTAssertEqual(MusicPromptBuilder.warnings(for: selection, lyricsAreInstrumental: true), [])
    }

    // MARK: - Catalog

    func testCatalogHasNoDuplicatesOrEmptyEntries() {
        let flat: [(String, [String])] = [
            ("genres", MusicPromptCatalog.genres.flatMap(\.items)),
            ("moods", MusicPromptCatalog.moods.flatMap(\.items)),
            ("instruments", MusicPromptCatalog.instruments.flatMap(\.items)),
            ("grooves", MusicPromptCatalog.grooves),
            ("voices", MusicPromptCatalog.voices),
            ("tones", MusicPromptCatalog.tones),
            ("deliveries", MusicPromptCatalog.deliveries),
            ("techniques", MusicPromptCatalog.techniques),
            ("emotions", MusicPromptCatalog.emotions),
            ("production", MusicPromptCatalog.production)
        ]
        for (name, items) in flat {
            XCTAssertFalse(items.isEmpty, name)
            XCTAssertEqual(Set(items).count, items.count, "\(name) has duplicates")
            XCTAssertTrue(items.allSatisfy { $0 == $0.trimmingCharacters(in: .whitespaces) && !$0.isEmpty }, name)
        }
    }

    func testArcSentencesAreCompleteSentences() {
        for arc in MusicArc.allCases {
            XCTAssertTrue(arc.sentence.hasSuffix("."), "\(arc)")
            XCTAssertTrue(arc.sentence.first!.isUppercase, "\(arc)")
        }
    }
}

final class MusicLyricsGuideTests: XCTestCase {
    func testAppendingToEmptyLyricsStartsWithTheTag() {
        XCTAssertEqual(MusicLyricsGuide.appending(["verse"], to: ""), "[verse]\n")
        XCTAssertEqual(MusicLyricsGuide.appending(["verse"], to: " \n\t"), "[verse]\n")
    }

    func testAppendingKeepsExistingWordsAndSeparatesWithABlankLine() {
        XCTAssertEqual(
            MusicLyricsGuide.appending(["chorus"], to: "[verse]\nline one\nline two\n\n\n"),
            "[verse]\nline one\nline two\n\n[chorus]\n")
    }

    func testAppendingReplacesTheBareInstrumentalToken() {
        XCTAssertEqual(MusicLyricsGuide.appending(["verse", "chorus"], to: "[instrumental]"), "[verse]\n\n[chorus]\n")
        XCTAssertEqual(
            MusicLyricsGuide.appending(["verse"], to: " [instrumental]\n"), "[verse]\n",
            "Surrounding whitespace must not defeat the token check")
    }

    func testAppendingNothingLeavesLyricsUntouched() {
        XCTAssertEqual(MusicLyricsGuide.appending([], to: "words"), "words")
    }

    func testEveryTagSitsAloneOnItsLine() {
        for layout in MusicLyricsGuide.layouts {
            let lyrics = MusicLyricsGuide.appending(layout, to: "first words")
            let tagLines = lyrics.split(separator: "\n").filter { $0.hasPrefix("[") }
            XCTAssertEqual(tagLines.count, layout.count)
            XCTAssertTrue(tagLines.allSatisfy { $0.hasSuffix("]") && $0.filter { $0 == "[" }.count == 1 })
        }
    }

    func testLayoutsOnlyUseKnownSectionTags() {
        let known = Set(MusicLyricsGuide.sectionTags)
        for layout in MusicLyricsGuide.layouts {
            XCTAssertTrue(Set(layout).isSubset(of: known), "\(layout)")
        }
    }

    func testAppendedSectionsAreNotMistakenForInstrumentalSongs() {
        let lyrics = MusicLyricsGuide.appending(["instrumental"], to: "[verse]\nwords")
        XCTAssertNotEqual(lyrics, AudioWorkspaceController.instrumentalLyrics)
    }
}
