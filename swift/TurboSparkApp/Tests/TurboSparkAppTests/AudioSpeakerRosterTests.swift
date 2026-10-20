import Foundation
import XCTest

@testable import TurboSparkApp

/// Speaker names and merges live on the take by voice key: one rename
/// reaches every line, revision and export, and undoing it leaves the raw
/// ids as the labeler wrote them.
final class AudioSpeakerRosterTests: XCTestCase {
    private func withEnglish(_ body: () throws -> Void) rethrows {
        let defaults = UserDefaults.standard
        let previous = defaults.object(forKey: AppLanguage.storageKey)
        defer {
            if let previous { defaults.set(previous, forKey: AppLanguage.storageKey) }
            else { defaults.removeObject(forKey: AppLanguage.storageKey) }
        }
        defaults.set(AppLanguage.english.rawValue, forKey: AppLanguage.storageKey)
        try body()
    }

    private func line(_ start: Double, _ end: Double, _ text: String, voice: String?) -> AudioLibrarySegment {
        AudioSpeakerRoster.assigning(AudioLibrarySegment(start: start, end: end, text: text, timing: "token"), to: voice)
    }

    /// A labeled meeting: a final revision and a user edit of it, three voices.
    private func meeting() -> AudioLibraryItem {
        var item = AudioLibraryItem(title: "Planning", kind: .recording)
        let lines = [
            line(0, 4, "Let's start with the release.", voice: "S1"),
            line(4, 9, "The migration took three hours.", voice: "S2"),
            line(9, 10, "Mm-hm.", voice: "S3"),
            line(10, 12, "Then we split it.", voice: "S1"),
        ]
        item.addTranscript(lines, source: .final)
        var edited = lines
        edited[1].text = "The migration took about three hours."
        item.addTranscript(edited, source: .user)
        return item
    }

    func testRenameReachesEveryLineAndEveryExport() throws {
        try withEnglish {
            var item = meeting()
            var roster = AudioSpeakerRoster(item)
            roster.rename("S2", to: "  Dana ")
            roster.apply(to: &item)
            XCTAssertEqual(item.speakerNames, ["S2": "Dana"])

            let text = try XCTUnwrap(String(data: try XCTUnwrap(AudioTranscriptExport.data(item, format: .text)), encoding: .utf8))
            XCTAssertEqual(text, """
                Speaker 1: Let's start with the release.
                Dana: The migration took about three hours.
                Speaker 3: Mm-hm.
                Speaker 1: Then we split it.
                """)
            let srt = try XCTUnwrap(String(data: try XCTUnwrap(AudioTranscriptExport.data(item, format: .srt)), encoding: .utf8))
            XCTAssertTrue(srt.contains("00:00:04,000 --> 00:00:09,000\nDana: The migration took about three hours.\n"), srt)
            let vtt = try XCTUnwrap(String(data: try XCTUnwrap(AudioTranscriptExport.data(item, format: .vtt)), encoding: .utf8))
            XCTAssertTrue(vtt.hasPrefix("WEBVTT\n\n"))
            XCTAssertTrue(vtt.contains("00:00:04.000 --> 00:00:09.000\nDana: The migration took about three hours.\n"), vtt)

            // JSON carries the name in every revision and keeps the raw id.
            let json = try JSONDecoder().decode(AudioLibraryItem.self, from: try XCTUnwrap(AudioTranscriptExport.data(item, format: .json)))
            for revision in json.transcripts {
                XCTAssertEqual(revision.segments.map(\.speaker), ["Speaker 1", "Dana", "Speaker 3", "Speaker 1"])
                XCTAssertEqual(revision.segments.map(\.speakerID), ["S1", "S2", "S3", "S1"])
            }
            XCTAssertEqual(json.speakerNames, ["S2": "Dana"])

            // The take itself is untouched: names never go into segments.
            XCTAssertEqual(item.transcripts.map { $0.segments.map(\.speaker) }, meeting().transcripts.map { $0.segments.map(\.speaker) })
        }
    }

    func testClearingANameRestoresTheDefaultLabel() {
        withEnglish {
            var item = meeting()
            var roster = AudioSpeakerRoster(item)
            roster.rename("S2", to: "Dana")
            roster.rename("S1", to: "Sam")
            roster.rename("S2", to: "   ")
            roster.rename("S1", to: "Speaker 1")
            XCTAssertEqual(roster.names, [:])
            roster.apply(to: &item)
            XCTAssertNil(item.speakerNames, "an empty map is not stored")
            XCTAssertEqual(roster.labeled(item.preferredTranscript?.segments ?? []).map(\.speaker),
                           ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 1"])
        }
    }

    func testMergeShowsTheTargetAndSeparateUndoesIt() {
        withEnglish {
            let segments = meeting().preferredTranscript?.segments ?? []
            var roster = AudioSpeakerRoster()
            roster.rename("S2", to: "Dana")
            roster.rename("S3", to: "Lee")
            XCTAssertTrue(roster.merge("S3", into: "S2"))
            XCTAssertEqual(roster.labeled(segments).map(\.speaker), ["Speaker 1", "Dana", "Dana", "Speaker 1"])
            XCTAssertEqual(roster.roots(in: segments), ["S1", "S2"])

            let summaries = AudioSpeakerSummary.summaries(segments, roster: roster)
            XCTAssertEqual(summaries.map(\.id), ["S1", "S2", "S3"], "a merged voice is listed after the shown ones")
            XCTAssertEqual(summaries.map(\.mergedInto), [nil, nil, "S2"])
            XCTAssertEqual(summaries[1].talkTime, 6, accuracy: 1e-9, "Dana's time includes the merged voice")
            XCTAssertEqual(summaries[2].talkTime, 1, accuracy: 1e-9)

            roster.separate("S3")
            XCTAssertEqual(roster.labeled(segments).map(\.speaker), ["Speaker 1", "Dana", "Lee", "Speaker 1"],
                           "separating restores the merged voice's own name")
        }
    }

    func testMergeChainsAndRefusesCycles() {
        withEnglish {
            var roster = AudioSpeakerRoster()
            XCTAssertTrue(roster.merge("S3", into: "S2"))
            XCTAssertTrue(roster.merge("S2", into: "S1"))
            XCTAssertEqual(roster.root("S3"), "S1")
            XCTAssertFalse(roster.merge("S1", into: "S3"), "S3 already shows as S1")
            XCTAssertFalse(roster.merge("S1", into: "S1"))
            XCTAssertEqual(roster.merges, ["S3": "S2", "S2": "S1"])

            // Only a damaged record holds a cycle: each voice shows as itself.
            let damaged = AudioSpeakerRoster(merges: ["S1": "S2", "S2": "S1"])
            XCTAssertEqual(damaged.root("S1"), "S1")
            XCTAssertEqual(damaged.root("S2"), "S2")
        }
    }

    /// A take saved before speaker ids has none of the new keys. It decodes,
    /// and its saved labels act as voices that can be renamed.
    func testATakeSavedBeforeSpeakerIDsDecodesAndCanBeRenamed() throws {
        try withEnglish {
            var old = AudioLibraryItem(title: "Old call", kind: .recording)
            old.addTranscript([
                AudioLibrarySegment(start: 0, end: 2, text: "Hello", speaker: "Speaker 1"),
                AudioLibrarySegment(start: 2, end: 4, text: "Hi", speaker: "Alice"),
                AudioLibrarySegment(start: 4, end: 5, text: "Bye", speaker: "Speaker 1"),
            ], source: .final)
            let json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(old)) as? [String: Any])
            for key in ["speakerNames", "speakerMerges", "speakerLabelParts"] { XCTAssertNil(json[key], key) }
            let transcripts = try XCTUnwrap(json["transcripts"] as? [[String: Any]])
            let segments = try XCTUnwrap(transcripts.first?["segments"] as? [[String: Any]])
            XCTAssertTrue(segments.allSatisfy { $0["speakerID"] == nil }, "an older build wrote no speakerID")

            var decoded = try JSONDecoder().decode(AudioLibraryItem.self, from: JSONSerialization.data(withJSONObject: json))
            XCTAssertNoThrow(try decoded.validate())
            XCTAssertEqual(decoded.preferredTranscript?.segments.map(AudioSpeakerRoster.key), ["Speaker 1", "Alice", "Speaker 1"])

            var roster = AudioSpeakerRoster(decoded)
            roster.rename("Speaker 1", to: "Sam")
            XCTAssertTrue(roster.merge("Alice", into: "Speaker 1"))
            roster.apply(to: &decoded)
            let text = try XCTUnwrap(String(data: try XCTUnwrap(AudioTranscriptExport.data(decoded, format: .text)), encoding: .utf8))
            XCTAssertEqual(text, "Sam: Hello\nSam: Hi\nSam: Bye")
        }
    }

    func testSummariesTakeTheLongestLineForQuoteAndSample() {
        withEnglish {
            let segments = [
                line(0, 2, "Short.", voice: "S1"),
                line(2, 20, "A long answer about the migration plan.", voice: "S1"),
                line(20, 21, "  ", voice: "S2"),
                line(21, 21, "Untimed.", voice: "S2"),
            ]
            let summaries = AudioSpeakerSummary.summaries(segments, roster: AudioSpeakerRoster())
            XCTAssertEqual(summaries.map(\.id), ["S1", "S2"])
            XCTAssertEqual(summaries[0].talkTime, 20, accuracy: 1e-9)
            XCTAssertEqual(summaries[0].quote, "A long answer about the migration plan.")
            XCTAssertEqual(summaries[0].sample, 2...8, "a sample is a few seconds from the line's start")
            XCTAssertEqual(summaries[1].quote, "Untimed.", "a blank line is never the quote")
            XCTAssertEqual(summaries[1].sample, 20...21)
        }
    }

    func testReassigningALineAndNewSpeakerIDs() {
        withEnglish {
            var item = meeting()
            item.speakerMerges = ["S9": "S1"]
            let unsaved = [line(0, 1, "New", voice: "S10")]
            XCTAssertEqual(AudioSpeakerRoster.nextID(in: item), "S10")
            XCTAssertEqual(AudioSpeakerRoster.nextID(in: item, also: unsaved), "S11")
            XCTAssertEqual(AudioSpeakerRoster.nextID(in: AudioLibraryItem(title: "Empty", kind: .recording)), "S1")

            let moved = AudioSpeakerRoster.assigning(item.preferredTranscript!.segments[0], to: "S2")
            XCTAssertEqual(moved.speakerID, "S2")
            XCTAssertEqual(moved.speaker, "Speaker 2")
            let cleared = AudioSpeakerRoster.assigning(moved, to: nil)
            XCTAssertNil(cleared.speakerID)
            XCTAssertNil(cleared.speaker)
            XCTAssertNil(AudioSpeakerRoster.key(cleared))
        }
    }

    /// Names saved through the controller survive a reload of the library.
    @MainActor func testControllerSavesNamesOnTheTake() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "speaker-roster" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let controller = AudioWorkspaceController(library: library, assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        let item = meeting()
        try controller.save(item)

        var roster = AudioSpeakerRoster(item)
        roster.rename("S1", to: "Sam")
        XCTAssertTrue(roster.merge("S3", into: "S2"))
        controller.applySpeakerRoster(roster, to: item.id)

        let saved = try XCTUnwrap(library.item(item.id))
        XCTAssertEqual(saved.speakerNames, ["S1": "Sam"])
        XCTAssertEqual(saved.speakerMerges, ["S3": "S2"])
        XCTAssertEqual(saved.transcripts, item.transcripts, "segments keep their raw ids")
    }

    /// Name the Speakers opens by itself only for a Meeting take whose final
    /// pass labeled speakers, while the Meeting pane is still showing.
    @MainActor func testOnlyALabeledMeetingTakeOnItsPaneAsksForNames() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "speaker-naming" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        func take(_ preset: AudioRecordingPreset, labeled: Bool) throws -> (id: UUID, revision: UUID) {
            var item = AudioLibraryItem(title: preset.rawValue, kind: .recording)
            item.recordingPreset = preset.rawValue
            let lines = [line(0, 2, "Hello", voice: "S1"), line(2, 4, "Hi", voice: "S2")]
            let revision = item.addTranscript(labeled ? lines : lines.map { AudioSpeakerRoster.assigning($0, to: nil) }, source: .final)
            try controller.save(item)
            return (item.id, revision)
        }
        let meeting = try take(.meeting, labeled: true)
        let unlabeled = try take(.meeting, labeled: false)
        let note = try take(.voiceNote, labeled: true)
        controller.selectPage(.transcribe)
        controller.selectSpeechToTextMode(.meeting)

        XCTAssertTrue(controller.offersSpeakerNaming(meeting.id, revision: meeting.revision))
        XCTAssertFalse(controller.offersSpeakerNaming(unlabeled.id, revision: unlabeled.revision), "no labels, nothing to name")
        XCTAssertFalse(controller.offersSpeakerNaming(note.id, revision: note.revision), "a labeled voice note is not a meeting")
        controller.selectSpeechToTextMode(.notes)
        XCTAssertFalse(controller.offersSpeakerNaming(meeting.id, revision: meeting.revision), "the Meeting pane is not showing")
        controller.selectSpeechToTextMode(.meeting)
        controller.selectPage(.library)
        XCTAssertFalse(controller.offersSpeakerNaming(meeting.id, revision: meeting.revision), "another page is showing")
    }
}
