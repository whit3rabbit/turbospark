import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

@MainActor
final class AudioLibraryOrganizationTests: XCTestCase {
    private var vault: ProfileVaultStore?
    private func store() throws -> (AudioLibraryStore, ProfileRepository, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "organization" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        self.vault = vault
        let repository = ProfileRepository(store: vault)
        return (AudioLibraryStore(repository: repository), repository, root)
    }

    func testOneUnreadableRecordDoesNotHideTheLibrary() throws {
        let (library, repository, root) = try store()
        defer { try? FileManager.default.removeItem(at: root) }
        try library.save(AudioLibraryItem(title: "Kept", kind: .recording))
        let badKey = "audio:item:\(UUID().uuidString)"
        try repository.saveAudioRecord(key: badKey, payload: Data("{not json".utf8), references: [])
        let loaded = try library.loadReporting()
        XCTAssertEqual(loaded.items.map(\.title), ["Kept"])
        XCTAssertEqual(loaded.unreadable, 1)
        let quarantined = try repository.rawRecords(prefix: "quarantine:\(badKey):")
        XCTAssertEqual(quarantined.count, 1, "the unreadable payload is kept aside")
        XCTAssertEqual(try repository.rawRecords(prefix: badKey).count, 1, "the original stays, with its asset references")
        _ = try library.loadReporting()
        XCTAssertEqual(try repository.rawRecords(prefix: "quarantine:\(badKey):").count, 1, "a record that stays broken is not copied again")
    }

    func testARecordSavedBeforeTagsStillDecodes() throws {
        var item = AudioLibraryItem(title: "Old", kind: .recording)
        item.tags = ["x"]; item.recordingPreset = "meeting"
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(item)) as? [String: Any])
        json.removeValue(forKey: "tags"); json.removeValue(forKey: "recordingPreset")
        let old = try JSONDecoder().decode(AudioLibraryItem.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertNil(old.tags)
        XCTAssertEqual(old.tagList, [])
        XCTAssertNoThrow(try old.validate())
        var preferences = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(AudioWorkspacePreferences())) as? [String: Any])
        preferences.removeValue(forKey: "recordingPreset"); preferences.removeValue(forKey: "presetCollections")
        XCTAssertNoThrow(try JSONDecoder().decode(AudioWorkspacePreferences.self, from: JSONSerialization.data(withJSONObject: preferences)))
    }

    func testTagsAreNormalizedAndBounded() {
        XCTAssertEqual(AudioLibraryOrganizer.normalizedTags(["  Team  sync ", "team sync", "", "Q3", "q3", String(repeating: "x", count: 41)]), ["Team sync", "Q3"])
        XCTAssertEqual(AudioLibraryOrganizer.normalizedTags((0..<30).map { "t\($0)" }).count, AudioLibraryOrganizer.maxTags)
        var item = AudioLibraryItem(title: "A", kind: .recording)
        item.setTags(["one"]); item.setTags([])
        XCTAssertNil(item.tags, "no tags is stored as absent")
        item.tags = (0..<257).map { "t\($0)" }
        XCTAssertThrowsError(try item.validate(), "only sizes no editor makes fail a load")
    }

    func testFiltersAndFacets() {
        func make(_ title: String, _ kind: AudioLibraryItem.Kind, favorite: Bool = false, transcript: Bool = false,
                  collection: String = "", tags: [String] = [], preset: String? = nil) -> AudioLibraryItem {
            var item = AudioLibraryItem(title: title, kind: kind)
            item.favorite = favorite; item.collection = collection; item.tags = tags.isEmpty ? nil : tags; item.recordingPreset = preset
            item.clips = [AudioLibraryClip(assetReference: "x", sampleRate: 16_000, channels: 1, frameCount: 16_000)]
            if transcript { item.addTranscript([AudioLibrarySegment(start: 0, end: 1, text: "hi")], source: .final) }
            return item
        }
        let items = [
            make("a", .recording, favorite: true, transcript: true, collection: "Meetings", tags: ["Team"]),
            make("b", .recording, collection: "Meetings", tags: ["team", "Q3"]),
            make("c", .imported),
            make("d", .recording, preset: AudioRecordingPreset.sound.rawValue),
            make("e", .music),
        ]
        func titles(_ f: AudioLibraryFilter, collection: String? = nil, tag: String? = nil) -> [String] {
            AudioLibraryOrganizer.filter(items, by: f, collection: collection, tag: tag).map(\.title)
        }
        XCTAssertEqual(titles(.all), ["a", "b", "c", "d", "e"])
        XCTAssertEqual(titles(.favorites), ["a"])
        XCTAssertEqual(titles(.recordings), ["a", "b", "d"])
        XCTAssertEqual(titles(.imported), ["c"])
        XCTAssertEqual(titles(.needsTranscript), ["b", "c"], "a sound take and generated music are not waiting on a transcript")
        XCTAssertEqual(titles(.all, collection: "Meetings"), ["a", "b"])
        XCTAssertEqual(titles(.all, tag: "TEAM"), ["a", "b"], "tags match ignoring case")
        XCTAssertEqual(AudioLibraryOrganizer.collections(items), [AudioFacet(name: "Meetings", count: 2)])
        XCTAssertEqual(AudioLibraryOrganizer.tags(items), [AudioFacet(name: "Team", count: 2), AudioFacet(name: "Q3", count: 1)])
    }

    func testDateGroupsFollowTheCalendarWeek() throws {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try XCTUnwrap(TimeZone(identifier: "UTC"))
        calendar.firstWeekday = 2
        let now = try XCTUnwrap(calendar.date(from: DateComponents(year: 2026, month: 10, day: 15, hour: 12)))
        func day(_ d: Int, _ h: Int = 9) -> Date { calendar.date(from: DateComponents(year: 2026, month: 10, day: d, hour: h))! }
        XCTAssertEqual(AudioDateGroup.group(day(15, 1), now: now, calendar: calendar), .today)
        XCTAssertEqual(AudioDateGroup.group(day(13), now: now, calendar: calendar), .thisWeek, "Tuesday, same Monday-first week")
        XCTAssertEqual(AudioDateGroup.group(day(11), now: now, calendar: calendar), .earlier, "Sunday ends the previous week")
        var items = [AudioLibraryItem(title: "old", kind: .recording), AudioLibraryItem(title: "new", kind: .recording), AudioLibraryItem(title: "mid", kind: .recording)]
        items[0].createdAt = day(1); items[1].createdAt = day(15, 11); items[2].createdAt = day(15, 8)
        let grouped = AudioLibraryOrganizer.grouped(items, now: now, calendar: calendar)
        XCTAssertEqual(grouped.map(\.group), [.today, .earlier], "empty groups are left out")
        XCTAssertEqual(grouped[0].items.map(\.title), ["new", "mid"], "newest first within a group")
    }

    func testPresetSetsSourcesAndFilesTakesIntoAStableCollection() throws {
        let (library, _, root) = try store()
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = AudioWorkspaceController(library: library, assets: ManagedAssetStore(vault: try XCTUnwrap(vault)))
        controller.applyRecordingPreset(.meeting)
        XCTAssertTrue(controller.captureMicrophone && controller.captureSystemAudio)
        XCTAssertEqual(controller.effectiveRecordingPreset, .meeting)
        let name = controller.collectionName(for: .meeting)
        XCTAssertEqual(controller.preferences.presetCollections?["meeting"], name, "named once, then kept")
        controller.preferences.presetCollections?["meeting"] = "Calls"
        XCTAssertEqual(controller.collectionName(for: .meeting), "Calls", "a saved name is never renamed by a language change")
        controller.captureSystemAudio = false
        XCTAssertNil(controller.effectiveRecordingPreset, "changed sources make the take a custom one")
        controller.applyRecordingPreset(.sound)
        XCTAssertFalse(controller.liveTranscription, "a sound recording has no speech to read")
    }

    func testAudioShortcutsAreListedAndMatchTheRecordButton() throws {
        let row = KeyboardShortcutCatalog.audio.rows.first { $0.label == "Start Recording" }
        XCTAssertEqual(row?.keys, "\u{21E7} \u{2318} R")
        let panel = try String(contentsOf: URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp/Audio/AudioRecordingPanel.swift"), encoding: .utf8)
        XCTAssertTrue(panel.contains(#".keyboardShortcut("r", modifiers: [.command, .shift])"#), "the pane lists the chord the button binds")
    }
}
