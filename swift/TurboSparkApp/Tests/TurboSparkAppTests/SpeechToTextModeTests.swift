import Foundation
import XCTest
@testable import TurboSparkApp

@MainActor
final class SpeechToTextModeTests: XCTestCase {
    private var vault: ProfileVaultStore?

    private func makeLibrary() throws -> (AudioLibraryStore, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "speech-to-text-modes" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        self.vault = vault
        return (AudioLibraryStore(repository: ProfileRepository(store: vault)), root)
    }

    private func makeController(_ library: AudioLibraryStore) throws -> AudioWorkspaceController {
        AudioWorkspaceController(library: library, assets: ManagedAssetStore(vault: try XCTUnwrap(vault)))
    }

    func testModesRecordWithTheirOwnPresets() {
        XCTAssertEqual(SpeechToTextMode.notes.recordingPreset, .voiceNote)
        XCTAssertEqual(SpeechToTextMode.meeting.recordingPreset, .meeting)
        XCTAssertEqual(SpeechToTextMode.anyAudio.recordingPreset, .appAudio)
        XCTAssertFalse(AudioRecordingPreset.appAudio.capturesMicrophone, "app audio is the other app only")
        XCTAssertTrue(AudioRecordingPreset.appAudio.capturesSystemAudio)
        XCTAssertTrue(AudioRecordingPreset.appAudio.liveText)
        XCTAssertTrue(AudioRecordingPreset.voiceNote.capturesMicrophone && !AudioRecordingPreset.voiceNote.capturesSystemAudio)
        XCTAssertFalse(AudioRecordingPreset.recordPageCases.contains(.appAudio), "the Record page does not offer app audio")
        XCTAssertEqual(Set(AudioRecordingPreset.recordPageCases + [.appAudio]), Set(AudioRecordingPreset.allCases))
    }

    func testOlderPreferencesAndUnknownModesFallBackToNotes() throws {
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(AudioWorkspacePreferences())) as? [String: Any])
        json.removeValue(forKey: "speechToTextMode")
        let old = try JSONDecoder().decode(AudioWorkspacePreferences.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertNil(old.speechToTextMode)
        let (library, root) = try makeLibrary()
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = try makeController(library)
        defer { controller.shutdown() }
        XCTAssertEqual(controller.speechToTextMode, .notes)
        controller.preferences.speechToTextMode = "from-a-newer-build"
        XCTAssertEqual(controller.speechToTextMode, .notes)
    }

    func testTheModeSurvivesAReload() throws {
        let (library, root) = try makeLibrary()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = try makeController(library)
        first.selectPage(.transcribe)
        first.selectSpeechToTextMode(.meeting)
        first.shutdown()
        let second = try makeController(library)
        defer { second.shutdown() }
        XCTAssertEqual(second.speechToTextMode, .meeting)
        XCTAssertTrue(second.captureMicrophone && second.captureSystemAudio, "a reopened meeting gets its sources back")
    }

    func testSourcesFollowTheModeOnlyWhileIdle() throws {
        let (library, root) = try makeLibrary()
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = try makeController(library)
        defer { controller.shutdown() }
        controller.selectPage(.transcribe)
        controller.selectSpeechToTextMode(.anyAudio)
        XCTAssertFalse(controller.captureMicrophone)
        XCTAssertTrue(controller.captureSystemAudio)
        controller.recordingSetupStatus = "Preparing recording..."
        controller.selectSpeechToTextMode(.notes)
        XCTAssertEqual(controller.speechToTextMode, .notes, "the view still switches")
        XCTAssertFalse(controller.captureMicrophone, "a running capture keeps its sources")
        XCTAssertTrue(controller.captureSystemAudio)
        controller.prepareRecordingSources()
        XCTAssertFalse(controller.captureMicrophone, "Start does not rewrite a running capture")
        controller.recordingSetupStatus = nil
        controller.liveTranscription = false
        controller.prepareRecordingSources()
        XCTAssertTrue(controller.captureMicrophone && !controller.captureSystemAudio)
        XCTAssertTrue(controller.liveTranscription, "Start turns live text back on for a mode that reads speech")
    }

    func testTheRecordPageKeepsItsOwnChoice() throws {
        let (library, root) = try makeLibrary()
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = try makeController(library)
        defer { controller.shutdown() }
        controller.selectPage(.record)
        controller.applyRecordingPreset(.sound)
        controller.selectPage(.transcribe)
        XCTAssertTrue(controller.liveTranscription, "Notes reads speech")
        XCTAssertEqual(controller.activeRecordingPreset, .voiceNote)
        controller.selectPage(.record)
        XCTAssertEqual(controller.preferences.recordingPreset, AudioRecordingPreset.sound.rawValue)
        XCTAssertFalse(controller.liveTranscription, "Sound comes back on the Record page")
        XCTAssertEqual(controller.activeRecordingPreset, .sound)
    }

    func testAMeetingWithoutTheMicrophoneIsStillAMeeting() async throws {
        let (library, root) = try makeLibrary()
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = try makeController(library)
        defer { controller.shutdown() }
        controller.recordingSetupDependencies.start = { _, _ in }
        controller.selectPage(.transcribe)
        controller.selectSpeechToTextMode(.meeting)
        controller.liveTranscription = false
        controller.captureMicrophone = false
        XCTAssertNil(controller.effectiveRecordingPreset, "the Record page's rule would call this custom")
        controller.startRecording()
        let id = try XCTUnwrap(controller.recordingID)
        XCTAssertEqual(controller.activeCaptureOrigin, .speechToText(.meeting))
        XCTAssertEqual(controller.liveSpeechToTextMode, .meeting)
        await controller.recordingSetupTask?.value
        controller.stopRecording()
        let take = try XCTUnwrap(controller.library.item(id))
        XCTAssertEqual(take.recordingPreset, AudioRecordingPreset.meeting.rawValue)
        XCTAssertEqual(take.collection, controller.collectionName(for: .meeting))
        XCTAssertNil(controller.activeCaptureOrigin, "no capture, no origin")
    }

    func testTheBadgeOpensThePageThatStartedTheCapture() throws {
        let (library, root) = try makeLibrary()
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = try makeController(library)
        defer { controller.shutdown() }
        controller.recordingSetupStatus = "Preparing recording..."
        controller.captureOrigin = .speechToText(.anyAudio)
        controller.selectPage(.music)
        controller.openActiveCapture()
        XCTAssertEqual(controller.page, .transcribe)
        XCTAssertEqual(controller.speechToTextMode, .anyAudio)
        controller.captureOrigin = .recordPage
        controller.selectPage(.music)
        controller.openActiveCapture()
        XCTAssertEqual(controller.page, .record)
        controller.recordingSetupStatus = nil
    }

    func testEachModeListsItsOwnItems() {
        func make(_ title: String, _ kind: AudioLibraryItem.Kind, preset: AudioRecordingPreset? = nil, collection: String = "",
                  source: UUID? = nil, transcript: Bool = false, minutesAgo: Double) -> AudioLibraryItem {
            var item = AudioLibraryItem(title: title, kind: kind)
            item.recordingPreset = preset?.rawValue; item.collection = collection; item.sourceID = source
            item.createdAt = Date(timeIntervalSince1970: 1_000_000 - minutesAgo * 60)
            if transcript { item.addTranscript([AudioLibrarySegment(start: 0, end: 1, text: "hi")], source: .final) }
            return item
        }
        let imported = make("file", .imported, minutesAgo: 9)
        let items = [
            make("note", .recording, preset: .voiceNote, transcript: true, minutesAgo: 1),
            make("call", .recording, preset: .meeting, minutesAgo: 2),
            make("old call", .recording, collection: "Calls", minutesAgo: 3),
            make("chime", .recording, preset: .sound, collection: "Calls", minutesAgo: 4),
            make("safari", .recording, preset: .appAudio, minutesAgo: 5),
            make("reference", .recording, minutesAgo: 6),
            imported,
            make("file transcript", .transcription, source: imported.id, transcript: true, minutesAgo: 8),
            make("orphan transcript", .transcription, source: UUID(), transcript: true, minutesAgo: 10),
            make("song", .music, minutesAgo: 11),
        ]
        let collections = ["meeting": "Calls"]
        func titles(_ mode: SpeechToTextMode) -> [String] {
            SpeechToTextLibrary.items(for: mode, in: items, collections: collections).map(\.title)
        }
        XCTAssertEqual(titles(.notes), ["note"])
        XCTAssertEqual(titles(.meeting), ["call", "old call"], "a stored preset wins over a matching collection")
        XCTAssertEqual(titles(.anyAudio), ["safari", "file", "orphan transcript"])
        let transcript = items[7]
        XCTAssertEqual(SpeechToTextLibrary.listSelection(transcript.id, in: items), imported.id, "a transcript selects its source row")
        XCTAssertEqual(SpeechToTextLibrary.transcriptItem(for: imported, in: items)?.id, transcript.id)
        XCTAssertEqual(SpeechToTextLibrary.transcriptItem(for: items[0], in: items)?.id, items[0].id, "a take holds its own transcript")
        XCTAssertNil(SpeechToTextLibrary.transcriptItem(for: items[1], in: items))
    }
}
