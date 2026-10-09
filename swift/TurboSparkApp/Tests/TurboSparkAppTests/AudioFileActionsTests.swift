import AppKit
import UniformTypeIdentifiers
import XCTest
@testable import TurboSparkApp

final class AudioFileActionsTests: XCTestCase {
    @MainActor func testDeleteRemovesSelectedTakeButKeepsSharedAudioAndExportedCopy() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root.appendingPathComponent("vault") }, profileIDProvider: { "delete-take" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let assets = ManagedAssetStore(vault: vault)
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let controller = AudioWorkspaceController(library: library, assets: assets)
        defer { controller.shutdown() }
        let wav = try AudioMediaIO.wav(samples: [Float](repeating: 0.25, count: 24_000), sampleRate: 24_000, channels: 1)
        let asset = try assets.store(data: wav, fileName: "take.wav", mimeType: "audio/wav")
        var item = AudioLibraryItem(title: "First take", kind: .music)
        item.status = .completed
        item.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 24_000, channels: 1, frameCount: 24_000)]
        var other = item
        other.id = UUID(); other.title = "Other take"
        try controller.save(item); try controller.save(other)
        controller.selectedID = item.id
        controller.isPlaying = true
        controller.pendingTranscriptEdits[item.id] = []
        let exported = root.appendingPathComponent("export.wav")
        try wav.write(to: exported)

        controller.delete(item)

        XCTAssertEqual(try library.load().map(\.id), [other.id])
        XCTAssertEqual(controller.items.map(\.id), [other.id])
        XCTAssertNil(controller.selectedID)
        XCTAssertFalse(controller.isPlaying)
        XCTAssertNil(controller.pendingTranscriptEdits[item.id])
        XCTAssertEqual(try Data(contentsOf: exported), wav)
        try library.collectUnusedAssets()
        XCTAssertEqual(try Data(contentsOf: assets.materializedURL(for: asset.storedReference)), wav)
    }

    @MainActor func testDeleteRechecksActiveRunExportAndRecordingAtConfirmationTime() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "delete-guard" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let controller = AudioWorkspaceController(library: library, assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        var item = AudioLibraryItem(title: "Take", kind: .music)
        item.status = .completed
        try controller.save(item)
        controller.activeRunID = item.id
        controller.isBusy = true
        controller.delete(item)
        XCTAssertNotNil(try library.item(item.id), "The confirmation holds an older completed snapshot")
        controller.activeRunID = nil; controller.isBusy = false
        controller.isExporting = true
        controller.delete(item)
        XCTAssertNotNil(try library.item(item.id))
        controller.isExporting = false
        controller.recordingSetupStatus = "Preparing"
        controller.delete(item)
        XCTAssertNotNil(try library.item(item.id))
        controller.recordingSetupStatus = nil
        controller.delete(item)
        XCTAssertNil(try library.item(item.id))
    }

    @MainActor func testSaveDialogFormatChangeKeepsFilenameAndFolder() throws {
        let chooser = AudioExportPanel(title: "Evening mix")
        let folder = FileManager.default.temporaryDirectory
        chooser.panel.directoryURL = folder
        chooser.panel.nameFieldStringValue = "My revised mix.wav"
        for preset in AudioExportPreset.allCases {
            chooser.formatPicker.selectItem(at: try XCTUnwrap(AudioExportPreset.allCases.firstIndex(of: preset)))
            NSApp.sendAction(try XCTUnwrap(chooser.formatPicker.action), to: chooser.formatPicker.target, from: chooser.formatPicker)
            XCTAssertEqual(chooser.preset, preset)
            XCTAssertEqual(chooser.formatPicker.selectedItem?.title, preset.title)
            XCTAssertEqual(chooser.panel.nameFieldStringValue, "My revised mix." + preset.fileExtension)
            XCTAssertTrue(try XCTUnwrap(chooser.panel.allowedContentTypes.first).tags[.filenameExtension]?.contains(preset.fileExtension) == true)
            XCTAssertEqual(chooser.panel.directoryURL?.standardizedFileURL, folder.standardizedFileURL)
        }
        XCTAssertTrue(chooser.panel.canCreateDirectories)
        XCTAssertNotNil(chooser.panel.accessoryView)
    }

    func testFailedExportPreservesExistingFileAndSuccessfulExportReplacesIt() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let target = root.appendingPathComponent("saved.wav")
        let source = root.appendingPathComponent("source.wav")
        let original = Data("original".utf8), replacement = Data("replacement".utf8)
        try original.write(to: target)
        XCTAssertThrowsError(try AudioMediaIO.commitExport(from: source, to: target))
        XCTAssertEqual(try Data(contentsOf: target), original)
        try replacement.write(to: source)
        try AudioMediaIO.commitExport(from: source, to: target)
        XCTAssertEqual(try Data(contentsOf: target), replacement)
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath: root.path)), ["saved.wav", "source.wav"])
    }
}
