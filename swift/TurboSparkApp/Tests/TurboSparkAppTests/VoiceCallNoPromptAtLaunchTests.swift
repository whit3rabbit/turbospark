import XCTest
@testable import TurboSparkApp

/// OC-08: the call asks macOS for nothing on its own. Source scans in the
/// style of `RecordingPermissionsTests`, plus the settings default.
final class VoiceCallNoPromptAtLaunchTests: XCTestCase {
    private func swiftFiles() throws -> [URL] {
        let sources = VoiceCallSources.directory()
        let walker = try XCTUnwrap(FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil))
        return walker.compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" }
    }

    private func callFiles() throws -> [URL] {
        try swiftFiles().filter { $0.lastPathComponent.hasPrefix("VoiceCall") || $0.lastPathComponent == "AppModel+VoiceCall.swift" }
    }

    func testTheCallSourcesNeverCallAPromptAPI() throws {
        let files = try callFiles()
        XCTAssertGreaterThanOrEqual(files.count, 7, "the scan must see the call sources it protects")
        var violations: [String] = []
        for file in files {
            for (index, line) in try String(contentsOf: file, encoding: .utf8).components(separatedBy: "\n").enumerated() {
                let code = line.trimmingCharacters(in: .whitespaces)
                if code.hasPrefix("//") || code.hasPrefix("///") { continue }
                for api in ["requestAccess(", "CGRequestScreenCaptureAccess(", "requestRecordPermission", "AXIsProcessTrustedWithOptions(", "SCShareableContent."]
                where code.contains(api) { violations.append("\(file.lastPathComponent):\(index + 1) \(api)") }
            }
        }
        XCTAssertEqual(violations, [])
    }

    /// The controller starts only from the host's Call action path, which sets
    /// the access request first. No launch, appear or settings path starts one.
    func testACallStartsOnlyFromTheCallActionPath() throws {
        var starters: [String] = []
        for file in try swiftFiles() {
            let text = try String(contentsOf: file, encoding: .utf8)
            for (index, line) in text.components(separatedBy: "\n").enumerated() {
                let code = line.trimmingCharacters(in: .whitespaces)
                if code.hasPrefix("//") { continue }
                if code.contains("call.start()") || code.contains("voiceCall.start()") {
                    starters.append("\(file.lastPathComponent):\(index + 1)")
                }
            }
        }
        XCTAssertFalse(starters.isEmpty, "the scan must see the start path it protects")
        XCTAssertTrue(starters.allSatisfy { $0.hasPrefix("VoiceCallHostView.swift") }, "\(starters)")
        let host = try String(contentsOf: VoiceCallSources.directory()
            .appendingPathComponent("Audio/VoiceCallHostView.swift"), encoding: .utf8)
        // Start is reached either with nothing missing or from the sheet's onReady.
        XCTAssertTrue(host.contains("guard !permissions.missing(needed).isEmpty else"))
        XCTAssertTrue(host.contains("RecordingAccessRequest(purpose: .call"))
        XCTAssertTrue(host.contains(".recordingAccessGate($accessRequest"))
    }

    func testNoLaunchPathTouchesTheCall() throws {
        // `voiceCall` is built lazily, so nothing at launch may name it
        // beyond the persisted flag and the section-change end.
        let sources = VoiceCallSources.directory()
        for name in ["State/AppModel+Persistence.swift", "App/TurboSparkApp.swift"] {
            let text = try String(contentsOf: sources.appendingPathComponent(name), encoding: .utf8)
            XCTAssertFalse(text.contains("voiceCall.start"), name)
            XCTAssertFalse(text.contains("AudioCaptureVoiceSource("), name)
        }
    }

    func testVoiceCallModeIsOffByDefaultAndOldSettingsFilesDecode() throws {
        XCTAssertFalse(MacAppSettings().voiceCallEnabled)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: Data("{}".utf8))
        XCTAssertFalse(decoded.voiceCallEnabled)
        var settings = MacAppSettings()
        settings.voiceCallEnabled = true
        let roundTrip = try JSONDecoder().decode(MacAppSettings.self, from: JSONEncoder().encode(settings))
        XCTAssertTrue(roundTrip.voiceCallEnabled)
    }

    func testTheSpeechModelStatusWordsEachBlocker() {
        typealias M = VoiceCallSpeechModel
        XCTAssertEqual(M.status(profileListed: true, canRun: true, unavailableReason: nil, installed: true, english: true), .ready)
        guard case .unavailable(_, let download) = M.status(profileListed: true, canRun: true, unavailableReason: nil, installed: false, english: true) else { return XCTFail() }
        XCTAssertTrue(download, "a missing install offers the download")
        guard case .unavailable(_, let notListed) = M.status(profileListed: false, canRun: true, unavailableReason: nil, installed: false, english: true) else { return XCTFail() }
        XCTAssertFalse(notListed)
        XCTAssertEqual(M.status(profileListed: true, canRun: false, unavailableReason: "needs M4", installed: true, english: true),
                       .unavailable(reason: "needs M4", canDownload: false))
        guard case .unavailable(_, let englishDownload) = M.status(profileListed: true, canRun: true, unavailableReason: nil, installed: true, english: false) else { return XCTFail() }
        XCTAssertFalse(englishDownload, "English-only is not fixed by a download")
    }

    func testTheCallCatalogChordIsRegisteredAndUnreserved() {
        let command = KeyboardShortcutCatalog.command(for: .toggleVoiceCall)
        XCTAssertEqual(command?.defaultChord.display, "\u{21E7} \u{2318} Y")
        XCTAssertEqual(command?.rebindable, true)
        XCTAssertFalse(command?.defaultChord.isReserved ?? true)
    }
}
