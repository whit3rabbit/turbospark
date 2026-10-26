import Foundation
import XCTest

@testable import TurboSparkApp

/// Every place that removes or forgets an audio install must also move the saved
/// choices off it and tell the app. A delete that skips `forgetChoices` leaves the
/// deleted id in the recipe and in the saved preferences, so the model stays in the
/// menu (the bug this guards). A delete that skips `onInstalledModelsChanged` leaves
/// Libraries and the Audio API pane listing it.
///
/// This reads every Swift file under `Sources/` with comments removed, like
/// `RawModelDeleteScanTests`, and checks the file that makes the engine call.
final class AudioDeleteRefreshScanTests: XCTestCase {
    private var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
    }

    private func sources() -> [(name: String, code: String)] {
        var files: [(String, String)] = []
        guard let walker = FileManager.default.enumerator(
            at: sourcesRoot, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { return [] }
        for case let url as URL in walker where url.pathExtension == "swift" {
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            files.append((url.lastPathComponent, RawModelDeleteScanTests.code(of: text).joined(separator: "\n")))
        }
        return files.sorted { $0.0 < $1.0 }
    }

    /// The scan must find the sites it guards, or it passes by reading nothing.
    func testTheDeleteSitesAreFound() {
        let files = sources()
        XCTAssertTrue(files.contains { $0.code.contains("AudioCatalog.delete(") }, "engine delete call not found")
    }

    /// The success branch follows the engine call within a few lines. A window,
    /// not the whole file: `removeLocalModel` also calls both, and one call site
    /// keeping them must not excuse another that lost them.
    func testEveryAudioDeleteMovesTheChoicesAndTellsTheApp() {
        for file in sources() {
            let lines = file.code.components(separatedBy: "\n")
            for (index, line) in lines.enumerated()
            where line.contains("AudioCatalog.delete(") || line.contains("AudioCatalog.deleteLegacy(") {
                let window = lines[index...].prefix(30).joined(separator: "\n")
                XCTAssertTrue(window.contains("forgetChoices(of:"),
                              "\(file.name):\(index + 1) removes an audio install without moving the saved choices off it")
                XCTAssertTrue(window.contains("onInstalledModelsChanged?("),
                              "\(file.name):\(index + 1) removes an audio install without telling the app")
            }
        }
    }
}
