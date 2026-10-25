import Foundation
import XCTest

@testable import TurboSparkApp

/// Task 6.8 (requirement 3.3): no code path that starts a load, or offers a model
/// to load, reads `AppModel.installed`.
///
/// `installed` is the engine's available-only list at the last `refreshModels()`
/// plus the rows scanned from other apps' folders. It can be stale for the moment
/// between a drive leaving and the reload that follows, and a picker reading it
/// can offer a model whose files are gone. `AppModel.loadableModels` is the list
/// that answers from live availability. A new consumer of `installed` fails this
/// test until somebody decides which kind it is, and writes the reason down.
///
/// The scan reads every Swift file under `Sources/` with comments and string
/// literals removed. It counts a read as `model.installed`, `appModel.installed`
/// or `self.installed` anywhere, and, in a file that extends `AppModel`, a bare
/// `installed` that is dotted into, assigned or iterated. A file may hold exactly
/// the number listed: a removed read must be removed from the list too, so the
/// list cannot go slack and let a later read through.
final class LoadPathSourceScanTests: XCTestCase {
    private var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
    }

    /// Files that may read `installed`, how many reads each holds, and why none
    /// of them starts a load or offers a model to load.
    static let allowList: [String: (reads: Int, reason: String)] = [
        "State/AppModel+MenuBar.swift": (
            1, "The menu bar dashboard's name for a model the server already serves; it loads nothing"),
        "Server/ServerFavorite.swift": (
            2,
            "Names the rows a favorite saves (the ones the server serves now) and merges the scanned rows "
                + "into favorite resolution, which attaches only rows that resolve as available"),
        "Installation/CatalogSheet.swift": (
            2, "The Installed Models list of the catalog sheet: Delete only, it has no Load"),
        "Installation/ModelDetailPaneView.swift": (
            2,
            "Looks up the installed row of a hub entry for Delete and for Load and Chat; "
                + "Load and Chat goes through the gated openChatWithModel"),
        "Installation/ModelHubView.swift": (4, "Installed badges, the installed count and the Installed filter"),
        "Installation/ModelManagerHeaderView.swift": (2, "The count and size in the manager's header"),
        "Installation/ModelManagerView.swift": (1, "Which row the manager list selects first"),
        "State/AppModel+Downloads.swift": (1, "Whether a failed download's alias is installed already: dedupe"),
        "State/AppModel+EjectCoordination.swift": (1, "The name a toast gives the loaded model"),
        "State/AppModel+ModelDiscovery.swift": (
            2, "The two places that WRITE the list: the engine's rows, then the rows merged with the scan"),
        "State/AppModel+Models.swift": (
            1, "Finds the row of a folder the user chose; the load goes through requestLoad"),
        "State/AppModel+Server.swift": (2, "The names the Server pane's model table shows for attached models"),
        "State/InstalledModelDisplayList.swift": (1, "The merge that builds the manager's display list"),
        "State/LoadableModels.swift": (
            1, "The filter itself: it takes the list as a parameter and is the one place that narrows it"),
    ]

    /// Files that must read `loadableModels`: the pickers, the load paths and the
    /// selection restoration.
    static let loadablePaths = [
        "Chrome/ModelLoaderControl.swift", "Generation/PromptContextPills.swift",
        "Diagnostics/InspectorOptionsSection.swift", "Server/ServerMenuDashboardView.swift",
        "Humanize/HumanizeView.swift", "State/AppModel.swift", "State/AppModel+Server.swift",
        "State/AppModel+SlashParity.swift", "State/AppModel+ModelDiscovery.swift",
        "State/AppModel+Gating.swift",
    ]

    private func swiftFiles() -> [URL] {
        var files: [URL] = []
        if let walker = FileManager.default.enumerator(
            at: sourcesRoot, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        {
            for case let url as URL in walker where url.pathExtension == "swift" { files.append(url) }
        }
        return files.sorted { $0.path < $1.path }
    }

    private func relativePath(_ url: URL) -> String {
        String(url.path.dropFirst(sourcesRoot.path.count + 1))
    }

    /// `line` with the inside of every string literal blanked, so a sentence that
    /// says "installed" is not a read of the list.
    static func blankingStrings(_ line: String) -> String {
        var result = ""
        var inString = false
        var escaped = false
        for character in line {
            if inString {
                if escaped {
                    escaped = false
                    result.append(" ")
                } else if character == "\\" {
                    escaped = true
                    result.append(" ")
                } else if character == "\"" {
                    inString = false
                    result.append("\"")
                } else {
                    result.append(" ")
                }
            } else {
                if character == "\"" { inString = true }
                result.append(character)
            }
        }
        return result
    }

    private static let explicit = try! NSRegularExpression(
        pattern: #"\b(?:model|appModel|self)\??\.installed\b"#)
    private static let bare = try! NSRegularExpression(
        pattern: #"(?<![\w.])installed(?:\s*\.[A-Za-z]|\s*=(?!=)|\s+where\b)"#)
    private static let declaration = try! NSRegularExpression(pattern: #"\b(?:let|var)\s+installed\b"#)

    /// The lines of `text` that read the list.
    static func reads(in text: String) -> [Int] {
        let code = RawModelDeleteScanTests.code(of: text)
        let joined = code.joined(separator: "\n")
        let extendsModel = joined.contains("extension AppModel") || joined.contains("class AppModel")
        var found: [Int] = []
        for (index, raw) in code.enumerated() {
            let line = blankingStrings(raw)
            let range = NSRange(line.startIndex..., in: line)
            var hit = explicit.firstMatch(in: line, range: range) != nil
            if !hit, extendsModel, declaration.firstMatch(in: line, range: range) == nil {
                hit = bare.firstMatch(in: line, range: range) != nil
            }
            if hit { found.append(index + 1) }
        }
        return found
    }

    func testNoLoadPathReadsTheUnfilteredInstalledList() throws {
        var problems: [String] = []
        var seen: Set<String> = []
        for url in swiftFiles() {
            let path = relativePath(url)
            let lines = Self.reads(in: try String(contentsOf: url, encoding: .utf8))
            let allowed = Self.allowList[path]?.reads ?? 0
            if allowed > 0 { seen.insert(path) }
            if lines.count != allowed {
                problems.append(
                    "\(path): \(lines.count) read(s) of installed at lines \(lines), the list allows \(allowed)")
            }
        }
        for path in Self.allowList.keys where !seen.contains(path) {
            problems.append("\(path) is on the allow-list and has no reads: remove it")
        }
        XCTAssertEqual(
            problems.sorted(), [],
            "a picker or a load path reads installed instead of loadableModels; or a display-only reader is new "
                + "and belongs on the allow-list with its reason")
    }

    func testEveryAllowedReaderGivesItsReason() {
        for (path, entry) in Self.allowList {
            XCTAssertGreaterThan(entry.reason.count, 20, "\(path) needs a real reason")
            XCTAssertGreaterThan(entry.reads, 0, path)
        }
    }

    func testThePickersAndLoadPathsReadTheLoadableList() throws {
        for path in Self.loadablePaths {
            let text = try String(contentsOf: sourcesRoot.appendingPathComponent(path), encoding: .utf8)
            let code = RawModelDeleteScanTests.code(of: text).joined(separator: "\n")
            XCTAssertTrue(code.contains("loadableModels"), "\(path) must read loadableModels")
        }
    }

    // MARK: The scan itself

    func testTheScanFlagsReadsAndSparesTheLoadableListAndSentences() {
        let picker = """
            ForEach(model.installed) { item in
                Text(item.alias)
            }
            """
        XCTAssertEqual(Self.reads(in: picker), [1])

        let loadable = "ForEach(model.loadableModels) { item in Text(item.alias) }"
        XCTAssertEqual(Self.reads(in: loadable), [])

        let sentence = #"Text("No models are installed. Open the Model Hub.", bundle: .module)"#
        XCTAssertEqual(Self.reads(in: sentence), [])

        let comment = "// model.installed is the stale list\nlet x = 1"
        XCTAssertEqual(Self.reads(in: comment), [])

        let bareRead = "extension AppModel {\n    var a: Int { installed.count }\n}"
        XCTAssertEqual(Self.reads(in: bareRead), [2])

        let bareWrite = "extension AppModel {\n    func f() { installed = [] }\n}"
        XCTAssertEqual(Self.reads(in: bareWrite), [2])

        let declared = "final class AppModel {\n    @Published public var installed: [InstalledModel] = []\n}"
        XCTAssertEqual(Self.reads(in: declared), [])

        let notAModelFile = "func f() { let installed = foo.bar; installed.count }"
        XCTAssertEqual(Self.reads(in: notAModelFile), [], "a local named installed outside AppModel is not the list")
    }
}
