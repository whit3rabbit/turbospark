import Foundation
import XCTest

@testable import TurboSparkApp

/// Task 6.4, the BINDING part of the 4.6 review: no code in the app may remove
/// an installed model's recorded path itself.
///
/// The old delete ran `FileManager.removeItem(model.path)` after ANY error from
/// the engine's alias delete. For a library row that deletes the recorded path
/// of a drive that is away (or a different drive mounted at the old mount
/// point), follows links planted on the drive, bypasses the engine's `busy`
/// refusal and leaves the registry and the marker listing a model that is gone.
/// The engine owns the bytes of every row it lists; the app only asks.
///
/// This scan reads every Swift file under `Sources/` (comments removed) and
/// fails when a `removeItem` call has the path of a model-shaped value in its
/// own statement or in the few lines that build its argument. It is specific
/// enough to pass for the app's own scratch files (profile archives, staging
/// folders, the settings records) and to fail if the fallback returns.
final class RawModelDeleteScanTests: XCTestCase {
    private var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
    }

    private func swiftFiles() -> [URL] {
        var files: [URL] = []
        if let walker = FileManager.default.enumerator(
            at: sourcesRoot, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        {
            for case let url as URL in walker where url.pathExtension == "swift" { files.append(url) }
        }
        return files.sorted { $0.path < $1.path }
    }

    /// The text with line and block comments removed. A `//` inside a string
    /// (a URL) cuts the rest of its line, which only matters for lines that
    /// have no `removeItem` before it, and keeps the line count.
    static func code(of text: String) -> [String] {
        var lines: [String] = []
        var inBlock = false
        for rawLine in text.components(separatedBy: "\n") {
            var line = rawLine
            if inBlock {
                if let end = line.range(of: "*/") {
                    line = String(line[end.upperBound...])
                    inBlock = false
                } else {
                    lines.append("")
                    continue
                }
            }
            while let start = line.range(of: "/*") {
                if let end = line.range(of: "*/", range: start.upperBound..<line.endIndex) {
                    line.replaceSubrange(start.lowerBound..<end.upperBound, with: " ")
                } else {
                    line = String(line[..<start.lowerBound])
                    inBlock = true
                    break
                }
            }
            if let comment = line.range(of: "//") { line = String(line[..<comment.lowerBound]) }
            lines.append(line)
        }
        return lines
    }

    /// Identifiers an installed model or image row is held in across the app.
    private static let modelNames = [
        "model", "installedModel", "installed", "image", "row", "entry", "item", "inst", "selected",
        "selectedModel", "candidate", "target",
    ]

    /// A `removeItem` call whose window (this line and the seven before it) names
    /// `<model-shaped identifier>.path`, or expands one.
    static func offences(in lines: [String]) -> [Int] {
        let names = modelNames.joined(separator: "|")
        let pathOfModel = try! NSRegularExpression(pattern: "\\b(\(names))\\??\\.path\\b")
        var found: [Int] = []
        for (index, line) in lines.enumerated() where line.contains("removeItem") {
            let window = lines[max(0, index - 7)...index].joined(separator: "\n")
            let range = NSRange(window.startIndex..., in: window)
            if pathOfModel.firstMatch(in: window, range: range) != nil { found.append(index + 1) }
        }
        return found
    }

    func testNoCodeRemovesAModelsRecordedPathItself() throws {
        var problems: [String] = []
        for url in swiftFiles() {
            let text = try String(contentsOf: url, encoding: .utf8)
            for line in Self.offences(in: Self.code(of: text)) {
                problems.append("\(url.lastPathComponent):\(line)")
            }
        }
        XCTAssertEqual(
            problems, [],
            "an engine row is deleted through TurboSparkCatalog.deleteInstall(installId:), never by removing its path")
    }

    func testTheModelFilesDoNotRemoveAnythingAtAll() throws {
        let guarded = [
            "State/AppModel+Models.swift", "State/AppModel+ModelDiscovery.swift",
            "State/AppModel+ImageModels.swift", "State/ModelLibraryService.swift",
            "State/ModelDeleteText.swift", "State/InstalledModelRowModel.swift",
            "State/InstalledModelDisplayList.swift", "Installation/InstalledModelDetailPaneView.swift",
            "Installation/InstalledModelActionBarView.swift", "Installation/ModelManagerView.swift",
            "Installation/ImageModelManagerView.swift", "Installation/CatalogSheet.swift",
            "Installation/ModelDetailPaneView.swift",
        ]
        for relative in guarded {
            let text = try String(contentsOf: sourcesRoot.appendingPathComponent(relative), encoding: .utf8)
            let code = Self.code(of: text).joined(separator: "\n")
            XCTAssertFalse(code.contains("removeItem"), "\(relative) must not remove files itself")
        }
    }

    func testTheAliasDeleteIsNotCalledFromTheApp() throws {
        // `TurboSparkCatalog.delete(alias)` is This Mac only and answers
        // `use_install_id` for a library row; every row goes through its id.
        var calls: [String] = []
        for url in swiftFiles() {
            let text = try String(contentsOf: url, encoding: .utf8)
            for (index, line) in Self.code(of: text).enumerated()
            where line.contains("TurboSparkCatalog.delete(") {
                calls.append("\(url.lastPathComponent):\(index + 1)")
            }
        }
        XCTAssertEqual(calls, [])
    }

    func testTheDeleteGoesThroughTheServiceSeamAndForgetsThroughIt() throws {
        let models = try String(
            contentsOf: sourcesRoot.appendingPathComponent("State/AppModel+Models.swift"), encoding: .utf8)
        let code = Self.code(of: models).joined(separator: "\n")
        XCTAssertTrue(code.contains("deleteInstall(installId:"))
        XCTAssertTrue(code.contains("forget(installId:"))
        let images = try String(
            contentsOf: sourcesRoot.appendingPathComponent("State/AppModel+ImageModels.swift"), encoding: .utf8)
        XCTAssertTrue(Self.code(of: images).joined(separator: "\n").contains("deleteImageInstall(installId:"))
    }

    // MARK: The scan itself

    func testTheScanFlagsTheOldFallbackAndSparesScratchFiles() {
        let oldFallback = """
            if !deleted {
                let expanded = ModelStorageManager.expandPath(model.path)
                if FileManager.default.fileExists(atPath: expanded) {
                    do {
                        try FileManager.default.removeItem(atPath: expanded)
                    } catch {}
                }
            }
            """
        XCTAssertEqual(Self.offences(in: oldFallback.components(separatedBy: "\n")), [5])

        let direct = "try FileManager.default.removeItem(atPath: installedModel.path)"
        XCTAssertEqual(Self.offences(in: [direct]), [1])

        let scratch = """
            let staging = FileManager.default.temporaryDirectory.appendingPathComponent("x")
            defer { try? FileManager.default.removeItem(at: staging) }
            """
        XCTAssertEqual(Self.offences(in: scratch.components(separatedBy: "\n")), [])
    }

    func testTheCommentStripperKeepsLineNumbers() {
        let lines = Self.code(of: "a\n// removeItem(model.path)\n/* removeItem(row.path)\n still */ b\nc")
        XCTAssertEqual(lines.count, 5)
        XCTAssertEqual(Self.offences(in: lines), [], "a comment that mentions a call is not a call")
    }
}
