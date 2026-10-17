import Foundation
import XCTest

@testable import TurboSparkApp

/// Task 6.3 (requirement 9.1): the app no longer offers moving the whole model
/// store, and no code path in it can reach the engine's relocation.
///
/// **THE OLD SHEET WAS A DATA-LOSS PATH, NOT ONLY A FEATURE.** It ran the
/// engine's `relocate_default_store`, which copies `models`, `installed.json`
/// and `models.json` out of the engine root and then deletes them from it.
/// Since task 1.1 the engine root is the home, and the home's `installed.json`
/// holds the registry rows of every library, so a user with a library who ran
/// the sheet lost that registry. Removing the only caller is what closes it,
/// which is why these tests scan the sources rather than trust a review: a
/// later change that wires the call back in must fail here.
///
/// The Rust export and the package wrapper stay (they have their own tests and
/// the C ABI is not changed by this task). Only the app is held to this rule.
final class MoveStoreRetirementTests: XCTestCase {
    /// `swift/TurboSparkApp`.
    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // Tests/TurboSparkAppTests/
            .deletingLastPathComponent()   // Tests/
            .deletingLastPathComponent()   // TurboSparkApp/
    }

    private var sourcesRoot: URL {
        packageRoot.appendingPathComponent("Sources")
    }

    /// Spellings that only the retired feature used. A hit in CODE means the
    /// sheet, its relocation call, or its row came back.
    ///
    /// - `ModelStoreMigrationSheet`: the sheet type.
    /// - `relocateStore`: the Swift package wrapper over the relocation export.
    /// - `ts_store_relocate`, `relocate_default_store`: the export itself, for
    ///   a caller that declared it by hand.
    /// - `StoreRelocation`: the wrapper's event and result types.
    /// - `canRelocateModelStore`: the sheet's idle gate.
    /// - `Move Store`, `Move Model Store`: the row's visible label and its
    ///   settings search title.
    static let retiredSpellings = [
        "ModelStoreMigrationSheet", "relocateStore", "ts_store_relocate",
        "relocate_default_store", "StoreRelocation", "canRelocateModelStore",
        "Move Store", "Move Model Store",
    ]

    /// The spellings found in `text` once comments are removed. Comments are
    /// skipped on purpose: `MoveStoreMigration` and `ModelStoreRootRecord`
    /// explain the history of the retired feature, and that explanation is the
    /// reason the conversion exists.
    static func retiredReferences(in text: String) -> Set<String> {
        let code = codeWithoutComments(text)
        return Set(retiredSpellings.filter { code.contains($0) })
    }

    /// Removes `//` and nested `/* */` comments, leaving string literals whole
    /// so a `//` inside a URL is not mistaken for one. Not a Swift parser: a
    /// quote inside an interpolation can flip the string state, which only
    /// risks keeping a comment (a visible false positive), never hiding code.
    static func codeWithoutComments(_ text: String) -> String {
        let chars = Array(text)
        var out = ""
        var i = 0
        var inString = false
        var blockDepth = 0
        while i < chars.count {
            let c = chars[i]
            let next: Character? = i + 1 < chars.count ? chars[i + 1] : nil
            if blockDepth > 0 {
                if c == "/", next == "*" {
                    blockDepth += 1
                    i += 2
                } else if c == "*", next == "/" {
                    blockDepth -= 1
                    i += 2
                } else {
                    i += 1
                }
                continue
            }
            if inString {
                out.append(c)
                if c == "\\", let next {
                    out.append(next)
                    i += 2
                    continue
                }
                if c == "\"" { inString = false }
                i += 1
                continue
            }
            if c == "\"" {
                inString = true
                out.append(c)
                i += 1
            } else if c == "/", next == "/" {
                while i < chars.count, chars[i] != "\n" { i += 1 }
            } else if c == "/", next == "*" {
                blockDepth = 1
                i += 2
            } else {
                out.append(c)
                i += 1
            }
        }
        return out
    }

    private func appSourceFiles() throws -> [URL] {
        let enumerator = try XCTUnwrap(
            FileManager.default.enumerator(at: sourcesRoot, includingPropertiesForKeys: nil))
        var files: [URL] = []
        for case let file as URL in enumerator where file.pathExtension == "swift" {
            files.append(file)
        }
        return files
    }

    // MARK: - The scanner itself

    /// Without this a scanner that stopped matching would turn the real scan
    /// into a pass for the wrong reason.
    func testTheScannerSeesCodeAndStringsButNotComments() {
        let sample = """
        // relocateStore named in a line comment
        /* ModelStoreMigrationSheet named in a block comment,
           /* nested canRelocateModelStore */ still a comment */
        let url = "https://example.com/ts_store_relocate" // Move Store in a trailing comment
        let label = Text("Move Model Store", bundle: .module)
        for try await event in TurboSparkCatalog.relocateStore(to: target) {}
        """
        XCTAssertEqual(
            Self.retiredReferences(in: sample),
            ["ts_store_relocate", "Move Model Store", "relocateStore"],
            "code and string literals count, comments (nested too) do not")
        XCTAssertEqual(
            Self.retiredReferences(in: "// relocateStore\n/* Move Store */\nlet x = 1\n"), [],
            "a file that only discusses the retired feature is clean")
    }

    // MARK: - The app can no longer reach the relocation

    func testNoAppSourceReferencesTheRetiredSheetOrTheRelocateExport() throws {
        let files = try appSourceFiles()
        XCTAssertGreaterThan(files.count, 200, "the scan stopped finding the app sources")

        var offenders: [String: Set<String>] = [:]
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            let hits = Self.retiredReferences(in: text)
            if !hits.isEmpty {
                offenders[file.lastPathComponent] = hits
            }
        }
        XCTAssertTrue(
            offenders.isEmpty,
            "the retired Move Store feature is referenced again: \(offenders)")
    }

    func testTheRetiredSheetFileIsGone() {
        let sheet = sourcesRoot.appendingPathComponent(
            "TurboSparkApp/Components/ModelStoreMigrationSheet.swift")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: sheet.path),
            "the sheet runs the engine relocation, which deletes the home's registry")
    }

    // MARK: - The Models and Storage pane and the settings search

    func testTheModelsPaneNoLongerOffersMoveStore() throws {
        let pane = sourcesRoot.appendingPathComponent(
            "TurboSparkApp/Components/ModelsSettingsPaneView.swift")
        let code = Self.codeWithoutComments(try String(contentsOf: pane, encoding: .utf8))

        XCTAssertFalse(code.contains("Move Store"), "no row, label or search title for the move")
        XCTAssertFalse(code.contains("showingStoreMigration"), "no state for the retired sheet")
        XCTAssertFalse(code.contains(".sheet(isPresented: $showingStoreMigration"))
        XCTAssertTrue(
            code.contains("TurboSpark Models Storage"),
            "the storage section keeps its heading and its path rows")
        XCTAssertFalse(
            code.contains("actions: { }") || code.contains("actions: {}"),
            "a header with an empty actions slot would leave a dangling spacer")
    }

    func testTheSettingsSearchCatalogNoLongerListsMoveStore() {
        let entries = SettingsControlCatalog.entries
        XCTAssertFalse(
            entries.contains { $0.title == "Move Model Store" },
            "the row is gone, so its search entry would open a pane with nothing to find")
        XCTAssertFalse(
            entries.contains {
                let title = $0.title.lowercased()
                return title.contains("move") && title.contains("store")
            },
            "no search entry may offer moving the store")

        // The regeneration removed one entry and no other.
        let models = Set(entries.filter { $0.pane == .models }.map(\.title))
        XCTAssertEqual(
            models,
            [
                "Additional Model Folders", "Hugging Face API Token",
                "Hugging Face Authentication", "LM Studio Library Integration",
                "Maximum allocation", "Minimum AutoFit context length",
                "Mirror Endpoint ($HF_ENDPOINT)", "Model loading guardrails",
            ])
        XCTAssertEqual(Set(entries.map(\.id)).count, entries.count, "search targets stay unique")
    }
}
