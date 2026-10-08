import Foundation
import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

/// Regression tests for the medium "Correctness" review findings (first third).
final class CorrectnessWave3Tests: XCTestCase {

    // MARK: Storage source

    func testStoreRootMatchRequiresPathComponentBoundary() {
        let roots = ["/Volumes/SSD/Models"]
        XCTAssertTrue(ModelFeatureDescriptor.isUnderStoreRoot(
            lowercasedPath: "/volumes/ssd/models/models/text/x", roots: roots))
        XCTAssertTrue(ModelFeatureDescriptor.isUnderStoreRoot(
            lowercasedPath: "/volumes/ssd/models", roots: roots))
        // A sibling that merely shares the prefix must not match.
        XCTAssertFalse(ModelFeatureDescriptor.isUnderStoreRoot(
            lowercasedPath: "/volumes/ssd/models-old/x", roots: roots))
        XCTAssertFalse(ModelFeatureDescriptor.isUnderStoreRoot(
            lowercasedPath: "/elsewhere/x", roots: roots))
    }

    // MARK: Window minimum size

    func testMinimumWindowSizeIsClampedToTheScreen() {
        let clamped = MainWindowMinimumSize.clamped(
            CGSize(width: 1622, height: 700), toScreen: CGSize(width: 1512, height: 900))
        XCTAssertEqual(clamped.width, 1512)
        XCTAssertEqual(clamped.height, 700)
        XCTAssertEqual(
            MainWindowMinimumSize.clamped(CGSize(width: 900, height: 600), toScreen: nil),
            CGSize(width: 900, height: 600))
    }

    // MARK: Element picker coordinates

    func testPickerPointAddsScrollOffsetBeforeUndoingZoom() {
        let p = BrowserViewportController.cssPoint(
            viewPoint: CGPoint(x: 100, y: 50),
            scrollOffset: CGPoint(x: 600, y: 0),
            scale: 2)
        XCTAssertEqual(p, CGPoint(x: 350, y: 25))
    }

    // MARK: Same-document navigation

    func testFragmentOnlyChangeIsSameDocument() throws {
        let page = try XCTUnwrap(URL(string: "https://example.com/docs"))
        let toc = try XCTUnwrap(URL(string: "https://example.com/docs#install"))
        let other = try XCTUnwrap(URL(string: "https://example.com/other#install"))
        XCTAssertTrue(WebKitBrowserEngine.isSameDocumentNavigation(from: page, to: toc))
        XCTAssertFalse(WebKitBrowserEngine.isSameDocumentNavigation(from: page, to: other))
        XCTAssertFalse(WebKitBrowserEngine.isSameDocumentNavigation(from: page, to: page))
        XCTAssertFalse(WebKitBrowserEngine.isSameDocumentNavigation(from: nil, to: toc))
    }

    // MARK: Symlink ordering

    private func makeTree() throws -> URL {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("wave3-\(UUID().uuidString)")
        // Nested under a directory named "build": ancestors must not matter.
        let base = root.appendingPathComponent("build/proj")
        try FileManager.default.createDirectory(
            at: base.appendingPathComponent("zz_real"), withIntermediateDirectories: true)
        try "hello".write(
            to: base.appendingPathComponent("zz_real/note.md"), atomically: true, encoding: .utf8)
        // Sorts before its target.
        try FileManager.default.createSymbolicLink(
            at: base.appendingPathComponent("aa_link"),
            withDestinationURL: base.appendingPathComponent("zz_real"))
        return base
    }

    func testFolderImportIgnoresSkipListedAncestorsAndLinkOrder() throws {
        let base = try makeTree()
        defer { try? FileManager.default.removeItem(at: base.deletingLastPathComponent().deletingLastPathComponent()) }
        let urls = AttachmentImporter.collectFolderURLs(from: base, maxFiles: 50)
        XCTAssertTrue(
            urls.contains { $0.lastPathComponent == "note.md" && $0.path.contains("zz_real") },
            "files under a real dir must survive an earlier-enumerated link: \(urls)")
    }

    func testProjectFileIndexKeepsRealDirectoryWhenLinkSortsFirst() throws {
        let base = try makeTree()
        defer { try? FileManager.default.removeItem(at: base.deletingLastPathComponent().deletingLastPathComponent()) }
        let entries = ProjectFileIndex.scan(root: base, maxEntries: 100, maxDepth: 8)
        XCTAssertTrue(
            entries.contains { $0.relativePath == "zz_real/note.md" },
            "got \(entries.map(\.relativePath))")
    }

    // MARK: openChat

    @MainActor
    func testOpenChatSelectsTargetWithoutFallbackChatCreation() {
        let model = AppModel()
        let projectA = AppProject(name: "A", rootDirectoryPath: nil)
        let projectB = AppProject(name: "B", rootDirectoryPath: nil)
        model.projects = [projectA, projectB]
        let inA = AppChat(projectID: projectA.id, title: "in A")
        let oldB = AppChat(projectID: projectB.id, title: "old B")
        let newerB = AppChat(projectID: projectB.id, title: "newer B")
        model.chats = [inA, oldB, newerB]
        model.selectedProjectID = projectA.id
        model.selectedChatID = inA.id
        let before = model.chats.count

        model.openChat(id: oldB.id)

        XCTAssertEqual(model.selectedProjectID, projectB.id)
        XCTAssertEqual(model.selectedChatID, oldB.id, "must not land on the project's newest chat")
        XCTAssertEqual(model.chats.count, before, "no empty chat may be created")
    }
}
