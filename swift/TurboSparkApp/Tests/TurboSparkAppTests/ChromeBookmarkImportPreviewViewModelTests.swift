import Foundation
import XCTest
@testable import TurboSparkApp

@MainActor
final class ChromeBookmarkImportPreviewViewModelTests: XCTestCase {
    func testPreviewAndSelectionDoNotWriteUntilExplicitConfirmation() throws {
        let (root, profileID) = try makeProfileRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let existing = BrowserBookmarkTree(sourceProfileDirectoryLabel: "Saved")
        var saveCount = 0
        var savedTrees: [BrowserBookmarkTree] = []
        let viewModel = ChromeBookmarkImportPreviewViewModel(
            importer: ChromeBookmarkImporter(profileRootURL: root),
            existingBookmarks: [existing]
        ) { trees in
            saveCount += 1
            savedTrees = trees
            return true
        }

        viewModel.scanProfiles()
        viewModel.setProfileSelected(profileID, selected: true)
        XCTAssertTrue(viewModel.previewSelectedProfiles())
        XCTAssertEqual(viewModel.stage, .preview)
        XCTAssertEqual(saveCount, 0)

        let guide = try XCTUnwrap(viewModel.rows(for: profileID).first { $0.title == "Guide" })
        viewModel.toggle(guide)
        XCTAssertTrue(viewModel.isSelected(guide))
        XCTAssertEqual(saveCount, 0)

        XCTAssertTrue(viewModel.confirmImport())
        XCTAssertTrue(viewModel.confirmImport(), "A repeated UI event must not duplicate the committed import.")

        XCTAssertEqual(saveCount, 1)
        XCTAssertEqual(savedTrees.first, existing)
        XCTAssertEqual(savedTrees.count, 2)
        XCTAssertEqual(viewModel.importReport?.importedBookmarkCount, 1)
    }

    func testFolderSelectionCanBeNarrowedToAnIndividualBookmark() throws {
        let (root, profileID) = try makeProfileRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let viewModel = ChromeBookmarkImportPreviewViewModel(
            importer: ChromeBookmarkImporter(profileRootURL: root)
        )
        viewModel.scanProfiles()
        viewModel.setProfileSelected(profileID, selected: true)
        XCTAssertTrue(viewModel.previewSelectedProfiles())
        let rows = viewModel.rows(for: profileID)
        let rootFolder = try XCTUnwrap(rows.first { $0.id == "root:bookmark_bar" })
        let guide = try XCTUnwrap(rows.first { $0.id == "bookmark:2" })

        viewModel.toggle(rootFolder)
        XCTAssertTrue(rows.allSatisfy { viewModel.isSelected($0) })
        viewModel.toggle(guide)

        XCTAssertFalse(viewModel.isSelected(guide))
        XCTAssertTrue(rows.filter { $0.id != guide.id && $0.id != rootFolder.id }
            .allSatisfy { viewModel.isSelected($0) })
        XCTAssertTrue(viewModel.confirmImport())
        XCTAssertEqual(viewModel.importReport?.importedBookmarkCount, 1)
        let imported = try XCTUnwrap(viewModel.importReport?.trees.first?.folders.first)
        XCTAssertTrue(imported.bookmarks.isEmpty, "The deselected Guide must not be imported via its parent")
        XCTAssertEqual(imported.folders.first?.bookmarks.map(\.id), ["bookmark:4"])
    }

    func testCancelAfterPreviewNeverSaves() throws {
        let (root, profileID) = try makeProfileRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        var saveCount = 0
        let viewModel = ChromeBookmarkImportPreviewViewModel(
            importer: ChromeBookmarkImporter(profileRootURL: root)
        ) { _ in
            saveCount += 1
            return true
        }
        viewModel.scanProfiles()
        viewModel.setProfileSelected(profileID, selected: true)
        XCTAssertTrue(viewModel.previewSelectedProfiles())
        let guide = try XCTUnwrap(viewModel.rows(for: profileID).first { $0.id == "bookmark:2" })
        viewModel.toggle(guide)

        viewModel.cancel()

        XCTAssertEqual(viewModel.stage, .chooseProfiles)
        XCTAssertTrue(viewModel.previews.isEmpty)
        XCTAssertEqual(saveCount, 0)
    }

    func testPartialIssuesRemainVisibleWhileRecoverableBookmarksImport() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let goodURL = root.appendingPathComponent("Default", isDirectory: true)
        let malformedURL = root.appendingPathComponent("Profile 2", isDirectory: true)
        try FileManager.default.createDirectory(at: goodURL, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: malformedURL, withIntermediateDirectories: true)
        try validBookmarks().write(to: goodURL.appendingPathComponent("Bookmarks"))
        try Data("invalid".utf8).write(to: malformedURL.appendingPathComponent("Bookmarks"))

        var saveCount = 0
        let importer = ChromeBookmarkImporter(profileRootURL: root)
        let viewModel = ChromeBookmarkImportPreviewViewModel(importer: importer) { _ in
            saveCount += 1
            return true
        }
        let profiles = importer.enumerateProfiles().profiles
        viewModel.scanProfiles()
        for profile in profiles { viewModel.setProfileSelected(profile.id, selected: true) }
        XCTAssertTrue(viewModel.previewSelectedProfiles())
        let guide = try XCTUnwrap(viewModel.rows(for: "Default").first { $0.id == "bookmark:2" })
        viewModel.toggle(guide)
        XCTAssertTrue(viewModel.confirmImport())
        XCTAssertEqual(saveCount, 1)
        XCTAssertEqual(viewModel.importReport?.skippedItemCount, 1)
        XCTAssertEqual(
            viewModel.importReport?.issues,
            [.malformedBookmarks(profileLabel: "Profile 2")]
        )
    }

    private func makeProfileRoot() throws -> (URL, String) {
        let root = try makeTemporaryDirectory()
        let profileURL = root.appendingPathComponent("Default", isDirectory: true)
        try FileManager.default.createDirectory(at: profileURL, withIntermediateDirectories: true)
        try validBookmarks().write(to: profileURL.appendingPathComponent("Bookmarks"))
        return (root, "Default")
    }

    private func validBookmarks() -> Data {
        Data(
            #"{"roots":{"bookmark_bar":{"type":"folder","id":"1","name":"Bookmarks bar","children":[{"type":"url","id":"2","name":"Guide","url":"https://docs.example/guide"},{"type":"folder","id":"3","name":"Work","children":[{"type":"url","id":"4","name":"Reference","url":"https://docs.example/reference"}]},{"type":"url","id":"5","name":"Unsafe","url":"javascript:alert(1)"}]}}}"#.utf8
        )
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
