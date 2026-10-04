import Foundation
import XCTest
@testable import TurboSparkApp

final class ChromeBookmarkImporterTests: XCTestCase {
    func testSelectingAFolderImportsItsDescendantsWithoutSelectingEveryBookmark() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let profileURL = root.appendingPathComponent("Default", isDirectory: true)
        try FileManager.default.createDirectory(at: profileURL, withIntermediateDirectories: true)
        try validBookmarks().write(to: profileURL.appendingPathComponent("Bookmarks"))
        let importer = ChromeBookmarkImporter(profileRootURL: root)
        let profileID = try XCTUnwrap(importer.enumerateProfiles().profiles.first?.id)
        _ = importer.preview(profileIDs: [profileID])

        let nested = importer.importConfirmed(selectionsByProfileID: [profileID: ["folder:3"]])
        XCTAssertEqual(nested.importedBookmarkCount, 1)
        let nestedRoot = try XCTUnwrap(nested.trees.first?.folders.first)
        XCTAssertTrue(nestedRoot.bookmarks.isEmpty, "A sibling bookmark was not selected.")
        XCTAssertEqual(nestedRoot.folders.first?.bookmarks.map(\.title), ["Reference"])

        let all = importer.importConfirmed(selectionsByProfileID: [profileID: ["root:bookmark_bar"]])
        XCTAssertEqual(all.importedBookmarkCount, 2)
        XCTAssertEqual(all.trees.first?.folders.first?.bookmarks.map(\.title), ["Guide"])
        XCTAssertEqual(all.trees.first?.folders.first?.folders.first?.bookmarks.map(\.title), ["Reference"])
    }

    func testEnumeratesSanitizedProfileAndImportsOnlyConfirmedEntries() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let profileURL = root.appendingPathComponent(" Profile\t1 ", isDirectory: true)
        try FileManager.default.createDirectory(at: profileURL, withIntermediateDirectories: true)
        try validBookmarks().write(to: profileURL.appendingPathComponent("Bookmarks"))

        let importer = ChromeBookmarkImporter(profileRootURL: root)
        let enumeration = importer.enumerateProfiles()
        let profile = try XCTUnwrap(enumeration.profiles.first)
        XCTAssertEqual(profile.displayLabel, "Profile1")

        let preview = try XCTUnwrap(importer.preview(profileIDs: [profile.id]).first)
        XCTAssertNil(preview.issue)
        XCTAssertTrue(preview.selectableIdentifiers.contains("bookmark:2"))
        XCTAssertTrue(preview.selectableIdentifiers.contains("folder:3"))
        XCTAssertEqual(preview.skippedItemCount, 1, "The javascript: bookmark must be skipped.")

        let imported = importer.importConfirmed(selectionsByProfileID: [
            profile.id: ["bookmark:2", "folder:3", "bookmark:4", "bookmark:missing"],
        ])

        XCTAssertEqual(imported.importedBookmarkCount, 2)
        XCTAssertEqual(imported.skippedItemCount, 2)
        XCTAssertEqual(imported.issues, [.unknownSelection(profileLabel: "Profile1", count: 1)])
        let tree = try XCTUnwrap(imported.trees.first)
        XCTAssertEqual(tree.sourceProfileDirectoryLabel, "Profile1")
        XCTAssertEqual(tree.folders.first?.bookmarks.map(\.title), ["Guide"])
        XCTAssertEqual(tree.folders.first?.folders.first?.bookmarks.map(\.title), ["Reference"])
    }

    func testCredentialStoreFilesAreNeverReadEvenWhenPresentBesideBookmarks() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let profileURL = root.appendingPathComponent("Default", isDirectory: true)
        try FileManager.default.createDirectory(at: profileURL, withIntermediateDirectories: true)
        try validBookmarks().write(to: profileURL.appendingPathComponent("Bookmarks"))
        try Data("credential material".utf8).write(to: profileURL.appendingPathComponent("Login Data"))
        try Data("profile metadata".utf8).write(to: root.appendingPathComponent("Local State"))
        let recorder = FileReadRecorder()
        let importer = ChromeBookmarkImporter(profileRootURL: root) { url in
            recorder.paths.append(url.standardizedFileURL.path)
            return try Data(contentsOf: url)
        }

        let profiles = importer.enumerateProfiles().profiles
        _ = importer.preview(profileIDs: Set(profiles.map(\.id)))

        XCTAssertEqual(recorder.paths, [profileURL.appendingPathComponent("Bookmarks").standardizedFileURL.path])
        XCTAssertTrue(ChromeBookmarkImporter.isAllowedProfileFile("Bookmarks"))
        XCTAssertFalse(ChromeBookmarkImporter.isAllowedProfileFile("Login Data"))
        XCTAssertFalse(ChromeBookmarkImporter.isAllowedProfileFile("Local State"))
    }

    func testMalformedAndUnreadableBookmarkFilesReportPartialFailure() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let profileURL = root.appendingPathComponent("Default", isDirectory: true)
        try FileManager.default.createDirectory(at: profileURL, withIntermediateDirectories: true)
        try Data("not json".utf8).write(to: profileURL.appendingPathComponent("Bookmarks"))

        let malformedImporter = ChromeBookmarkImporter(profileRootURL: root)
        let profileID = try XCTUnwrap(malformedImporter.enumerateProfiles().profiles.first?.id)
        let malformed = try XCTUnwrap(malformedImporter.preview(profileIDs: [profileID]).first)
        XCTAssertEqual(malformed.issue, .malformedBookmarks(profileLabel: "Default"))
        XCTAssertEqual(
            malformedImporter.importConfirmed(selectionsByProfileID: [profileID: []]).issues,
            [.malformedBookmarks(profileLabel: "Default")]
        )

        let unreadableImporter = ChromeBookmarkImporter(profileRootURL: root) { _ in
            throw CocoaError(.fileReadNoPermission)
        }
        let unreadableID = try XCTUnwrap(unreadableImporter.enumerateProfiles().profiles.first?.id)
        let unreadable = try XCTUnwrap(unreadableImporter.preview(profileIDs: [unreadableID]).first)
        XCTAssertEqual(unreadable.issue, .bookmarksFileUnreadable(profileLabel: "Default"))
    }

    func testMalformedChildDoesNotDiscardRecoverableBookmarks() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let profileURL = root.appendingPathComponent("Default", isDirectory: true)
        try FileManager.default.createDirectory(at: profileURL, withIntermediateDirectories: true)
        let contents = #"{"roots":{"bookmark_bar":{"name":"Bookmarks bar","children":[{"type":"url","id":"2","name":"Guide","url":"https://docs.example/guide"},null,{"type":"url","id":"4","name":"Reference","url":"https://docs.example/reference"}]}}}"#
        try Data(contents.utf8).write(to: profileURL.appendingPathComponent("Bookmarks"))
        let importer = ChromeBookmarkImporter(profileRootURL: root)
        let profileID = try XCTUnwrap(importer.enumerateProfiles().profiles.first?.id)
        let preview = try XCTUnwrap(importer.preview(profileIDs: [profileID]).first)

        XCTAssertEqual(preview.folders.first?.bookmarks.map(\.title), ["Guide", "Reference"])
        XCTAssertEqual(preview.skippedItemCount, 1)
        let imported = importer.importConfirmed(selectionsByProfileID: [profileID: ["root:bookmark_bar"]])
        XCTAssertEqual(imported.importedBookmarkCount, 2)
        XCTAssertEqual(imported.skippedItemCount, 1)
    }

    func testAbsentRootAndOversizeBookmarkFilesAreReported() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let missingRoot = root.appendingPathComponent("missing", isDirectory: true)
        XCTAssertEqual(
            ChromeBookmarkImporter(profileRootURL: missingRoot).enumerateProfiles().issues,
            [.profileRootUnavailable]
        )

        let profileRoot = root.appendingPathComponent("Chrome", isDirectory: true)
        let profileURL = profileRoot.appendingPathComponent("Default", isDirectory: true)
        try FileManager.default.createDirectory(at: profileURL, withIntermediateDirectories: true)
        let oversized = Data(repeating: 0x20, count: ChromeBookmarkImporter.maximumBookmarkFileBytes + 1)
        try oversized.write(to: profileURL.appendingPathComponent("Bookmarks"))
        let importer = ChromeBookmarkImporter(profileRootURL: profileRoot)
        let profileID = try XCTUnwrap(importer.enumerateProfiles().profiles.first?.id)

        let preview = try XCTUnwrap(importer.preview(profileIDs: [profileID]).first)
        XCTAssertEqual(preview.issue, .bookmarksFileTooLarge(profileLabel: "Default"))
        XCTAssertTrue(importer.importConfirmed(selectionsByProfileID: [profileID: []]).trees.isEmpty)
    }

    func testMissingBookmarksProfileIsNotExposedAsImportable() throws {
        let root = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("Default", isDirectory: true),
            withIntermediateDirectories: true
        )

        let result = ChromeBookmarkImporter(profileRootURL: root).enumerateProfiles()
        XCTAssertTrue(result.profiles.isEmpty)
        XCTAssertEqual(result.issues, [.noProfilesFound])
    }

    private func validBookmarks() -> Data {
        Data(
            #"{"roots":{"bookmark_bar":{"type":"folder","id":"1","name":"Bookmarks bar","children":[{"type":"url","id":"2","name":"Guide","url":"https://docs.example/guide"},{"type":"folder","id":"3","name":"Work","children":[{"type":"url","id":"4","name":"Reference","url":"https://docs.example/reference"}]},{"type":"url","id":"5","name":"Unsafe","url":"javascript:alert(1)"}]},"other":{"type":"folder","id":"6","name":"Other","children":[]}}}"#.utf8
        )
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

private final class FileReadRecorder {
    var paths: [String] = []
}
