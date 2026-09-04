import XCTest
@testable import TurboSparkApp

@MainActor
final class SystemPermissionsTests: XCTestCase {
    func testSystemFolderTypeDefinitions() {
        for folder in SystemFolderType.allCases {
            XCTAssertFalse(folder.rawValue.isEmpty)
            XCTAssertFalse(folder.systemImage.isEmpty)
            XCTAssertFalse(folder.pathDisplay.isEmpty)
            XCTAssertFalse(folder.description.isEmpty)
            XCTAssertNotNil(folder.defaultURL, "Folder defaultURL should resolve for \(folder.rawValue)")
        }
    }

    func testSystemPermissionsManagerStatusProbe() {
        let manager = SystemPermissionsManager()
        manager.refreshAllStatuses()

        for folder in SystemFolderType.allCases {
            let status = manager.folderStatuses[folder]
            XCTAssertNotNil(status, "Status should be populated for \(folder.rawValue)")
            // **THE HOME ROW IS NOT PINNED TO `.granted` ANY MORE**
            // (state#94). This case used to assert exactly that, and it
            // passed because the synchronous refresh called
            // `checkFolderStatus`, whose home probe lists `~` -- which is not
            // TCC-gated, so the row could only ever read `.granted` whatever
            // the user had allowed (`swift/CLAUDE.md` Gotcha 22's badge that
            // cannot fail, pinned as though it were behaviour). state#62
            // taught `probe` to ask about the PROTECTED CHILDREN instead and
            // fixed the background path only; pointing this one at the same
            // probe reddened the assertion, which is the defect reproducing
            // rather than a regression.
            XCTAssertEqual(
                status, SystemPermissionsManager.probe(folder),
                "The synchronous refresh and the background one must answer the same question.")
        }
    }

    func testEvaluatePathAccess() {
        let manager = SystemPermissionsManager()

        // Existing directory (home)
        let homeURL = FileManager.default.homeDirectoryForCurrentUser
        let homeStatus = manager.evaluatePathAccess(url: homeURL)
        XCTAssertEqual(homeStatus, .granted)

        // Non-existent directory
        let fakeURL = URL(fileURLWithPath: "/path/that/definitely/does/not/exist/turbospark_12345")
        let fakeStatus = manager.evaluatePathAccess(url: fakeURL)
        XCTAssertEqual(fakeStatus, .notFound)
    }

    func testCustomFolderGrantAndRemove() {
        let manager = SystemPermissionsManager()
        let testPath = "/tmp/turbospark_test_project_perm"

        // Ensure clean start
        manager.removeCustomFolder(id: testPath)
        let initialCount = manager.customFolders.count

        // Add
        manager.grantCustomFolder(path: testPath)
        XCTAssertEqual(manager.customFolders.count, initialCount + 1)
        XCTAssertTrue(manager.customFolders.contains(where: { $0.path == testPath }))

        // Adding same path again should not duplicate
        manager.grantCustomFolder(path: testPath)
        XCTAssertEqual(manager.customFolders.count, initialCount + 1)

        // Remove
        manager.removeCustomFolder(id: testPath)
        XCTAssertEqual(manager.customFolders.count, initialCount)
        XCTAssertFalse(manager.customFolders.contains(where: { $0.path == testPath }))
    }

    func testGrantedCustomFolderEncoding() throws {
        let folder = GrantedCustomFolder(path: "/Users/dev/workspace/app", name: "app")
        let data = try JSONEncoder().encode(folder)
        let decoded = try JSONDecoder().decode(GrantedCustomFolder.self, from: data)

        XCTAssertEqual(decoded.path, folder.path)
        XCTAssertEqual(decoded.name, folder.name)
        XCTAssertEqual(decoded.id, folder.path)
    }

    func testMacPrivacyPaneURLs() {
        let filesURL = MacPrivacyPane.filesAndFolders.url
        XCTAssertNotNil(filesURL)
        XCTAssertTrue(filesURL?.absoluteString.contains("Privacy_FilesAndFolders") == true)

        let allFilesURL = MacPrivacyPane.fullDiskAccess.url
        XCTAssertNotNil(allFilesURL)
        XCTAssertTrue(allFilesURL?.absoluteString.contains("Privacy_AllFiles") == true)

        let accessURL = MacPrivacyPane.accessibility.url
        XCTAssertNotNil(accessURL)
        XCTAssertTrue(accessURL?.absoluteString.contains("Privacy_Accessibility") == true)
    }
}
