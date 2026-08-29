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
            // On standard macOS systems, home and documents exist
            if folder == .home {
                XCTAssertEqual(status, .granted, "Home folder should be readable")
            }
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
