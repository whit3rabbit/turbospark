import Foundation
import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

/// Tests for folder import into chat attachments.
final class FolderAttachmentTests: XCTestCase {

    @MainActor
    func testImportFolderAttachesSupportedFiles() async throws {
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ts-folder-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Create sample files
        let swiftFile = tempDir.appendingPathComponent("Main.swift")
        try "print(\"Hello world\")".write(to: swiftFile, atomically: true, encoding: .utf8)

        let mdFile = tempDir.appendingPathComponent("README.md")
        try "# Test Project\nSome documentation".write(to: mdFile, atomically: true, encoding: .utf8)

        let txtFile = tempDir.appendingPathComponent("notes.txt")
        try "Important note".write(to: txtFile, atomically: true, encoding: .utf8)

        // Subdirectory with another file
        let subDir = tempDir.appendingPathComponent("src")
        try FileManager.default.createDirectory(at: subDir, withIntermediateDirectories: true)
        let subFile = subDir.appendingPathComponent("helper.py")
        try "def run(): pass".write(to: subFile, atomically: true, encoding: .utf8)

        // Excluded directory
        let gitDir = tempDir.appendingPathComponent(".git")
        try FileManager.default.createDirectory(at: gitDir, withIntermediateDirectories: true)
        let gitFile = gitDir.appendingPathComponent("config.txt")
        try "git config".write(to: gitFile, atomically: true, encoding: .utf8)

        let appModel = AppModel()
        let outcome = await AttachmentImporter.importFolder(tempDir, into: appModel, chatID: nil)

        XCTAssertEqual(outcome.importedCount, 4, "Should import 4 files, ignoring the one in .git")
        XCTAssertNil(outcome.errorText)
        XCTAssertEqual(appModel.promptAttachments.count, 4)

        let names = Set(appModel.promptAttachments.map(\.fileName))
        XCTAssertTrue(names.contains("Main.swift"))
        XCTAssertTrue(names.contains("README.md"))
        XCTAssertTrue(names.contains("notes.txt"))
        XCTAssertTrue(names.contains("helper.py"))
        XCTAssertFalse(names.contains("config.txt"))
    }

    @MainActor
    func testImportFolderWithNoSupportedFilesReportsFailure() async throws {
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ts-empty-folder-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let binaryFile = tempDir.appendingPathComponent("sample.bin")
        try Data([0x00, 0x01, 0x02, 0x03]).write(to: binaryFile)

        let appModel = AppModel()
        let initialCount = appModel.promptAttachments.count
        let outcome = await AttachmentImporter.importFolder(tempDir, into: appModel, chatID: nil)

        XCTAssertEqual(outcome.importedCount, 0)
        XCTAssertNotNil(outcome.errorText)
        XCTAssertEqual(appModel.promptAttachments.count, initialCount)
    }

    @MainActor
    func testImportFolderDefendsAgainstSymlinkCyclesAndEscapes() async throws {
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ts-symlink-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Normal file
        let regularFile = tempDir.appendingPathComponent("regular.txt")
        try "normal content".write(to: regularFile, atomically: true, encoding: .utf8)

        // Circular directory symlink (pointing back to tempDir)
        let cycleDir = tempDir.appendingPathComponent("cycle_dir")
        try? FileManager.default.createSymbolicLink(at: cycleDir, withDestinationURL: tempDir)

        // Escaping symlink pointing outside tempDir (e.g. /etc/hosts)
        let escapeLink = tempDir.appendingPathComponent("escape.txt")
        try? FileManager.default.createSymbolicLink(
            at: escapeLink,
            withDestinationURL: URL(fileURLWithPath: "/etc/hosts")
        )

        let appModel = AppModel()
        let outcome = await AttachmentImporter.importFolder(tempDir, into: appModel, chatID: nil)

        // Only regular.txt should be imported; cycle_dir skipped, escape.txt refused by containment
        XCTAssertEqual(outcome.importedCount, 1)
        let names = appModel.promptAttachments.map(\.fileName)
        XCTAssertTrue(names.contains("regular.txt"))
        XCTAssertFalse(names.contains("escape.txt"))
    }

    func testFlowingTextXMLParserPreservesTableCellsAndRows() throws {
        let xml = """
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:tbl>
              <w:tr>
                <w:tc><w:p><w:t>Header1</w:t></w:p></w:tc>
                <w:tc><w:p><w:t>Header2</w:t></w:p></w:tc>
              </w:tr>
              <w:tr>
                <w:tc><w:p><w:t>Val1</w:t></w:p></w:tc>
                <w:tc><w:p><w:t>Val2</w:t></w:p></w:tc>
              </w:tr>
            </w:tbl>
          </w:body>
        </w:document>
        """
        let parser = FlowingTextXMLParser()
        let result = try parser.parse(Data(xml.utf8))
        XCTAssertTrue(result.contains("Header1\tHeader2"))
        XCTAssertTrue(result.contains("Val1\tVal2"))
    }
}
