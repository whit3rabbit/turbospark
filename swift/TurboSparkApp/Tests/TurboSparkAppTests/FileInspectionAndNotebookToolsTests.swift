import XCTest
@testable import TurboSparkApp

final class FileInspectionAndNotebookToolsTests: XCTestCase {
    private func makeProject() throws -> (AppProject, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let project = AppProject(name: "test-notebook-snip", rootDirectoryPath: dir.path)
        return (project, dir)
    }

    // MARK: - notebook_edit / notebookedit

    func testNotebookEditReplaceCell() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let sampleNotebook = """
        {
          "cells": [
            {
              "cell_type": "markdown",
              "metadata": {},
              "source": ["# Old Title\\n"]
            },
            {
              "cell_type": "code",
              "metadata": {},
              "source": ["print('hello world')\\n"],
              "outputs": [],
              "execution_count": null
            }
          ],
          "metadata": {},
          "nbformat": 4,
          "nbformat_minor": 2
        }
        """
        let notebookURL = dir.appendingPathComponent("notebook.ipynb")
        try sampleNotebook.write(to: notebookURL, atomically: true, encoding: .utf8)

        for alias in ["notebook_edit", "notebookedit"] {
            let call = AppToolCall(
                name: alias,
                arguments: [
                    "path": "notebook.ipynb",
                    "cell_index": "0",
                    "action": "replace",
                    "source": "# Brand New Title\n"
                ],
                category: .fileWrite
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should succeed: \(result.output)")

            let updated = try String(contentsOf: notebookURL, encoding: .utf8)
            XCTAssertTrue(updated.contains("Brand New Title"))
        }
    }

    func testNotebookEditInsertAndDeleteCell() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let sampleNotebook = """
        {
          "cells": [
            {
              "cell_type": "code",
              "metadata": {},
              "source": ["x = 1\\n"],
              "outputs": [],
              "execution_count": null
            }
          ],
          "metadata": {},
          "nbformat": 4,
          "nbformat_minor": 2
        }
        """
        let notebookURL = dir.appendingPathComponent("notebook2.ipynb")
        try sampleNotebook.write(to: notebookURL, atomically: true, encoding: .utf8)

        // Insert new markdown cell
        let insertCall = AppToolCall(
            name: "notebook_edit",
            arguments: [
                "path": "notebook2.ipynb",
                "cell_index": "0",
                "action": "insert",
                "cell_type": "markdown",
                "source": "## Introduction\n"
            ],
            category: .fileWrite
        )
        let insertResult = await AppToolRegistry.execute(call: insertCall, in: project)
        XCTAssertFalse(insertResult.isError, "Insert should succeed: \(insertResult.output)")

        let insertedContent = try String(contentsOf: notebookURL, encoding: .utf8)
        XCTAssertTrue(insertedContent.contains("Introduction"))

        // Delete the second cell (the original code cell)
        let deleteCall = AppToolCall(
            name: "notebook_edit",
            arguments: [
                "path": "notebook2.ipynb",
                "cell_index": "1",
                "action": "delete"
            ],
            category: .fileWrite
        )
        let deleteResult = await AppToolRegistry.execute(call: deleteCall, in: project)
        XCTAssertFalse(deleteResult.isError, "Delete should succeed: \(deleteResult.output)")

        let deletedContent = try String(contentsOf: notebookURL, encoding: .utf8)
        XCTAssertFalse(deletedContent.contains("x = 1"))
    }

    func testNotebookEditOutOfBoundsIndex() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let sampleNotebook = """
        {
          "cells": [],
          "metadata": {},
          "nbformat": 4,
          "nbformat_minor": 2
        }
        """
        let notebookURL = dir.appendingPathComponent("empty.ipynb")
        try sampleNotebook.write(to: notebookURL, atomically: true, encoding: .utf8)

        let call = AppToolCall(
            name: "notebook_edit",
            arguments: [
                "path": "empty.ipynb",
                "cell_index": "5",
                "action": "replace",
                "source": "content"
            ],
            category: .fileWrite
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertTrue(result.isError, "Out of bounds cell index must report error")
    }

    // MARK: - snip / extract_snippet

    func testSnipExtractionAndBounds() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let content = (1...50).map { "Line \($0): data" }.joined(separator: "\n")
        let fileURL = dir.appendingPathComponent("data.txt")
        try content.write(to: fileURL, atomically: true, encoding: .utf8)

        for alias in ["snip", "extract_snippet"] {
            let call = AppToolCall(
                name: alias,
                arguments: [
                    "path": "data.txt",
                    "start_line": "10",
                    "end_line": "15"
                ],
                category: .fileRead
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should extract snippet: \(result.output)")
            XCTAssertTrue(result.output.contains("Line 10"))
            XCTAssertTrue(result.output.contains("Line 15"))
            XCTAssertFalse(result.output.contains("Line 9:"))
            XCTAssertFalse(result.output.contains("Line 16:"))
        }
    }

    func testSnipBeyondEOF() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let content = "Alpha\nBeta\nGamma"
        try content.write(to: dir.appendingPathComponent("short.txt"), atomically: true, encoding: .utf8)

        let call = AppToolCall(
            name: "snip",
            arguments: [
                "path": "short.txt",
                "start_line": "100",
                "end_line": "120"
            ],
            category: .fileRead
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertTrue(result.isError, "Snippet start line beyond EOF must report error")
    }

    // MARK: - send_user_file / senduserfile

    func testSendUserFileSuccessAndRefusal() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        var presentedURL: URL?
        SendUserFileExecutor.onFileSent = { _, url, _ in
            presentedURL = url
        }
        defer { SendUserFileExecutor.onFileSent = nil }

        let fileURL = dir.appendingPathComponent("report.pdf")
        try "dummy pdf bytes".write(to: fileURL, atomically: true, encoding: .utf8)

        for alias in ["send_user_file", "senduserfile"] {
            let call = AppToolCall(
                name: alias,
                arguments: ["path": "report.pdf"],
                category: .fileRead
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should succeed: \(result.output)")
            XCTAssertEqual(presentedURL?.lastPathComponent, "report.pdf")
        }

        // Missing file
        let missingCall = AppToolCall(
            name: "send_user_file",
            arguments: ["path": "missing.pdf"],
            category: .fileRead
        )
        let missingResult = await AppToolRegistry.execute(call: missingCall, in: project)
        XCTAssertTrue(missingResult.isError, "Sending non-existent file must report error")
    }
}
