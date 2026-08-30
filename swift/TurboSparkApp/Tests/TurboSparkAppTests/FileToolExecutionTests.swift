import XCTest
@testable import TurboSparkApp

/// Regression tests for `read_file`'s line-bounds handling (state#1):
/// `allLines[(sLine - 1)..<eLine]` trapped the process whenever a
/// model-supplied `limit`/`end_line` combination produced `eLine < sLine`,
/// which happens the moment `limit` is read as the COUNT it conventionally
/// is (Claude/OpenAI's `offset` + `limit`) rather than as an absolute line
/// number.
final class ReadFileBoundsTests: XCTestCase {
    private func makeProject(fileContents: String, fileName: String = "sample.txt") throws -> (AppProject, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let fileURL = dir.appendingPathComponent(fileName)
        try fileContents.write(to: fileURL, atomically: true, encoding: .utf8)
        let project = AppProject(name: "test", rootDirectoryPath: dir.path)
        return (project, dir)
    }

    func testOffsetNearEndOfFileWithSmallLimitDoesNotTrap() async throws {
        let lines = (1...10).map { "line\($0)" }.joined(separator: "\n")
        let (project, dir) = try makeProject(fileContents: lines)
        defer { try? FileManager.default.removeItem(at: dir) }

        // Under the OLD code, `end_line` was always an absolute line number,
        // so `start_line: 8, limit: 2` (limit aliased into `end_line`)
        // computed `eLine = 2`, `sLine = 8`, and
        // `allLines[(8-1)..<2]` == `allLines[7..<2]` trapped the process.
        let call = AppToolCall(
            name: "read_file",
            arguments: ["path": "sample.txt", "start_line": "8", "limit": "2"],
            category: .fileRead
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "read_file must not error (or trap) on a valid offset+limit pair: \(result.output)")
        XCTAssertTrue(result.output.contains("line8"))
    }

    func testStartLineBeyondFileEndReturnsMessageNotTrap() async throws {
        let lines = (1...5).map { "line\($0)" }.joined(separator: "\n")
        let (project, dir) = try makeProject(fileContents: lines)
        defer { try? FileManager.default.removeItem(at: dir) }

        let call = AppToolCall(
            name: "read_file",
            arguments: ["path": "sample.txt", "start_line": "500", "limit": "10"],
            category: .fileRead
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.output.contains("out of bounds"))
    }

    func testEndLineConventionStillReadsAnAbsoluteRangeStartingAtLineOne() async throws {
        // `start_line=1` is the case where "count" and "absolute end line"
        // conventions coincide, which is what the tool's own documented
        // usage example (`AppToolRegistry.standardTools`) relies on.
        let lines = (1...200).map { "line\($0)" }.joined(separator: "\n")
        let (project, dir) = try makeProject(fileContents: lines)
        defer { try? FileManager.default.removeItem(at: dir) }

        let call = AppToolCall(
            name: "read_file",
            arguments: ["path": "sample.txt", "start_line": "1", "end_line": "100"],
            category: .fileRead
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.output.contains("line1\n")) // trailing \n excludes line10, line100, ...
        XCTAssertTrue(result.output.contains("line100"))
        XCTAssertFalse(result.output.contains("line101"))
    }
}

/// Regression tests for the tolerant `AppToolCall`/`AppToolResult` decoders
/// (state#4): the synthesized decoder they replaced required every field,
/// so ONE tool call written before a field like `category` existed threw
/// while decoding the `toolCalls` array inside a chat message, which took
/// the WHOLE chat archive down with it even though `AppChatMessage` and
/// `AppChat` are already tolerant one level up.
final class ToolCallTolerantDecodeTests: XCTestCase {
    func testAppToolCallDecodesWithMissingNewerFields() throws {
        // No `approvalID`, `category`, `riskAssessment`, or `createdAt` --
        // the shape a call would have had before those fields existed.
        let json = """
        {"id":"5C36DA02-0000-0000-0000-000000000000","name":"run_command","arguments":{"command":"ls"},"rawInvocation":"","status":"completed"}
        """
        let data = json.data(using: .utf8)!
        let call = try JSONDecoder().decode(AppToolCall.self, from: data)
        XCTAssertEqual(call.name, "run_command")
        XCTAssertEqual(call.category, .fileRead) // documented default
        XCTAssertNil(call.riskAssessment)
    }

    func testAppToolResultDecodesWithMissingNewerFields() throws {
        let json = """
        {"id":"5C36DA02-0000-0000-0000-000000000001","callID":"5C36DA02-0000-0000-0000-000000000002","output":"done"}
        """
        let data = json.data(using: .utf8)!
        let result = try JSONDecoder().decode(AppToolResult.self, from: data)
        XCTAssertEqual(result.output, "done")
        XCTAssertFalse(result.isError)
        XCTAssertEqual(result.durationSeconds, 0.0)
    }

    func testAChatMessageWithAPreCategoryToolCallStillDecodesTheWholeArchive() throws {
        // The exact failure shape from `swift/CLAUDE.md` Gotcha 13, one
        // level down: a message whose EMBEDDED tool call predates a field
        // that was later added to `AppToolCall` must not fail the message,
        // and a message that fails must not fail the whole archive.
        let json = """
        {
          "selectedChatID": "5C36DA02-0000-0000-0000-000000000010",
          "chats": [
            {
              "id": "5C36DA02-0000-0000-0000-000000000011",
              "title": "Old chat",
              "draft": "",
              "messages": [
                {
                  "role": "assistant",
                  "content": "Invoking tool",
                  "toolCalls": [
                    {"id":"5C36DA02-0000-0000-0000-000000000012","name":"run_command","arguments":{},"rawInvocation":"","status":"completed"}
                  ],
                  "toolResults": [
                    {"id":"5C36DA02-0000-0000-0000-000000000013","callID":"5C36DA02-0000-0000-0000-000000000012","output":"ok"}
                  ]
                }
              ]
            }
          ]
        }
        """
        let data = json.data(using: .utf8)!
        let archive = try JSONDecoder().decode(AppChatArchive.self, from: data)
        XCTAssertEqual(archive.chats.count, 1)
        XCTAssertEqual(archive.chats.first?.messages.count, 1)
        XCTAssertEqual(archive.chats.first?.messages.first?.toolCalls.first?.name, "run_command")
    }
}
