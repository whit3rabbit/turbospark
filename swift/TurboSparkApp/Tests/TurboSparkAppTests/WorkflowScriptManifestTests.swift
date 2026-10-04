import CryptoKit
import Foundation
import XCTest
@testable import TurboSparkApp

final class WorkflowScriptManifestTests: XCTestCase {
    func testManifestPreservesActorRolePromptAndBindsItToSourceIdentity() throws {
        let firstSource = """
        async function workflow() {
          agent("writer", "Drafts a concise answer");
        }
        """
        let changedRoleSource = """
        async function workflow() {
          agent("writer", "Checks a concise answer");
        }
        """

        let firstResult = WorkflowScriptChecker.check(
            source: firstSource, name: "Role", args: [:]
        )
        let changedRoleResult = WorkflowScriptChecker.check(
            source: changedRoleSource, name: "Role", args: [:]
        )
        let first = try XCTUnwrap(firstResult.checked)
        let changedRole = try XCTUnwrap(changedRoleResult.checked)
        let encodedManifest = try JSONEncoder().encode(first.descriptor.manifest)
        let manifestObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encodedManifest) as? [String: Any]
        )
        let actors = try XCTUnwrap(manifestObject["actors"] as? [[String: String]])

        XCTAssertEqual(actors, [[
            "name": "writer",
            "rolePrompt": "Drafts a concise answer",
        ]])
        XCTAssertNotEqual(first.descriptor.sourceHash, changedRole.descriptor.sourceHash)
    }

    func testCheckBuildsPinnedManifestAndStableCheckedDescriptor() throws {
        let fixture = try makeExecutableFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let source = script(executablePath: fixture.symlink.path)
        let arguments = ["path": "Sources/main.swift", "mode": "safe", "note": "review this"]
        let first = WorkflowScriptChecker.check(source: source, name: "Build review", args: arguments)
        let checked = try XCTUnwrap(first.checked, first.diagnostics.map(\.message).joined(separator: "\n"))
        let descriptor = checked.descriptor
        let manifest = descriptor.manifest

        XCTAssertTrue(first.diagnostics.isEmpty)
        XCTAssertEqual(checked.ast.facadeVersion, WorkflowFacade.version)
        XCTAssertEqual(descriptor.source, source)
        XCTAssertEqual(descriptor.args.allValues, arguments)
        XCTAssertEqual(descriptor.facadeVersion, WorkflowFacade.version)
        XCTAssertEqual(descriptor.sourceHash, sha256(Data(source.utf8)))
        XCTAssertEqual(descriptor.sourceHash,
                       WorkflowScriptChecker.check(source: source, name: "Build review", args: arguments)
                           .checked?.descriptor.sourceHash)

        XCTAssertEqual(manifest.phases, ["draft", "review"])
        XCTAssertEqual(manifest.actors, [
            WorkflowActorSpec(name: "writer", rolePrompt: "Drafts an answer"),
            WorkflowActorSpec(name: "reviewer", rolePrompt: "Reviews an answer"),
        ])
        XCTAssertEqual(manifest.commandPins.count, 1)
        let pin = try XCTUnwrap(manifest.commandPins.first)
        XCTAssertEqual(pin.commandKey, "build")
        XCTAssertEqual(pin.executable.canonicalPath, fixture.executable.resolvingSymlinksInPath().path)
        XCTAssertEqual(pin.executable.sha256, sha256(fixture.contents))
        XCTAssertEqual(pin.workingDirectory, "workspace")
        XCTAssertEqual(pin.argvTemplate.count, 4)

        guard case .fixed("build") = pin.argvTemplate[0],
              case .dynamic(let pathRule) = pin.argvTemplate[1],
              case .dynamic(let modeRule) = pin.argvTemplate[2],
              case .dynamic(let noteRule) = pin.argvTemplate[3]
        else {
            return XCTFail("Expected fixed and named argument slots in declaration order")
        }
        XCTAssertEqual(pathRule.name, "input")
        XCTAssertEqual(pathRule.kind, .workspaceInputPath)
        XCTAssertGreaterThan(pathRule.maximumBytes, 0)
        XCTAssertEqual(modeRule.name, "mode")
        XCTAssertEqual(modeRule.kind, .allowedValue)
        XCTAssertEqual(modeRule.allowedValues, ["safe", "fast"])
        XCTAssertGreaterThan(modeRule.maximumBytes, 0)
        XCTAssertEqual(noteRule.name, "note")
        XCTAssertEqual(noteRule.kind, .boundedText)
        XCTAssertEqual(noteRule.maximumBytes, 64)

        for lane in ["writer", "reviewer"] {
            let indices = manifest.siteTable.filter { $0.lane == lane }.map(\.siteIndex)
            XCTAssertEqual(indices, Array(0..<indices.count), "\(lane) call sites need stable lane-local indices")
            XCTAssertFalse(indices.isEmpty, "Expected an actor call site for \(lane)")
        }
        XCTAssertTrue(manifest.siteTable.contains { $0.lane == "main" })
        XCTAssertEqual(Set(manifest.siteTable).count, manifest.siteTable.count)
    }

    func testCheckFailsClosedWhenExecutableIdentityCannotBeCaptured() {
        let source = script(executablePath: "/tmp/turbospark-workflow-missing-executable")
        let result = WorkflowScriptChecker.check(source: source, name: "Missing tool", args: [:])

        XCTAssertNil(result.checked)
        XCTAssertEqual(result.diagnostics.map(\.rule), [.executableIdentityUnavailable])
        XCTAssertTrue(result.diagnostics.first?.message.contains("executable") == true)
    }

    func testManifestIncludesBothBranchesAndOneSiteForARepeatedLoopCall() throws {
        let source = """
        async function workflow() {
          agent("writer", "Drafts an answer");
          if (args.enabled) {
            const answer = await ask("writer", "Draft", { answer: "string" });
            await report(answer);
          } else {
            await artifact({ fallback: true });
          }
          for (const item of args.items) {
            await report(item);
          }
        }
        """
        let result = WorkflowScriptChecker.check(source: source, name: "Branches", args: [:])
        let manifest = try XCTUnwrap(result.checked?.descriptor.manifest,
                                     result.diagnostics.map(\.message).joined(separator: "\n"))

        XCTAssertEqual(manifest.siteTable.filter { $0.lane == "writer" }.map(\.siteIndex), [0])
        XCTAssertEqual(manifest.siteTable.filter { $0.lane == "main" }.map(\.siteIndex), [0, 1, 2])
    }

    func testDefinitionCheckReturnsManifestWithoutRunDescriptorAndChecksDeclaredArguments() throws {
        let valid = """
        async function workflow() {
          const title = args.title;
          await report(title);
        }
        """
        let definition = WorkflowScriptChecker.checkDefinition(
            source: valid,
            name: "Saved report",
            declarations: [.init(name: "title", required: true, maximumUTF8Bytes: 128)])

        XCTAssertTrue(definition.isValid, definition.diagnostics.map(\.message).joined(separator: "\n"))
        XCTAssertNotNil(definition.ast)
        XCTAssertNotNil(definition.manifest)
        XCTAssertTrue(definition.manifest?.commandPins.isEmpty == true)

        let invalid = WorkflowScriptChecker.checkDefinition(source: valid, name: "Saved report", declarations: [])
        XCTAssertNotNil(invalid.ast)
        XCTAssertNil(invalid.manifest)
        XCTAssertEqual(invalid.diagnostics.map(\.rule), [.undeclaredWorkflowArgument])
    }

    private func makeExecutableFixture() throws -> (root: URL, executable: URL, symlink: URL, contents: Data) {
        let manager = FileManager.default
        let root = manager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try manager.createDirectory(at: root, withIntermediateDirectories: true)
        let executable = root.appendingPathComponent("tool")
        let contents = Data("#!/bin/sh\nexit 0\n".utf8)
        try contents.write(to: executable)
        try manager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
        let symlink = root.appendingPathComponent("tool-link")
        try manager.createSymbolicLink(at: symlink, withDestinationURL: executable)
        return (root, executable, symlink, contents)
    }

    private func script(executablePath: String) -> String {
        """
        async function workflow() {
          agent("writer", "Drafts an answer");
          agent("reviewer", "Reviews an answer");
          command("build", { executable: "\(executablePath)", workingDirectory: "workspace", argv: ["build", { name: "input", kind: "workspaceInputPath" }, { name: "mode", kind: "allowedValue", values: ["safe", "fast"] }, { name: "note", kind: "boundedText", maximumBytes: 64 }] });
          phase("draft");
          phase("review");
          phase("draft");
          const shape = { answer: "string" };
          const answer = await ask("writer", "Draft the result", shape);
          const graph = parallel([{ id: "review", actor: "reviewer", prompt: answer, shape: shape, dependsOn: [], maxRetries: 1 }]);
          const joined = await join(graph);
          const review = await criticLoop({ producer: { actor: "writer", prompt: answer, shape: shape }, critic: { actor: "reviewer", prompt: joined }, verdictField: "verdict", feedbackField: "feedback", maxIterations: 3 });
          const files = await world.read(glob("Sources/**"));
          const result = await run("build", { input: args.path, mode: args.mode, note: args.note });
          await report({ answer: joined });
          await artifact({ review: review });
        }
        """
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
