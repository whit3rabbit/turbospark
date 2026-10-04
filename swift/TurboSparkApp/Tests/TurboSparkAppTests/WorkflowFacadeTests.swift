import Foundation
import XCTest
@testable import TurboSparkApp

final class WorkflowFacadeTests: XCTestCase {
    func testDeclarationAndVersionedReferenceEnumerateTheSameMemberSet() throws {
        let reference = try WorkflowFacade.loadAPIReference()
        let documentedMembers = reference
            .components(separatedBy: .newlines)
            .compactMap { line -> String? in
                guard line.hasPrefix("### `"), line.hasSuffix("`") else {
                    return nil
                }
                return String(line.dropFirst(5).dropLast())
            }

        XCTAssertEqual(WorkflowFacade.version, 1)
        XCTAssertEqual(WorkflowFacade.entryForm, "async function workflow() { ... }")
        XCTAssertEqual(Set(WorkflowFacade.members.map(\.rawValue)), Set(documentedMembers))
        XCTAssertEqual(documentedMembers.count, WorkflowFacade.members.count)

        let canonicalProductions = try section(
            "## Canonical statement productions",
            before: "## Accepted source forms",
            in: reference)
        let requiredForms = [
            "agent(\"name\", \"role\");",
            "const value = await ask(\"actor\", \"prompt\", shape);",
            "const graph = parallel([{ id: \"node\", actor: \"name\", prompt: value, shape: shape, dependsOn: [\"prior\"], maxRetries: 1 }]);",
            "const values = await join(group);",
            "const review = await criticLoop(policy);",
            "const review = await criticLoop({ producer: { actor: \"writer\", prompt: value, shape: shape }, critic: { actor: \"reviewer\", prompt: value }, verdictField: \"verdict\", feedbackField: \"feedback\", maxIterations: 3 });",
            "phase(\"name\");",
            "const value = await world.read(operation);",
            "command(\"key\", { executable: \"/absolute/path\", workingDirectory: \"workspace\", argv: [\"fixed\", { name: \"path\", kind: \"workspaceInputPath\" }] });",
            "const value = await run(\"commandKey\", args);",
            "await report(value);",
            "await artifact(value);"
        ]
        for form in requiredForms {
            XCTAssertTrue(canonicalProductions.contains(form), "Missing canonical production: \(form)")
        }
    }

    func testRunDispatchFixturesEnforceLiteralKeyAndDynamicValuesOnly() throws {
        let reference = try WorkflowFacade.loadAPIReference()
        let fixtures = try runDispatchFixtures(in: reference)

        XCTAssertEqual(fixtures.count, 3)
        XCTAssertTrue(fixtures[0].source.contains("run(\"build\""))
        XCTAssertTrue(fixtures[1].source.contains("executable:"))
        XCTAssertTrue(fixtures[2].source.contains("run(commandName"))

        for fixture in fixtures {
            let accepted: Bool
            do {
                try WorkflowFacade.validateRunDispatch(
                    commandKeyForm: fixture.commandKeyForm,
                    argumentForm: fixture.argumentForm)
                accepted = true
            } catch {
                accepted = false
            }
            XCTAssertEqual(accepted, fixture.accepted, fixture.source)
        }
    }

    private func runDispatchFixtures(
        in reference: String
    ) throws -> [WorkflowFacade.RunDispatchShape] {
        let startMarker = "<!-- run-dispatch-fixtures:start -->"
        let endMarker = "<!-- run-dispatch-fixtures:end -->"
        guard let start = reference.range(of: startMarker),
              let end = reference.range(of: endMarker, range: start.upperBound..<reference.endIndex)
        else {
            throw FixtureError.missingBlock
        }

        let json = String(reference[start.upperBound..<end.lowerBound])
        return try JSONDecoder().decode([WorkflowFacade.RunDispatchShape].self, from: Data(json.utf8))
    }

    private func section(
        _ startHeading: String,
        before endHeading: String,
        in reference: String
    ) throws -> String {
        guard let start = reference.range(of: startHeading),
              let end = reference.range(of: endHeading, range: start.upperBound..<reference.endIndex)
        else {
            throw FixtureError.missingBlock
        }
        return String(reference[start.lowerBound..<end.lowerBound])
    }

    private enum FixtureError: Error {
        case missingBlock
    }
}
