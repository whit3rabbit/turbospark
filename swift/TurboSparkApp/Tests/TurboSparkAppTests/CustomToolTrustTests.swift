import XCTest
@testable import TurboSparkApp

final class CustomToolTrustTests: XCTestCase {
    private func properties(_ names: String...) -> JSONSchema {
        JSONSchema.object(
            properties: Dictionary(uniqueKeysWithValues: names.map { ($0, JSONSchemaProperty(type: "string")) }))
    }

    private func projectTool(parameters: JSONSchema, description: String = "d", enabled: Bool = true)
        -> CustomToolDefinition
    {
        CustomToolDefinition(
            name: "build", toolDescription: description,
            scope: .projectLocal(projectPath: "/tmp/proj"), parameters: parameters,
            execution: CustomToolExecution(type: .command, command: "echo {{msg}}"),
            isEnabled: enabled, sourcePath: "/tmp/proj/.turbospark/tools/build.json")
    }

    func testFingerprintChangesWhenParametersChange() {
        let a = projectTool(parameters: properties("msg"))
        let b = projectTool(parameters: properties("msg", "extra"))
        XCTAssertNotEqual(CustomToolTrustStore.fingerprint(of: a), CustomToolTrustStore.fingerprint(of: b))
        let store = CustomToolTrustStore(fileURL: nil)
        store.trust(a)
        XCTAssertTrue(store.isTrusted(a))
        XCTAssertFalse(store.isTrusted(b), "adding a declared parameter must ask the user again")
    }

    func testFingerprintChangesWhenDescriptionOrEnabledChange() {
        let base = CustomToolTrustStore.fingerprint(of: projectTool(parameters: properties("msg")))
        XCTAssertNotEqual(
            base, CustomToolTrustStore.fingerprint(
                of: projectTool(parameters: properties("msg"), description: "other")))
        XCTAssertNotEqual(
            base, CustomToolTrustStore.fingerprint(
                of: projectTool(parameters: properties("msg"), enabled: false)))
    }

    func testFingerprintIsStableForIdenticalDefinitions() {
        XCTAssertEqual(
            CustomToolTrustStore.fingerprint(of: projectTool(parameters: properties("a", "b"))),
            CustomToolTrustStore.fingerprint(of: projectTool(parameters: properties("b", "a"))))
    }
}
