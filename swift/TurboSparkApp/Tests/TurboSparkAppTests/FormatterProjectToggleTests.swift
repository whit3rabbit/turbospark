import XCTest
@testable import TurboSparkApp

/// OC-11 per-project toggle: the project can only NARROW the app-wide switch,
/// and an archive from before the field loads as "inherit".
final class FormatterProjectToggleTests: XCTestCase {
    // MARK: persistence

    private func decodeArchive(_ json: String) throws -> AppProjectArchive {
        try JSONDecoder().decode(AppProjectArchive.self, from: Data(json.utf8))
    }

    func testArchiveWrittenBeforeTheFieldDecodesAsInherit() throws {
        let archive = try decodeArchive("""
        {"selectedProjectID": null, "projects": [
          {"id": "4C3B1B0E-0D43-4C1B-9E52-6F1A1A1A1A11", "name": "Legacy", "maxAutonomousSteps": 7}
        ]}
        """)
        XCTAssertEqual(archive.projects.count, 1)
        XCTAssertEqual(archive.projects[0].name, "Legacy")
        XCTAssertEqual(archive.projects[0].maxAutonomousSteps, 7, "the rest of the row still decodes")
        XCTAssertEqual(archive.projects[0].formatterMode, .inherit)
    }

    func testUnknownOrWrongTypedValueDecodesAsInheritNotAThrow() throws {
        for raw in ["\"on\"", "\"sometimes\"", "7", "null", "true"] {
            let archive = try decodeArchive("""
            {"projects": [{"name": "P", "formatterMode": \(raw)}]}
            """)
            XCTAssertEqual(archive.projects.first?.formatterMode, .inherit, "value \(raw)")
        }
    }

    /// Mutation check: make the decode fall back to `.off` and the legacy
    /// archive test above fails; drop the key from `CodingKeys` and this fails.
    func testOffSurvivesAnArchiveRoundTrip() throws {
        var project = AppProject(name: "P")
        project.formatterMode = .off
        let data = try JSONEncoder().encode(AppProjectArchive(projects: [project, AppProject(name: "Q")]))
        let back = try JSONDecoder().decode(AppProjectArchive.self, from: data)
        XCTAssertEqual(back.projects.map(\.formatterMode), [.off, .inherit])
    }

    func testNewProjectsInheritByDefault() {
        XCTAssertEqual(AppProject(name: "P").formatterMode, .inherit)
    }

    // MARK: precedence

    /// Mutation check: change `&&` to `||` in `allowsPass` and the first two
    /// assertions fail.
    func testBothSwitchesMustAllowThePass() {
        // Project off beats app on.
        XCTAssertFalse(ProjectFormatterMode.allowsPass(appWideEnabled: true, project: .off))
        // App off beats project inherit: a project can never widen.
        XCTAssertFalse(ProjectFormatterMode.allowsPass(appWideEnabled: false, project: .inherit))
        XCTAssertFalse(ProjectFormatterMode.allowsPass(appWideEnabled: false, project: .off))
        XCTAssertTrue(ProjectFormatterMode.allowsPass(appWideEnabled: true, project: .inherit))
    }

    // MARK: the step honors both

    private func step(enabled: Bool, row: FormatterRow) -> FormatterPostEditStep {
        var step = FormatterPostEditStep()
        step.isEnabled = { enabled }
        step.rows = { [row] }
        step.searchPath = { nil }
        return step
    }

    private func context(_ fixture: FormatterFixture, mode: ProjectFormatterMode) -> PostEditContext {
        PostEditContext(
            toolName: "write_file", relativePaths: ["Sources/A.swift"], rootURL: fixture.project,
            chatID: UUID(), formatterMode: mode)
    }

    /// Mutation check: delete the `context.formatterMode` test in the step and
    /// the project-off case formats the file.
    func testProjectOffStopsTheFormatterWhileAppIsOn() async throws {
        let fixture = try FormatterFixture()
        let row = try fixture.row(FormatterFixture.upcase)
        let note = await step(enabled: true, row: row).run(context(fixture, mode: .off))
        XCTAssertNil(note)
        XCTAssertEqual(try fixture.read("Sources/A.swift"), "let a = 1\n")
    }

    func testAppOffStopsTheFormatterWhileProjectInherits() async throws {
        let fixture = try FormatterFixture()
        let row = try fixture.row(FormatterFixture.upcase)
        let note = await step(enabled: false, row: row).run(context(fixture, mode: .inherit))
        XCTAssertNil(note)
        XCTAssertEqual(try fixture.read("Sources/A.swift"), "let a = 1\n")
    }

    /// The control: with both allowing it the same fixture does format, so the
    /// two refusals above are the toggles and not a broken fixture.
    func testBothAllowingFormatsTheFile() async throws {
        let fixture = try FormatterFixture()
        let row = try fixture.row(FormatterFixture.upcase)
        let note = await step(enabled: true, row: row).run(context(fixture, mode: .inherit))
        XCTAssertNotNil(note)
        XCTAssertEqual(try fixture.read("Sources/A.swift"), "LET A = 1\n")
    }

    /// The seam hands the project's mode to the step. Mutation check: build
    /// the context without `formatterMode:` in `PostEditSeam.note` and the
    /// project-off case formats.
    func testSeamCarriesTheProjectModeToTheStep() async throws {
        let fixture = try FormatterFixture()
        let row = try fixture.row(FormatterFixture.upcase)
        var project = AppProject(name: "p", rootDirectoryPath: fixture.project.path)
        let edit = AppToolCall(
            name: "write_file", arguments: ["path": "Sources/A.swift", "content": "x"], category: .fileWrite)
        project.formatterMode = .off
        let off = await PostEditSeam.note(
            call: edit, isError: false, project: project, chatID: UUID(),
            steps: [step(enabled: true, row: row)])
        XCTAssertNil(off)
        XCTAssertEqual(try fixture.read("Sources/A.swift"), "let a = 1\n")
        project.formatterMode = .inherit
        let on = await PostEditSeam.note(
            call: edit, isError: false, project: project, chatID: UUID(),
            steps: [step(enabled: true, row: row)])
        XCTAssertNotNil(on)
    }
}
