import XCTest

@testable import TurboSparkApp

/// The test-results summary on a finished shell card (component audit,
/// test-results). Fixtures are captured `cargo test` and `swift test` runs
/// (toolchains of Nov 2026), trimmed only where noted, so the parser is held
/// to the shapes the runners really print. The durations in the mixed
/// fixtures were edited to sum to 1.0 s; every other byte is as printed. Every refusal case matters as
/// much as a parse: no summary is better than a wrong one.
final class TestRunSummaryTests: XCTestCase {

    // MARK: - Fixtures

    /// `cargo test --no-fail-fast` on a crate with a failing unit test
    /// binary, a passing integration binary, and a failing doc-test, with
    /// the compile lines interleaved.
    static let cargoMixed = """
           Compiling cfix v0.1.0 (/p/cfix)
            Finished `test` profile [unoptimized + debuginfo] target(s) in 0.84s
             Running unittests src/lib.rs (target/debug/deps/cfix-3a7c75ef04029b8b)

        running 4 tests
        test tests::adds ... ok
        test tests::slow ... ignored
        test tests::panics_msg ... FAILED
        test tests::breaks ... FAILED

        failures:

        ---- tests::panics_msg stdout ----

        thread 'tests::panics_msg' (1526009) panicked at src/lib.rs:19:31:
        boom here

        ---- tests::breaks stdout ----

        thread 'tests::breaks' (1526008) panicked at src/lib.rs:17:27:
        assertion `left == right` failed
          left: 3
         right: 4
        note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


        failures:
            tests::breaks
            tests::panics_msg

        test result: FAILED. 1 passed; 2 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.25s

        error: test failed, to rerun pass `--lib`
             Running tests/it.rs (target/debug/deps/it-a880ae6b68ba79da)

        running 2 tests
        test integration_ok ... ok
        test integration_two ... ok

        test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.50s

           Doc-tests cfix

        running 2 tests
        test src/lib.rs - ok (line 8) ... ok
        test src/lib.rs - add (line 3) ... FAILED

        failures:

        ---- src/lib.rs - add (line 3) stdout ----
        Test executable failed (exit status: 101).

        stderr:

        thread 'main' (1526125) panicked at /var/folders/T/rustdoctest/doctest_bundle_2024.rs:6:1:
        assertion `left == right` failed
          left: 2
         right: 3
        note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace



        failures:
            src/lib.rs - add (line 3)

        test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.25s

        all doctests ran in 0.44s; merged doctests compilation took 0.26s
        error: doctest failed, to rerun pass `--doc`
        error: 2 targets failed:
            `--lib`
            `--doc`
        """

    /// `cargo test -q --lib`: no per-test lines, only the failures sections.
    static let cargoQuiet = """

        running 4 tests
        .i 2/4
        tests::panics_msg --- FAILED
        tests::breaks --- FAILED

        failures:

        ---- tests::panics_msg stdout ----

        thread 'tests::panics_msg' (1526405) panicked at src/lib.rs:19:31:
        boom here
        note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

        ---- tests::breaks stdout ----

        thread 'tests::breaks' (1526404) panicked at src/lib.rs:17:27:
        assertion `left == right` failed
          left: 3
         right: 4


        failures:
            tests::breaks
            tests::panics_msg

        test result: FAILED. 1 passed; 2 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

        error: test failed, to rerun pass `--lib`
        """

    /// `swift test` on a package with XCTest and swift-testing tests, the
    /// build progress cut to two lines and the absolute paths shortened.
    static let swiftMixed = """
        Building for debugging...
        [126/127] SFixTests-product
        Build complete! (8.24 sec)
        Test Suite 'All tests' started at 2026-11-19 20:22:07.027.
        Test Suite 'SFixTests.xctest' started at 2026-11-19 20:22:07.028.
        Test Suite 'XTests' started at 2026-11-19 20:22:07.028.
        Test Case '-[SFixTests.XTests testFails]' started.
        /p/Tests/SFixTests/XTests.swift:5: error: -[SFixTests.XTests testFails] : XCTAssertEqual failed: ("1") is not equal to ("2")
        /p/Tests/SFixTests/XTests.swift:5: error: -[SFixTests.XTests testFails] : XCTAssertTrue failed - second
        Test Case '-[SFixTests.XTests testFails]' failed (0.961 seconds).
        Test Case '-[SFixTests.XTests testPasses]' started.
        Test Case '-[SFixTests.XTests testPasses]' passed (0.000 seconds).
        Test Case '-[SFixTests.XTests testSkips]' started.
        /p/Tests/SFixTests/XTests.swift:6: -[SFixTests.XTests testSkips] : Test skipped - not here
        Test Case '-[SFixTests.XTests testSkips]' skipped (0.001 seconds).
        Test Suite 'XTests' failed at 2026-11-19 20:22:07.991.
        \t Executed 3 tests, with 1 test skipped and 2 failures (0 unexpected) in 0.962 (0.963) seconds
        Test Suite 'SFixTests.xctest' failed at 2026-11-19 20:22:07.991.
        \t Executed 3 tests, with 1 test skipped and 2 failures (0 unexpected) in 0.962 (0.963) seconds
        Test Suite 'All tests' failed at 2026-11-19 20:22:07.991.
        \t Executed 3 tests, with 1 test skipped and 2 failures (0 unexpected) in 0.962 (0.963) seconds
        \u{25C7} Test run started.
        \u{21B3} Testing Library Version: 2403
        \u{21B3} Target Platform: arm64e-apple-macos14.0
        \u{25C7} Test "Named failure" started.
        \u{25C7} Suite Group started.
        \u{25C7} Test addsUp() started.
        \u{25C7} Test inner() started.
        \u{279C} Test skipped() skipped: "later"
        \u{2714} Test inner() passed after 0.001 seconds.
        \u{2714} Suite Group passed after 0.001 seconds.
        \u{2714} Test addsUp() passed after 0.001 seconds.
        \u{2718} Test "Named failure" recorded an issue at SFixTests.swift:4:44: Expectation failed: 1 + 1 == 3
        \u{21B3} 1 + 1 == 3 \u{2192} false
        \u{21B3}   1 + 1 \u{2192} 2
        \u{2718} Test "Named failure" failed after 0.001 seconds with 1 issue.
        \u{2718} Test run with 4 tests in 1 suite failed after 0.038 seconds with 1 issue.
        Note: Some test targets reported failures:
          - SFixTests (XCTest)
          - SFixTests (Swift Testing)
        """

    /// `swift test --filter testPasses`: XCTest only, under 'Selected tests'.
    static let xctestSelectedPass = """
        Building for debugging...
        Build complete! (0.49 sec)
        Test Suite 'Selected tests' started at 2026-11-19 20:22:16.183.
        Test Suite 'SFixTests.xctest' started at 2026-11-19 20:22:16.184.
        Test Suite 'XTests' started at 2026-11-19 20:22:16.184.
        Test Case '-[SFixTests.XTests testPasses]' started.
        Test Case '-[SFixTests.XTests testPasses]' passed (0.000 seconds).
        Test Suite 'XTests' passed at 2026-11-19 20:22:16.184.
        \t Executed 1 test, with 0 failures (0 unexpected) in 0.000 (0.001) seconds
        Test Suite 'SFixTests.xctest' passed at 2026-11-19 20:22:16.184.
        \t Executed 1 test, with 0 failures (0 unexpected) in 0.000 (0.001) seconds
        Test Suite 'Selected tests' passed at 2026-11-19 20:22:16.184.
        \t Executed 1 test, with 0 failures (0 unexpected) in 0.000 (0.001) seconds
        """

    /// `swift test --filter "addsUp|inner"` with the SF Symbols prefixes a
    /// terminal gets, here as the private-use code points.
    static let swiftTestingPass = """
        Build complete! (0.20 sec)
        \u{10F048} Test run started.
        \u{10F048} Test addsUp() started.
        \u{10F048} Suite Group started.
        \u{10F048} Test inner() started.
        \u{10105B} Test addsUp() passed after 0.001 seconds.
        \u{10105B} Test inner() passed after 0.001 seconds.
        \u{10105B} Suite Group passed after 0.001 seconds.
        \u{10105B} Test run with 2 tests in 1 suite passed after 0.001 seconds.
        """

    // MARK: - cargo test

    func testCargoSumsEveryBinaryIncludingDocTests() throws {
        let summary = try XCTUnwrap(TestRunSummary.parse(Self.cargoMixed))
        XCTAssertEqual(summary.passed, 4)
        XCTAssertEqual(summary.failed, 3)
        XCTAssertEqual(summary.skipped, 1)
        XCTAssertEqual(try XCTUnwrap(summary.durationSeconds), 1.0, accuracy: 0.0001)
        XCTAssertFalse(summary.allPassed)
    }

    func testCargoFailuresCarryTheirPanicMessage() throws {
        let summary = try XCTUnwrap(TestRunSummary.parse(Self.cargoMixed))
        XCTAssertEqual(summary.failures, [
            .init(name: "tests::breaks", message: "assertion `left == right` failed"),
            .init(name: "tests::panics_msg", message: "boom here"),
            .init(name: "src/lib.rs - add (line 3)", message: "assertion `left == right` failed"),
        ])
    }

    func testCargoQuietModeStillNamesFailures() throws {
        let summary = try XCTUnwrap(TestRunSummary.parse(Self.cargoQuiet))
        XCTAssertEqual(summary.passed, 1)
        XCTAssertEqual(summary.failed, 2)
        XCTAssertEqual(summary.skipped, 1)
        XCTAssertEqual(summary.failures.map(\.name), ["tests::breaks", "tests::panics_msg"])
        XCTAssertEqual(summary.failures.first?.message, "assertion `left == right` failed")
    }

    func testCargoPassingRunHasNoFailures() throws {
        let output = """
            running 2 tests
            test a ... ok
            test b ... ok

            test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.01s
            """
        let summary = try XCTUnwrap(TestRunSummary.parse(output))
        XCTAssertEqual(summary.passed, 2)
        XCTAssertEqual(summary.failed, 0)
        XCTAssertTrue(summary.allPassed)
        XCTAssertEqual(summary.failures, [])
    }

    func testCargoOldPanicFormatTakesTheQuotedMessage() throws {
        let output = """
            running 1 test
            test t ... FAILED

            failures:

            ---- t stdout ----
            thread 't' panicked at 'boom, again', src/lib.rs:3:5

            failures:
                t

            test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
            """
        let summary = try XCTUnwrap(TestRunSummary.parse(output))
        XCTAssertEqual(summary.failures, [.init(name: "t", message: "boom, again")])
    }

    // MARK: - swift test

    func testSwiftTestSumsXCTestAndSwiftTesting() throws {
        let summary = try XCTUnwrap(TestRunSummary.parse(Self.swiftMixed))
        // XCTest: 1 passed, 1 failed, 1 skipped. swift-testing: 2, 1, 1.
        XCTAssertEqual(summary.passed, 3)
        XCTAssertEqual(summary.failed, 2)
        XCTAssertEqual(summary.skipped, 2)
        XCTAssertEqual(try XCTUnwrap(summary.durationSeconds), 1.0, accuracy: 0.0001)
    }

    func testXCTestCountsFailedTestsNotFailedAssertions() throws {
        // "2 failures" are two assertions in one test.
        let summary = try XCTUnwrap(TestRunSummary.parse(Self.swiftMixed))
        XCTAssertEqual(summary.failures, [
            .init(name: "XTests.testFails", message: "XCTAssertEqual failed: (\"1\") is not equal to (\"2\")"),
            .init(name: "Named failure", message: "Expectation failed: 1 + 1 == 3"),
        ])
    }

    func testXCTestSelectedTestsPass() throws {
        let summary = try XCTUnwrap(TestRunSummary.parse(Self.xctestSelectedPass))
        XCTAssertEqual(summary.passed, 1)
        XCTAssertEqual(summary.failed, 0)
        XCTAssertEqual(summary.skipped, 0)
        XCTAssertTrue(summary.allPassed)
    }

    func testSwiftTestingAloneWithSymbolPrefixes() throws {
        let summary = try XCTUnwrap(TestRunSummary.parse(Self.swiftTestingPass))
        XCTAssertEqual(summary.passed, 2)
        XCTAssertEqual(summary.failed, 0)
        XCTAssertEqual(try XCTUnwrap(summary.durationSeconds), 0.001, accuracy: 0.00001)
    }

    func testANSIColoredOutputParses() throws {
        let colored = Self.cargoQuiet
            .replacingOccurrences(of: "FAILED.", with: "\u{1B}[31mFAILED\u{1B}[0m.")
        let summary = try XCTUnwrap(TestRunSummary.parse(colored))
        XCTAssertEqual(summary.failed, 2)
    }

    // MARK: - Refusals (nil when unsure)

    func testOutputWithoutATestRunIsNil() {
        XCTAssertNil(TestRunSummary.parse("a\nb\nc"))
        XCTAssertNil(TestRunSummary.parse(""))
        // A build failure never reaches a summary line.
        XCTAssertNil(TestRunSummary.parse("""
               Compiling cfix v0.1.0 (/p/cfix)
            error[E0425]: cannot find value `x` in this scope
            error: could not compile `cfix` (lib test) due to 1 previous error
            """))
    }

    func testCargoBinaryThatCrashedBeforeItsResultIsNil() {
        // The second binary died (signal), so its counts are unknown and a
        // sum of the first alone would understate the run.
        let output = """
            running 1 test
            test a ... ok

            test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

            running 3 tests
            test b ... ok
            error: test failed, to rerun pass `--test it`

            Caused by:
              process didn't exit successfully: `target/debug/deps/it-1` (signal: 11, SIGSEGV)
            """
        XCTAssertNil(TestRunSummary.parse(output))
    }

    func testCargoCountsThatDisagreeWithTheHeaderAreNil() {
        let output = """
            running 3 tests
            test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
            """
        XCTAssertNil(TestRunSummary.parse(output))
    }

    func testCargoFailureWhoseNameListIsMissingIsNil() {
        // The failed count says 2 but only one name is listed: the list was
        // cut, so the names would be incomplete.
        let output = """
            running 2 tests

            failures:
                a

            test result: FAILED. 0 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
            """
        XCTAssertNil(TestRunSummary.parse(output))
    }

    func testCompactedOutputIsNil() {
        let output = Self.cargoMixed.replacingOccurrences(
            of: "   Doc-tests cfix", with: "... [9000 chars truncated] ...")
        XCTAssertNil(TestRunSummary.parse(output))
    }

    func testXCTestThatCrashedBeforeTheTopSummaryIsNil() {
        let output = """
            Test Suite 'All tests' started at 2026-11-19 20:22:07.027.
            Test Case '-[M.T testA]' started.
            Test Case '-[M.T testA]' passed (0.000 seconds).
            Test Case '-[M.T testB]' started.
            Fatal error: boom
            error: Exited with unexpected signal code 5
            """
        XCTAssertNil(TestRunSummary.parse(output))
    }

    func testXCTestCaseLinesThatDisagreeWithTheTotalAreNil() {
        // Executed 2, but only one case line survived.
        let output = """
            Test Case '-[M.T testA]' passed (0.000 seconds).
            Test Suite 'All tests' passed at 2026-11-19 20:22:16.184.
            \t Executed 2 tests, with 0 failures (0 unexpected) in 0.000 (0.001) seconds
            """
        XCTAssertNil(TestRunSummary.parse(output))
    }

    func testXCTestFailureOutsideAnyFailedCaseIsNil() {
        // The suite reports a failure but every case passed (a class-level
        // setUp failure, say): no test name to show for it.
        let output = """
            Test Case '-[M.T testA]' passed (0.000 seconds).
            Test Suite 'All tests' failed at 2026-11-19 20:22:16.184.
            \t Executed 1 test, with 1 failure (1 unexpected) in 0.000 (0.001) seconds
            """
        XCTAssertNil(TestRunSummary.parse(output))
    }

    func testSwiftTestingRunThatNeverFinishedIsNil() {
        let output = """
            \u{25C7} Test run started.
            \u{2714} Test a() passed after 0.001 seconds.
            Fatal error: crashed
            """
        XCTAssertNil(TestRunSummary.parse(output))
    }

    func testSwiftTestingFailedRunWithNoFailedTestIsNil() {
        // The run failed on an issue recorded outside any test: the counts
        // cannot say which test broke, so no summary.
        let output = """
            \u{2714} Test a() passed after 0.001 seconds.
            \u{2718} Test run with 1 test failed after 0.002 seconds with 1 issue.
            """
        XCTAssertNil(TestRunSummary.parse(output))
    }

    func testEmptyRunIsNil() {
        // "0 passed" would read as a green run that tested nothing.
        let output = """
            running 0 tests

            test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 5 filtered out; finished in 0.00s
            """
        XCTAssertNil(TestRunSummary.parse(output))
    }
}
