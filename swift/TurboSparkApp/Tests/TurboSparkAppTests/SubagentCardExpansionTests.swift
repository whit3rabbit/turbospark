import XCTest
@testable import TurboSparkApp

/// Finished subagent cards fold by default; running ones stay open; the
/// user's own toggle always wins.
final class SubagentCardExpansionTests: XCTestCase {
    func testRunningStaysExpandedFinishedFolds() {
        XCTAssertTrue(SubagentCardExpansion.isExpanded(isRunning: true, userChoice: nil))
        XCTAssertFalse(SubagentCardExpansion.isExpanded(isRunning: false, userChoice: nil))
    }

    func testUserChoiceWins() {
        XCTAssertTrue(SubagentCardExpansion.isExpanded(isRunning: false, userChoice: true))
        XCTAssertFalse(SubagentCardExpansion.isExpanded(isRunning: true, userChoice: false))
    }

    func testResultSummaryIsFirstNonBlankLine() {
        XCTAssertEqual(SubagentCardExpansion.resultSummary("\n  Found 3 files.  \nDetails"), "Found 3 files.")
        XCTAssertNil(SubagentCardExpansion.resultSummary(" \n\t\n"))
        XCTAssertNil(SubagentCardExpansion.resultSummary(""))
    }
}
