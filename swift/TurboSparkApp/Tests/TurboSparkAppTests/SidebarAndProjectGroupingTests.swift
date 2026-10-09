import Foundation
import XCTest

@testable import TurboSparkApp

final class SidebarAndProjectGroupingTests: XCTestCase {
    func testCompactTimestampsFollowTheSelectedLanguage() {
        let now = Date(timeIntervalSince1970: 1_800_000_000)
        let date = now.addingTimeInterval(-52 * 86400)
        XCTAssertEqual(
            MessageTimestampFormatter.compactRelativeString(
                for: date, relativeTo: now, locale: Locale(identifier: "fr")),
            "52j")
        XCTAssertEqual(
            MessageTimestampFormatter.compactRelativeString(
                for: date, relativeTo: now, locale: Locale(identifier: "en")),
            "52d")
    }

    func testCompactRelativeTimestamps() {
        let now = Date()

        // Less than 60 seconds
        let secondsAgo = now.addingTimeInterval(-25)
        XCTAssertEqual(MessageTimestampFormatter.compactRelativeString(for: secondsAgo, relativeTo: now, locale: Locale(identifier: "en")), "Just now")

        // Minutes
        let fiveMinutesAgo = now.addingTimeInterval(-300)
        XCTAssertEqual(MessageTimestampFormatter.compactRelativeString(for: fiveMinutesAgo, relativeTo: now, locale: Locale(identifier: "en")), "5m")

        let fortyFiveMinutesAgo = now.addingTimeInterval(-2700)
        XCTAssertEqual(MessageTimestampFormatter.compactRelativeString(for: fortyFiveMinutesAgo, relativeTo: now, locale: Locale(identifier: "en")), "45m")

        // Hours
        let oneHourAgo = now.addingTimeInterval(-3600)
        XCTAssertEqual(MessageTimestampFormatter.compactRelativeString(for: oneHourAgo, relativeTo: now, locale: Locale(identifier: "en")), "1h")

        let twoHoursAgo = now.addingTimeInterval(-7200)
        XCTAssertEqual(MessageTimestampFormatter.compactRelativeString(for: twoHoursAgo, relativeTo: now, locale: Locale(identifier: "en")), "2h")

        // Days
        let fiftyTwoDaysAgo = now.addingTimeInterval(-52 * 86400)
        XCTAssertEqual(MessageTimestampFormatter.compactRelativeString(for: fiftyTwoDaysAgo, relativeTo: now, locale: Locale(identifier: "en")), "52d")

        let sixtyDaysAgo = now.addingTimeInterval(-60 * 86400)
        XCTAssertEqual(MessageTimestampFormatter.compactRelativeString(for: sixtyDaysAgo, relativeTo: now, locale: Locale(identifier: "en")), "60d")
    }

    @MainActor
    func testProjectGroupingAndFiltering() {
        let projectA = AppProject(name: "rabbit-writes")
        let projectB = AppProject(name: "mrefrust")

        let chatA1 = AppChat(projectID: projectA.id, title: "Drafting docs")
        let chatB1 = AppChat(projectID: projectB.id, title: "Implement User Profiles in Swift")
        let chatB2 = AppChat(projectID: projectB.id, title: "Align Swift Agent Plugins")
        let unassigned = AppChat(projectID: nil, title: "General quick note")

        let allChats = [chatA1, chatB1, chatB2, unassigned]

        // Grouping by project
        let projectAChats = allChats.filter { $0.projectID == projectA.id }
        XCTAssertEqual(projectAChats.count, 1)
        XCTAssertEqual(projectAChats.first?.title, "Drafting docs")

        let projectBChats = allChats.filter { $0.projectID == projectB.id }
        XCTAssertEqual(projectBChats.count, 2)

        let unassignedChats = allChats.filter { $0.projectID == nil }
        XCTAssertEqual(unassignedChats.count, 1)
        XCTAssertEqual(unassignedChats.first?.title, "General quick note")

        // Filtering by search term matching task title
        let searchSwift = "Swift"
        let matchingBChats = projectBChats.filter {
            $0.title.localizedCaseInsensitiveContains(searchSwift)
        }
        XCTAssertEqual(matchingBChats.count, 2)

        let searchPlugins = "Plugins"
        let matchingPluginChats = projectBChats.filter {
            $0.title.localizedCaseInsensitiveContains(searchPlugins)
        }
        XCTAssertEqual(matchingPluginChats.count, 1)
        XCTAssertEqual(matchingPluginChats.first?.title, "Align Swift Agent Plugins")

        // Filtering by search term matching project name
        let searchRabbit = "rabbit"
        XCTAssertTrue(projectA.name.localizedCaseInsensitiveContains(searchRabbit))
        XCTAssertFalse(projectB.name.localizedCaseInsensitiveContains(searchRabbit))
    }
}
