import Foundation
@testable import TurboSparkApp
import XCTest

final class GreetingProviderTests: XCTestCase {
    private var provider: GreetingProvider!

    override func setUp() {
        super.setUp()
        provider = GreetingProvider.shared
    }

    func testCatalogHasRichGreetingCount() {
        let greetings = provider.allGreetings
        XCTAssertGreaterThanOrEqual(greetings.count, 60, "Expected at least 50-100 greetings in catalog")
        XCTAssertEqual(greetings.count, 100, "Expected exactly 100 greetings")

        let categories = Set(greetings.map(\.category))
        XCTAssertTrue(categories.contains(.earlyMorning))
        XCTAssertTrue(categories.contains(.morning))
        XCTAssertTrue(categories.contains(.afternoon))
        XCTAssertTrue(categories.contains(.evening))
        XCTAssertTrue(categories.contains(.night))
        XCTAssertTrue(categories.contains(.anytime))
    }

    func testTimeSlotResolutionByHour() {
        XCTAssertEqual(GreetingProvider.slot(forHour: 4), .earlyMorning)
        XCTAssertEqual(GreetingProvider.slot(forHour: 5), .earlyMorning)
        XCTAssertEqual(GreetingProvider.slot(forHour: 6), .earlyMorning)

        XCTAssertEqual(GreetingProvider.slot(forHour: 7), .morning)
        XCTAssertEqual(GreetingProvider.slot(forHour: 9), .morning)
        XCTAssertEqual(GreetingProvider.slot(forHour: 11), .morning)

        XCTAssertEqual(GreetingProvider.slot(forHour: 12), .afternoon)
        XCTAssertEqual(GreetingProvider.slot(forHour: 14), .afternoon)
        XCTAssertEqual(GreetingProvider.slot(forHour: 16), .afternoon)

        XCTAssertEqual(GreetingProvider.slot(forHour: 17), .evening)
        XCTAssertEqual(GreetingProvider.slot(forHour: 19), .evening)
        XCTAssertEqual(GreetingProvider.slot(forHour: 21), .evening)

        XCTAssertEqual(GreetingProvider.slot(forHour: 22), .night)
        XCTAssertEqual(GreetingProvider.slot(forHour: 23), .night)
        XCTAssertEqual(GreetingProvider.slot(forHour: 0), .night)
        XCTAssertEqual(GreetingProvider.slot(forHour: 3), .night)
    }

    func testMorningSlotNeverReturnsEveningOrAfternoonGreetings() {
        let morningGreetings = provider.greetings(forSlot: .morning)
        XCTAssertFalse(morningGreetings.isEmpty)

        for greeting in morningGreetings {
            XCTAssertTrue(
                greeting.category == .morning || greeting.category == .anytime,
                "Greeting '\(greeting.id)' with category '\(greeting.category)' must not appear in morning slot"
            )
            XCTAssertNotEqual(greeting.category, .evening)
            XCTAssertNotEqual(greeting.category, .afternoon)
            XCTAssertNotEqual(greeting.category, .night)
        }
    }

    func testEveningSlotNeverReturnsMorningOrAfternoonGreetings() {
        let eveningGreetings = provider.greetings(forSlot: .evening)
        XCTAssertFalse(eveningGreetings.isEmpty)

        for greeting in eveningGreetings {
            XCTAssertTrue(
                greeting.category == .evening || greeting.category == .anytime,
                "Greeting '\(greeting.id)' with category '\(greeting.category)' must not appear in evening slot"
            )
            XCTAssertNotEqual(greeting.category, .morning)
            XCTAssertNotEqual(greeting.category, .earlyMorning)
            XCTAssertNotEqual(greeting.category, .afternoon)
        }
    }

    func testAfternoonSlotNeverReturnsMorningOrNightGreetings() {
        let afternoonGreetings = provider.greetings(forSlot: .afternoon)
        XCTAssertFalse(afternoonGreetings.isEmpty)

        for greeting in afternoonGreetings {
            XCTAssertTrue(
                greeting.category == .afternoon || greeting.category == .anytime,
                "Greeting '\(greeting.id)' with category '\(greeting.category)' must not appear in afternoon slot"
            )
            XCTAssertNotEqual(greeting.category, .morning)
            XCTAssertNotEqual(greeting.category, .night)
            XCTAssertNotEqual(greeting.category, .evening)
        }
    }

    func testNightSlotNeverReturnsDaytimeGreetings() {
        let nightGreetings = provider.greetings(forSlot: .night)
        XCTAssertFalse(nightGreetings.isEmpty)

        for greeting in nightGreetings {
            XCTAssertTrue(
                greeting.category == .night || greeting.category == .anytime,
                "Greeting '\(greeting.id)' with category '\(greeting.category)' must not appear in night slot"
            )
            XCTAssertNotEqual(greeting.category, .morning)
            XCTAssertNotEqual(greeting.category, .afternoon)
        }
    }

    func testAllGreetingsCoverAllSupportedLanguages() {
        let concreteLanguages = AppLanguage.allCases.filter { $0 != .system }

        for greeting in provider.allGreetings {
            for lang in concreteLanguages {
                let text = greeting.text(for: lang)
                XCTAssertFalse(
                    text.isEmpty,
                    "Greeting '\(greeting.id)' has empty text for language '\(lang.rawValue)'"
                )
                XCTAssertNotEqual(
                    text,
                    greeting.id,
                    "Greeting '\(greeting.id)' fell back to ID for language '\(lang.rawValue)'"
                )
            }
        }
    }

    func testRandomGreetingProducesNonEmptyString() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!

        // Construct 9:00 AM UTC
        var components = DateComponents()
        components.year = 2026
        components.month = 9
        components.day = 5
        components.hour = 9
        let morningDate = calendar.date(from: components)!

        let greetingEn = provider.randomGreeting(for: morningDate, language: .english, calendar: calendar)
        XCTAssertFalse(greetingEn.isEmpty)

        let greetingEs = provider.randomGreeting(for: morningDate, language: .spanish, calendar: calendar)
        XCTAssertFalse(greetingEs.isEmpty)

        let greetingJa = provider.randomGreeting(for: morningDate, language: .japanese, calendar: calendar)
        XCTAssertFalse(greetingJa.isEmpty)

        let greetingHe = provider.randomGreeting(for: morningDate, language: .hebrew, calendar: calendar)
        XCTAssertFalse(greetingHe.isEmpty)
    }

    func testDeterministicGreetingCycling() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!

        var components = DateComponents()
        components.year = 2026
        components.month = 9
        components.day = 5
        components.hour = 14 // afternoon
        let afternoonDate = calendar.date(from: components)!

        let first = provider.greeting(at: 0, for: afternoonDate, language: .english, calendar: calendar)
        let second = provider.greeting(at: 1, for: afternoonDate, language: .english, calendar: calendar)
        let looped = provider.greeting(at: 0, for: afternoonDate, language: .english, calendar: calendar)

        XCTAssertFalse(first.isEmpty)
        XCTAssertFalse(second.isEmpty)
        XCTAssertEqual(first, looped)
    }
}
