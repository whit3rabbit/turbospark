import XCTest
@testable import TurboSparkApp

/// Pins the drive speed policy of the model-libraries design (requirements
/// 5.2 to 5.5 and 5.7). Every table row carries the number it comes from, so a
/// failing row names the rule that moved.
final class DriveSpeedAdvisorTests: XCTestCase {
    private let gib: Int64 = 1 << 30
    private let gbps: UInt64 = 1_000_000_000

    private func usb(_ bitsPerSecond: UInt64?) -> DriveConnection {
        DriveConnection(kind: .usb, bitsPerSecond: bitsPerSecond)
    }

    private func thunderbolt(_ bitsPerSecond: UInt64? = nil) -> DriveConnection {
        DriveConnection(kind: .thunderbolt, bitsPerSecond: bitsPerSecond)
    }

    private let internalDrive = DriveConnection(kind: .internalDrive, bitsPerSecond: nil)

    // MARK: Library level: the slow-link rule (5.2)

    func testSlowLinkRuleTable() {
        struct Row {
            let label: String
            let connection: DriveConnection
            let measured: Double?
            let slow: Bool
        }
        let rows = [
            Row(label: "USB 2.0 exactly (480 Mb/s)", connection: usb(480_000_000), measured: nil, slow: true),
            Row(label: "one bit above USB 2.0", connection: usb(480_000_001), measured: nil, slow: false),
            Row(label: "USB full speed (12 Mb/s)", connection: usb(12_000_000), measured: nil, slow: true),
            Row(label: "5 Gb/s", connection: usb(5 * gbps), measured: nil, slow: false),
            Row(label: "10 Gb/s", connection: usb(10 * gbps), measured: nil, slow: false),
            Row(label: "speed unknown, nothing measured", connection: usb(nil), measured: nil, slow: false),
            Row(label: "Thunderbolt, speed unknown", connection: thunderbolt(), measured: nil, slow: false),
            Row(label: "other kind, speed unknown", connection: DriveConnection(kind: .other("FireWire"), bitsPerSecond: nil), measured: nil, slow: false),
            Row(label: "a zero rate is unknown, not slow", connection: usb(0), measured: nil, slow: false),
            Row(label: "measured just under 150 MB/s", connection: usb(10 * gbps), measured: 149.999, slow: true),
            Row(label: "measured exactly 150 MB/s", connection: usb(10 * gbps), measured: 150, slow: false),
            Row(label: "measured just over 150 MB/s", connection: usb(10 * gbps), measured: 150.001, slow: false),
            Row(label: "measured slow on an unknown link", connection: usb(nil), measured: 100, slow: true),
            Row(label: "a fast measurement does not clear USB 2.0", connection: usb(480_000_000), measured: 400, slow: true),
            Row(label: "a zero measurement is ignored", connection: usb(10 * gbps), measured: 0, slow: false),
            Row(label: "a negative measurement is ignored", connection: usb(10 * gbps), measured: -5, slow: false),
            Row(label: "a NaN measurement is ignored", connection: usb(10 * gbps), measured: .nan, slow: false),
        ]
        for row in rows {
            let advice = DriveSpeedAdvisor.library(row.connection, measuredMBps: row.measured)
            XCTAssertEqual(advice.slowLink, row.slow, row.label)
        }
    }

    // MARK: Library level: the effective speed (5.3)

    func testNominalRateTable() throws {
        struct Row {
            let label: String
            let connection: DriveConnection
            let expected: Double?
        }
        let rows = [
            Row(label: "USB 2.0 is 40 MB/s", connection: usb(480_000_000), expected: 40),
            Row(label: "USB full speed scales down", connection: usb(12_000_000), expected: 1),
            Row(label: "between USB 2.0 and 5 Gb/s keeps the lowest tier", connection: usb(1 * gbps), expected: 40),
            Row(label: "just under 5 Gb/s", connection: usb(5 * gbps - 1), expected: 40),
            Row(label: "5 Gb/s is 400 MB/s", connection: usb(5 * gbps), expected: 400),
            Row(label: "just under 10 Gb/s", connection: usb(10 * gbps - 1), expected: 400),
            Row(label: "10 Gb/s is 700 MB/s", connection: usb(10 * gbps), expected: 700),
            Row(label: "just under 20 Gb/s", connection: usb(20 * gbps - 1), expected: 700),
            Row(label: "20 Gb/s is 2,500 MB/s", connection: usb(20 * gbps), expected: 2_500),
            Row(label: "40 Gb/s is 2,500 MB/s", connection: usb(40 * gbps), expected: 2_500),
            Row(label: "Thunderbolt without a rate is 2,500 MB/s", connection: thunderbolt(), expected: 2_500),
            Row(label: "Thunderbolt is 2,500 MB/s whatever rate is reported", connection: thunderbolt(10 * gbps), expected: 2_500),
            Row(label: "the rate counts for any kind but Thunderbolt", connection: DriveConnection(kind: .other("x"), bitsPerSecond: 10 * gbps), expected: 700),
            Row(label: "USB with no rate has no estimate", connection: usb(nil), expected: nil),
            Row(label: "unknown kind with no rate has no estimate", connection: DriveConnection(kind: .unknown, bitsPerSecond: nil), expected: nil),
            Row(label: "other kind with no rate has no estimate", connection: DriveConnection(kind: .other("x"), bitsPerSecond: nil), expected: nil),
            Row(label: "a zero rate has no estimate", connection: usb(0), expected: nil),
            Row(label: "internal has no estimate", connection: internalDrive, expected: nil),
        ]
        for row in rows {
            let advice = DriveSpeedAdvisor.library(row.connection, measuredMBps: nil)
            if let expected = row.expected {
                XCTAssertEqual(try XCTUnwrap(advice.effectiveMBps, row.label), expected, accuracy: 1e-9, row.label)
            } else {
                XCTAssertNil(advice.effectiveMBps, row.label)
            }
        }
    }

    func testAMeasurementReplacesTheNominalRate() throws {
        let slowCable = DriveSpeedAdvisor.library(usb(480_000_000), measuredMBps: 250)
        XCTAssertEqual(try XCTUnwrap(slowCable.effectiveMBps), 250, accuracy: 1e-9)
        let fastCable = DriveSpeedAdvisor.library(usb(10 * gbps), measuredMBps: 90)
        XCTAssertEqual(try XCTUnwrap(fastCable.effectiveMBps), 90, accuracy: 1e-9)
        let unknownLink = DriveSpeedAdvisor.library(usb(nil), measuredMBps: 320)
        XCTAssertEqual(try XCTUnwrap(unknownLink.effectiveMBps), 320, accuracy: 1e-9)
        let junk = DriveSpeedAdvisor.library(usb(10 * gbps), measuredMBps: -1)
        XCTAssertEqual(try XCTUnwrap(junk.effectiveMBps), 700, accuracy: 1e-9, "an unusable measurement falls back to the nominal rate")
    }

    // MARK: This Mac gets no advice (5.7)

    func testThisMacGetsNoAdviceAtEitherLevel() {
        let hostile = DriveConnection(kind: .internalDrive, bitsPerSecond: 480_000_000)
        let library = DriveSpeedAdvisor.library(hostile, measuredMBps: 10)
        XCTAssertEqual(library, LibraryAdvice(slowLink: false, effectiveMBps: nil))

        // Pass a library advice that CLAIMS a slow link: the model level must
        // short-circuit on the connection and not trust it.
        let claimed = LibraryAdvice(slowLink: true, effectiveMBps: 10)
        let model = DriveSpeedAdvisor.model(
            installBytes: 200 * gib, isMoE: true, physicalMemoryBytes: UInt64(36 * gib),
            library: claimed, connection: internalDrive)
        XCTAssertEqual(model, ModelAdvice(estimatedLoadSeconds: nil, streamsFromDrive: false, warnBeforeLoad: false))
    }

    // MARK: Model level: when a model streams from the drive (5.4)

    func testStreamsFromDriveBoundary() {
        struct Row {
            let label: String
            let isMoE: Bool
            let installBytes: Int64
            let physical: UInt64
            let streams: Bool
        }
        let memory = UInt64(36 * gib)
        let rows = [
            Row(label: "exactly physical memory minus 4 GiB", isMoE: true, installBytes: 32 * gib, physical: memory, streams: false),
            Row(label: "one byte over the boundary", isMoE: true, installBytes: 32 * gib + 1, physical: memory, streams: true),
            Row(label: "one byte under the boundary", isMoE: true, installBytes: 32 * gib - 1, physical: memory, streams: false),
            Row(label: "a dense model never streams", isMoE: false, installBytes: 64 * gib, physical: memory, streams: false),
            Row(label: "memory of exactly 4 GiB leaves a zero budget", isMoE: true, installBytes: 1, physical: UInt64(4 * gib), streams: true),
            Row(label: "an empty install does not stream", isMoE: true, installBytes: 0, physical: UInt64(4 * gib), streams: false),
            Row(label: "memory below the reserve does not underflow", isMoE: true, installBytes: 1, physical: UInt64(3 * gib), streams: true),
            Row(label: "memory of 4 GiB plus one byte", isMoE: true, installBytes: 1, physical: UInt64(4 * gib) + 1, streams: false),
            Row(label: "memory of 4 GiB plus one byte, two bytes installed", isMoE: true, installBytes: 2, physical: UInt64(4 * gib) + 1, streams: true),
            Row(label: "a negative size is treated as empty", isMoE: true, installBytes: -1, physical: UInt64(4 * gib), streams: false),
        ]
        let connection = usb(10 * gbps)
        for row in rows {
            let library = DriveSpeedAdvisor.library(connection, measuredMBps: nil)
            let advice = DriveSpeedAdvisor.model(
                installBytes: row.installBytes, isMoE: row.isMoE, physicalMemoryBytes: row.physical,
                library: library, connection: connection)
            XCTAssertEqual(advice.streamsFromDrive, row.streams, row.label)
        }
    }

    // MARK: Model level: the warn-before-load rule (5.4, 5.5)

    func testWarnBeforeLoadTable() {
        struct Row {
            let label: String
            let connection: DriveConnection
            let measured: Double?
            let isMoE: Bool
            let installBytes: Int64
            let streams: Bool
            let warn: Bool
        }
        let memory = UInt64(36 * gib)
        let big = 40 * gib   // above 36 GiB - 4 GiB: a mixture-of-experts model streams
        let small = 10 * gib
        let rows = [
            Row(label: "streaming MoE on 10 Gb/s", connection: usb(10 * gbps), measured: nil, isMoE: true, installBytes: big, streams: true, warn: false),
            Row(label: "streaming MoE on 5 Gb/s", connection: usb(5 * gbps), measured: nil, isMoE: true, installBytes: big, streams: true, warn: true),
            Row(label: "streaming MoE on 20 Gb/s", connection: usb(20 * gbps), measured: nil, isMoE: true, installBytes: big, streams: true, warn: false),
            Row(label: "streaming MoE on Thunderbolt", connection: thunderbolt(), measured: nil, isMoE: true, installBytes: big, streams: true, warn: false),
            Row(label: "streaming MoE measured just under 500 MB/s", connection: usb(10 * gbps), measured: 499.999, isMoE: true, installBytes: big, streams: true, warn: true),
            Row(label: "streaming MoE measured exactly 500 MB/s", connection: usb(10 * gbps), measured: 500, isMoE: true, installBytes: big, streams: true, warn: false),
            Row(label: "streaming MoE on 20 Gb/s measured 300 MB/s", connection: usb(20 * gbps), measured: 300, isMoE: true, installBytes: big, streams: true, warn: true),
            Row(label: "streaming MoE on Thunderbolt measured 450 MB/s", connection: thunderbolt(), measured: 450, isMoE: true, installBytes: big, streams: true, warn: true),
            Row(label: "streaming MoE, link and speed unknown", connection: usb(nil), measured: nil, isMoE: true, installBytes: big, streams: true, warn: false),
            Row(label: "streaming MoE, link unknown, measured 450 MB/s", connection: usb(nil), measured: 450, isMoE: true, installBytes: big, streams: true, warn: true),
            Row(label: "streaming MoE on USB 2.0", connection: usb(480_000_000), measured: nil, isMoE: true, installBytes: big, streams: true, warn: true),
            Row(label: "streaming MoE on 10 Gb/s measured 149 MB/s", connection: usb(10 * gbps), measured: 149, isMoE: true, installBytes: big, streams: true, warn: true),
            Row(label: "streaming MoE on 5 Gb/s measured 520 MB/s: the link is still below 10 Gb/s", connection: usb(5 * gbps), measured: 520, isMoE: true, installBytes: big, streams: true, warn: true),
            Row(label: "MoE that fits in memory on 5 Gb/s", connection: usb(5 * gbps), measured: nil, isMoE: true, installBytes: small, streams: false, warn: false),
            Row(label: "MoE that fits in memory on 10 Gb/s", connection: usb(10 * gbps), measured: nil, isMoE: true, installBytes: small, streams: false, warn: false),
            Row(label: "MoE that fits in memory on USB 2.0 still warns: the load itself is slow", connection: usb(480_000_000), measured: nil, isMoE: true, installBytes: small, streams: false, warn: true),
            Row(label: "dense model on 5 Gb/s is not warned about decode speed", connection: usb(5 * gbps), measured: nil, isMoE: false, installBytes: big, streams: false, warn: false),
            Row(label: "dense model on USB 2.0 warns because the load is slow, not because it streams", connection: usb(480_000_000), measured: nil, isMoE: false, installBytes: big, streams: false, warn: true),
            Row(label: "dense model on a link measured at 100 MB/s", connection: usb(10 * gbps), measured: 100, isMoE: false, installBytes: big, streams: false, warn: true),
            Row(label: "dense model on a link measured at 300 MB/s", connection: usb(10 * gbps), measured: 300, isMoE: false, installBytes: big, streams: false, warn: false),
        ]
        for row in rows {
            let library = DriveSpeedAdvisor.library(row.connection, measuredMBps: row.measured)
            let advice = DriveSpeedAdvisor.model(
                installBytes: row.installBytes, isMoE: row.isMoE, physicalMemoryBytes: memory,
                library: library, connection: row.connection)
            XCTAssertEqual(advice.streamsFromDrive, row.streams, "streams: \(row.label)")
            XCTAssertEqual(advice.warnBeforeLoad, row.warn, "warn: \(row.label)")
        }
    }

    // MARK: Model level: the estimate arithmetic (5.3)

    func testEstimatedLoadSecondsTable() {
        struct Row {
            let label: String
            let bytes: Int64
            let connection: DriveConnection
            let measured: Double?
            let seconds: Int?
        }
        let rows = [
            Row(label: "10 GB at 700 MB/s rounds 14.3 s up", bytes: 10_000_000_000, connection: usb(10 * gbps), measured: nil, seconds: 15),
            Row(label: "28 GB at 700 MB/s is exactly 40 s", bytes: 28_000_000_000, connection: usb(10 * gbps), measured: nil, seconds: 40),
            Row(label: "20 GB over USB 2.0 at 40 MB/s", bytes: 20_000_000_000, connection: usb(480_000_000), measured: nil, seconds: 500),
            Row(label: "20 GB at 400 MB/s", bytes: 20_000_000_000, connection: usb(5 * gbps), measured: nil, seconds: 50),
            Row(label: "5 GB on Thunderbolt at 2,500 MB/s", bytes: 5_000_000_000, connection: thunderbolt(), measured: nil, seconds: 2),
            Row(label: "a measurement beats the nominal rate", bytes: 10_000_000_000, connection: usb(10 * gbps), measured: 250, seconds: 40),
            Row(label: "a measurement beats a slow link", bytes: 10_000_000_000, connection: usb(480_000_000), measured: 1_000, seconds: 10),
            Row(label: "one byte takes one second", bytes: 1, connection: usb(10 * gbps), measured: nil, seconds: 1),
            Row(label: "an empty install has no estimate", bytes: 0, connection: usb(10 * gbps), measured: nil, seconds: nil),
            Row(label: "a negative size has no estimate", bytes: -1, connection: usb(10 * gbps), measured: nil, seconds: nil),
            Row(label: "an unknown speed gives no estimate", bytes: 10_000_000_000, connection: usb(nil), measured: nil, seconds: nil),
            Row(label: "an unknown link with a measurement", bytes: 10_000_000_000, connection: usb(nil), measured: 100, seconds: 100),
            Row(label: "internal has no estimate", bytes: 10_000_000_000, connection: DriveConnection(kind: .internalDrive, bitsPerSecond: nil), measured: 100, seconds: nil),
            Row(label: "an absurd result is not an Int overflow", bytes: Int64.max, connection: usb(nil), measured: 0.000_001, seconds: nil),
        ]
        for row in rows {
            let library = DriveSpeedAdvisor.library(row.connection, measuredMBps: row.measured)
            let advice = DriveSpeedAdvisor.model(
                installBytes: row.bytes, isMoE: false, physicalMemoryBytes: UInt64(36 * gib),
                library: library, connection: row.connection)
            XCTAssertEqual(advice.estimatedLoadSeconds, row.seconds, row.label)
        }
    }

    // MARK: Types

    func testConnectionTypesCanCrossActors() {
        func requireSendable<T: Sendable>(_ value: T) -> T { value }
        let connection = requireSendable(DriveConnection(kind: .other("x"), bitsPerSecond: 1, volumeUUID: "u", isLocal: false))
        XCTAssertEqual(connection, DriveConnection(kind: .other("x"), bitsPerSecond: 1, volumeUUID: "u", isLocal: false))
        XCTAssertNotEqual(DriveLinkKind.other("a"), DriveLinkKind.other("b"))
        _ = requireSendable(LibraryAdvice(slowLink: false, effectiveMBps: nil))
        _ = requireSendable(ModelAdvice(estimatedLoadSeconds: nil, streamsFromDrive: false, warnBeforeLoad: false))
    }
}
