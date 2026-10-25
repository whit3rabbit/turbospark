import XCTest
@testable import TurboSparkApp

/// A thread-safe integer the test hooks write from the reader's thread.
private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = 0
    var value: Int { lock.lock(); defer { lock.unlock() }; return stored }
    func set(_ newValue: Int) { lock.lock(); stored = newValue; lock.unlock() }
    func increment() { lock.lock(); stored += 1; lock.unlock() }
}

/// Records one value written by a hook on another thread.
private final class Box<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Value?
    var value: Value? { lock.lock(); defer { lock.unlock() }; return stored }
    func set(_ newValue: Value) { lock.lock(); stored = newValue; lock.unlock() }
}

final class DriveSpeedTestTests: XCTestCase {
    private let mib: Int64 = 1 << 20

    // MARK: Fixtures

    private func makeDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("DriveSpeedTestTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    @discardableResult
    private func writeFile(_ root: URL, _ relative: String, mib: Int64) throws -> URL {
        try writeFile(root, relative, bytes: mib * self.mib)
    }

    @discardableResult
    private func writeFile(_ root: URL, _ relative: String, bytes: Int64) throws -> URL {
        let url = root.appendingPathComponent(relative)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        // A hole reads back as zeros without a device read; the tests only need
        // the size, the type and a readable file, so nothing is written.
        try handle.truncate(atOffset: UInt64(bytes))
        return url
    }

    /// Small limits so a test needs megabytes, not the real 256 MiB and 1 GiB.
    private func smallConfiguration() -> DriveSpeedTest.Configuration {
        var configuration = DriveSpeedTest.Configuration()
        configuration.minimumFileBytes = mib
        configuration.maximumReadBytes = 8 * mib
        configuration.chunkBytes = Int(mib)
        return configuration
    }

    private func measure(
        _ root: URL, isInternal: Bool = false,
        _ configuration: DriveSpeedTest.Configuration? = nil,
        isCancelled: @escaping () -> Bool = { false }
    ) throws -> DriveSpeedResult {
        try DriveSpeedTest.measure(
            libraryRoot: root, isInternal: isInternal,
            configuration: configuration ?? smallConfiguration(), isCancelled: isCancelled)
    }

    // MARK: The design's numbers

    func testDefaultsAreTheDesignNumbers() {
        let configuration = DriveSpeedTest.Configuration()
        XCTAssertEqual(configuration.minimumFileBytes, 256 * mib, "design: refuses libraries with no file over 256 MiB")
        XCTAssertEqual(configuration.maximumReadBytes, 1_024 * mib, "design: reads up to 1 GiB")
        XCTAssertEqual(configuration.chunkBytes, Int(8 * mib), "design: 8 MiB chunks")
        XCTAssertEqual(configuration.timeLimit, 10, "design: at most 10 s")
        XCTAssertLessThan(
            configuration.timeLimit + configuration.walkTimeLimit, 15,
            "requirement 5.6: the whole test finishes in under 15 s")
    }

    // MARK: A read returns a positive rate

    func testReadsTheLargestFileAndReportsAPositiveRate() throws {
        let root = try makeDirectory()
        try writeFile(root, "text/small.gturbo/model_weights.bin", mib: 2)
        try writeFile(root, "text/big.gturbo/model_weights.bin", mib: 5)
        try writeFile(root, "image/other.gturbo/deep/weights.bin", mib: 3)

        let started = Date()
        let result = try measure(root)
        let elapsed = Date().timeIntervalSince(started)

        XCTAssertEqual(result.bytesRead, 5 * mib, "read the 5 MiB file, not the 2 MiB or 3 MiB ones")
        XCTAssertGreaterThan(result.bytesPerSecond, 0)
        XCTAssertGreaterThan(result.seconds, 0)
        XCTAssertLessThanOrEqual(result.seconds, DriveSpeedTest.Configuration().timeLimit)
        XCTAssertLessThan(elapsed, 15)
        XCTAssertEqual(result.megabytesPerSecond, result.bytesPerSecond / 1_000_000, accuracy: 1e-6, "MB/s is decimal, like the advisor's 150 MB/s")
    }

    func testTheDefaultLimitsAcceptAFileOfExactlyTheMinimumSize() throws {
        let root = try makeDirectory()
        try writeFile(root, "text/m.gturbo/model_weights.bin", mib: 256)

        let started = Date()
        let result = try measure(root, DriveSpeedTest.Configuration())

        XCTAssertEqual(result.bytesRead, 256 * mib, "the whole file: it is under the 1 GiB limit")
        XCTAssertGreaterThan(result.bytesPerSecond, 0)
        XCTAssertLessThan(Date().timeIntervalSince(started), 15)
    }

    func testTheDefaultLimitsRefuseAFileOneByteUnderTheMinimum() throws {
        let root = try makeDirectory()
        try writeFile(root, "text/m.gturbo/model_weights.bin", bytes: 256 * mib - 1)
        XCTAssertThrowsError(try measure(root, DriveSpeedTest.Configuration())) {
            XCTAssertEqual($0 as? DriveSpeedTestError, .noLargeFile)
        }
    }

    func testStopsAtTheByteLimit() throws {
        let root = try makeDirectory()
        try writeFile(root, "text/m.gturbo/model_weights.bin", mib: 6)
        var configuration = smallConfiguration()
        configuration.maximumReadBytes = 2 * mib
        let result = try measure(root, configuration)
        XCTAssertEqual(result.bytesRead, 2 * mib)
    }

    func testStopsAtTheTimeLimitUsingTheInjectedClock() throws {
        let root = try makeDirectory()
        try writeFile(root, "text/m.gturbo/model_weights.bin", mib: 8)
        let chunks = Counter()
        var configuration = smallConfiguration()
        configuration.timeLimit = 10
        configuration.walkTimeLimit = 1_000_000
        // Each chunk read costs four seconds of fake time. The walk reads the
        // clock too, at zero chunks, so it never sees time pass.
        configuration.now = { Double(chunks.value) * 4 }
        configuration.onChunk = { chunks.set($0) }

        let result = try measure(root, configuration)

        // 4 s and 8 s are under the limit, 12 s is over: three chunks.
        XCTAssertEqual(result.bytesRead, 3 * mib)
        XCTAssertEqual(result.seconds, 12, accuracy: 1e-9)
        XCTAssertEqual(result.bytesPerSecond, Double(3 * mib) / 12, accuracy: 1e-6)
    }

    func testAlwaysReadsOneChunkEvenWithAZeroTimeLimit() throws {
        let root = try makeDirectory()
        try writeFile(root, "text/m.gturbo/model_weights.bin", mib: 4)
        var configuration = smallConfiguration()
        configuration.timeLimit = 0
        let result = try measure(root, configuration)
        XCTAssertEqual(result.bytesRead, mib, "a rate needs at least one chunk")
        XCTAssertGreaterThan(result.bytesPerSecond, 0)
    }

    // MARK: Uncached, off the main thread

    func testOpensTheFileWithCachingDisabled() throws {
        let root = try makeDirectory()
        try writeFile(root, "text/m.gturbo/model_weights.bin", mib: 2)
        let flags = Box<Int32>()
        var configuration = smallConfiguration()
        configuration.onOpen = { descriptor in flags.set(fcntl(descriptor, F_GETFL)) }
        _ = try measure(root, configuration)
        // The public headers have no F_GETNOCACHE. The kernel reports the
        // setting through F_GETFL as FNOCACHE (0x40000), which a probe on this
        // machine showed appearing after F_NOCACHE and going away after
        // F_NOCACHE 0.
        let fNoCache: Int32 = 0x40000
        let seen = try XCTUnwrap(flags.value, "the open hook ran")
        XCTAssertNotEqual(seen & fNoCache, 0, "F_NOCACHE is on while the file is read")
    }

    @MainActor
    func testTheAsyncEntryPointReadsOffTheMainThread() async throws {
        let root = try makeDirectory()
        try writeFile(root, "text/m.gturbo/model_weights.bin", mib: 2)
        let onMain = Box<Bool>()
        var configuration = smallConfiguration()
        configuration.onChunk = { _ in onMain.set(Thread.isMainThread) }

        let result = try await DriveSpeedTest.measureReadSpeed(libraryRoot: root, isInternal: false, configuration: configuration)

        XCTAssertGreaterThan(result.bytesPerSecond, 0)
        XCTAssertEqual(onMain.value, false, "reads must not block the main thread")
    }

    // MARK: Where the read starts

    func testTheStartOffsetIsChunkAlignedAndLeavesRoomForTheWholeRead() {
        let chunk: Int64 = 1_000
        // 10_050 bytes, 4_000 to read: starts of 0 ... 6_050 fit, aligned to
        // 1_000 that is slots 0 ... 6.
        XCTAssertEqual(DriveSpeedTest.startOffset(fileBytes: 10_050, readBytes: 4_000, chunkBytes: chunk, pickSlot: { _ in 0 }), 0)
        XCTAssertEqual(DriveSpeedTest.startOffset(fileBytes: 10_050, readBytes: 4_000, chunkBytes: chunk, pickSlot: { $0 }), 6_000)
        XCTAssertEqual(DriveSpeedTest.startOffset(fileBytes: 10_050, readBytes: 4_000, chunkBytes: chunk, pickSlot: { _ in 3 }), 3_000)
        XCTAssertEqual(DriveSpeedTest.startOffset(fileBytes: 10_050, readBytes: 4_000, chunkBytes: chunk, pickSlot: { _ in 99 }), 6_000, "a picker that overshoots is clamped")
        XCTAssertEqual(DriveSpeedTest.startOffset(fileBytes: 10_050, readBytes: 4_000, chunkBytes: chunk, pickSlot: { _ in -4 }), 0, "and so is one that undershoots")
        XCTAssertEqual(DriveSpeedTest.startOffset(fileBytes: 4_000, readBytes: 4_000, chunkBytes: chunk, pickSlot: { $0 }), 0, "reading the whole file starts at the top")
    }

    func testTheReadStartsAtThePickedOffset() throws {
        let root = try makeDirectory()
        try writeFile(root, "text/m.gturbo/model_weights.bin", mib: 6)
        var configuration = smallConfiguration()
        configuration.maximumReadBytes = 2 * mib
        let slots = Box<Int64>()
        configuration.pickSlot = { highest in slots.set(highest); return highest }
        let result = try measure(root, configuration)
        XCTAssertEqual(slots.value, 4, "6 MiB file, 2 MiB read, 1 MiB chunks: slots 0 ... 4")
        XCTAssertEqual(result.bytesRead, 2 * mib, "starting at slot 4 leaves exactly the 2 MiB to read")
    }

    // MARK: Refusals

    func testRefusesThisMacWithoutTouchingTheDisk() throws {
        let root = try makeDirectory()
        try writeFile(root, "text/m.gturbo/model_weights.bin", mib: 4)
        let opened = Counter()
        var configuration = smallConfiguration()
        configuration.onOpen = { _ in opened.increment() }
        configuration.onChunk = { _ in opened.increment() }

        XCTAssertThrowsError(try measure(root, isInternal: true, configuration)) {
            XCTAssertEqual($0 as? DriveSpeedTestError, .thisMac)
        }
        XCTAssertEqual(opened.value, 0, "nothing was opened or read")
    }

    func testRefusesAsynchronouslyForThisMac() async throws {
        let root = try makeDirectory()
        try writeFile(root, "text/m.gturbo/model_weights.bin", mib: 4)
        do {
            _ = try await DriveSpeedTest.measureReadSpeed(libraryRoot: root, isInternal: true, configuration: smallConfiguration())
            XCTFail("This Mac must be refused")
        } catch {
            XCTAssertEqual(error as? DriveSpeedTestError, .thisMac)
        }
    }

    func testRefusesWhenNoFileIsLargeEnough() throws {
        let root = try makeDirectory()
        try writeFile(root, "text/a.gturbo/model_weights.bin", bytes: mib - 1)
        try writeFile(root, "text/a.gturbo/manifest.json", bytes: 100)
        XCTAssertThrowsError(try measure(root)) {
            XCTAssertEqual($0 as? DriveSpeedTestError, .noLargeFile, "one byte under the minimum")
        }

        try writeFile(root, "text/b.gturbo/model_weights.bin", bytes: mib)
        XCTAssertNoThrow(try measure(root), "exactly the minimum is large enough")
    }

    func testRefusesAnEmptyLibrary() throws {
        let root = try makeDirectory()
        XCTAssertThrowsError(try measure(root)) {
            XCTAssertEqual($0 as? DriveSpeedTestError, .noLargeFile)
        }
    }

    func testRefusesALibraryThatIsNotThere() throws {
        let root = try makeDirectory().appendingPathComponent("gone")
        XCTAssertThrowsError(try measure(root)) {
            guard case .unreadable? = $0 as? DriveSpeedTestError else {
                return XCTFail("expected unreadable, got \($0)")
            }
        }
    }

    func testIgnoresHiddenFoldersSuchAsStagingAndTheDownloadCache() throws {
        let root = try makeDirectory()
        try writeFile(root, ".staging/move-1/model_weights.bin", mib: 7)
        try writeFile(root, ".download-cache/blob", mib: 7)
        try writeFile(root, "text/a.gturbo/model_weights.bin", mib: 2)
        let result = try measure(root)
        XCTAssertEqual(result.bytesRead, 2 * mib, "a half-finished install is not the model being measured")
    }

    // MARK: Drive contents are untrusted: no links are followed

    func testNeverFollowsSymbolicLinks() throws {
        let outside = try makeDirectory()
        try writeFile(outside, "huge.bin", mib: 7)
        try writeFile(outside, "folder/huge2.bin", mib: 7)
        let root = try makeDirectory()
        try writeFile(root, "text/real.gturbo/model_weights.bin", mib: 2)
        let manager = FileManager.default
        try manager.createSymbolicLink(at: root.appendingPathComponent("text/linked.bin"),
                                       withDestinationURL: outside.appendingPathComponent("huge.bin"))
        try manager.createSymbolicLink(at: root.appendingPathComponent("linked-folder"),
                                       withDestinationURL: outside.appendingPathComponent("folder"))
        try manager.createSymbolicLink(at: root.appendingPathComponent("text/real.gturbo/also.bin"),
                                       withDestinationURL: outside.appendingPathComponent("huge.bin"))

        let result = try measure(root)

        XCTAssertEqual(result.bytesRead, 2 * mib, "the 2 MiB file in the library, never the 7 MiB files behind the links")
    }

    func testALibraryThatOnlyHoldsLinksHasNoLargeFile() throws {
        let outside = try makeDirectory()
        try writeFile(outside, "huge.bin", mib: 7)
        let root = try makeDirectory()
        try FileManager.default.createDirectory(at: root.appendingPathComponent("text"), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("text/linked.bin"),
                                                   withDestinationURL: outside.appendingPathComponent("huge.bin"))
        XCTAssertThrowsError(try measure(root)) {
            XCTAssertEqual($0 as? DriveSpeedTestError, .noLargeFile)
        }
    }

    func testRefusesALibraryRootThatIsItselfALink() throws {
        let real = try makeDirectory()
        try writeFile(real, "text/m.gturbo/model_weights.bin", mib: 4)
        let holder = try makeDirectory()
        let link = holder.appendingPathComponent("library")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)
        XCTAssertThrowsError(try measure(link)) {
            guard case .unreadable? = $0 as? DriveSpeedTestError else {
                return XCTFail("expected unreadable, got \($0)")
            }
        }
    }

    func testTheFileIsOpenedWithoutFollowingALinkSwappedInAfterTheWalk() throws {
        let outside = try makeDirectory()
        try writeFile(outside, "huge.bin", mib: 7)
        let root = try makeDirectory()
        let target = try writeFile(root, "text/m.gturbo/model_weights.bin", mib: 3)
        var configuration = smallConfiguration()
        // Swap the real file for a link the moment the walk is done and the
        // open is about to happen. The walk's decision must not be trusted at
        // open time.
        configuration.afterScan = {
            try? FileManager.default.removeItem(at: target)
            try? FileManager.default.createSymbolicLink(at: target, withDestinationURL: outside.appendingPathComponent("huge.bin"))
        }
        XCTAssertThrowsError(try measure(root, configuration)) {
            guard case .unreadable? = $0 as? DriveSpeedTestError else {
                return XCTFail("expected unreadable, got \($0)")
            }
        }
    }

    // MARK: The walk is bounded

    func testTheWalkReportsTheLargestFileAndItsPlace() throws {
        let root = try makeDirectory()
        try writeFile(root, "text/a.gturbo/model_weights.bin", mib: 2)
        try writeFile(root, "image/b.gturbo/shards/part.bin", mib: 4)
        let outcome = try DriveSpeedTest.scanLargest(libraryRoot: root, configuration: smallConfiguration(), isCancelled: { false })
        XCTAssertEqual(outcome.best?.size, 4 * mib)
        XCTAssertEqual(outcome.best?.relativePath, "image/b.gturbo/shards/part.bin")
        XCTAssertFalse(outcome.truncated)
    }

    func testTheWalkStopsAtTheEntryLimit() throws {
        let root = try makeDirectory()
        for index in 0..<40 {
            try writeFile(root, "files/f\(index).bin", bytes: 10)
        }
        var configuration = smallConfiguration()
        configuration.maximumEntries = 10
        let capped = try DriveSpeedTest.scanLargest(libraryRoot: root, configuration: configuration, isCancelled: { false })
        XCTAssertEqual(capped.visited, 10, "the folder itself and nine files, then it stops")
        XCTAssertTrue(capped.truncated)

        configuration.maximumEntries = 41
        let full = try DriveSpeedTest.scanLargest(libraryRoot: root, configuration: configuration, isCancelled: { false })
        XCTAssertEqual(full.visited, 41, "one folder and forty files")
        XCTAssertFalse(full.truncated, "reaching the limit exactly is not truncation")
    }

    func testTheWalkStopsAtTheDepthLimit() throws {
        let root = try makeDirectory()
        try writeFile(root, "a/b/near.bin", mib: 2)
        try writeFile(root, "a/b/c/over.bin", mib: 5)
        try writeFile(root, "a/b/c/d/far.bin", mib: 6)
        var configuration = smallConfiguration()
        configuration.maximumDepth = 2
        let outcome = try DriveSpeedTest.scanLargest(libraryRoot: root, configuration: configuration, isCancelled: { false })
        XCTAssertEqual(outcome.best?.relativePath, "a/b/near.bin", "the folders a and b are scanned, c is exactly one level too deep")

        configuration.maximumDepth = 3
        let one = try DriveSpeedTest.scanLargest(libraryRoot: root, configuration: configuration, isCancelled: { false })
        XCTAssertEqual(one.best?.relativePath, "a/b/c/over.bin", "one more level reaches c but not d")

        configuration.maximumDepth = 4
        let deeper = try DriveSpeedTest.scanLargest(libraryRoot: root, configuration: configuration, isCancelled: { false })
        XCTAssertEqual(deeper.best?.relativePath, "a/b/c/d/far.bin")
    }

    func testTheWalkStopsAtItsTimeLimit() throws {
        let root = try makeDirectory()
        for index in 0..<30 {
            try writeFile(root, "f\(index).bin", bytes: 10)
        }
        var configuration = smallConfiguration()
        configuration.walkTimeLimit = 3
        // Every reading of the clock is ten seconds later than the last.
        let ticks = Counter()
        configuration.now = { ticks.increment(); return Double(ticks.value) * 10 }
        let outcome = try DriveSpeedTest.scanLargest(libraryRoot: root, configuration: configuration, isCancelled: { false })
        XCTAssertTrue(outcome.truncated)
        XCTAssertLessThan(outcome.visited, 30)
    }

    // MARK: Cancellation

    func testCancellationIsCheckedBetweenChunks() throws {
        let root = try makeDirectory()
        try writeFile(root, "text/m.gturbo/model_weights.bin", mib: 8)
        let chunks = Counter()
        var configuration = smallConfiguration()
        configuration.onChunk = { chunks.set($0) }

        XCTAssertThrowsError(try measure(root, configuration, isCancelled: { chunks.value >= 2 })) {
            XCTAssertTrue($0 is CancellationError)
        }
        XCTAssertEqual(chunks.value, 2, "stopped at the first check after the second chunk, not at the end")
    }

    func testCancellationBeforeTheStartReadsNothing() throws {
        let root = try makeDirectory()
        try writeFile(root, "text/m.gturbo/model_weights.bin", mib: 4)
        let touched = Counter()
        var configuration = smallConfiguration()
        configuration.onOpen = { _ in touched.increment() }
        configuration.onChunk = { _ in touched.increment() }
        XCTAssertThrowsError(try measure(root, configuration, isCancelled: { true })) {
            XCTAssertTrue($0 is CancellationError)
        }
        XCTAssertEqual(touched.value, 0)
    }

    func testCancellingTheTaskStopsTheReadBetweenChunks() async throws {
        let root = try makeDirectory()
        try writeFile(root, "text/m.gturbo/model_weights.bin", mib: 8)
        let reached = expectation(description: "the second chunk was read")
        let proceed = DispatchSemaphore(value: 0)
        let chunks = Counter()
        var configuration = smallConfiguration()
        configuration.onChunk = { count in
            chunks.set(count)
            if count == 2 {
                reached.fulfill()
                proceed.wait()
            }
        }
        let library = root
        let settings = configuration
        let task = Task {
            try await DriveSpeedTest.measureReadSpeed(libraryRoot: library, isInternal: false, configuration: settings)
        }

        await fulfillment(of: [reached], timeout: 30)
        task.cancel()
        proceed.signal()

        do {
            _ = try await task.value
            XCTFail("a cancelled speed test must not return a rate")
        } catch {
            XCTAssertTrue(error is CancellationError, "got \(error)")
        }
        XCTAssertEqual(chunks.value, 2)
    }
}
