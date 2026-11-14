import Compression
import CryptoKit
import Foundation
import Network
import XCTest
@testable import TurboSparkApp

/// ZP-05 `cdnZip`: a loopback fixture server stands in for the CDN, the real
/// downloader, extractor, installer and ledger run over a temp plugin root.
/// Plain http to 127.0.0.1 is allowed only through the internal test policy;
/// production (`.production`) refuses it, and a test pins that.

// MARK: - Zip builder (test side, with the attributes the writer lacks)

struct TestZipEntry {
    var name: String
    var data: Data = Data()
    var method: UInt16 = 8
    /// Full Unix st_mode (type bits included), or nil for a non-Unix host.
    var mode: UInt32? = 0o100644
    var declaredSize: UInt32?
    var crcOverride: UInt32?
    var flags: UInt16 = 0
}

enum TestZip {
    static func build(_ entries: [TestZipEntry], overlapSecondOntoFirst: Bool = false) -> Data {
        func le16(_ v: UInt16) -> [UInt8] { [UInt8(v & 0xFF), UInt8(v >> 8)] }
        func le32(_ v: UInt32) -> [UInt8] {
            [UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF), UInt8((v >> 16) & 0xFF), UInt8(v >> 24)]
        }
        var archive: [UInt8] = []
        var central: [UInt8] = []
        var firstOffset: UInt32 = 0
        for (index, entry) in entries.enumerated() {
            let plain = [UInt8](entry.data)
            var payload = plain
            if entry.method == 8, !plain.isEmpty {
                var out = [UInt8](repeating: 0, count: plain.count + 4_096)
                let n = compression_encode_buffer(&out, out.count, plain, plain.count, nil, COMPRESSION_ZLIB)
                payload = Array(out.prefix(n))
            }
            let crc = entry.crcOverride ?? CRC32.checksum(plain)
            let name = [UInt8](entry.name.utf8)
            let offset = UInt32(archive.count)
            if index == 0 { firstOffset = offset }
            let size = entry.declaredSize ?? UInt32(plain.count)
            // Overlap bomb: the second member's central record points at the
            // first member's local header instead of its own.
            let centralOffset = (overlapSecondOntoFirst && index == 1) ? firstOffset : offset
            archive += le32(0x0403_4B50) + le16(20) + le16(entry.flags) + le16(entry.method)
            archive += le16(0) + le16(0x21) + le32(crc) + le32(UInt32(payload.count)) + le32(size)
            archive += le16(UInt16(name.count)) + le16(0) + name + payload
            let madeBy: UInt16 = entry.mode != nil ? (3 << 8) | 20 : 20
            central += le32(0x0201_4B50) + le16(madeBy) + le16(20) + le16(entry.flags) + le16(entry.method)
            central += le16(0) + le16(0x21) + le32(crc) + le32(UInt32(payload.count)) + le32(size)
            central += le16(UInt16(name.count)) + le16(0) + le16(0) + le16(0) + le16(0)
            central += le32((entry.mode ?? 0) << 16) + le32(centralOffset) + name
        }
        var tail = le32(0x0605_4B50) + le16(0) + le16(0)
        tail += le16(UInt16(entries.count)) + le16(UInt16(entries.count))
        tail += le32(UInt32(central.count)) + le32(UInt32(archive.count)) + le16(0)
        return Data(archive + central + tail)
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// A normal plugin: manifest, one skill, and a hook file that must stay
    /// inert (installing never runs it).
    static func goodPlugin(version: String = "1.2.3") -> Data {
        build([
            TestZipEntry(name: ".claude-plugin/plugin.json",
                data: Data("{\"name\": \"cdnp\", \"version\": \"\(version)\"}".utf8)),
            TestZipEntry(name: "skills/hi/SKILL.md", data: Data("---\ndescription: Hi\n---\nhello hello hello".utf8)),
            TestZipEntry(name: "hooks/hooks.json", data: Data("{\"hooks\": {}}".utf8), method: 0),
            TestZipEntry(name: "bin/run.sh", data: Data("#!/bin/sh\ntouch RAN\n".utf8), mode: 0o100755),
        ])
    }
}

// MARK: - Loopback server

final class CdnFixtureServer {
    struct Route {
        var status = "200 OK"
        var headers: [String] = []
        var body = Data()
        var omitContentLength = false
    }

    private let listener: NWListener
    private let queue = DispatchQueue(label: "CdnFixtureServer")
    private let lock = NSLock()
    private var routes: [String: Route] = [:]
    private var hits: [String: Int] = [:]
    private(set) var port: UInt16 = 0

    init() throws {
        listener = try NWListener(using: .tcp, on: .any)
        let ready = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { [weak self, listener] state in
            if case .ready = state { self?.port = listener.port?.rawValue ?? 0; ready.signal() }
            if case .failed = state { ready.signal() }
        }
        listener.newConnectionHandler = { [weak self] in self?.serve($0) }
        listener.start(queue: queue)
        guard ready.wait(timeout: .now() + 5) == .success, port != 0 else {
            throw NSError(domain: "CdnFixtureServer", code: 1)
        }
    }

    deinit { listener.cancel() }

    func set(_ path: String, _ route: Route) { lock.lock(); routes[path] = route; lock.unlock() }
    func hitCount(_ path: String) -> Int { lock.lock(); defer { lock.unlock() }; return hits[path, default: 0] }
    func url(_ path: String) -> String { "http://127.0.0.1:\(port)\(path)" }

    private func serve(_ connection: NWConnection) {
        connection.start(queue: queue)
        connection.receive(minimumIncompleteLength: 1, maximumLength: 8_192) { [weak self] data, _, _, _ in
            guard let self, let data, let text = String(data: data, encoding: .utf8) else {
                connection.cancel(); return
            }
            let target = text.components(separatedBy: " ").dropFirst().first ?? "/"
            self.lock.lock()
            self.hits[target, default: 0] += 1
            let route = self.routes[target] ?? Route(status: "404 Not Found", body: Data("missing".utf8))
            self.lock.unlock()
            var head = "HTTP/1.1 \(route.status)\r\n"
            for header in route.headers { head += header + "\r\n" }
            if !route.omitContentLength { head += "Content-Length: \(route.body.count)\r\n" }
            head += "Connection: close\r\n\r\n"
            var payload = Data(head.utf8)
            payload.append(route.body)
            connection.send(content: payload, completion: .contentProcessed { _ in connection.cancel() })
        }
    }
}

// MARK: - Tests

final class CdnZipSourceTests: XCTestCase {
    var base: URL!
    var root: URL!
    var server: CdnFixtureServer!
    var manager: PluginMarketplaceManager!

    override func setUpWithError() throws {
        base = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tsp-cdn-\(UUID().uuidString)", isDirectory: true)
        root = base.appendingPathComponent("plugins", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        server = try CdnFixtureServer()
        manager = PluginMarketplaceManager(
            root: root,
            cdnZipPolicy: CdnZipDownloader.Policy(allowLoopbackHTTP: true, maxDownloadBytes: 1_048_576))
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: base)
    }

    private func entry(url: String, sha256: String?, type: String = "cdnZip") -> PluginManifestParser.MarketplaceEntry {
        var source: [String: Any] = ["type": type, "url": url]
        if let sha256 { source["sha256"] = sha256 }
        return PluginManifestParser.MarketplaceEntry(
            name: "cdnp", sourceValue: source, strict: true, raw: ["name": "cdnp"])
    }

    private func install(
        _ path: String, sha256 explicit: String? = nil, zip: Data? = nil
    ) async throws -> PluginMarketplaceManager.InstallOutcome {
        try await manager.install(
            entry: entry(url: server.url(path), sha256: explicit ?? zip.map(TestZip.sha256)),
            marketplaceName: "cdn-market", checkoutDirectory: nil,
            marketplaceSource: .url(url: "https://example.invalid/m.json", headers: nil),
            scope: "user")
    }

    private func serve(_ zip: Data, at path: String = "/p.zip") {
        server.set(path, .init(headers: ["Content-Type: application/zip"], body: zip))
    }

    /// Nothing may be installed: no ledger record, no cache directory.
    private func assertNothingInstalled(_ message: String = "", file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertTrue(
            (manager.ledger.load().plugins["cdnp@cdn-market"] ?? []).isEmpty,
            "ledger must stay empty. \(message)", file: file, line: line)
        let cache = manager.installCacheDirectory.appendingPathComponent("cdn-market")
        XCTAssertFalse(FileManager.default.fileExists(atPath: cache.path), message, file: file, line: line)
    }

    private func assertFails<T>(
        _ expression: @autoclosure () async throws -> T, matching: (Error) -> Bool,
        _ message: String, file: StaticString = #filePath, line: UInt = #line
    ) async {
        do {
            _ = try await expression()
            XCTFail("expected a refusal: \(message)", file: file, line: line)
        } catch {
            XCTAssertTrue(matching(error), "wrong error for \(message): \(error)", file: file, line: line)
        }
    }

    // MARK: source decoding

    func testCdnZipAndBuiltinDecodeAndRoundTrip() throws {
        let cdn = try JSONDecoder().decode(
            MarketplaceSource.self,
            from: Data(#"{"type":"cdnZip","url":"https://cdn.example/p.zip","sha256":"ab"}"#.utf8))
        XCTAssertEqual(cdn, .cdnZip(url: "https://cdn.example/p.zip", sha256: "ab"))
        let again = try JSONDecoder().decode(MarketplaceSource.self, from: try JSONEncoder().encode(cdn))
        XCTAssertEqual(again, cdn)
        XCTAssertEqual(
            try JSONDecoder().decode(MarketplaceSource.self, from: Data(#"{"type":"builtin"}"#.utf8)),
            .builtin)
        // A row with no digest still decodes, so one bad row cannot take a
        // whole marketplace file down; the install refuses it instead.
        XCTAssertEqual(
            try JSONDecoder().decode(
                MarketplaceSource.self, from: Data(#"{"type":"cdnZip","url":"https://x/y.zip"}"#.utf8)),
            .cdnZip(url: "https://x/y.zip", sha256: ""))
    }

    func testExistingMarketplaceFilesStillLoad() throws {
        let file = #"""
        {"a":{"type":"github","repo":"o/r"},"b":{"type":"directory","path":"/x"},
         "c":{"type":"url","url":"https://e/x.json"},"d":{"type":"somethingNew","url":"https://e/z"}}
        """#
        let decoded = try JSONDecoder().decode([String: MarketplaceSource].self, from: Data(file.utf8))
        XCTAssertEqual(decoded.count, 4)
        XCTAssertEqual(decoded["b"], .directory(path: "/x"))
        XCTAssertEqual(decoded["d"], .url(url: "https://e/z", headers: nil), "unknown kinds keep the old fallback")
    }

    // MARK: good install

    func testGoodZipInstallsThroughTheAtomicInstallerAndRunsNothing() async throws {
        let zip = TestZip.goodPlugin()
        serve(zip)
        let outcome = try await install("/p.zip", zip: zip)
        XCTAssertEqual(outcome.pluginID, "cdnp@cdn-market")
        XCTAssertEqual(outcome.version, "1.2.3")
        XCTAssertTrue(outcome.installPath.hasPrefix(
            root.appendingPathComponent("cache/cdn-market/cdnp/1.2.3").path))
        let fm = FileManager.default
        XCTAssertTrue(fm.fileExists(atPath: outcome.installPath + "/skills/hi/SKILL.md"))
        XCTAssertTrue(fm.fileExists(atPath: outcome.installPath + "/hooks/hooks.json"))
        let script = outcome.installPath + "/bin/run.sh"
        XCTAssertEqual(
            (try fm.attributesOfItem(atPath: script)[.posixPermissions] as? NSNumber)?.intValue, 0o755)
        XCTAssertFalse(fm.fileExists(atPath: outcome.installPath + "/RAN"), "install never runs plugin code")
        XCTAssertEqual((manager.ledger.load().plugins["cdnp@cdn-market"] ?? []).count, 1)
        XCTAssertEqual(server.hitCount("/p.zip"), 1)
    }

    // MARK: digest

    func testWrongDigestRefusesAndWritesNothing() async throws {
        let zip = TestZip.goodPlugin()
        serve(zip)
        let wrong = String(repeating: "0", count: 64)
        await assertFails(try await install("/p.zip", sha256: wrong),
            matching: { ($0 as? CdnZipDownloader.Failure).map { if case .digestMismatch = $0 { return true } else { return false } } ?? false },
            "digest mismatch")
        assertNothingInstalled("a mismatch must not extract or install")
    }

    func testDigestIsOverTheExactBytes() async throws {
        let zip = TestZip.goodPlugin()
        var tampered = zip
        tampered[tampered.startIndex + 40] ^= 0x01
        serve(tampered)
        await assertFails(try await install("/p.zip", sha256: TestZip.sha256(zip)),
            matching: { $0 is CdnZipDownloader.Failure }, "one flipped byte")
        assertNothingInstalled()
    }

    func testMalformedDigestsAreRefusedBeforeAnyRequest() async throws {
        let zip = TestZip.goodPlugin()
        serve(zip)
        let good = TestZip.sha256(zip)
        for bad in [nil, "", "abc123", String(good.prefix(63)), good.uppercased(), good + "0", "zz" + String(good.dropFirst(2))] {
            await assertFails(try await install("/p.zip", sha256: bad ?? "", zip: nil),
                matching: { ($0 as? CdnZipDownloader.Failure) == .invalidDigest }, "digest \(bad ?? "nil")")
        }
        XCTAssertEqual(server.hitCount("/p.zip"), 0, "a bad digest never reaches the network")
        assertNothingInstalled()
    }

    func testMissingDigestKeyIsRefused() async throws {
        let zip = TestZip.goodPlugin()
        serve(zip)
        await assertFails(
            try await manager.install(
                entry: entry(url: server.url("/p.zip"), sha256: nil),
                marketplaceName: "cdn-market", checkoutDirectory: nil,
                marketplaceSource: .url(url: "https://example.invalid/m.json", headers: nil), scope: "user"),
            matching: { ($0 as? CdnZipDownloader.Failure) == .invalidDigest }, "no sha256 key")
        XCTAssertEqual(server.hitCount("/p.zip"), 0)
    }

    // MARK: transport

    func testHttpIsRefusedInProduction() async throws {
        let digest = TestZip.sha256(Data("x".utf8))
        XCTAssertThrowsError(try CdnZipDownloader.validate(
            url: "http://cdn.example/p.zip", sha256: digest)) {
            XCTAssertEqual($0 as? CdnZipDownloader.Failure, .insecureURL("http://cdn.example/p.zip"))
        }
        // Even the loopback address is refused without the test policy.
        XCTAssertThrowsError(try CdnZipDownloader.validate(
            url: "http://127.0.0.1:1/p.zip", sha256: digest, policy: .production))
        // And a non-loopback host stays refused under the test policy.
        XCTAssertThrowsError(try CdnZipDownloader.validate(
            url: "http://cdn.example/p.zip", sha256: digest,
            policy: .init(allowLoopbackHTTP: true)))
        XCTAssertNoThrow(try CdnZipDownloader.validate(url: "https://cdn.example/p.zip", sha256: digest))
        XCTAssertThrowsError(try CdnZipDownloader.validate(
            url: "https://user:pw@cdn.example/p.zip", sha256: digest), "credentials in the URL")
        XCTAssertThrowsError(try CdnZipDownloader.validate(url: "file:///etc/passwd", sha256: digest))
    }

    func testRedirectToHttpIsRefused() async throws {
        let zip = TestZip.goodPlugin()
        serve(zip, at: "/real.zip")
        server.set("/r.zip", .init(status: "302 Found", headers: ["Location: \(server.url("/real.zip"))"]))
        await assertFails(try await install("/r.zip", zip: zip),
            matching: {
                if case .redirectRefused = ($0 as? CdnZipDownloader.Failure) ?? .timedOut { return true }
                return false
            }, "redirect to a plain http target")
        XCTAssertEqual(server.hitCount("/real.zip"), 0, "the http target is never requested")
        assertNothingInstalled()
    }

    func testRedirectVerdictBoundsHopsAndSchemes() {
        let policy = CdnZipDownloader.Policy(maxRedirects: 2)
        let https = URL(string: "https://cdn.example/a.zip")
        XCTAssertNil(CdnZipDownloader.redirectVerdict(to: https, hop: 1, policy: policy))
        XCTAssertNil(CdnZipDownloader.redirectVerdict(to: https, hop: 2, policy: policy))
        XCTAssertNotNil(CdnZipDownloader.redirectVerdict(to: https, hop: 3, policy: policy), "hop bound")
        XCTAssertNotNil(CdnZipDownloader.redirectVerdict(
            to: URL(string: "http://cdn.example/a.zip"), hop: 1, policy: policy))
        XCTAssertNotNil(CdnZipDownloader.redirectVerdict(
            to: URL(string: "http://127.0.0.1/a.zip"), hop: 1,
            policy: .init(allowLoopbackHTTP: true)), "the loopback allowance never covers a redirect")
        XCTAssertNotNil(CdnZipDownloader.redirectVerdict(
            to: URL(string: "https://u:p@cdn.example/a.zip"), hop: 1, policy: policy))
        XCTAssertNotNil(CdnZipDownloader.redirectVerdict(to: URL(string: "ftp://cdn.example/a"), hop: 1, policy: policy))
    }

    func testOversizeDownloadIsRefusedByHeaderAndWithoutOne() async throws {
        let big = Data(repeating: 0x41, count: 2_000_000)
        server.set("/big.zip", .init(body: big))
        await assertFails(try await install("/big.zip", sha256: TestZip.sha256(big)),
            matching: { ($0 as? CdnZipDownloader.Failure) == .tooLarge(1_048_576) }, "Content-Length over the cap")
        // A server that omits Content-Length is stopped at the cap while streaming.
        server.set("/chunk.zip", .init(body: big, omitContentLength: true))
        await assertFails(try await install("/chunk.zip", sha256: TestZip.sha256(big)),
            matching: { ($0 as? CdnZipDownloader.Failure) == .tooLarge(1_048_576) }, "no Content-Length")
        assertNothingInstalled()
    }

    func testNon200StatusIsRefused() async throws {
        await assertFails(try await install("/absent.zip", sha256: String(repeating: "a", count: 64)),
            matching: { ($0 as? CdnZipDownloader.Failure) == .badStatus(404) }, "404")
        assertNothingInstalled()
    }

    // MARK: hostile archives (end to end)

    private func assertHostile(
        _ zip: Data, _ message: String, file: StaticString = #filePath, line: UInt = #line
    ) async {
        serve(zip, at: "/h.zip")
        await assertFails(try await install("/h.zip", zip: zip),
            matching: { $0 is ZipSafeExtractor.Failure }, message, file: file, line: line)
        assertNothingInstalled(message, file: file, line: line)
    }

    func testZipSlipEntryIsRefusedAndNothingEscapes() async throws {
        await assertHostile(TestZip.build([
            TestZipEntry(name: ".claude-plugin/plugin.json", data: Data("{\"name\":\"cdnp\"}".utf8)),
            TestZipEntry(name: "../../escaped.txt", data: Data("pwn".utf8)),
        ]), "dot-dot entry")
        XCTAssertFalse(FileManager.default.fileExists(atPath: base.appendingPathComponent("escaped.txt").path))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: base.deletingLastPathComponent().appendingPathComponent("escaped.txt").path))
        await assertHostile(TestZip.build([
            TestZipEntry(name: "/tmp/tsp-abs-escape.txt", data: Data("pwn".utf8)),
        ]), "absolute entry")
        await assertHostile(TestZip.build([
            TestZipEntry(name: "a\\..\\b.txt", data: Data("pwn".utf8)),
        ]), "backslash entry")
    }

    func testSymlinkAndSpecialFileEntriesAreRefused() async throws {
        await assertHostile(TestZip.build([
            TestZipEntry(name: "link", data: Data("/etc".utf8), method: 0, mode: 0o120777),
        ]), "symlink")
        await assertHostile(TestZip.build([
            TestZipEntry(name: "dev", data: Data(), method: 0, mode: 0o060644),
        ]), "block device")
        await assertHostile(TestZip.build([
            TestZipEntry(name: "pipe", data: Data(), method: 0, mode: 0o010644),
        ]), "fifo")
    }

    func testCompressionBombIsRefusedByRatio() async throws {
        await assertHostile(TestZip.build([
            TestZipEntry(name: "zeros.bin", data: Data(count: 8 * 1_024 * 1_024)),
        ]), "ratio bomb")
    }

    func testLyingSizeTableDoesNotLetAnEntryExpandPastItsDeclaredSize() async throws {
        // Declares 10 bytes, inflates to 1 MiB.
        await assertHostile(TestZip.build([
            TestZipEntry(name: "liar.bin", data: Data(count: 1_048_576), declaredSize: 10),
        ]), "declared size is a lie")
    }

    func testEncryptedAndUnsupportedMethodsAreRefused() async throws {
        await assertHostile(TestZip.build([
            TestZipEntry(name: "enc.bin", data: Data("x".utf8), method: 0, flags: 1),
        ]), "encrypted")
        await assertHostile(TestZip.build([
            TestZipEntry(name: "m.bin", data: Data("x".utf8), method: 12),
        ]), "bzip2")
    }

    // MARK: extractor guards, unit level

    private func extract(_ zip: Data, limits: ZipSafeExtractor.Limits = .init()) throws -> URL {
        let archive = base.appendingPathComponent("u-\(UUID().uuidString).zip")
        try zip.write(to: archive)
        let out = base.appendingPathComponent("out-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        try ZipSafeExtractor.extract(archive: archive, to: out, limits: limits)
        return out
    }

    private func expectFailure(
        _ zip: Data, limits: ZipSafeExtractor.Limits = .init(), _ message: String,
        _ check: (ZipSafeExtractor.Failure) -> Bool = { _ in true },
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertThrowsError(try extract(zip, limits: limits), message, file: file, line: line) {
            guard let failure = $0 as? ZipSafeExtractor.Failure else {
                return XCTFail("not an extractor failure: \($0)", file: file, line: line)
            }
            XCTAssertTrue(check(failure), "\(message): \(failure)", file: file, line: line)
        }
    }

    func testExtractorLimits() throws {
        let three = (0..<3).map { TestZipEntry(name: "f\($0).txt", data: Data("hello".utf8)) }
        expectFailure(TestZip.build(three), limits: .init(maxEntries: 2), "entry count") {
            if case .tooManyEntries = $0 { return true } else { return false }
        }
        expectFailure(TestZip.build([TestZipEntry(name: "a", data: Data(count: 2_000), method: 0)]),
            limits: .init(maxEntryBytes: 1_000), "per-entry size") {
            if case .entryTooLarge = $0 { return true } else { return false }
        }
        expectFailure(TestZip.build([
            TestZipEntry(name: "a", data: Data(count: 600), method: 0),
            TestZipEntry(name: "b", data: Data(count: 600), method: 0)]),
            limits: .init(maxTotalBytes: 1_000), "total size") { $0 == .totalTooLarge }
        expectFailure(TestZip.build([TestZipEntry(name: "a", data: Data(repeating: 7, count: 500_000))]),
            "ratio") { if case .ratioExceeded = $0 { return true } else { return false } }
    }

    func testExtractorRefusesDuplicatesCaseFoldedAndFileDirConflicts() throws {
        expectFailure(TestZip.build([
            TestZipEntry(name: "A.txt", data: Data("1".utf8)),
            TestZipEntry(name: "a.TXT", data: Data("2".utf8))]), "case-fold duplicate") {
            if case .duplicateEntry = $0 { return true } else { return false }
        }
        expectFailure(TestZip.build([
            TestZipEntry(name: "x", data: Data("1".utf8)),
            TestZipEntry(name: "x/y", data: Data("2".utf8))]), "file then dir")
        expectFailure(TestZip.build([
            TestZipEntry(name: "a/./b", data: Data("1".utf8))]), "dot component")
    }

    func testExtractorRefusesOverlappingMembersAndBadCRC() throws {
        let a = TestZipEntry(name: "a.txt", data: Data("first".utf8), method: 0)
        let b = TestZipEntry(name: "b.txt", data: Data("second".utf8), method: 0)
        expectFailure(TestZip.build([a, b], overlapSecondOntoFirst: true), "overlap")
        expectFailure(TestZip.build([
            TestZipEntry(name: "c.txt", data: Data("x".utf8), method: 0, crcOverride: 1234)]), "crc") {
            if case .corrupt = $0 { return true } else { return false }
        }
    }

    func testExtractorRefusesTruncatedAndNonZipInput() throws {
        let zip = TestZip.goodPlugin()
        expectFailure(Data(zip.prefix(zip.count / 2)), "truncated") { $0 == .notAZip }
        expectFailure(Data("not a zip at all, just text padding padding".utf8), "text") { $0 == .notAZip }
        expectFailure(zip + Data([0, 1, 2]), "trailing bytes after the end record") { $0 == .notAZip }
    }

    func testExtractorStripsSpecialPermissionBits() throws {
        let out = try extract(TestZip.build([
            TestZipEntry(name: "s.sh", data: Data("x".utf8), method: 0, mode: 0o104755)]))
        let perms = (try FileManager.default.attributesOfItem(
            atPath: out.appendingPathComponent("s.sh").path)[.posixPermissions] as? NSNumber)?.intValue
        XCTAssertEqual(perms, 0o755, "setuid is dropped")
    }

    func testExtractorAcceptsDirectoriesAndNestedFiles() throws {
        let out = try extract(TestZip.build([
            TestZipEntry(name: "d/", data: Data(), method: 0, mode: 0o040755),
            TestZipEntry(name: "d/e/f.txt", data: Data("deep deep deep deep".utf8))]))
        XCTAssertEqual(
            try String(contentsOf: out.appendingPathComponent("d/e/f.txt"), encoding: .utf8),
            "deep deep deep deep")
    }

    // MARK: other sources

    func testBuiltinAndCdnZipCannotBeUsedAsCatalogs() async throws {
        do {
            _ = try await manager.fetchMarketplace(
                name: "x", source: .cdnZip(url: "https://cdn.example/p.zip", sha256: String(repeating: "a", count: 64)))
            XCTFail("a cdnZip is not a catalog")
        } catch {}
    }
}
