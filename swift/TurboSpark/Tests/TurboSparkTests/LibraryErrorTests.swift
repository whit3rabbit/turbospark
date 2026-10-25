import XCTest

@testable import TurboSpark

/// Every stable code the engine puts first in `ts_last_error()` as
/// "<code>: <sentence>" maps to one typed case, keyed on the PREFIX. The code
/// lists are the ones in tasks.md (4.1, 4.3, 4.4 and 4.6 notes) and the
/// header's "model libraries" block.
final class LibraryErrorTests: XCTestCase {

    private func fault(_ message: String, status: TurboSparkError.Code = .invalidArgument)
        -> LibraryFault?
    {
        LibraryFault(TurboSparkError(code: status, message: message))
    }

    /// Parses "<code>: a sentence" and checks the code, the sentence and the
    /// status all arrive.
    private func assertMaps(
        _ wire: String, to expected: LibraryErrorCode,
        status: TurboSparkError.Code = .invalidArgument,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        guard let parsed = fault("\(wire): the sentence", status: status) else {
            return XCTFail("\(wire) did not parse as a library fault", file: file, line: line)
        }
        XCTAssertEqual(parsed.code, expected, "code \(wire)", file: file, line: line)
        XCTAssertEqual(parsed.sentence, "the sentence", file: file, line: line)
        XCTAssertEqual(parsed.status, status, file: file, line: line)
        XCTAssertEqual(expected.wireCode, wire, "wireCode of \(wire)", file: file, line: line)
    }

    // MARK: - Location refusals (add, import) - the 4.1 note

    func testEveryLocationRefusalCodeMapsToItsRefusal() {
        let table: [(String, LibraryRefusal)] = [
            ("same_disk_as_home", .sameDiskAsHome),
            ("network", .network),
            ("small_file_limit", .smallFileLimit),
            ("read_only", .readOnly),
            ("nested", .nested),
            ("inside_home", .insideHome),
            ("unknown_volume", .unknownVolume),
            ("not_a_directory", .notADirectory),
        ]
        for (wire, refusal) in table {
            assertMaps(wire, to: .refusal(refusal))
            XCTAssertEqual(refusal.rawValue, wire)
            XCTAssertEqual(LibraryErrorCode.refusal(refusal).refusal, refusal)
        }
        // The table is the whole set: a refusal added to the enum without a
        // wire code in the header would have no row here.
        XCTAssertEqual(Set(table.map(\.1)), Set(LibraryRefusal.allCases))
    }

    // MARK: - Library errors - the 4.1 note

    func testEveryLibraryErrorCodeMapsToItsCase() {
        let table: [(String, LibraryErrorCode)] = [
            ("unknown_library", .unknownLibrary),
            ("internal_library", .internalLibrary),
            ("not_connected", .notConnected),
            ("marker", .marker),
            ("io", .io),
        ]
        for (wire, code) in table { assertMaps(wire, to: code) }
    }

    // MARK: - Install errors - the 4.1 and 4.3 notes

    func testEveryInstallErrorCodeMapsToItsCase() {
        let table: [(String, LibraryErrorCode)] = [
            ("not_connected", .notConnected),
            ("library_read_only", .libraryReadOnly),
            ("library_disconnected", .libraryDisconnected),
            ("unknown_library", .unknownLibrary),
            ("alias_elsewhere", .aliasElsewhere),
            ("home_only", .homeOnly),
            ("io", .io),
        ]
        for (wire, code) in table { assertMaps(wire, to: code) }
    }

    // MARK: - Move errors - the 4.4 note

    func testEveryMoveErrorCodeMapsToItsCase() {
        let invalid: [(String, LibraryErrorCode)] = [
            ("not_connected", .notConnected),
            ("library_read_only", .libraryReadOnly),
            ("home_only", .homeOnly),
            ("same_library", .sameLibrary),
            ("unknown_install", .unknownInstall),
            ("unknown_library", .unknownLibrary),
            ("destination_taken", .destinationTaken),
            ("insufficient_space", .insufficientSpace),
            ("verify_failed", .verifyFailed),
            ("free_space_unknown", .freeSpaceUnknown),
            ("io", .io),
        ]
        for (wire, code) in invalid { assertMaps(wire, to: code) }
        // The two with a status of their own.
        assertMaps("cancelled", to: .cancelled, status: .cancelled)
        assertMaps("busy", to: .busy, status: .busy)
    }

    // MARK: - Delete errors - the 4.6 note

    func testEveryDeleteErrorCodeMapsToItsCase() {
        let table: [(String, LibraryErrorCode)] = [
            ("not_connected", .notConnected),
            ("library_read_only", .libraryReadOnly),
            ("unknown_install", .unknownInstall),
            ("unknown_library", .unknownLibrary),
            ("use_install_id", .useInstallId),
            ("io", .io),
        ]
        for (wire, code) in table { assertMaps(wire, to: code) }
        assertMaps("busy", to: .busy, status: .busy)
    }

    func testEveryKnownCodeHasADistinctWireSpelling() {
        let known: [LibraryErrorCode] =
            LibraryRefusal.allCases.map(LibraryErrorCode.refusal) + [
                .unknownLibrary, .internalLibrary, .notConnected, .marker, .io,
                .libraryDisconnected, .libraryReadOnly, .homeOnly, .aliasElsewhere,
                .unknownInstall, .useInstallId, .sameLibrary, .destinationTaken,
                .insufficientSpace, .verifyFailed, .freeSpaceUnknown, .busy, .cancelled,
            ]
        let spellings = known.map(\.wireCode)
        XCTAssertEqual(Set(spellings).count, spellings.count)
        for code in known {
            XCTAssertEqual(LibraryErrorCode(wireCode: code.wireCode), code)
        }
    }

    // MARK: - Unknown codes and messages that carry no code

    func testAnUnknownCodeKeepsItsSpellingAndSentence() throws {
        let parsed = try XCTUnwrap(fault("quota_exceeded: the drive is full"))
        XCTAssertEqual(parsed.code, .unrecognized("quota_exceeded"))
        XCTAssertEqual(parsed.code.wireCode, "quota_exceeded")
        XCTAssertEqual(parsed.sentence, "the drive is full")
    }

    func testMessagesWithoutACodeAreNotLibraryFaults() {
        // Install failures keep their old, code-less form. Reading a prefix
        // out of them would invent a code the engine never sent.
        XCTAssertNil(fault("options is not valid install options: unknown field `lib`"))
        XCTAssertNil(fault("install cancelled", status: .cancelled))
        XCTAssertNil(fault("disk full"))
        XCTAssertNil(fault(""))
        XCTAssertNil(fault(": no code at all"))
        XCTAssertNil(fault("Not_A_Code: uppercase is not a code"))
        XCTAssertNil(fault("io:no space after the colon"))
        XCTAssertNil(fault("a-b: punctuation is not a code"))
        XCTAssertNil(fault("io"), "a bare word is not a coded message")
    }

    func testAnOverlongPrefixIsNotACode() {
        let long = String(repeating: "a", count: 80)
        XCTAssertNil(fault("\(long): sentence"))
    }

    func testTheSentenceMayItselfContainColons() throws {
        let parsed = try XCTUnwrap(fault("io: could not read /Volumes/a:b: permission denied"))
        XCTAssertEqual(parsed.code, .io)
        XCTAssertEqual(parsed.sentence, "could not read /Volumes/a:b: permission denied")
    }

    func testACodeWithNoSentenceKeepsAnEmptySentence() throws {
        let parsed = try XCTUnwrap(fault("io: "))
        XCTAssertEqual(parsed.code, .io)
        XCTAssertEqual(parsed.sentence, "")
        let bare = try XCTUnwrap(fault("busy:", status: .busy))
        XCTAssertEqual(bare.code, .busy)
        XCTAssertEqual(bare.sentence, "")
    }

    // MARK: - busy: the prefix, not the status

    func testBusyIsKeyedOnThePrefixNotOnTheStatus() throws {
        // A library change refused while an install runs.
        let library = TurboSparkError(
            code: .busy,
            message: "busy: cannot change model libraries while 1 install is running")
        XCTAssertEqual(library.libraryFault?.code, .busy)

        // The audio device or session is busy: the same status, no prefix.
        // `AudioWorkspaceController+Installs` keys on this one.
        let audio = TurboSparkError(code: .busy, message: "audio device is busy; retry")
        XCTAssertEqual(audio.code, .busy)
        XCTAssertNil(audio.libraryFault, "the audio busy status carries no library code")

        // The prefix decides: the status alone must not.
        XCTAssertNil(
            TurboSparkError(code: .busy, message: "native audio session is busy").libraryFault)
    }

    func testCancelledIsKeyedOnThePrefixToo() throws {
        let move = TurboSparkError(code: .cancelled, message: "cancelled: the move was cancelled")
        XCTAssertEqual(move.libraryFault?.code, .cancelled)
        // An install cancelled by ts_install_cancel is not a library fault.
        let install = TurboSparkError(code: .cancelled, message: "install cancelled")
        XCTAssertNil(install.libraryFault)
    }

    func testTheFaultOfALiveErrorRoundTripsThroughTheLibrary() {
        // Real engine errors are covered in LibraryEngineTests; here the
        // package's own bridge for an error the engine did not write.
        let error = TurboSparkError(code: .invalidArgument, message: "not_connected: Studio is away")
        XCTAssertEqual(error.libraryFault?.code, .notConnected)
        XCTAssertEqual(error.libraryFault?.status, .invalidArgument)
    }

    // MARK: - Untrusted text

    func testControlCharactersAreEscapedForDisplay() {
        XCTAssertEqual(UntrustedText.display("a\u{1B}[31mred\n"), #"a\u{1b}[31mred\n"#)
        XCTAssertEqual(UntrustedText.display("tab\there\r"), #"tab\there\r"#)
        XCTAssertEqual(UntrustedText.display("nul\u{0}x"), #"nul\u{0}x"#)
        XCTAssertEqual(UntrustedText.display("del\u{7F}"), #"del\u{7f}"#)
    }

    func testBidirectionalFormattingCharactersAreEscapedForDisplay() {
        // A right-to-left override can make `gnp.exe` read as `exe.png`.
        XCTAssertEqual(UntrustedText.display("evil\u{202E}gnp.exe"), #"evil\u{202e}gnp.exe"#)
        let marks: [Unicode.Scalar] = [
            "\u{061C}", "\u{200E}", "\u{200F}", "\u{202A}", "\u{202B}", "\u{202C}",
            "\u{202D}", "\u{202E}", "\u{2066}", "\u{2067}", "\u{2068}", "\u{2069}",
        ]
        for mark in marks {
            let shown = UntrustedText.display("a\(mark)b")
            XCTAssertFalse(shown.unicodeScalars.contains(mark), "U+\(String(mark.value, radix: 16))")
            XCTAssertTrue(shown.hasPrefix("a\\u{") && shown.hasSuffix("}b"), shown)
        }
    }

    func testLineAndParagraphSeparatorsAreEscapedForDisplay() {
        let shown = UntrustedText.display("a\u{2028}b\u{2029}c")
        XCTAssertEqual(shown, #"a\u{2028}b\u{2029}c"#)
    }

    func testOrdinaryTextIsLeftAsItIs() {
        XCTAssertEqual(UntrustedText.display("Studio"), "Studio")
        XCTAssertEqual(UntrustedText.display("Caf\u{E9} \u{2603} drive"), "Caf\u{E9} \u{2603} drive")
        XCTAssertEqual(UntrustedText.display(""), "")
    }

    func testLongTextIsCutToTheLimitWithAnEllipsis() {
        let long = String(repeating: "a", count: 200)
        let shown = UntrustedText.display(long)
        XCTAssertEqual(shown.count, 120)
        XCTAssertEqual(shown, String(repeating: "a", count: 117) + "...")

        let exact = String(repeating: "b", count: 120)
        XCTAssertEqual(UntrustedText.display(exact), exact)
        XCTAssertEqual(UntrustedText.display(exact + "b").count, 120)
        XCTAssertEqual(UntrustedText.display(long, limit: 10), "aaaaaaa...")
    }

    func testEscapingCountsTowardTheLimitAndNeverLeavesTheRawCharacter() {
        let hostile = String(repeating: "\u{202E}", count: 50)
        let shown = UntrustedText.display(hostile)
        XCTAssertEqual(shown.count, 120)
        XCTAssertFalse(shown.unicodeScalars.contains("\u{202E}"))
    }

    func testTruncationDoesNotSplitACharacterThatIsSeveralScalars() {
        // "e" plus a combining accent is one character on screen.
        let accents = String(repeating: "e\u{301}", count: 130)
        let shown = UntrustedText.display(accents)
        XCTAssertEqual(shown.count, 120)
        XCTAssertTrue(shown.hasSuffix("..."))
        XCTAssertEqual(String(shown.dropLast(3)), String(repeating: "e\u{301}", count: 117))
    }

    func testTheFaultExposesTheRawSentenceAndAShortEscapedOne() throws {
        let raw = "marker: bad \u{1B}[2J" + String(repeating: "x", count: 300)
        let parsed = try XCTUnwrap(fault(raw))
        XCTAssertEqual(parsed.code, .marker)
        XCTAssertTrue(parsed.sentence.contains("\u{1B}"), "the sentence stays raw")
        XCTAssertEqual(parsed.sentence.count, "bad \u{1B}[2J".count + 300)
        XCTAssertFalse(parsed.displaySentence.contains("\u{1B}"))
        XCTAssertEqual(parsed.displaySentence.count, 120)
        XCTAssertTrue(parsed.displaySentence.hasPrefix(#"bad \u{1b}[2J"#))
    }
}
