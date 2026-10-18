import Foundation
import XCTest

@testable import TurboSparkApp

/// Task 6.11: the strings the move sheet adds, in all 21 languages, and the scans
/// that keep the move code honest.
///
/// `LocalizationParityTests` holds every catalog key to full translation and
/// matching specifiers, but its source scan skips interpolated literals, and a key
/// missing there renders its English source silently. So each key is named here,
/// each sentence is rendered through every compiled bundle (`ModelMoveTextTests`),
/// and a scan resolves EVERY literal the new files pass to `Text` or
/// `AppStrings.localized`, interpolated ones included, against the catalog.
final class ModelMoveStringsTests: XCTestCase {
    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var sourcesRoot: URL { packageRoot.appendingPathComponent("Sources/TurboSparkApp") }

    /// Keys this change added to the catalog.
    static let newKeys: [String] = [
        "Move to...",
        "Move %@",
        "Move to",
        "Move",
        "Space required",
        "Space available",
        "There is no other library connected to move this model to.",
        "TurboSpark copies the model, checks the copy, and only then removes the original.",
        "Keep the drives connected until the move finishes.",
        "Preparing to move...",
        "Copying...",
        "Verifying and finishing...",
        "Cancelling...",
        "This model is loaded in Chat. Unload it, then move it.",
        "This model is loaded in Image Generation. Unload it, then move it.",
        "This model is attached to the server. Detach it there, then move it.",
        "Humanize is using this model. Unload it there, then move it.",
        "This model is loaded by Audio. Try again after Audio releases it.",
        "A reply is being generated. Try again when it finishes.",
        "A model is loading. Try again when it finishes.",
        "The model is already in that library.",
        "%@ has only %@ free, but this model needs %@.",
        "TurboSpark could not read how much space %@ has free.",
        "TurboSpark could not work out how big this model is.",
        "The original was kept.",
        "%@ was disconnected during the move.",
        "A drive was disconnected during the move.",
        "%@ or %@ is read-only, so the model cannot be moved.",
        "%@ does not have enough free space for this model.",
        "This model is being moved. Try again when the move finishes.",
        "This kind of model can only be stored on This Mac.",
        "%@ already has a model with this name.",
        "The copied files did not match the original, so the move was stopped.",
        "The move stopped because of a file system error.",
        "TurboSpark could not move the model.",
        "Move cancelled. The original was not changed.",
        "Moved %@ to %@.",
        "%@ may still list this model. The model works from its new place.",
        "The original library may still list this model. The model works from its new place.",
        "The original folder could not be removed. It is left at %@. You can delete it yourself. The model works from its new place.",
    ]

    /// Keys the change reuses from the existing catalog.
    static let reusedKeys: [String] = [
        "Cancel", "Close", "Done", "Retry", "This Mac", "Unknown library", "%@ free", "Unknown",
        "Location", "Measuring...", "%@ of %@",
        "A model is installing or moving. Try again when it finishes.",
        "%@ is not connected. Connect it or choose another library.",
        "'%@' is no longer in the model list.", "That library is no longer in the list.",
    ]

    /// The files this change added. A literal in any of them must be a key, and
    /// none of them may touch a model's files.
    static let newSourceFiles = [
        "State/ModelMovePlan.swift", "State/ModelMoveText.swift", "State/MoveProgress.swift",
        "State/AppModel+Moves.swift", "State/ModelMoveSheetModel.swift", "Components/ModelMoveSheet.swift",
    ]

    private var languages: [AppLanguage] { AppLanguage.allCases.filter { $0 != .system } }

    private func loadStrings() throws -> [String: [String: Any]] {
        let url = packageRoot.appendingPathComponent("Localization/Localizable.xcstrings")
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        let root = try XCTUnwrap(object as? [String: Any])
        return try XCTUnwrap(root["strings"] as? [String: [String: Any]])
    }

    private func value(_ localization: Any?) -> String? {
        let node = localization as? [String: Any]
        return (node?["stringUnit"] as? [String: Any])?["value"] as? String
    }

    private func specifiers(_ text: String) -> [String] {
        let regex = try! NSRegularExpression(pattern: #"%(\d+\$)?(lld|ld|lf|d|@|f)"#)
        let range = NSRange(text.startIndex..., in: text)
        return regex.matches(in: text, range: range).map { String(text[Range($0.range, in: text)!]) }.sorted()
    }

    func testEveryKeyIsInTheCatalogInAllLanguages() throws {
        let strings = try loadStrings()
        XCTAssertEqual(languages.count, 21)
        var problems: [String] = []
        for key in Self.newKeys + Self.reusedKeys {
            guard let localizations = strings[key]?["localizations"] as? [String: Any] else {
                problems.append("missing key: \(key)")
                continue
            }
            for language in languages where (value(localizations[language.rawValue]) ?? "").isEmpty {
                problems.append("\(language.rawValue): empty or missing: \(key)")
            }
        }
        XCTAssertEqual(problems, [])
    }

    func testTheNewKeysKeepTheirSpecifiersInEveryLanguage() throws {
        let strings = try loadStrings()
        var problems: [String] = []
        for key in Self.newKeys {
            guard let localizations = strings[key]?["localizations"] as? [String: Any] else { continue }
            let expected = specifiers(key)
            for (language, node) in localizations {
                if let text = value(node), specifiers(text) != expected {
                    problems.append("\(language): \(key.prefix(50)) -> \(text.prefix(60))")
                }
            }
        }
        XCTAssertEqual(problems.sorted(), [])
    }

    /// Plain specifiers are filled in the order the call site passes its
    /// arguments, so a translation may not reorder them (the multiset check cannot
    /// see an order). The way to keep the order is to write each translation around
    /// the English order, and the way to stay in it is to use no positional
    /// specifier at all, which this pins.
    func testNoNewKeyUsesAPositionalSpecifier() throws {
        let strings = try loadStrings()
        for key in Self.newKeys {
            let localizations = strings[key]?["localizations"] as? [String: Any] ?? [:]
            for (language, node) in localizations {
                XCTAssertFalse(
                    (value(node) ?? "").contains("$@"), "\(language): \(key.prefix(40)) uses a positional specifier")
            }
        }
    }

    func testTheEnglishValueIsTheKey() throws {
        let strings = try loadStrings()
        for key in Self.newKeys {
            XCTAssertEqual(value((strings[key]?["localizations"] as? [String: Any])?["en"]), key, key)
        }
    }

    func testNoTranslationInANonLatinLanguageIsTheEnglishSentence() throws {
        let strings = try loadStrings()
        let nonLatin = ["ar", "he", "hi", "ja", "ko", "ru", "uk", "zh-Hans", "zh-Hant"]
        var problems: [String] = []
        for key in Self.newKeys {
            let localizations = strings[key]?["localizations"] as? [String: Any] ?? [:]
            for language in nonLatin where value(localizations[language]) == key { problems.append("\(language): \(key)") }
        }
        XCTAssertEqual(problems, [])
    }

    func testTheNewKeysCompileIntoEveryLanguagesBundle() {
        for language in languages where language != .english {
            let bundle = AppStrings.bundle(for: language)
            for key in Self.newKeys {
                let text = bundle.localizedString(forKey: key, value: "MISSING", table: nil)
                XCTAssertNotEqual(text, "MISSING", "\(language.rawValue): \(key.prefix(40)) did not compile")
            }
        }
    }

    /// A translation that is the English sentence in a Latin-script language is
    /// not caught by the script gate; the sentence must at least differ from
    /// English there, except for the short words that are the same in a language.
    func testNoSentenceIsLeftInEnglishInAnyLanguage() throws {
        let strings = try loadStrings()
        let sameInSomeLanguages: Set<String> = ["Move", "Copying...", "Move to...", "Move to"]
        var problems: [String] = []
        for key in Self.newKeys where !sameInSomeLanguages.contains(key) {
            let localizations = strings[key]?["localizations"] as? [String: Any] ?? [:]
            for language in languages where language != .english {
                if value(localizations[language.rawValue]) == key { problems.append("\(language.rawValue): \(key)") }
            }
        }
        XCTAssertEqual(problems, [])
    }

    // MARK: - The phase line

    func testEveryPhaseTitleIsACatalogKey() throws {
        let strings = try loadStrings()
        for phase in MoveProgress.Phase.allCases {
            XCTAssertNotNil(strings[phase.title], phase.title)
            XCTAssertTrue(Self.newKeys.contains(phase.title), "\(phase.title) is listed with this change")
        }
        XCTAssertEqual(MoveProgress.Phase.allCases.count, 4)
    }

    // MARK: - Every literal the new files pass is a key

    private func catalogKey(of source: [Character], from start: Int) -> (key: String, end: Int)? {
        var out = ""
        var index = start
        while index < source.count {
            let character = source[index]
            if character == "\"" { return (out, index) }
            guard character == "\\", index + 1 < source.count else {
                out.append(character)
                index += 1
                continue
            }
            let next = source[index + 1]
            if next == "(" {
                var depth = 1
                var cursor = index + 2
                var inner = ""
                while cursor < source.count, depth > 0 {
                    if source[cursor] == "(" { depth += 1 }
                    if source[cursor] == ")" { depth -= 1 }
                    if depth > 0 { inner.append(source[cursor]) }
                    cursor += 1
                }
                out += inner.lowercased().hasSuffix("count") ? "%lld" : "%@"
                index = cursor
            } else if next == "u", index + 2 < source.count, source[index + 2] == "{",
                let close = source[(index + 3)...].firstIndex(of: "}"),
                let value = UInt32(String(source[(index + 3)..<close]), radix: 16),
                let scalar = Unicode.Scalar(value)
            {
                out.append(Character(scalar))
                index = close + 1
            } else {
                out.append(next == "n" ? "\n" : next)
                index += 2
            }
        }
        return nil
    }

    private func sourceText(_ file: String) throws -> String {
        MoveStoreRetirementTests.codeWithoutComments(
            try String(contentsOf: sourcesRoot.appendingPathComponent(file), encoding: .utf8))
    }

    func testEveryLiteralTheNewFilesPassToTextOrLocalizedIsACatalogKey() throws {
        let strings = try loadStrings()
        var checked = 0
        var problems: [String] = []
        for file in Self.newSourceFiles {
            let code = try sourceText(file)
            let characters = Array(code)
            for trigger in ["Text(\"", "AppStrings.localized(\""] {
                var search = code.startIndex..<code.endIndex
                while let found = code.range(of: trigger, range: search) {
                    search = found.upperBound..<code.endIndex
                    let offset = code.distance(from: code.startIndex, to: found.upperBound)
                    guard let (key, end) = catalogKey(of: characters, from: offset) else {
                        problems.append("\(file): unterminated literal after \(trigger)")
                        continue
                    }
                    // `Text("x", bundle: .module)` only: a bare literal renders English.
                    if trigger == "Text(\"" {
                        let tail = String(characters[(end + 1)...].prefix(400))
                        let closing = tail.firstIndex(of: ")").map { String(tail[..<$0]) } ?? tail
                        if !closing.contains("bundle: .module") {
                            problems.append("\(file): Text literal without bundle: .module -> \(key.prefix(50))")
                        }
                    }
                    checked += 1
                    if strings[key] == nil { problems.append("\(file): no catalog key -> \(key)") }
                }
            }
        }
        XCTAssertGreaterThanOrEqual(checked, 20, "the scan found almost nothing: is it looking at the right files?")
        XCTAssertEqual(problems, [])
    }

    func testTheNewFilesNeverPassABareLiteralToAKeyOnlyInitializer() throws {
        let triggers = [
            ".help(\"", ".accessibilityLabel(\"", ".accessibilityHint(\"", "Button(\"", "Label(\"",
            "ProgressView(\"", ".confirmationDialog(\"", ".alert(\"", "Picker(\"", "Menu(\"",
        ]
        for file in Self.newSourceFiles {
            let code = try sourceText(file)
            for trigger in triggers {
                XCTAssertFalse(code.contains(trigger), "\(file) passes a bare literal to \(trigger)")
            }
        }
    }

    // MARK: - The app never touches a model's files

    /// Only the engine moves, copies, renames or deletes a model folder
    /// (requirement 7.1): the app asks, and reads what the engine says. A move
    /// that did any of it itself would race the engine's own verification and
    /// could delete a drive's files through a path the drive no longer holds.
    func testNothingInTheMoveCodeMovesCopiesRenamesOrRemovesAFile() throws {
        let forbidden = [
            "moveItem", "copyItem", "removeItem", "trashItem", "replaceItem", "linkItem", "createSymbolicLink",
            "createDirectory", "createFile", "FileManager", "NSWorkspace", "unlink(", "rename(", "rmdir(", "mkdir(",
            "FileHandle(forWriting", "FileHandle(forUpdating", ".write(to", "Process(",
        ]
        for file in Self.newSourceFiles {
            let code = try sourceText(file)
            for word in forbidden {
                XCTAssertFalse(code.contains(word), "\(file) uses \(word): only the engine touches a model's files")
            }
        }
    }

    /// A move of the loaded model is refused (requirement 7.5), so nothing in the
    /// move code needs to unload a model, and nothing in it may load one: the sheet
    /// must not turn a move into a load.
    func testNothingInTheMoveCodeLoadsOrUnloadsAModel() throws {
        let forbidden = [
            "requestLoad", "unloadModel", "selectModel(", "beginChatLoad", "loadAnyway", "openChatWithModel",
            "performLoad", "confirmLoad",
        ]
        for file in Self.newSourceFiles {
            let code = try sourceText(file)
            for word in forbidden {
                XCTAssertFalse(code.contains(word), "\(file) uses \(word): a move never loads or unloads a model")
            }
        }
    }

    func testTheMoveDriverReachesTheEngineOnlyThroughTheService() throws {
        let code = try sourceText("State/AppModel+Moves.swift")
        XCTAssertTrue(code.contains("service.move("), "the move goes through the library seam")
        XCTAssertTrue(code.contains("cancelMove()"))
        XCTAssertFalse(code.contains("TurboSparkLibraries"), "never around the seam")
        XCTAssertFalse(code.contains("TurboSparkCatalog"), "never around the seam")
    }
}
