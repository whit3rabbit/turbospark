import Foundation

// Excel (.xlsx/.xlsm) render model. Cached values only: a formula shows the
// value Excel last stored, never a recalculation. Not rendered: theme.xml
// (default-palette fallback in `OfficeColor`), conditional formats, borders,
// merged ranges (cells render individually), vertical alignment, hidden
// rows and columns, data validation, charts (counted, not drawn), pivot
// tables, images, frozen COLUMNS (rows are pinned, columns scroll).

struct XlsxCell: Equatable {
    var column: Int
    var text: String
    var bold = false
    var italic = false
    var colorHex: String?
    var fillHex: String?
    var alignment: DocxAlignment?
    /// Numbers right-align by default in Excel; text left.
    var isNumeric = false
    var comment: String?
}

struct XlsxRow: Equatable {
    /// 1-based sheet row number.
    var number: Int
    var cells: [XlsxCell]
}

struct XlsxSheet {
    var name: String
    var rows: [XlsxRow] = []
    /// Column widths in character units, indexed by 0-based column.
    var columnWidths: [Int: Double] = [:]
    var columnCount = 0
    var frozenRows = 0
    var frozenColumns = 0
    var truncated = false
}

struct XlsxDocument {
    var sheets: [XlsxSheet] = []
    var hiddenSheetCount = 0
    var notes = OfficeRenderNotes()

    private struct CellFormat {
        var numberFormat = "General"
        var bold = false
        var italic = false
        var colorHex: String?
        var fillHex: String?
        var alignment: DocxAlignment?
    }

    static func parse(_ package: OfficePackage) throws -> XlsxDocument {
        let workbookPart = try package.mainPart(fallback: "xl/workbook.xml")
        let workbook = try package.xml(workbookPart)
        let workbookRels = try package.relationships(of: workbookPart)
        let date1904 = workbook.child("workbookPr")?.attrs["date1904"].map { $0 == "1" || $0 == "true" } ?? false

        let shared = try sharedStrings(package)
        let formats = try cellFormats(package)

        var document = XlsxDocument()
        document.notes.externalReferences = workbookRels.values.filter(\.isExternal).count
        document.notes.unrenderedCharts = package.archive.entries.filter {
            $0.hasPrefix("xl/charts/chart") && $0.hasSuffix(".xml")
        }.count

        for sheetNode in workbook.child("sheets")?.all("sheet") ?? [] {
            try package.checkDeadline()
            if (sheetNode.attrs["state"] ?? "visible") != "visible" {
                document.hiddenSheetCount += 1
                continue
            }
            guard document.sheets.count < package.limits.maximumSheets else {
                document.notes.truncated = true
                break
            }
            guard let id = sheetNode.attrs["id"], let rel = workbookRels[id],
                  !rel.isExternal, let entry = rel.entry else { continue }
            let name = String((sheetNode.attrs["name"] ?? "Sheet").prefix(64))
            document.sheets.append(try parseSheet(
                entry: entry, name: name, package: package,
                shared: shared, formats: formats, date1904: date1904))
        }
        guard !document.sheets.isEmpty || document.hiddenSheetCount > 0 else {
            throw OfficeRenderError.missingPart("xl/worksheets")
        }
        if document.sheets.contains(where: \.truncated) { document.notes.truncated = true }
        return document
    }

    // MARK: Shared parts

    private static func sharedStrings(_ package: OfficePackage) throws -> [String] {
        guard let root = try package.optionalXML("xl/sharedStrings.xml") else { return [] }
        var strings: [String] = []
        for item in root.all("si") {
            if strings.count >= package.limits.maximumSharedStrings { break }
            // `t` directly under `si`, or under each rich-text `r`. Phonetic
            // runs (`rPh`) are skipped so furigana does not double the text.
            var text = item.child("t")?.text ?? ""
            for run in item.all("r") { text += run.child("t")?.text ?? "" }
            strings.append(String(text.prefix(package.limits.maximumCellCharacters)))
        }
        return strings
    }

    private static func cellFormats(_ package: OfficePackage) throws -> [CellFormat] {
        guard let root = try package.optionalXML("xl/styles.xml") else { return [] }
        var customFormats: [Int: String] = [:]
        for node in root.child("numFmts")?.all("numFmt") ?? [] {
            if let id = node.attrs["numFmtId"].flatMap(Int.init), let code = node.attrs["formatCode"] {
                customFormats[id] = code
            }
        }
        var fonts: [(bold: Bool, italic: Bool, color: String?)] = []
        for font in root.child("fonts")?.all("font") ?? [] {
            fonts.append((font.toggle("b") ?? false, font.toggle("i") ?? false,
                          OfficeColor.resolve(font.child("color"))))
        }
        var fills: [String?] = []
        for fill in root.child("fills")?.all("fill") ?? [] {
            if let pattern = fill.child("patternFill"), pattern.attrs["patternType"] == "solid" {
                fills.append(OfficeColor.resolve(pattern.child("fgColor")))
            } else {
                fills.append(nil)
            }
        }
        var formats: [CellFormat] = []
        for xf in root.child("cellXfs")?.all("xf") ?? [] {
            var format = CellFormat()
            let numberID = xf.attrs["numFmtId"].flatMap(Int.init) ?? 0
            format.numberFormat = customFormats[numberID]
                ?? OfficeNumberFormat.builtIn[numberID] ?? "General"
            if let index = xf.attrs["fontId"].flatMap(Int.init), fonts.indices.contains(index) {
                format.bold = fonts[index].bold
                format.italic = fonts[index].italic
                format.colorHex = fonts[index].color
            }
            if let index = xf.attrs["fillId"].flatMap(Int.init), fills.indices.contains(index) {
                format.fillHex = fills[index]
            }
            switch xf.child("alignment")?.attrs["horizontal"] {
            case "center", "centerContinuous": format.alignment = .center
            case "right": format.alignment = .trailing
            case "left": format.alignment = .leading
            case "justify": format.alignment = .justified
            default: break
            }
            formats.append(format)
        }
        return formats
    }

    // MARK: Sheet

    private static func parseSheet(
        entry: String, name: String, package: OfficePackage,
        shared: [String], formats: [CellFormat], date1904: Bool
    ) throws -> XlsxSheet {
        let root = try package.xml(entry)
        var sheet = XlsxSheet(name: name)
        let limits = package.limits

        if let pane = root.child("sheetViews")?.child("sheetView")?.child("pane"),
           let state = pane.attrs["state"], state.hasPrefix("frozen") {
            sheet.frozenRows = smallCount(pane.attrs["ySplit"])
            sheet.frozenColumns = smallCount(pane.attrs["xSplit"])
        }
        for col in root.child("cols")?.all("col") ?? [] {
            guard let first = col.attrs["min"].flatMap(Int.init),
                  let last = col.attrs["max"].flatMap(Int.init),
                  let width = col.attrs["width"].flatMap(Double.init) else { continue }
            // A hostile `max="16384"` must not become 16k dictionary writes.
            for index in max(first, 1)...max(min(last, limits.maximumSheetColumns), max(first, 1))
            where index <= limits.maximumSheetColumns {
                sheet.columnWidths[index - 1] = min(max(width, 2), 80)
            }
        }

        let comments = try commentMap(sheetEntry: entry, package: package)
        var cellCount = 0
        var nextRow = 0
        var widest = 0
        for rowNode in root.child("sheetData")?.all("row") ?? [] {
            nextRow = rowNode.attrs["r"].flatMap(Int.init) ?? (nextRow + 1)
            if sheet.rows.count >= limits.maximumSheetRows {
                sheet.truncated = true
                break
            }
            var cells: [XlsxCell] = []
            var nextColumn = 0
            for cellNode in rowNode.all("c") {
                var column = nextColumn
                if let reference = cellNode.attrs["r"], let parsed = columnIndex(of: reference) {
                    column = parsed
                }
                nextColumn = column + 1
                guard column < limits.maximumSheetColumns else { sheet.truncated = true; continue }
                guard cellCount < limits.maximumSheetCells else { sheet.truncated = true; break }
                let format = cellNode.attrs["s"].flatMap(Int.init).flatMap {
                    formats.indices.contains($0) ? formats[$0] : nil
                } ?? CellFormat()
                let (text, numeric) = display(cellNode, shared: shared, format: format,
                                              date1904: date1904, limit: limits.maximumCellCharacters)
                let comment = cellNode.attrs["r"].flatMap { comments[$0] }
                guard !text.isEmpty || format.fillHex != nil || comment != nil else { continue }
                cellCount += 1
                cells.append(XlsxCell(
                    column: column, text: text, bold: format.bold, italic: format.italic,
                    colorHex: format.colorHex, fillHex: format.fillHex,
                    alignment: format.alignment, isNumeric: numeric, comment: comment))
                widest = max(widest, column + 1)
            }
            if !cells.isEmpty { sheet.rows.append(XlsxRow(number: nextRow, cells: cells)) }
        }
        sheet.columnCount = widest
        return sheet
    }

    /// The text to show for one cell, plus whether it is numeric.
    private static func display(
        _ node: XMLNode, shared: [String], format: CellFormat, date1904: Bool, limit: Int
    ) -> (String, Bool) {
        let value = node.child("v")?.text ?? ""
        let type = node.attrs["t"] ?? "n"
        var text: String
        var numeric = false
        switch type {
        case "s":
            text = Int(value).flatMap { shared.indices.contains($0) ? shared[$0] : nil } ?? ""
        case "inlineStr":
            let inline = node.child("is")
            text = (inline?.child("t")?.text ?? "") + (inline?.all("r").map { $0.child("t")?.text ?? "" }.joined() ?? "")
        case "b":
            text = value == "1" ? "TRUE" : "FALSE"
        case "str", "e", "d":
            text = value
        default:
            numeric = true
            text = OfficeNumberFormat.format(value, code: format.numberFormat, date1904: date1904)
        }
        return (String(text.prefix(limit)), numeric)
    }

    /// A split position clamped to 0...20. `Int(Double)` traps on a hostile
    /// `1e300`, so the range is checked before converting.
    private static func smallCount(_ value: String?) -> Int {
        guard let value, let number = Double(value), number.isFinite, number >= 0 else { return 0 }
        return Int(min(number, 20))
    }

    /// `B7` to 1; nil when the reference has no leading letters.
    static func columnIndex(of reference: String) -> Int? {
        var value = 0
        var seen = false
        for scalar in reference.unicodeScalars {
            guard scalar.value >= 65, scalar.value <= 90 || (scalar.value >= 97 && scalar.value <= 122) else { break }
            value = value * 26 + Int(scalar.properties.uppercaseMapping.unicodeScalars.first!.value) - 64
            seen = true
            if value > 20_000 { return nil }
        }
        return seen ? value - 1 : nil
    }

    private static func commentMap(sheetEntry: String, package: OfficePackage) throws -> [String: String] {
        let rels = try package.relationships(of: sheetEntry)
        guard let rel = rels.values.first(where: { $0.kind == "comments" }), let entry = rel.entry,
              let root = try package.optionalXML(entry) else { return [:] }
        var map: [String: String] = [:]
        for comment in root.child("commentList")?.all("comment") ?? [] {
            guard let reference = comment.attrs["ref"], map.count < 2_000 else { continue }
            let body = comment.child("text")
            let text = (body?.child("t")?.text ?? "") + (body?.all("r").map { $0.child("t")?.text ?? "" }.joined() ?? "")
            map[reference] = String(text.prefix(500))
        }
        return map
    }
}

// MARK: - Number formats

/// A conservative SpreadsheetML number formatter. It covers General, fixed
/// decimals, thousands separators, percent, literal prefixes and suffixes
/// (currency), negative sections, and the common date and time tokens. Scientific,
/// fractions, elapsed-time and conditional sections fall back to the cached
/// value as General. Separators are fixed US style (no locale lookup), so the
/// output is the same on every machine and in tests.
enum OfficeNumberFormat {
    static let builtIn: [Int: String] = [
        0: "General", 1: "0", 2: "0.00", 3: "#,##0", 4: "#,##0.00", 9: "0%", 10: "0.00%",
        14: "m/d/yyyy", 15: "d-mmm-yy", 16: "d-mmm", 17: "mmm-yy",
        18: "h:mm AM/PM", 19: "h:mm:ss AM/PM", 20: "h:mm", 21: "h:mm:ss", 22: "m/d/yyyy h:mm",
        37: "#,##0 ;(#,##0)", 38: "#,##0 ;[Red](#,##0)", 39: "#,##0.00;(#,##0.00)",
        40: "#,##0.00;[Red](#,##0.00)", 45: "mm:ss", 49: "@",
    ]

    static func format(_ raw: String, code: String, date1904: Bool) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        guard let number = Double(trimmed), number.isFinite else { return raw }
        let sections = splitSections(code)
        guard let first = sections.first, !first.isEmpty else { return general(number) }

        var section = first
        let magnitude = abs(number)
        var negativeHandled = false
        if number < 0, sections.count > 1 {
            section = sections[1]
            negativeHandled = true
        } else if number == 0, sections.count > 2 {
            section = sections[2]
        }

        let tokens = tokenize(section)
        if tokens.contains(where: { $0.kind == .unsupported }) { return general(number) }
        if tokens.contains(where: { $0.kind == .date }) {
            return formatDate(tokens, serial: number, date1904: date1904) ?? general(number)
        }
        if !tokens.contains(where: { $0.kind == .digit }) {
            // General, text, or literal-only section: the cached value as is.
            return general(number)
        }
        return formatNumber(tokens, value: magnitude, negative: number < 0 && !negativeHandled)
    }

    static func general(_ value: Double) -> String {
        if value == value.rounded(), abs(value) < 1e15 { return String(Int64(value)) }
        var text = String(format: "%.10g", value)
        if text.contains("e") { text = String(format: "%.6E", value) }
        return text
    }

    // MARK: Tokens

    private enum Kind { case literal, digit, date, general, unsupported }
    private struct Token {
        var kind: Kind
        var text: String
    }

    /// Splits on `;` outside quotes and brackets.
    private static func splitSections(_ code: String) -> [String] {
        var sections: [String] = []
        var current = ""
        var quoted = false
        var bracketed = false
        var escaped = false
        for character in code {
            if escaped { current.append(character); escaped = false; continue }
            switch character {
            case "\\": current.append(character); escaped = true
            case "\"": quoted.toggle(); current.append(character)
            case "[" where !quoted: bracketed = true; current.append(character)
            case "]" where !quoted: bracketed = false; current.append(character)
            case ";" where !quoted && !bracketed: sections.append(current); current = ""
            default: current.append(character)
            }
        }
        sections.append(current)
        return sections
    }

    private static func tokenize(_ section: String) -> [Token] {
        var tokens: [Token] = []
        let chars = Array(section)
        var index = 0
        while index < chars.count {
            let character = chars[index]
            switch character {
            case "\"":
                var text = ""
                index += 1
                while index < chars.count, chars[index] != "\"" { text.append(chars[index]); index += 1 }
                tokens.append(Token(kind: .literal, text: text))
            case "\\":
                index += 1
                if index < chars.count { tokens.append(Token(kind: .literal, text: String(chars[index]))) }
            case "_":
                index += 1
                tokens.append(Token(kind: .literal, text: " "))
            case "*":
                index += 1
            case "[":
                var inner = ""
                index += 1
                while index < chars.count, chars[index] != "]" { inner.append(chars[index]); index += 1 }
                let lower = inner.lowercased()
                if inner.hasPrefix("$") {
                    // [$EUR-407]: the symbol before the dash is the literal.
                    let symbol = inner.dropFirst().split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? ""
                    if !symbol.isEmpty { tokens.append(Token(kind: .literal, text: symbol)) }
                } else if lower.hasPrefix("h") || lower.hasPrefix("m") || lower.hasPrefix("s")
                            || lower.hasPrefix("<") || lower.hasPrefix(">") || lower.hasPrefix("=") {
                    tokens.append(Token(kind: .unsupported, text: inner))
                }
                // Colors such as [Red] are dropped.
            case "0", "#", "?":
                tokens.append(Token(kind: .digit, text: String(character)))
            case ".", ",":
                tokens.append(Token(kind: .digit, text: String(character)))
            case "%":
                tokens.append(Token(kind: .digit, text: "%"))
            case "E", "e":
                if index + 1 < chars.count, chars[index + 1] == "+" || chars[index + 1] == "-" {
                    tokens.append(Token(kind: .unsupported, text: "E"))
                    index += 1
                } else {
                    tokens.append(Token(kind: .literal, text: String(character)))
                }
            case "/":
                // A fraction when digit placeholders surround it; else a date separator.
                if tokens.last?.kind == .digit, index + 1 < chars.count, "0#?".contains(chars[index + 1]) {
                    tokens.append(Token(kind: .unsupported, text: "/"))
                } else {
                    tokens.append(Token(kind: .literal, text: "/"))
                }
            case "y", "Y", "d", "D", "h", "H", "s", "S":
                var run = String(character)
                while index + 1 < chars.count, chars[index + 1].lowercased() == String(character).lowercased() {
                    run.append(chars[index + 1]); index += 1
                }
                tokens.append(Token(kind: .date, text: run))
            case "m", "M":
                var run = String(character)
                while index + 1 < chars.count, chars[index + 1].lowercased() == "m" {
                    run.append(chars[index + 1]); index += 1
                }
                tokens.append(Token(kind: .date, text: run))
            case "a", "A":
                let rest = String(chars[index...]).uppercased()
                if rest.hasPrefix("AM/PM") {
                    tokens.append(Token(kind: .date, text: "AM/PM")); index += 4
                } else if rest.hasPrefix("A/P") {
                    tokens.append(Token(kind: .date, text: "A/P")); index += 2
                } else {
                    tokens.append(Token(kind: .literal, text: String(character)))
                }
            case "G", "g":
                let rest = String(chars[index...]).lowercased()
                if rest.hasPrefix("general") {
                    tokens.append(Token(kind: .general, text: "General")); index += 6
                } else {
                    tokens.append(Token(kind: .literal, text: String(character)))
                }
            case "@":
                tokens.append(Token(kind: .general, text: "@"))
            default:
                tokens.append(Token(kind: .literal, text: String(character)))
            }
            index += 1
        }
        return tokens
    }

    // MARK: Numbers

    private static func formatNumber(_ tokens: [Token], value: Double, negative: Bool) -> String {
        var value = value
        let percents = tokens.filter { $0.kind == .digit && $0.text == "%" }.count
        for _ in 0..<percents { value *= 100 }
        var seenPoint = false
        var minimumIntegerDigits = 0
        var decimals = 0
        var grouping = false
        for (offset, token) in tokens.enumerated() where token.kind == .digit {
            switch token.text {
            case ".": seenPoint = true
            case ",":
                // A comma between digit placeholders groups thousands.
                let before = tokens[..<offset].contains { $0.kind == .digit && "0#?".contains($0.text) }
                let after = tokens[(offset + 1)...].contains { $0.kind == .digit && "0#?".contains($0.text) }
                if before && after && !seenPoint { grouping = true }
            case "0", "#", "?":
                if seenPoint { decimals += 1 } else {
                    if token.text == "0" { minimumIntegerDigits += 1 }
                }
            default: break
            }
        }
        let fixed = String(format: "%.\(min(decimals, 15))f", value)
        var parts = fixed.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        var integer = parts[0]
        if integer == "0", minimumIntegerDigits == 0, decimals > 0 { integer = "" }
        while integer.count < minimumIntegerDigits { integer = "0" + integer }
        if grouping {
            var grouped = ""
            for (offset, character) in integer.reversed().enumerated() {
                if offset > 0, offset % 3 == 0 { grouped.append(",") }
                grouped.append(character)
            }
            integer = String(grouped.reversed())
        }
        parts[0] = integer
        let number = parts.count > 1 ? integer + "." + parts[1] : integer

        var output = negative ? "-" : ""
        var emitted = false
        for token in tokens {
            switch token.kind {
            case .literal: output += token.text
            case .digit:
                if token.text == "%" { output += "%" }
                else if !emitted { output += number; emitted = true }
            default: break
            }
        }
        return output
    }

    // MARK: Dates

    private static let monthNames = ["January", "February", "March", "April", "May", "June", "July",
                                     "August", "September", "October", "November", "December"]
    private static let dayNames = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    private static func formatDate(_ tokens: [Token], serial: Double, date1904: Bool) -> String? {
        guard serial >= 0, serial < 2_958_466 else { return nil }
        var days = Int(serial.rounded(.down))
        let fraction = serial - Double(days)
        var seconds = Int((fraction * 86_400).rounded())
        if seconds >= 86_400 { seconds -= 86_400; days += 1 }
        // Days since 1970-01-01. The 1900 system treats 1900 as a leap year
        // (the Lotus 1-2-3 bug), so serials from 61 sit one day further on.
        let epoch: Int
        if date1904 { epoch = days - 24_107 } else { epoch = days >= 61 ? days - 25_569 : days - 25_568 }
        let civil = civilDate(fromDays: epoch)
        let weekday = ((epoch % 7) + 11) % 7   // 1970-01-01 was a Thursday (4)
        let hour = seconds / 3_600
        let minute = (seconds % 3_600) / 60
        let second = seconds % 60
        let twelveHour = tokens.contains { $0.text.uppercased() == "AM/PM" || $0.text.uppercased() == "A/P" }

        var output = ""
        for (offset, token) in tokens.enumerated() {
            let lower = token.text.lowercased()
            switch token.kind {
            case .literal: output += token.text
            case .digit:
                // Punctuation the tokenizer reads as a number placeholder.
                if token.text == "," || token.text == "." || token.text == "%" { output += token.text }
            case .date:
                switch lower {
                case "yyyy", "yyy": output += String(format: "%04d", civil.year)
                case "yy", "y": output += String(format: "%02d", civil.year % 100)
                case "dddd": output += dayNames[weekday]
                case "ddd": output += String(dayNames[weekday].prefix(3))
                case "dd": output += String(format: "%02d", civil.day)
                case "d": output += String(civil.day)
                case "h", "hh":
                    let value = twelveHour ? (hour % 12 == 0 ? 12 : hour % 12) : hour
                    output += lower == "hh" ? String(format: "%02d", value) : String(value)
                case "s", "ss": output += lower == "ss" ? String(format: "%02d", second) : String(second)
                case "am/pm": output += hour < 12 ? "AM" : "PM"
                case "a/p": output += hour < 12 ? "A" : "P"
                case "mmmm": output += monthNames[civil.month - 1]
                case "mmmmm": output += String(monthNames[civil.month - 1].prefix(1))
                case "mmm": output += String(monthNames[civil.month - 1].prefix(3))
                case "m", "mm":
                    // Minutes when adjacent to an hour or second token.
                    let previousDate = tokens[..<offset].last { $0.kind == .date }
                    let nextDate = tokens[(offset + 1)...].first { $0.kind == .date }
                    let isMinute = (previousDate?.text.lowercased().hasPrefix("h") ?? false)
                        || (nextDate?.text.lowercased().hasPrefix("s") ?? false)
                    if isMinute {
                        output += lower == "mm" ? String(format: "%02d", minute) : String(minute)
                    } else {
                        output += lower == "mm" ? String(format: "%02d", civil.month) : String(civil.month)
                    }
                default: return nil
                }
            default: break
            }
        }
        return output
    }

    /// Howard Hinnant's civil-from-days; exact for the proleptic Gregorian calendar.
    private static func civilDate(fromDays days: Int) -> (year: Int, month: Int, day: Int) {
        let z = days + 719_468
        let era = (z >= 0 ? z : z - 146_096) / 146_097
        let dayOfEra = z - era * 146_097
        let yearOfEra = (dayOfEra - dayOfEra / 1_460 + dayOfEra / 36_524 - dayOfEra / 146_096) / 365
        let year = yearOfEra + era * 400
        let dayOfYear = dayOfEra - (365 * yearOfEra + yearOfEra / 4 - yearOfEra / 100)
        let mp = (5 * dayOfYear + 2) / 153
        let day = dayOfYear - (153 * mp + 2) / 5 + 1
        let month = mp < 10 ? mp + 3 : mp - 9
        return (month <= 2 ? year + 1 : year, month, day)
    }
}
