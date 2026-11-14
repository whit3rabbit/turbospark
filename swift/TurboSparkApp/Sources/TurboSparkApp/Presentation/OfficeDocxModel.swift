import Foundation

// Word (.docx) render model. Reads word/document.xml, styles.xml and
// numbering.xml through `OfficePackage`. Not rendered, by design: themes
// (a run's literal rgb is used, which Word writes beside a theme color),
// headers and footers, footnotes and endnotes, comments, tracked-deletion
// text, fields' computed values, text boxes, SmartArt, charts, columns.

struct DocxRun: Equatable {
    var text: String
    var bold = false
    var italic = false
    var underline = false
    var strike = false
    var colorHex: String?
    /// Points.
    var size: Double?
}

enum DocxAlignment: Equatable { case leading, center, trailing, justified }

struct DocxParagraph: Equatable {
    var runs: [DocxRun] = []
    /// 1...6, nil for body text. A Title style maps to 1.
    var headingLevel: Int?
    /// A bullet or a counter such as `2.`; nil when the paragraph is not a list item.
    var marker: String?
    var listLevel = 0
    var alignment = DocxAlignment.leading
    /// Indices into `DocxDocument.images`.
    var imageIndices: [Int] = []

    var plainText: String { runs.map(\.text).joined() }
}

struct DocxCell: Equatable {
    var paragraphs: [DocxParagraph]
    var shadingHex: String?
}

struct DocxTable: Equatable {
    var rows: [[DocxCell]]
}

enum DocxBlock: Equatable {
    case paragraph(DocxParagraph)
    case table(DocxTable)
}

struct DocxOutlineEntry: Equatable {
    let blockIndex: Int
    let level: Int
    let title: String
}

struct DocxDocument {
    var blocks: [DocxBlock] = []
    var images: [Data] = []
    var outline: [DocxOutlineEntry] = []
    var notes = OfficeRenderNotes()

    // MARK: Parsing

    private struct StyleProps {
        var name = ""
        var basedOn: String?
        var bold: Bool?
        var italic: Bool?
        var underline: Bool?
        var strike: Bool?
        var colorHex: String?
        var size: Double?
        var numberingID: String?
        var numberingLevel: Int?
        var alignment: DocxAlignment?
    }

    static func parse(_ package: OfficePackage) throws -> DocxDocument {
        let mainPart = try package.mainPart(fallback: "word/document.xml")
        let root = try package.xml(mainPart)
        guard let body = root.child("body") else {
            throw OfficeRenderError.malformedXML(part: mainPart)
        }
        var context = try Context(package: package, mainPart: mainPart)
        var document = DocxDocument()
        document.notes.unrenderedCharts = package.archive.entries.filter {
            $0.hasPrefix("word/charts/chart") && $0.hasSuffix(".xml")
        }.count
        context.walk(body.children, into: &document)
        document.notes.externalReferences = context.externalReferences
        return document
    }

    private struct Context {
        let package: OfficePackage
        let relationships: [String: OfficePackage.Relationship]
        var styles: [String: StyleProps] = [:]
        var defaultSize: Double?
        var numberFormats: [String: [Int: String]] = [:]   // numId -> level -> numFmt
        var counters: [String: [Int: Int]] = [:]
        var imageIndexByEntry: [String: Int] = [:]
        var externalReferences = 0
        var tableCells = 0

        init(package: OfficePackage, mainPart: String) throws {
            self.package = package
            relationships = try package.relationships(of: mainPart)
            externalReferences = relationships.values.filter(\.isExternal).count
            if let stylesRoot = try package.optionalXML("word/styles.xml") {
                Self.readStyles(stylesRoot, into: &styles, defaultSize: &defaultSize)
            }
            if let numbering = try package.optionalXML("word/numbering.xml") {
                numberFormats = Self.readNumbering(numbering)
            }
        }

        static func runProps(_ rPr: XMLNode?, into props: inout StyleProps) {
            guard let rPr else { return }
            if let value = rPr.toggle("b") { props.bold = value }
            if let value = rPr.toggle("i") { props.italic = value }
            if let value = rPr.toggle("strike") { props.strike = value }
            if let underline = rPr.child("u") {
                props.underline = (underline.attrs["val"] ?? "single") != "none"
            }
            // Word writes the resolved literal beside a theme color, so
            // `val` is the theme fallback. "auto" means the default text color.
            if let color = rPr.child("color") { props.colorHex = OfficeColor.hex(color.attrs["val"]) }
            if let size = rPr.child("sz")?.attrs["val"].flatMap(Double.init), size > 0 {
                props.size = min(size / 2, 200)
            }
        }

        static func readStyles(
            _ root: XMLNode, into styles: inout [String: StyleProps], defaultSize: inout Double?
        ) {
            if let defaults = root.child("docDefaults")?.child("rPrDefault")?.child("rPr"),
               let size = defaults.child("sz")?.attrs["val"].flatMap(Double.init), size > 0 {
                defaultSize = min(size / 2, 200)
            }
            for node in root.all("style") {
                guard let id = node.attrs["styleId"] else { continue }
                var props = StyleProps()
                props.name = node.child("name")?.attrs["val"] ?? ""
                props.basedOn = node.child("basedOn")?.attrs["val"]
                runProps(node.child("rPr"), into: &props)
                if let pPr = node.child("pPr") {
                    props.alignment = alignment(pPr.child("jc")?.attrs["val"])
                    if let numPr = pPr.child("numPr") {
                        props.numberingID = numPr.child("numId")?.attrs["val"]
                        props.numberingLevel = numPr.child("ilvl")?.attrs["val"].flatMap(Int.init)
                    }
                }
                styles[id] = props
            }
        }

        static func readNumbering(_ root: XMLNode) -> [String: [Int: String]] {
            var abstracts: [String: [Int: String]] = [:]
            for abstract in root.all("abstractNum") {
                guard let id = abstract.attrs["abstractNumId"] else { continue }
                var levels: [Int: String] = [:]
                for level in abstract.all("lvl") {
                    if let index = level.attrs["ilvl"].flatMap(Int.init) {
                        levels[index] = level.child("numFmt")?.attrs["val"] ?? "decimal"
                    }
                }
                abstracts[id] = levels
            }
            var result: [String: [Int: String]] = [:]
            for number in root.all("num") {
                if let id = number.attrs["numId"],
                   let abstractID = number.child("abstractNumId")?.attrs["val"] {
                    result[id] = abstracts[abstractID] ?? [:]
                }
            }
            return result
        }

        static func alignment(_ value: String?) -> DocxAlignment? {
            switch value {
            case "center": return .center
            case "right", "end": return .trailing
            case "both", "distribute": return .justified
            case "left", "start": return .leading
            default: return nil
            }
        }

        /// A style with its `basedOn` chain folded in, base first. Bounded
        /// depth so a cyclic chain in a hostile file cannot loop.
        func effective(_ styleID: String?) -> StyleProps {
            var chain: [StyleProps] = []
            var current = styleID
            var seen: Set<String> = []
            while let id = current, chain.count < 8, seen.insert(id).inserted, let style = styles[id] {
                chain.append(style)
                current = style.basedOn
            }
            var merged = StyleProps()
            for style in chain.reversed() {
                merged.name = style.name
                merged.bold = style.bold ?? merged.bold
                merged.italic = style.italic ?? merged.italic
                merged.underline = style.underline ?? merged.underline
                merged.strike = style.strike ?? merged.strike
                merged.colorHex = style.colorHex ?? merged.colorHex
                merged.size = style.size ?? merged.size
                merged.numberingID = style.numberingID ?? merged.numberingID
                merged.numberingLevel = style.numberingLevel ?? merged.numberingLevel
                merged.alignment = style.alignment ?? merged.alignment
            }
            return merged
        }

        static func headingLevel(styleID: String?, style: StyleProps) -> Int? {
            let name = style.name.lowercased()
            if name == "title" { return 1 }
            for source in [name, (styleID ?? "").lowercased()] {
                for prefix in ["heading ", "heading"] where source.hasPrefix(prefix) {
                    if let level = Int(source.dropFirst(prefix.count)), (1...9).contains(level) {
                        return min(level, 6)
                    }
                }
            }
            return nil
        }

        // MARK: Blocks

        mutating func walk(_ nodes: [XMLNode], into document: inout DocxDocument) {
            for node in nodes {
                if document.blocks.count >= package.limits.maximumDocxBlocks {
                    document.notes.truncated = true
                    return
                }
                switch node.name {
                case "p":
                    let paragraph = paragraph(node, into: &document)
                    document.blocks.append(.paragraph(paragraph))
                    if let level = paragraph.headingLevel, !paragraph.plainText.isEmpty {
                        document.outline.append(DocxOutlineEntry(
                            blockIndex: document.blocks.count - 1,
                            level: level,
                            title: String(paragraph.plainText.prefix(120))))
                    }
                case "tbl":
                    if let table = table(node, into: &document) {
                        document.blocks.append(.table(table))
                    }
                case "sdt":
                    if let content = node.child("sdtContent") { walk(content.children, into: &document) }
                default:
                    break
                }
            }
        }

        mutating func paragraph(_ node: XMLNode, into document: inout DocxDocument) -> DocxParagraph {
            let pPr = node.child("pPr")
            let styleID = pPr?.child("pStyle")?.attrs["val"]
            let style = effective(styleID)
            var result = DocxParagraph()
            result.headingLevel = Self.headingLevel(styleID: styleID, style: style)
            result.alignment = Self.alignment(pPr?.child("jc")?.attrs["val"]) ?? style.alignment ?? .leading

            // Direct numPr wins over the style's. numId 0 switches numbering off.
            var numberingID = style.numberingID
            var level = style.numberingLevel ?? 0
            if let numPr = pPr?.child("numPr") {
                numberingID = numPr.child("numId")?.attrs["val"] ?? numberingID
                level = numPr.child("ilvl")?.attrs["val"].flatMap(Int.init) ?? level
            }
            if let id = numberingID, id != "0" {
                result.marker = marker(numberID: id, level: level)
                result.listLevel = min(max(level, 0), 8)
            }

            var base = StyleProps()
            base.bold = style.bold
            base.italic = style.italic
            base.underline = style.underline
            base.strike = style.strike
            base.colorHex = style.colorHex
            base.size = style.size ?? defaultSize
            collectRuns(node.children, base: base, into: &result, document: &document)
            return result
        }

        mutating func collectRuns(
            _ nodes: [XMLNode], base: StyleProps,
            into paragraph: inout DocxParagraph, document: inout DocxDocument
        ) {
            for node in nodes {
                switch node.name {
                case "r":
                    run(node, base: base, into: &paragraph, document: &document)
                case "hyperlink", "ins", "smartTag", "fldSimple", "sdt", "sdtContent", "customXml":
                    // Hyperlinks show their text only; the target is never
                    // resolved or followed. `w:del` is skipped on purpose.
                    collectRuns(node.children, base: base, into: &paragraph, document: &document)
                default:
                    break
                }
            }
        }

        mutating func run(
            _ node: XMLNode, base: StyleProps,
            into paragraph: inout DocxParagraph, document: inout DocxDocument
        ) {
            var props = base
            let rPr = node.child("rPr")
            if let characterStyle = rPr?.child("rStyle")?.attrs["val"] {
                let resolved = effective(characterStyle)
                props.bold = resolved.bold ?? props.bold
                props.italic = resolved.italic ?? props.italic
                props.underline = resolved.underline ?? props.underline
                props.colorHex = resolved.colorHex ?? props.colorHex
                props.size = resolved.size ?? props.size
            }
            Self.runProps(rPr, into: &props)

            var text = ""
            for child in node.children {
                switch child.name {
                case "t": text += child.text
                case "tab": text += "\t"
                case "br", "cr": text += "\n"
                case "noBreakHyphen": text += "\u{2011}"
                case "drawing", "pict":
                    if let index = image(in: child, document: &document) {
                        paragraph.imageIndices.append(index)
                    }
                default: break
                }
            }
            guard !text.isEmpty else { return }
            paragraph.runs.append(DocxRun(
                text: text,
                bold: props.bold ?? false,
                italic: props.italic ?? false,
                underline: props.underline ?? false,
                strike: props.strike ?? false,
                colorHex: props.colorHex,
                size: props.size))
        }

        /// The first `blip` under a drawing, resolved through the main part's
        /// relationships to a raster entry INSIDE the archive. Anything else
        /// (external link, vector format, oversized) is skipped.
        mutating func image(in node: XMLNode, document: inout DocxDocument) -> Int? {
            guard let blip = firstDescendant("blip", of: node),
                  let id = blip.attrs["embed"],
                  let rel = relationships[id], !rel.isExternal, let entry = rel.entry else {
                return nil
            }
            if let existing = imageIndexByEntry[entry] { return existing }
            let ext = (entry as NSString).pathExtension.lowercased()
            guard ["png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff"].contains(ext),
                  document.images.count < package.limits.maximumDocxImages,
                  (package.archive.entrySizes[entry] ?? 0) <= package.limits.maximumImageBytes,
                  let data = try? package.data(entry) else {
                document.notes.truncated = document.notes.truncated
                    || document.images.count >= package.limits.maximumDocxImages
                return nil
            }
            document.images.append(data)
            imageIndexByEntry[entry] = document.images.count - 1
            return document.images.count - 1
        }

        func firstDescendant(_ name: String, of node: XMLNode, depth: Int = 0) -> XMLNode? {
            if node.name == name { return node }
            guard depth < 12 else { return nil }
            for child in node.children {
                if let found = firstDescendant(name, of: child, depth: depth + 1) { return found }
            }
            return nil
        }

        mutating func table(_ node: XMLNode, into document: inout DocxDocument) -> DocxTable? {
            var rows: [[DocxCell]] = []
            for row in node.all("tr") {
                var cells: [DocxCell] = []
                for cell in row.all("tc") {
                    tableCells += 1
                    if tableCells > package.limits.maximumDocxTableCells {
                        // Keep the cells already read, including a partial row.
                        document.notes.truncated = true
                        if !cells.isEmpty { rows.append(cells) }
                        return rows.isEmpty ? nil : DocxTable(rows: rows)
                    }
                    var paragraphs: [DocxParagraph] = []
                    // Nested tables are flattened to their paragraphs.
                    for child in cell.children {
                        if child.name == "p" {
                            paragraphs.append(paragraph(child, into: &document))
                        } else if child.name == "tbl" {
                            for nested in child.all("tr").flatMap({ $0.all("tc") }).flatMap({ $0.all("p") }) {
                                paragraphs.append(paragraph(nested, into: &document))
                            }
                        }
                    }
                    let fill = cell.child("tcPr")?.child("shd")?.attrs["fill"]
                    cells.append(DocxCell(paragraphs: paragraphs, shadingHex: OfficeColor.hex(fill)))
                }
                rows.append(cells)
            }
            return rows.isEmpty ? nil : DocxTable(rows: rows)
        }

        mutating func marker(numberID: String, level: Int) -> String {
            let format = numberFormats[numberID]?[level] ?? "decimal"
            if format == "bullet" || format == "none" { return "\u{2022}" }
            var levels = counters[numberID] ?? [:]
            let value = (levels[level] ?? 0) + 1
            levels[level] = value
            for deeper in levels.keys where deeper > level { levels[deeper] = nil }
            counters[numberID] = levels
            switch format {
            case "lowerLetter": return Self.letters(value, upper: false) + "."
            case "upperLetter": return Self.letters(value, upper: true) + "."
            case "lowerRoman": return Self.roman(value).lowercased() + "."
            case "upperRoman": return Self.roman(value) + "."
            default: return "\(value)."
            }
        }

        static func letters(_ value: Int, upper: Bool) -> String {
            let base = upper ? 65 : 97
            var number = value - 1
            var result = ""
            repeat {
                result = String(UnicodeScalar(base + number % 26)!) + result
                number = number / 26 - 1
            } while number >= 0 && result.count < 4
            return result
        }

        static func roman(_ value: Int) -> String {
            guard value > 0, value < 4_000 else { return String(value) }
            let table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
                         (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
            var number = value
            var result = ""
            for (amount, symbol) in table {
                while number >= amount { result += symbol; number -= amount }
            }
            return result
        }
    }
}
