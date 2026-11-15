import Foundation

// PowerPoint (.pptx/.pptm) render model. Slides in presentation order, shapes
// as positioned frames in EMU. Not rendered: layouts and masters (a shape
// with no own geometry falls back to fixed placeholder boxes), backgrounds
// (slides draw on white), themes and theme fonts, gradients, effects,
// transitions, animations, SmartArt, charts (placeholder), media, notes,
// rotation, and text autofit (sizes are as written).

struct SlideRun: Equatable {
    var text: String
    var bold = false
    var italic = false
    var underline = false
    var colorHex: String?
    /// Points at the slide's own coordinate space.
    var size: Double?
}

struct SlideParagraph: Equatable {
    var runs: [SlideRun] = []
    var alignment = DocxAlignment.leading
    var bulleted = false
}

enum SlideShapeKind: Equatable {
    case text, rectangle, ellipse, line, picture, table, chart
}

struct SlideShape: Equatable {
    var kind: SlideShapeKind
    /// EMU, in slide space.
    var x: Double
    var y: Double
    var width: Double
    var height: Double
    var fillHex: String?
    var lineHex: String?
    var paragraphs: [SlideParagraph] = []
    var tableRows: [[String]] = []
    var imageData: Data?
}

struct Slide {
    var shapes: [SlideShape] = []
}

struct PptxDocument {
    /// Slide size in EMU.
    var width = 9_144_000.0
    var height = 6_858_000.0
    var slides: [Slide] = []
    var notes = OfficeRenderNotes()

    static func parse(_ package: OfficePackage) throws -> PptxDocument {
        let presentationPart = try package.mainPart(fallback: "ppt/presentation.xml")
        let root = try package.xml(presentationPart)
        let rels = try package.relationships(of: presentationPart)
        var document = PptxDocument()
        document.notes.externalReferences = rels.values.filter(\.isExternal).count
        if let size = root.child("sldSz"),
           let cx = size.attrs["cx"].flatMap(Double.init), let cy = size.attrs["cy"].flatMap(Double.init),
           cx >= 914_400, cy >= 914_400, cx <= 60_000_000, cy <= 60_000_000 {
            document.width = cx
            document.height = cy
        }

        var entries: [String] = []
        for id in root.child("sldIdLst")?.all("sldId") ?? [] {
            guard let relationship = id.attrs["id"], let rel = rels[relationship],
                  !rel.isExternal, let entry = rel.entry else { continue }
            entries.append(entry)
        }
        if entries.isEmpty {
            // No usable slide list: fall back to numeric file order.
            entries = package.archive.entries.compactMap { entry -> (Int, String)? in
                guard entry.hasPrefix("ppt/slides/slide"), entry.hasSuffix(".xml") else { return nil }
                let name = ((entry as NSString).lastPathComponent as NSString).deletingPathExtension
                return Int(name.dropFirst(5)).map { ($0, entry) }
            }.sorted { $0.0 < $1.0 }.map(\.1)
        }
        guard !entries.isEmpty else { throw OfficeRenderError.missingPart("ppt/slides") }
        if entries.count > package.limits.maximumSlides {
            entries = Array(entries.prefix(package.limits.maximumSlides))
            document.notes.truncated = true
        }
        document.notes.unrenderedCharts = package.archive.entries.filter {
            $0.hasPrefix("ppt/charts/chart") && $0.hasSuffix(".xml")
        }.count

        for entry in entries {
            try package.checkDeadline()
            let slideRoot = try package.xml(entry)
            let slideRels = try package.relationships(of: entry)
            var builder = SlideBuilder(package: package, relationships: slideRels, document: document)
            builder.walk(slideRoot.child("cSld")?.child("spTree")?.children ?? [], transform: .identity)
            if builder.truncated { document.notes.truncated = true }
            document.slides.append(Slide(shapes: builder.shapes))
        }
        return document
    }

    /// Group transform: child space (`chOff`, `chExt`) onto parent space.
    private struct Transform {
        var offsetX = 0.0, offsetY = 0.0, scaleX = 1.0, scaleY = 1.0
        static let identity = Transform()
        func apply(x: Double, y: Double, w: Double, h: Double) -> (Double, Double, Double, Double) {
            (offsetX + x * scaleX, offsetY + y * scaleY, w * scaleX, h * scaleY)
        }
    }

    private struct SlideBuilder {
        let package: OfficePackage
        let relationships: [String: OfficePackage.Relationship]
        let document: PptxDocument
        var shapes: [SlideShape] = []
        var truncated = false

        init(package: OfficePackage, relationships: [String: OfficePackage.Relationship],
             document: PptxDocument) {
            self.package = package
            self.relationships = relationships
            self.document = document
        }

        mutating func walk(_ nodes: [XMLNode], transform: Transform, depth: Int = 0) {
            guard depth < 8 else { return }
            for node in nodes {
                if shapes.count >= package.limits.maximumShapesPerSlide {
                    truncated = true
                    return
                }
                switch node.name {
                case "sp": shape(node, transform: transform)
                case "cxnSp": connector(node, transform: transform)
                case "pic": picture(node, transform: transform)
                case "graphicFrame": frame(node, transform: transform)
                case "grpSp":
                    let props = node.child("grpSpPr")?.child("xfrm")
                    var inner = transform
                    if let off = props?.child("off"), let ext = props?.child("ext"),
                       let chOff = props?.child("chOff"), let chExt = props?.child("chExt"),
                       let ox = number(off, "x"), let oy = number(off, "y"),
                       let cx = number(ext, "cx"), let cy = number(ext, "cy"),
                       let px = number(chOff, "x"), let py = number(chOff, "y"),
                       let pcx = number(chExt, "cx"), let pcy = number(chExt, "cy"),
                       pcx > 0, pcy > 0 {
                        let scaleX = cx / pcx, scaleY = cy / pcy
                        let origin = transform.apply(x: ox, y: oy, w: 0, h: 0)
                        inner = Transform(
                            offsetX: origin.0 - px * scaleX * transform.scaleX,
                            offsetY: origin.1 - py * scaleY * transform.scaleY,
                            scaleX: scaleX * transform.scaleX,
                            scaleY: scaleY * transform.scaleY)
                    }
                    walk(node.children, transform: inner, depth: depth + 1)
                default: break
                }
            }
        }

        func number(_ node: XMLNode, _ key: String) -> Double? {
            guard let value = node.attrs[key].flatMap(Double.init), value.isFinite,
                  abs(value) < 1e11 else { return nil }
            return value
        }

        /// Frame from `xfrm`, or the fixed placeholder box when the shape
        /// inherits geometry from a layout this model does not read.
        func geometry(_ node: XMLNode, spPr: XMLNode?, transform: Transform)
            -> (x: Double, y: Double, w: Double, h: Double) {
            if let xfrm = spPr?.child("xfrm") ?? node.child("xfrm"),
               let off = xfrm.child("off"), let ext = xfrm.child("ext"),
               let x = number(off, "x"), let y = number(off, "y"),
               let w = number(ext, "cx"), let h = number(ext, "cy"), w >= 0, h >= 0 {
                let mapped = transform.apply(x: x, y: y, w: w, h: h)
                return (mapped.0, mapped.1, mapped.2, mapped.3)
            }
            let type = node.child("nvSpPr")?.child("nvPr")?.child("ph")?.attrs["type"] ?? "body"
            let width = document.width, height = document.height
            switch type {
            case "title", "ctrTitle": return (width * 0.05, height * 0.05, width * 0.9, height * 0.18)
            case "subTitle": return (width * 0.1, height * 0.42, width * 0.8, height * 0.2)
            default: return (width * 0.05, height * 0.27, width * 0.9, height * 0.65)
            }
        }

        func fill(_ spPr: XMLNode?) -> String? {
            guard let color = spPr?.child("solidFill") else { return nil }
            return solid(color)
        }

        /// srgbClr, or a schemeClr via the default-theme fallback.
        func solid(_ fill: XMLNode) -> String? {
            if let rgb = fill.child("srgbClr") { return OfficeColor.hex(rgb.attrs["val"]) }
            if let scheme = fill.child("schemeClr")?.attrs["val"] {
                let map = ["lt1": 0, "bg1": 0, "dk1": 1, "tx1": 1, "lt2": 2, "bg2": 2, "dk2": 3, "tx2": 3,
                           "accent1": 4, "accent2": 5, "accent3": 6, "accent4": 7, "accent5": 8, "accent6": 9,
                           "hlink": 10, "folHlink": 11]
                if let index = map[scheme] { return OfficeColor.defaultTheme[index] }
            }
            return nil
        }

        func lineColor(_ spPr: XMLNode?) -> String? {
            guard let line = spPr?.child("ln"), line.child("noFill") == nil,
                  let fill = line.child("solidFill") else { return nil }
            return solid(fill)
        }

        mutating func shape(_ node: XMLNode, transform: Transform) {
            let spPr = node.child("spPr")
            let frame = geometry(node, spPr: spPr, transform: transform)
            let preset = spPr?.child("prstGeom")?.attrs["prst"] ?? "rect"
            let paragraphs = text(node.child("txBody"))
            var kind: SlideShapeKind = preset == "ellipse" ? .ellipse : .rectangle
            if preset == "line" || preset == "straightConnector1" { kind = .line }
            let fillHex = fill(spPr)
            let lineHex = lineColor(spPr)
            // A plain text box (or placeholder) draws no outline of its own.
            if kind == .rectangle, fillHex == nil, lineHex == nil, !paragraphs.isEmpty { kind = .text }
            if paragraphs.isEmpty, fillHex == nil, lineHex == nil { return }
            shapes.append(SlideShape(
                kind: kind, x: frame.x, y: frame.y, width: frame.w, height: frame.h,
                fillHex: fillHex, lineHex: lineHex, paragraphs: paragraphs))
        }

        mutating func connector(_ node: XMLNode, transform: Transform) {
            let spPr = node.child("spPr")
            let frame = geometry(node, spPr: spPr, transform: transform)
            shapes.append(SlideShape(
                kind: .line, x: frame.x, y: frame.y, width: frame.w, height: frame.h,
                lineHex: lineColor(spPr) ?? "000000"))
        }

        mutating func picture(_ node: XMLNode, transform: Transform) {
            let spPr = node.child("spPr")
            let frame = geometry(node, spPr: spPr, transform: transform)
            var data: Data?
            if let id = node.child("blipFill")?.child("blip")?.attrs["embed"],
               let rel = relationships[id], !rel.isExternal, let entry = rel.entry,
               ["png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff"]
                .contains((entry as NSString).pathExtension.lowercased()),
               (package.archive.entrySizes[entry] ?? Int.max) <= package.limits.maximumImageBytes {
                data = try? package.data(entry)
            }
            shapes.append(SlideShape(
                kind: .picture, x: frame.x, y: frame.y, width: frame.w, height: frame.h, imageData: data))
        }

        mutating func frame(_ node: XMLNode, transform: Transform) {
            guard let xfrm = node.child("xfrm"), let off = xfrm.child("off"), let ext = xfrm.child("ext"),
                  let x = number(off, "x"), let y = number(off, "y"),
                  let w = number(ext, "cx"), let h = number(ext, "cy") else { return }
            let mapped = transform.apply(x: x, y: y, w: w, h: h)
            let graphicData = node.child("graphic")?.child("graphicData")
            if let table = graphicData?.child("tbl") {
                var rows: [[String]] = []
                var cells = 0
                for row in table.all("tr") {
                    var texts: [String] = []
                    for cell in row.all("tc") {
                        cells += 1
                        if cells > package.limits.maximumTableCells { truncated = true; break }
                        texts.append(text(cell.child("txBody")).map(\.plainText).joined(separator: "\n"))
                    }
                    rows.append(texts)
                    if cells > package.limits.maximumTableCells { break }
                }
                shapes.append(SlideShape(
                    kind: .table, x: mapped.0, y: mapped.1, width: mapped.2, height: mapped.3, tableRows: rows))
            } else {
                shapes.append(SlideShape(
                    kind: .chart, x: mapped.0, y: mapped.1, width: mapped.2, height: mapped.3))
            }
        }

        func text(_ body: XMLNode?) -> [SlideParagraph] {
            var result: [SlideParagraph] = []
            for node in body?.all("p") ?? [] {
                var paragraph = SlideParagraph()
                switch node.child("pPr")?.attrs["algn"] {
                case "ctr": paragraph.alignment = .center
                case "r": paragraph.alignment = .trailing
                case "just": paragraph.alignment = .justified
                default: break
                }
                if let pPr = node.child("pPr"), pPr.child("buChar") != nil || pPr.child("buAutoNum") != nil {
                    paragraph.bulleted = true
                }
                for child in node.children {
                    switch child.name {
                    case "r":
                        let rPr = child.child("rPr")
                        let value = child.child("t")?.text ?? ""
                        guard !value.isEmpty else { continue }
                        paragraph.runs.append(SlideRun(
                            text: String(value.prefix(package.limits.maximumNodeTextCharacters)),
                            bold: rPr?.attrs["b"] == "1",
                            italic: rPr?.attrs["i"] == "1",
                            underline: (rPr?.attrs["u"] ?? "none") != "none",
                            colorHex: rPr?.child("solidFill").flatMap(solid),
                            size: rPr?.attrs["sz"].flatMap(Double.init).map { min(max($0 / 100, 1), 400) }))
                    case "br":
                        paragraph.runs.append(SlideRun(text: "\n"))
                    default: break
                    }
                }
                result.append(paragraph)
            }
            // Drop trailing empty paragraphs so an empty placeholder is not a shape.
            while let last = result.last, last.runs.isEmpty { result.removeLast() }
            return result
        }
    }
}

extension SlideParagraph {
    var plainText: String { runs.map(\.text).joined() }
}
