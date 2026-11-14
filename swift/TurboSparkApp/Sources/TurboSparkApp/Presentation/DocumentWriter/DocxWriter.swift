import Foundation

/// ZP-09: a pure function from a validated spec (and already-loaded images)
/// to docx bytes. A deliberate subset: headings, styled runs, bullet and
/// numbered lists, tables, PNG/JPEG images, page breaks. No headers, footers,
/// fields, comments, tracked changes, hyperlinks or macros.
enum DocxWriter {
    private static let wordNS = "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\""
    private static let rootNamespaces = wordNS
        + " xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\""
        + " xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\""
        + " xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\""
        + " xmlns:pic=\"http://schemas.openxmlformats.org/drawingml/2006/picture\""

    static func write(
        spec: DocumentSpec,
        images: [String: DocumentImage],
        limits: DocumentWriteLimits = .default
    ) throws -> Data {
        var state = BodyState(pageSize: spec.pageSize)
        var body = ""
        for block in spec.blocks {
            body += try render(block, state: &state, images: images)
        }
        body += sectionProperties(spec.pageSize)

        var package = OOXMLPackage()
        package.addRelationships(for: "", [("rId1", OOXMLPackage.RelationshipType.officeDocument, "word/document.xml")])
        package.addXML(
            "word/document.xml",
            "<w:document \(rootNamespaces)><w:body>\(body)</w:body></w:document>",
            type: OOXMLPackage.ContentType.docx)
        package.addXML("word/styles.xml", stylesXML, type: OOXMLPackage.ContentType.docxStyles)
        package.addXML(
            "word/numbering.xml", numberingXML(orderedLists: state.orderedListCount),
            type: OOXMLPackage.ContentType.docxNumbering)

        var relationships: [(id: String, type: String, target: String)] = [
            ("rId1", OOXMLPackage.RelationshipType.styles, "styles.xml"),
            ("rId2", OOXMLPackage.RelationshipType.numbering, "numbering.xml"),
        ]
        for (index, media) in state.media.enumerated() {
            let name = "image\(index + 1).\(media.kind.fileExtension)"
            package.addBinary("word/media/\(name)", media.data, extension: media.kind.fileExtension,
                              contentType: media.kind.contentType)
            relationships.append((media.relationshipID, OOXMLPackage.RelationshipType.image, "media/\(name)"))
        }
        package.addRelationships(for: "word/document.xml", relationships)
        let output = try package.serialize()
        guard output.count <= limits.maximumOutputBytes else {
            throw DocumentWriteError.overLimit("the docx is \(output.count) bytes (maximum \(limits.maximumOutputBytes))")
        }
        return output
    }

    // MARK: Body

    private struct Media {
        let relationshipID: String
        let kind: DocumentImage.Kind
        let data: Data
    }

    private struct BodyState {
        let pageSize: DocumentPageSize
        var orderedListCount = 0
        var media: [Media] = []
        var mediaIndexByPath: [String: Int] = [:]
        var drawingID = 1
    }

    private static func render(
        _ block: DocumentBlock, state: inout BodyState, images: [String: DocumentImage]
    ) throws -> String {
        switch block {
        case .heading(let level, let runs):
            return "<w:p><w:pPr><w:pStyle w:val=\"Heading\(level)\"/></w:pPr>\(runsXML(runs))</w:p>"
        case .paragraph(let runs, let alignment):
            return "<w:p>\(paragraphProperties(style: nil, alignment: alignment))\(runsXML(runs))</w:p>"
        case .list(let ordered, let items):
            var numberID = 1
            if ordered {
                state.orderedListCount += 1
                numberID = 1 + state.orderedListCount
            }
            return items.map {
                "<w:p><w:pPr><w:pStyle w:val=\"ListParagraph\"/><w:numPr><w:ilvl w:val=\"0\"/>"
                    + "<w:numId w:val=\"\(numberID)\"/></w:numPr></w:pPr>\(runsXML($0))</w:p>"
            }.joined()
        case .table(let header, let rows):
            return tableXML(header: header, rows: rows, pageSize: state.pageSize)
        case .image(let reference):
            guard let image = images[reference.path] else {
                throw DocumentWriteError.invalidSpec("image '\(reference.path)' was not loaded")
            }
            return imageXML(reference, image: image, state: &state)
        case .pageBreak:
            return "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"
        }
    }

    private static func paragraphProperties(style: String?, alignment: DocumentAlignment) -> String {
        var inner = style.map { "<w:pStyle w:val=\"\($0)\"/>" } ?? ""
        switch alignment {
        case .left: break
        case .center: inner += "<w:jc w:val=\"center\"/>"
        case .right: inner += "<w:jc w:val=\"right\"/>"
        case .justify: inner += "<w:jc w:val=\"both\"/>"
        }
        return inner.isEmpty ? "" : "<w:pPr>\(inner)</w:pPr>"
    }

    static func runsXML(_ runs: [DocumentRun]) -> String {
        runs.map { run in
            var properties = ""
            if run.bold { properties += "<w:b/>" }
            if run.italic { properties += "<w:i/>" }
            if run.strike { properties += "<w:strike/>" }
            if let color = run.color { properties += "<w:color w:val=\"\(color.uppercased())\"/>" }
            if let size = run.size { properties += "<w:sz w:val=\"\(Int((size * 2).rounded()))\"/>" }
            if run.underline { properties += "<w:u w:val=\"single\"/>" }
            return "<w:r>" + (properties.isEmpty ? "" : "<w:rPr>\(properties)</w:rPr>") + textXML(run.text) + "</w:r>"
        }.joined()
    }

    /// Text with tabs and line breaks as their elements; everything else
    /// escaped inside `w:t`.
    private static func textXML(_ text: String) -> String {
        var out = ""
        var buffer = ""
        func flush() {
            if !buffer.isEmpty {
                out += "<w:t xml:space=\"preserve\">\(OOXMLText.escape(buffer))</w:t>"
                buffer = ""
            }
        }
        for character in text {
            switch character {
            case "\t": flush(); out += "<w:tab/>"
            case "\n", "\r\n", "\r": flush(); out += "<w:br/>"
            default: buffer.append(character)
            }
        }
        flush()
        return out
    }

    private static func tableXML(header: [String], rows: [[String]], pageSize: DocumentPageSize) -> String {
        let columns = max(header.count, rows.map(\.count).max() ?? 1)
        let usable = 9_360   // dxa between 1 inch margins on letter; a4 is scaled by the section margins
        let widthPerColumn = usable / columns
        var xml = "<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"\(usable)\" w:type=\"dxa\"/>"
            + "<w:tblLayout w:type=\"fixed\"/></w:tblPr><w:tblGrid>"
        xml += String(repeating: "<w:gridCol w:w=\"\(widthPerColumn)\"/>", count: columns)
        xml += "</w:tblGrid>"
        func row(_ cells: [String], isHeader: Bool) -> String {
            var out = "<w:tr>" + (isHeader ? "<w:trPr><w:tblHeader/></w:trPr>" : "")
            for index in 0..<columns {
                let text = index < cells.count ? cells[index] : ""
                out += "<w:tc><w:tcPr><w:tcW w:w=\"\(widthPerColumn)\" w:type=\"dxa\"/>"
                out += isHeader ? "<w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"F2F2F2\"/>" : ""
                out += "</w:tcPr><w:p>"
                out += runsXML([{ var run = DocumentRun(text: text); run.bold = isHeader; return run }()])
                out += "</w:p></w:tc>"
            }
            return out + "</w:tr>"
        }
        if !header.isEmpty { xml += row(header, isHeader: true) }
        for cells in rows { xml += row(cells, isHeader: false) }
        // A table must be followed by a paragraph before the section end.
        return xml + "</w:tbl><w:p/>"
    }

    private static func imageXML(
        _ reference: DocumentImageRef, image: DocumentImage, state: inout BodyState
    ) -> String {
        let index: Int
        if let existing = state.mediaIndexByPath[reference.path] {
            index = existing
        } else {
            state.media.append(Media(relationshipID: "rId\(state.media.count + 3)", kind: image.kind, data: image.data))
            index = state.media.count - 1
            state.mediaIndexByPath[reference.path] = index
        }
        let media = state.media[index]
        let size = image.displaySize(requestedWidth: reference.widthPoints, maximumWidth: 468)
        let cx = Int(size.width * 12_700), cy = Int(size.height * 12_700)
        let id = state.drawingID
        state.drawingID += 1
        let alt = OOXMLText.escape(reference.alt)
        let fileName = "image\(index + 1).\(image.kind.fileExtension)"
        return "<w:p><w:r><w:drawing><wp:inline distT=\"0\" distB=\"0\" distL=\"0\" distR=\"0\">"
            + "<wp:extent cx=\"\(cx)\" cy=\"\(cy)\"/>"
            + "<wp:docPr id=\"\(id)\" name=\"Picture \(id)\" descr=\"\(alt)\"/>"
            + "<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect=\"1\"/></wp:cNvGraphicFramePr>"
            + "<a:graphic><a:graphicData uri=\"http://schemas.openxmlformats.org/drawingml/2006/picture\">"
            + "<pic:pic><pic:nvPicPr><pic:cNvPr id=\"\(id)\" name=\"\(fileName)\"/><pic:cNvPicPr/></pic:nvPicPr>"
            + "<pic:blipFill><a:blip r:embed=\"\(media.relationshipID)\"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>"
            + "<pic:spPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"\(cx)\" cy=\"\(cy)\"/></a:xfrm>"
            + "<a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>"
            + "</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"
    }

    private static func sectionProperties(_ page: DocumentPageSize) -> String {
        let width = Int(page.size.width * 20), height = Int(page.size.height * 20)
        return "<w:sectPr><w:pgSz w:w=\"\(width)\" w:h=\"\(height)\"/>"
            + "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
            + "w:header=\"720\" w:footer=\"720\" w:gutter=\"0\"/></w:sectPr>"
    }

    // MARK: Static parts

    private static var stylesXML: String {
        var xml = "<w:styles \(wordNS)>"
        xml += "<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii=\"Calibri\" w:hAnsi=\"Calibri\" "
            + "w:eastAsia=\"Calibri\" w:cs=\"Calibri\"/><w:sz w:val=\"22\"/><w:szCs w:val=\"22\"/>"
            + "<w:lang w:val=\"en-US\"/></w:rPr></w:rPrDefault>"
            + "<w:pPrDefault><w:pPr><w:spacing w:after=\"120\" w:line=\"264\" w:lineRule=\"auto\"/></w:pPr></w:pPrDefault>"
            + "</w:docDefaults>"
        xml += "<w:style w:type=\"paragraph\" w:default=\"1\" w:styleId=\"Normal\"><w:name w:val=\"Normal\"/><w:qFormat/></w:style>"
        let sizes = [1: 40, 2: 32, 3: 28, 4: 24, 5: 22, 6: 22]
        for level in 1...6 {
            xml += "<w:style w:type=\"paragraph\" w:styleId=\"Heading\(level)\"><w:name w:val=\"heading \(level)\"/>"
                + "<w:basedOn w:val=\"Normal\"/><w:next w:val=\"Normal\"/><w:qFormat/>"
                + "<w:pPr><w:keepNext/><w:spacing w:before=\"\(level == 1 ? 360 : 240)\" w:after=\"120\"/>"
                + "<w:outlineLvl w:val=\"\(level - 1)\"/></w:pPr>"
                + "<w:rPr><w:b/><w:sz w:val=\"\(sizes[level] ?? 22)\"/><w:szCs w:val=\"\(sizes[level] ?? 22)\"/></w:rPr></w:style>"
        }
        xml += "<w:style w:type=\"paragraph\" w:styleId=\"ListParagraph\"><w:name w:val=\"List Paragraph\"/>"
            + "<w:basedOn w:val=\"Normal\"/><w:qFormat/><w:pPr><w:spacing w:after=\"60\"/>"
            + "<w:ind w:left=\"720\"/></w:pPr></w:style>"
        xml += "<w:style w:type=\"table\" w:default=\"1\" w:styleId=\"TableNormal\"><w:name w:val=\"Normal Table\"/>"
            + "<w:tblPr><w:tblInd w:w=\"0\" w:type=\"dxa\"/><w:tblCellMar><w:top w:w=\"0\" w:type=\"dxa\"/>"
            + "<w:left w:w=\"108\" w:type=\"dxa\"/><w:bottom w:w=\"0\" w:type=\"dxa\"/>"
            + "<w:right w:w=\"108\" w:type=\"dxa\"/></w:tblCellMar></w:tblPr></w:style>"
        xml += "<w:style w:type=\"table\" w:styleId=\"TableGrid\"><w:name w:val=\"Table Grid\"/>"
            + "<w:basedOn w:val=\"TableNormal\"/><w:tblPr><w:tblBorders>"
        for edge in ["top", "left", "bottom", "right", "insideH", "insideV"] {
            xml += "<w:\(edge) w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"A6A6A6\"/>"
        }
        xml += "</w:tblBorders></w:tblPr></w:style></w:styles>"
        return xml
    }

    /// abstractNum 0 bullets, abstractNum 1 decimal. numId 1 is the bullet
    /// list; each ordered list gets its own numId (2...) with a start
    /// override, so numbering restarts at 1 per list.
    private static func numberingXML(orderedLists: Int) -> String {
        var xml = "<w:numbering \(wordNS)>"
        xml += "<w:abstractNum w:abstractNumId=\"0\"><w:multiLevelType w:val=\"hybridMultilevel\"/>"
            + "<w:lvl w:ilvl=\"0\"><w:start w:val=\"1\"/><w:numFmt w:val=\"bullet\"/><w:lvlText w:val=\"\u{2022}\"/>"
            + "<w:lvlJc w:val=\"left\"/><w:pPr><w:ind w:left=\"720\" w:hanging=\"360\"/></w:pPr></w:lvl></w:abstractNum>"
        xml += "<w:abstractNum w:abstractNumId=\"1\"><w:multiLevelType w:val=\"hybridMultilevel\"/>"
            + "<w:lvl w:ilvl=\"0\"><w:start w:val=\"1\"/><w:numFmt w:val=\"decimal\"/><w:lvlText w:val=\"%1.\"/>"
            + "<w:lvlJc w:val=\"left\"/><w:pPr><w:ind w:left=\"720\" w:hanging=\"360\"/></w:pPr></w:lvl></w:abstractNum>"
        xml += "<w:num w:numId=\"1\"><w:abstractNumId w:val=\"0\"/></w:num>"
        for index in 0..<orderedLists {
            xml += "<w:num w:numId=\"\(index + 2)\"><w:abstractNumId w:val=\"1\"/>"
                + "<w:lvlOverride w:ilvl=\"0\"><w:startOverride w:val=\"1\"/></w:lvlOverride></w:num>"
        }
        return xml + "</w:numbering>"
    }
}
