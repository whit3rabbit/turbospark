import Foundation

/// ZP-09: a pure function from a validated spec to pptx bytes. v1 is title
/// slides and content slides (title, bullets with up to three levels, plain
/// paragraphs). **Deferred:** pictures, tables, shapes, charts, notes,
/// transitions, custom layouts. One master, one layout, one theme, all
/// generated here; no external relationship exists anywhere in the package.
enum PptxWriter {
    private static let nsA = "xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\""
    private static let nsR = "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\""
    private static let nsP = "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\""

    /// 16:9, EMU.
    static let slideWidth = 12_192_000
    static let slideHeight = 6_858_000

    static func write(spec: DocumentSpec, limits: DocumentWriteLimits = .default) throws -> Data {
        var package = OOXMLPackage()
        package.addRelationships(for: "", [("rId1", OOXMLPackage.RelationshipType.officeDocument, "ppt/presentation.xml")])

        var presentation = "<p:presentation \(nsA) \(nsR) \(nsP)>"
        presentation += "<p:sldMasterIdLst><p:sldMasterId id=\"2147483648\" r:id=\"rId1\"/></p:sldMasterIdLst><p:sldIdLst>"
        var presentationRels: [(id: String, type: String, target: String)] = [
            ("rId1", OOXMLPackage.RelationshipType.slideMaster, "slideMasters/slideMaster1.xml"),
        ]
        for (index, slide) in spec.slides.enumerated() {
            let id = "rId\(index + 2)"
            presentation += "<p:sldId id=\"\(256 + index)\" r:id=\"\(id)\"/>"
            presentationRels.append((id, OOXMLPackage.RelationshipType.slide, "slides/slide\(index + 1).xml"))
            package.addXML("ppt/slides/slide\(index + 1).xml", slideXML(slide), type: OOXMLPackage.ContentType.pptxSlide)
            package.addRelationships(for: "ppt/slides/slide\(index + 1).xml", [
                ("rId1", OOXMLPackage.RelationshipType.slideLayout, "../slideLayouts/slideLayout1.xml"),
            ])
        }
        presentationRels.append(("rId\(spec.slides.count + 2)", OOXMLPackage.RelationshipType.theme, "theme/theme1.xml"))
        presentation += "</p:sldIdLst><p:sldSz cx=\"\(slideWidth)\" cy=\"\(slideHeight)\"/>"
        presentation += "<p:notesSz cx=\"6858000\" cy=\"9144000\"/></p:presentation>"
        package.addXML("ppt/presentation.xml", presentation, type: OOXMLPackage.ContentType.pptx)
        package.addRelationships(for: "ppt/presentation.xml", presentationRels)

        package.addXML("ppt/slideMasters/slideMaster1.xml", masterXML, type: OOXMLPackage.ContentType.pptxMaster)
        package.addRelationships(for: "ppt/slideMasters/slideMaster1.xml", [
            ("rId1", OOXMLPackage.RelationshipType.slideLayout, "../slideLayouts/slideLayout1.xml"),
            ("rId2", OOXMLPackage.RelationshipType.theme, "../theme/theme1.xml"),
        ])
        package.addXML("ppt/slideLayouts/slideLayout1.xml", layoutXML, type: OOXMLPackage.ContentType.pptxLayout)
        package.addRelationships(for: "ppt/slideLayouts/slideLayout1.xml", [
            ("rId1", OOXMLPackage.RelationshipType.slideMaster, "../slideMasters/slideMaster1.xml"),
        ])
        package.addXML("ppt/theme/theme1.xml", themeXML, type: OOXMLPackage.ContentType.theme)

        let output = try package.serialize()
        guard output.count <= limits.maximumOutputBytes else {
            throw DocumentWriteError.overLimit("the pptx is \(output.count) bytes (maximum \(limits.maximumOutputBytes))")
        }
        return output
    }

    // MARK: Slides

    private static func frame(_ x: Int, _ y: Int, _ cx: Int, _ cy: Int) -> String {
        "<a:xfrm><a:off x=\"\(x)\" y=\"\(y)\"/><a:ext cx=\"\(cx)\" cy=\"\(cy)\"/></a:xfrm>"
    }

    private static func paragraph(
        _ text: String, size: Int, bold: Bool = false, bullet: Bool = false, level: Int = 0, center: Bool = false
    ) -> String {
        var properties = ""
        if bullet {
            let indent = 342_900 + level * 342_900
            properties = "<a:pPr marL=\"\(indent)\" indent=\"-342900\"\(level > 0 ? " lvl=\"\(level)\"" : "")>"
                + "<a:buFont typeface=\"Arial\"/><a:buChar char=\"\u{2022}\"/></a:pPr>"
        } else {
            properties = "<a:pPr marL=\"0\" indent=\"0\"\(center ? " algn=\"ctr\"" : "")><a:buNone/></a:pPr>"
        }
        return "<a:p>\(properties)<a:r><a:rPr lang=\"en-US\" sz=\"\(size * 100)\"\(bold ? " b=\"1\"" : "") dirty=\"0\"/>"
            + "<a:t>\(OOXMLText.escape(text))</a:t></a:r></a:p>"
    }

    private static func shape(
        id: Int, name: String, placeholder: String, frame: String, paragraphs: String, anchorCenter: Bool = false
    ) -> String {
        "<p:sp><p:nvSpPr><p:cNvPr id=\"\(id)\" name=\"\(name)\"/><p:cNvSpPr><a:spLocks noGrp=\"1\"/></p:cNvSpPr>"
            + "<p:nvPr>\(placeholder)</p:nvPr></p:nvSpPr><p:spPr>\(frame)</p:spPr>"
            + "<p:txBody><a:bodyPr\(anchorCenter ? " anchor=\"ctr\"" : "")><a:normAutofit/></a:bodyPr><a:lstStyle/>\(paragraphs)</p:txBody></p:sp>"
    }

    static func slideXML(_ slide: SlideSpec) -> String {
        var shapes = ""
        switch slide.layout {
        case .title:
            shapes += shape(
                id: 2, name: "Title", placeholder: "<p:ph type=\"title\"/>",
                frame: frame(914_400, 2_130_000, 10_363_200, 1_470_000),
                paragraphs: paragraph(slide.title, size: 44, bold: true, center: true), anchorCenter: true)
            if let subtitle = slide.subtitle, !subtitle.isEmpty {
                shapes += shape(
                    id: 3, name: "Subtitle", placeholder: "<p:ph idx=\"1\"/>",
                    frame: frame(914_400, 3_800_000, 10_363_200, 1_000_000),
                    paragraphs: paragraph(subtitle, size: 24, center: true))
            }
        case .content:
            shapes += shape(
                id: 2, name: "Title", placeholder: "<p:ph type=\"title\"/>",
                frame: frame(609_600, 365_760, 10_972_800, 1_000_000),
                paragraphs: paragraph(slide.title, size: 36, bold: true), anchorCenter: true)
            let count = slide.bullets.count + slide.paragraphs.count
            let size = count > 10 ? 16 : (count > 7 ? 20 : 24)
            var body = slide.paragraphs.map { paragraph($0, size: size) }.joined()
            body += slide.bullets.map { paragraph($0.text, size: size, bullet: true, level: $0.level) }.joined()
            if body.isEmpty { body = "<a:p><a:endParaRPr lang=\"en-US\"/></a:p>" }
            shapes += shape(
                id: 3, name: "Content", placeholder: "<p:ph idx=\"1\"/>",
                frame: frame(609_600, 1_500_000, 10_972_800, 4_700_000), paragraphs: body)
        }
        return "<p:sld \(nsA) \(nsR) \(nsP)><p:cSld><p:spTree>"
            + "<p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
            + "<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/><a:chOff x=\"0\" y=\"0\"/>"
            + "<a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>\(shapes)</p:spTree></p:cSld>"
            + "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"
    }

    // MARK: Master, layout, theme

    private static let groupHeader = "<p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
        + "<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/><a:chOff x=\"0\" y=\"0\"/>"
        + "<a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>"

    private static func placeholderShape(id: Int, name: String, placeholder: String, frame: String, prompt: String) -> String {
        "<p:sp><p:nvSpPr><p:cNvPr id=\"\(id)\" name=\"\(name)\"/><p:cNvSpPr><a:spLocks noGrp=\"1\"/></p:cNvSpPr>"
            + "<p:nvPr>\(placeholder)</p:nvPr></p:nvSpPr><p:spPr>\(frame)</p:spPr>"
            + "<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang=\"en-US\"/><a:t>\(prompt)</a:t></a:r></a:p></p:txBody></p:sp>"
    }

    private static var masterXML: String {
        let title = placeholderShape(
            id: 2, name: "Title Placeholder 1", placeholder: "<p:ph type=\"title\"/>",
            frame: frame(609_600, 365_760, 10_972_800, 1_000_000), prompt: "Title")
        let body = placeholderShape(
            id: 3, name: "Text Placeholder 2", placeholder: "<p:ph type=\"body\" idx=\"1\"/>",
            frame: frame(609_600, 1_500_000, 10_972_800, 4_700_000), prompt: "Text")
        var xml = "<p:sldMaster \(nsA) \(nsR) \(nsP)><p:cSld><p:bg><p:bgRef idx=\"1001\"><a:schemeClr val=\"bg1\"/></p:bgRef></p:bg>"
        xml += "<p:spTree>\(groupHeader)\(title)\(body)</p:spTree></p:cSld>"
        xml += "<p:clrMap bg1=\"lt1\" tx1=\"dk1\" bg2=\"lt2\" tx2=\"dk2\" accent1=\"accent1\" accent2=\"accent2\" "
            + "accent3=\"accent3\" accent4=\"accent4\" accent5=\"accent5\" accent6=\"accent6\" hlink=\"hlink\" folHlink=\"folHlink\"/>"
        xml += "<p:sldLayoutIdLst><p:sldLayoutId id=\"2147483649\" r:id=\"rId1\"/></p:sldLayoutIdLst>"
        xml += "<p:txStyles><p:titleStyle><a:lvl1pPr algn=\"l\"><a:defRPr sz=\"3600\" b=\"1\"><a:solidFill><a:schemeClr val=\"tx1\"/></a:solidFill>"
            + "<a:latin typeface=\"+mj-lt\"/></a:defRPr></a:lvl1pPr></p:titleStyle>"
        xml += "<p:bodyStyle><a:lvl1pPr marL=\"342900\" indent=\"-342900\" algn=\"l\"><a:buFont typeface=\"Arial\"/><a:buChar char=\"\u{2022}\"/>"
            + "<a:defRPr sz=\"2400\"><a:solidFill><a:schemeClr val=\"tx1\"/></a:solidFill><a:latin typeface=\"+mn-lt\"/></a:defRPr></a:lvl1pPr></p:bodyStyle>"
        xml += "<p:otherStyle><a:defPPr><a:defRPr lang=\"en-US\"/></a:defPPr></p:otherStyle></p:txStyles></p:sldMaster>"
        return xml
    }

    private static var layoutXML: String {
        let title = placeholderShape(
            id: 2, name: "Title 1", placeholder: "<p:ph type=\"title\"/>",
            frame: frame(609_600, 365_760, 10_972_800, 1_000_000), prompt: "Title")
        let body = placeholderShape(
            id: 3, name: "Content Placeholder 2", placeholder: "<p:ph idx=\"1\"/>",
            frame: frame(609_600, 1_500_000, 10_972_800, 4_700_000), prompt: "Text")
        return "<p:sldLayout \(nsA) \(nsR) \(nsP) type=\"obj\" preserve=\"1\"><p:cSld name=\"Title and Content\">"
            + "<p:spTree>\(groupHeader)\(title)\(body)</p:spTree></p:cSld>"
            + "<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>"
    }

    private static var themeXML: String {
        func color(_ name: String, _ hex: String) -> String { "<a:\(name)><a:srgbClr val=\"\(hex)\"/></a:\(name)>" }
        var xml = "<a:theme \(nsA) name=\"Office Theme\"><a:themeElements><a:clrScheme name=\"Office\">"
        xml += "<a:dk1><a:sysClr val=\"windowText\" lastClr=\"000000\"/></a:dk1>"
        xml += "<a:lt1><a:sysClr val=\"window\" lastClr=\"FFFFFF\"/></a:lt1>"
        xml += color("dk2", "44546A") + color("lt2", "E7E6E6")
        xml += color("accent1", "4472C4") + color("accent2", "ED7D31") + color("accent3", "A5A5A5")
        xml += color("accent4", "FFC000") + color("accent5", "5B9BD5") + color("accent6", "70AD47")
        xml += color("hlink", "0563C1") + color("folHlink", "954F72")
        xml += "</a:clrScheme><a:fontScheme name=\"Office\">"
        xml += "<a:majorFont><a:latin typeface=\"Calibri Light\"/><a:ea typeface=\"\"/><a:cs typeface=\"\"/></a:majorFont>"
        xml += "<a:minorFont><a:latin typeface=\"Calibri\"/><a:ea typeface=\"\"/><a:cs typeface=\"\"/></a:minorFont>"
        xml += "</a:fontScheme><a:fmtScheme name=\"Office\"><a:fillStyleLst>"
        xml += String(repeating: "<a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill>", count: 3)
        xml += "</a:fillStyleLst><a:lnStyleLst>"
        for width in [6350, 12700, 19050] {
            xml += "<a:ln w=\"\(width)\"><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill></a:ln>"
        }
        xml += "</a:lnStyleLst><a:effectStyleLst>"
        xml += String(repeating: "<a:effectStyle><a:effectLst/></a:effectStyle>", count: 3)
        xml += "</a:effectStyleLst><a:bgFillStyleLst>"
        xml += String(repeating: "<a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill>", count: 3)
        xml += "</a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>"
        return xml
    }
}
