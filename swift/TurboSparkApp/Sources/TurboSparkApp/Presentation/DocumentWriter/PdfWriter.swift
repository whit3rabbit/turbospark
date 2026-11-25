import CoreGraphics
import CoreText
import Foundation
import ImageIO

/// ZP-09: a native PDF writer over Core Graphics and Core Text. It lays out
/// headings, styled paragraphs, lists, equal-column tables, PNG/JPEG images
/// and page breaks, flowing across pages.
///
/// Scripts are shaped by Core Text, so CJK uses the system fallback fonts and
/// right-to-left text uses Core Text's bidi handling. Neither is verified
/// against a reference rendering here, which is a stated limit (see the draft
/// and `swift/docs/SWIFT_DOCUMENTS.md`). Table rows never split across pages;
/// a row taller than a page is an error. No LaTeX, no forms, no links.
enum PdfWriter {
    private static let margin = 54.0
    private static let blockGap = 6.0

    static func write(
        spec: DocumentSpec,
        images: [String: DocumentImage],
        limits: DocumentWriteLimits = .default
    ) throws -> Data {
        let page = spec.pageSize.size
        let output = NSMutableData()
        var mediaBox = CGRect(x: 0, y: 0, width: page.width, height: page.height)
        var auxiliary: [CFString: Any] = [kCGPDFContextCreator: "TurboSpark"]
        if let title = spec.title, !title.isEmpty { auxiliary[kCGPDFContextTitle] = title }
        guard let consumer = CGDataConsumer(data: output),
              let context = CGContext(consumer: consumer, mediaBox: &mediaBox, auxiliary as CFDictionary) else {
            throw DocumentWriteError.unsupported("Core Graphics could not create a PDF context")
        }
        var layout = Layout(context: context, pageWidth: page.width, pageHeight: page.height, pageLimit: limits.maximumPdfPages)
        try layout.beginPage()
        for block in spec.blocks {
            try draw(block, layout: &layout, images: images)
        }
        layout.endPage()
        context.closePDF()
        let data = output as Data
        guard data.count <= limits.maximumOutputBytes else {
            throw DocumentWriteError.overLimit("the pdf is \(data.count) bytes (maximum \(limits.maximumOutputBytes))")
        }
        return data
    }

    // MARK: Layout state

    struct Layout {
        let context: CGContext
        let pageWidth: Double
        let pageHeight: Double
        let pageLimit: Int
        /// Distance from the top edge of the page to the next free line.
        var cursor = PdfWriter.margin
        var pages = 0
        var pageOpen = false

        var contentWidth: Double { pageWidth - 2 * PdfWriter.margin }
        var contentBottom: Double { pageHeight - PdfWriter.margin }
        var remaining: Double { contentBottom - cursor }
        var isAtTop: Bool { cursor <= PdfWriter.margin + 0.5 }

        init(context: CGContext, pageWidth: Double, pageHeight: Double, pageLimit: Int) {
            self.context = context
            self.pageWidth = pageWidth
            self.pageHeight = pageHeight
            self.pageLimit = pageLimit
        }

        mutating func beginPage() throws {
            guard pages < pageLimit else {
                throw DocumentWriteError.overLimit("the pdf would pass \(pageLimit) pages")
            }
            context.beginPDFPage(nil)
            pages += 1
            pageOpen = true
            cursor = PdfWriter.margin
        }

        mutating func endPage() {
            if pageOpen { context.endPDFPage(); pageOpen = false }
        }

        mutating func newPage() throws {
            endPage()
            try beginPage()
        }
    }

    // MARK: Blocks

    private static func draw(_ block: DocumentBlock, layout: inout Layout, images: [String: DocumentImage]) throws {
        switch block {
        case .heading(let level, let runs):
            let size = [1: 22.0, 2: 18.0, 3: 15.0][level] ?? 13.0
            if !layout.isAtTop { layout.cursor += 6 }
            try drawText(
                attributed(runs, baseSize: size, forceBold: true, alignment: .left),
                layout: &layout, indent: 0, width: layout.contentWidth)
            layout.cursor += blockGap
        case .paragraph(let runs, let alignment):
            try drawText(attributed(runs, baseSize: 11, forceBold: false, alignment: alignment),
                         layout: &layout, indent: 0, width: layout.contentWidth)
            layout.cursor += blockGap
        case .list(let ordered, let items):
            for (index, item) in items.enumerated() {
                let marker = ordered ? "\(index + 1)." : "\u{2022}"
                var runs = item
                runs.insert(DocumentRun(text: marker + "\t"), at: 0)
                try drawText(
                    attributed(runs, baseSize: 11, forceBold: false, alignment: .left, hangingIndent: 22),
                    layout: &layout, indent: 0, width: layout.contentWidth)
                layout.cursor += 2
            }
            layout.cursor += blockGap
        case .table(let header, let rows):
            try drawTable(header: header, rows: rows, layout: &layout)
            layout.cursor += blockGap
        case .image(let reference):
            guard let image = images[reference.path] else {
                throw DocumentWriteError.invalidSpec("image '\(reference.path)' was not loaded")
            }
            try drawImage(reference, image: image, layout: &layout)
            layout.cursor += blockGap
        case .pageBreak:
            if !layout.isAtTop { try layout.newPage() }
        }
    }

    // MARK: Text

    private static func color(_ hex: String?) -> CGColor {
        guard let hex, DocumentSpecValidation.isHexColor(hex), let value = UInt32(hex, radix: 16) else {
            return CGColor(gray: 0, alpha: 1)
        }
        return CGColor(
            red: CGFloat((value >> 16) & 0xFF) / 255, green: CGFloat((value >> 8) & 0xFF) / 255,
            blue: CGFloat(value & 0xFF) / 255, alpha: 1)
    }

    private static func font(size: Double, bold: Bool, italic: Bool) -> CTFont {
        let base = CTFontCreateWithName("Helvetica" as CFString, CGFloat(size), nil)
        var traits: CTFontSymbolicTraits = []
        if bold { traits.insert(.boldTrait) }
        if italic { traits.insert(.italicTrait) }
        guard !traits.isEmpty,
              let styled = CTFontCreateCopyWithSymbolicTraits(base, CGFloat(size), nil, traits, traits) else { return base }
        return styled
    }

    private static func paragraphStyle(alignment: DocumentAlignment, hangingIndent: Double) -> CTParagraphStyle {
        var textAlignment: CTTextAlignment
        switch alignment {
        case .left: textAlignment = .left
        case .center: textAlignment = .center
        case .right: textAlignment = .right
        case .justify: textAlignment = .justified
        }
        var headIndent = CGFloat(hangingIndent)
        var firstLine = CGFloat(0)
        var lineSpacing = CGFloat(2)
        let tab = CGFloat(hangingIndent)
        let tabStop = CTTextTabCreate(.left, Double(tab), nil)
        var tabs = [tabStop] as CFArray
        return withUnsafeMutablePointer(to: &textAlignment) { alignPointer in
            withUnsafeMutablePointer(to: &headIndent) { headPointer in
                withUnsafeMutablePointer(to: &firstLine) { firstPointer in
                    withUnsafeMutablePointer(to: &lineSpacing) { spacingPointer in
                        withUnsafeMutablePointer(to: &tabs) { tabsPointer in
                            let settings = [
                                CTParagraphStyleSetting(spec: .alignment, valueSize: MemoryLayout<CTTextAlignment>.size, value: alignPointer),
                                CTParagraphStyleSetting(spec: .headIndent, valueSize: MemoryLayout<CGFloat>.size, value: headPointer),
                                CTParagraphStyleSetting(spec: .firstLineHeadIndent, valueSize: MemoryLayout<CGFloat>.size, value: firstPointer),
                                CTParagraphStyleSetting(spec: .lineSpacingAdjustment, valueSize: MemoryLayout<CGFloat>.size, value: spacingPointer),
                                CTParagraphStyleSetting(spec: .tabStops, valueSize: MemoryLayout<CFArray>.size, value: tabsPointer),
                            ]
                            return CTParagraphStyleCreate(settings, hangingIndent > 0 ? settings.count : settings.count - 1)
                        }
                    }
                }
            }
        }
    }

    static func attributed(
        _ runs: [DocumentRun], baseSize: Double, forceBold: Bool,
        alignment: DocumentAlignment, hangingIndent: Double = 0
    ) -> NSAttributedString {
        let style = paragraphStyle(alignment: alignment, hangingIndent: hangingIndent)
        let result = NSMutableAttributedString()
        for run in runs {
            let size = run.size ?? baseSize
            var attributes: [NSAttributedString.Key: Any] = [
                NSAttributedString.Key(kCTFontAttributeName as String): font(size: size, bold: run.bold || forceBold, italic: run.italic),
                NSAttributedString.Key(kCTForegroundColorAttributeName as String): color(run.color),
                NSAttributedString.Key(kCTParagraphStyleAttributeName as String): style,
            ]
            if run.underline {
                attributes[NSAttributedString.Key(kCTUnderlineStyleAttributeName as String)] =
                    CTUnderlineStyle.single.rawValue
            }
            // Core Text draws a control character as a missing-glyph box.
            let cleaned = String(run.text.unicodeScalars.filter {
                $0.value >= 0x20 || $0.value == 0x09 || $0.value == 0x0A
            })
            result.append(NSAttributedString(string: cleaned, attributes: attributes))
        }
        return result
    }

    /// Flows `text` from the cursor, opening pages as needed.
    private static func drawText(
        _ text: NSAttributedString, layout: inout Layout, indent: Double, width: Double
    ) throws {
        guard text.length > 0 else { return }
        let setter = CTFramesetterCreateWithAttributedString(text)
        var location = 0
        while location < text.length {
            if layout.remaining < 24, !layout.isAtTop { try layout.newPage(); continue }
            let available = layout.remaining
            let rect = CGRect(
                x: margin + indent, y: layout.pageHeight - layout.cursor - available, width: width, height: available)
            let frame = CTFramesetterCreateFrame(
                setter, CFRange(location: location, length: 0), CGPath(rect: rect, transform: nil), nil)
            let visible = CTFrameGetVisibleStringRange(frame)
            if visible.length == 0 {
                guard !layout.isAtTop else {
                    throw DocumentWriteError.unsupported("a line of text does not fit on one page")
                }
                try layout.newPage()
                continue
            }
            CTFrameDraw(frame, layout.context)
            let used = CTFramesetterSuggestFrameSizeWithConstraints(
                setter, CFRange(location: location, length: visible.length), nil,
                CGSize(width: width, height: available), nil).height
            layout.cursor += Double(used)
            location += visible.length
            if location < text.length { try layout.newPage() }
        }
    }

    // MARK: Table

    private static func drawTable(header: [String], rows: [[String]], layout: inout Layout) throws {
        let columns = max(header.count, rows.map(\.count).max() ?? 1)
        let columnWidth = layout.contentWidth / Double(columns)
        let padding = 4.0
        let allRows: [(cells: [String], isHeader: Bool)] =
            (header.isEmpty ? [] : [(header, true)]) + rows.map { ($0, false) }

        for row in allRows {
            let texts: [NSAttributedString] = (0..<columns).map { index in
                var run = DocumentRun(text: index < row.cells.count ? row.cells[index] : "")
                run.bold = row.isHeader
                return attributed([run], baseSize: 10, forceBold: false, alignment: .left)
            }
            let heights = texts.map { text -> Double in
                guard text.length > 0 else { return 12 }
                let setter = CTFramesetterCreateWithAttributedString(text)
                return Double(CTFramesetterSuggestFrameSizeWithConstraints(
                    setter, CFRange(location: 0, length: 0), nil,
                    CGSize(width: columnWidth - 2 * padding, height: .greatestFiniteMagnitude), nil).height)
            }
            let rowHeight = (heights.max() ?? 12) + 2 * padding
            guard rowHeight <= layout.contentBottom - Self.margin else {
                throw DocumentWriteError.overLimit("a table row is taller than a page")
            }
            if rowHeight > layout.remaining { try layout.newPage() }

            let top = layout.cursor
            for (index, text) in texts.enumerated() {
                let x = margin + Double(index) * columnWidth
                let cell = CGRect(x: x, y: layout.pageHeight - top - rowHeight, width: columnWidth, height: rowHeight)
                if row.isHeader {
                    layout.context.setFillColor(CGColor(gray: 0.95, alpha: 1))
                    layout.context.fill(cell)
                }
                layout.context.setStrokeColor(CGColor(gray: 0.65, alpha: 1))
                layout.context.setLineWidth(0.5)
                layout.context.stroke(cell)
                if text.length > 0 {
                    let inner = cell.insetBy(dx: padding, dy: padding)
                    let frame = CTFramesetterCreateFrame(
                        CTFramesetterCreateWithAttributedString(text), CFRange(location: 0, length: 0),
                        CGPath(rect: inner, transform: nil), nil)
                    CTFrameDraw(frame, layout.context)
                }
            }
            layout.cursor += rowHeight
        }
    }

    // MARK: Image

    private static func drawImage(_ reference: DocumentImageRef, image: DocumentImage, layout: inout Layout) throws {
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw DocumentWriteError.invalidSpec("image '\(reference.path)' could not be decoded")
        }
        let pageContentHeight = layout.contentBottom - margin
        var size = image.displaySize(requestedWidth: reference.widthPoints, maximumWidth: layout.contentWidth)
        if size.height > pageContentHeight {
            let scale = pageContentHeight / size.height
            size = (size.width * scale, pageContentHeight)
        }
        if size.height > layout.remaining, !layout.isAtTop { try layout.newPage() }
        let rect = CGRect(
            x: margin, y: layout.pageHeight - layout.cursor - size.height, width: size.width, height: size.height)
        layout.context.draw(cgImage, in: rect)
        layout.cursor += size.height
    }
}
