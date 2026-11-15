import Foundation

/// Limit and shape checks run on a decoded spec BEFORE any writer builds a
/// part. Throws `DocumentWriteError`; never trims silently, because a model
/// that asked for 3,000 rows should be told it got none, not 1,000.
enum DocumentSpecValidation {
    static func validate(
        _ spec: DocumentSpec, format: DocumentFormat,
        limits: DocumentWriteLimits = .default
    ) throws {
        if let title = spec.title, title.count > limits.maximumTitleCharacters {
            throw DocumentWriteError.overLimit("title is over \(limits.maximumTitleCharacters) characters")
        }
        switch format {
        case .docx, .pdf:
            guard !spec.blocks.isEmpty else {
                throw DocumentWriteError.invalidSpec("'blocks' is empty: a \(format.rawValue) needs at least one block")
            }
            guard spec.sheets.isEmpty, spec.slides.isEmpty else {
                throw DocumentWriteError.invalidSpec("a \(format.rawValue) takes 'blocks', not 'sheets' or 'slides'")
            }
            try validateBlocks(spec.blocks, limits: limits)
        case .xlsx:
            guard !spec.sheets.isEmpty else {
                throw DocumentWriteError.invalidSpec("'sheets' is empty: an xlsx needs at least one sheet")
            }
            guard spec.blocks.isEmpty, spec.slides.isEmpty else {
                throw DocumentWriteError.invalidSpec("an xlsx takes 'sheets', not 'blocks' or 'slides'")
            }
            try validateSheets(spec.sheets, limits: limits)
        case .pptx:
            guard !spec.slides.isEmpty else {
                throw DocumentWriteError.invalidSpec("'slides' is empty: a pptx needs at least one slide")
            }
            guard spec.blocks.isEmpty, spec.sheets.isEmpty else {
                throw DocumentWriteError.invalidSpec("a pptx takes 'slides', not 'blocks' or 'sheets'")
            }
            try validateSlides(spec.slides, limits: limits)
        }
    }

    /// `RRGGBB`, exactly. No `#`, no names, no theme tokens: the value lands in
    /// an XML attribute and a PDF color, so it is restricted to hex digits.
    static func isHexColor(_ value: String) -> Bool {
        value.utf8.count == 6 && value.utf8.allSatisfy {
            ($0 >= 48 && $0 <= 57) || ($0 >= 65 && $0 <= 70) || ($0 >= 97 && $0 <= 102)
        }
    }

    private static func validateRuns(_ runs: [DocumentRun], limits: DocumentWriteLimits) throws {
        guard runs.count <= limits.maximumRunsPerParagraph else {
            throw DocumentWriteError.overLimit("a paragraph has \(runs.count) runs (maximum \(limits.maximumRunsPerParagraph))")
        }
        var characters = 0
        for run in runs {
            characters += run.text.count
            if let color = run.color, !isHexColor(color) {
                throw DocumentWriteError.invalidSpec("run color must be RRGGBB hex, got '\(color.prefix(12))'")
            }
            if let size = run.size, !(size >= 6 && size <= 72) {
                throw DocumentWriteError.invalidSpec("run size must be 6 to 72 points")
            }
        }
        guard characters <= limits.maximumParagraphCharacters else {
            throw DocumentWriteError.overLimit("a paragraph has \(characters) characters (maximum \(limits.maximumParagraphCharacters))")
        }
    }

    private static func validateBlocks(_ blocks: [DocumentBlock], limits: DocumentWriteLimits) throws {
        guard blocks.count <= limits.maximumBlocks else {
            throw DocumentWriteError.overLimit("\(blocks.count) blocks (maximum \(limits.maximumBlocks))")
        }
        var tableCells = 0
        var images = 0
        for block in blocks {
            switch block {
            case .heading(_, let runs), .paragraph(let runs, _):
                try validateRuns(runs, limits: limits)
            case .list(_, let items):
                guard !items.isEmpty, items.count <= limits.maximumListItems else {
                    throw DocumentWriteError.overLimit("a list needs 1 to \(limits.maximumListItems) items")
                }
                for item in items { try validateRuns(item, limits: limits) }
            case .table(let header, let rows):
                let width = max(header.count, rows.map(\.count).max() ?? 0)
                guard width > 0, width <= limits.maximumTableColumns else {
                    throw DocumentWriteError.overLimit("a table needs 1 to \(limits.maximumTableColumns) columns")
                }
                guard rows.count <= limits.maximumTableRows else {
                    throw DocumentWriteError.overLimit("a table has \(rows.count) rows (maximum \(limits.maximumTableRows))")
                }
                tableCells += width * (rows.count + (header.isEmpty ? 0 : 1))
                guard tableCells <= limits.maximumTableCells else {
                    throw DocumentWriteError.overLimit("tables hold \(tableCells) cells (maximum \(limits.maximumTableCells))")
                }
                for text in header + rows.flatMap({ $0 }) where text.count > limits.maximumCellCharacters {
                    throw DocumentWriteError.overLimit("a table cell is over \(limits.maximumCellCharacters) characters")
                }
            case .image(let reference):
                images += 1
                guard images <= limits.maximumImages else {
                    throw DocumentWriteError.overLimit("more than \(limits.maximumImages) images")
                }
                if let width = reference.widthPoints, !(width >= 16 && width <= 468) {
                    throw DocumentWriteError.invalidSpec("image width_pt must be 16 to 468")
                }
                guard reference.alt.count <= 500 else {
                    throw DocumentWriteError.overLimit("image alt text is over 500 characters")
                }
            case .pageBreak:
                break
            }
        }
    }

    /// Excel's own rules: 1 to 31 characters, none of `[]:*?/\`, no leading or
    /// trailing apostrophe, unique ignoring case.
    static func validateSheetName(_ name: String) throws {
        guard !name.isEmpty, name.count <= 31 else {
            throw DocumentWriteError.invalidSpec("a sheet name needs 1 to 31 characters, got '\(name.prefix(40))'")
        }
        if name.contains(where: { "[]:*?/\\".contains($0) }) || name.hasPrefix("'") || name.hasSuffix("'") {
            throw DocumentWriteError.invalidSpec("sheet name '\(name)' has a character Excel refuses ([ ] : * ? / \\ or an edge apostrophe)")
        }
    }

    private static func validateSheets(_ sheets: [SheetSpec], limits: DocumentWriteLimits) throws {
        guard sheets.count <= limits.maximumSheets else {
            throw DocumentWriteError.overLimit("\(sheets.count) sheets (maximum \(limits.maximumSheets))")
        }
        var seen: Set<String> = []
        var cells = 0
        for sheet in sheets {
            try validateSheetName(sheet.name)
            guard seen.insert(sheet.name.lowercased()).inserted else {
                throw DocumentWriteError.invalidSpec("duplicate sheet name '\(sheet.name)' (names compare ignoring case)")
            }
            guard sheet.rows.count <= limits.maximumSheetRows else {
                throw DocumentWriteError.overLimit("sheet '\(sheet.name)' has \(sheet.rows.count) rows (maximum \(limits.maximumSheetRows))")
            }
            guard (0...20).contains(sheet.freezeRows) else {
                throw DocumentWriteError.invalidSpec("freeze_rows must be 0 to 20")
            }
            guard sheet.columnWidths.count <= limits.maximumSheetColumns,
                  sheet.columnWidths.allSatisfy({ $0 >= 2 && $0 <= 80 }) else {
                throw DocumentWriteError.invalidSpec("columns widths must be 2 to 80 characters, at most \(limits.maximumSheetColumns)")
            }
            for row in sheet.rows {
                guard row.count <= limits.maximumSheetColumns else {
                    throw DocumentWriteError.overLimit("a row has \(row.count) cells (maximum \(limits.maximumSheetColumns))")
                }
                for cell in row {
                    cells += 1
                    if case .text(let text) = cell.value, text.count > limits.maximumSheetCellCharacters {
                        throw DocumentWriteError.overLimit("a cell is over \(limits.maximumSheetCellCharacters) characters")
                    }
                    for color in [cell.color, cell.fill].compactMap({ $0 }) where !isHexColor(color) {
                        throw DocumentWriteError.invalidSpec("cell color must be RRGGBB hex, got '\(color.prefix(12))'")
                    }
                }
            }
        }
        guard cells <= limits.maximumWorkbookCells else {
            throw DocumentWriteError.overLimit("\(cells) cells (maximum \(limits.maximumWorkbookCells))")
        }
    }

    private static func validateSlides(_ slides: [SlideSpec], limits: DocumentWriteLimits) throws {
        guard slides.count <= limits.maximumSlides else {
            throw DocumentWriteError.overLimit("\(slides.count) slides (maximum \(limits.maximumSlides))")
        }
        for slide in slides {
            let strings = [slide.title, slide.subtitle ?? ""] + slide.bullets.map(\.text) + slide.paragraphs
            guard slide.bullets.count + slide.paragraphs.count <= limits.maximumSlideItems else {
                throw DocumentWriteError.overLimit("a slide has more than \(limits.maximumSlideItems) bullets and paragraphs")
            }
            for text in strings where text.count > limits.maximumSlideItemCharacters {
                throw DocumentWriteError.overLimit("slide text is over \(limits.maximumSlideItemCharacters) characters")
            }
            for bullet in slide.bullets where !(0...2).contains(bullet.level) {
                throw DocumentWriteError.invalidSpec("bullet level must be 0 to 2")
            }
        }
    }
}
