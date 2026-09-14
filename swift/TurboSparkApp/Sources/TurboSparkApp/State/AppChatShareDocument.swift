import AppKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

struct AppChatExportOptions: Equatable {
    var includeToolDetails = false
}

/// A value snapshot taken before the save panel opens. The format writers
/// never consult live model state, and remote attachments are never fetched.
struct AppChatShareDocument {
    struct Attachment {
        let name: String
        let reference: String
        let png: Data?
    }

    let chat: AppChat
    let attachments: [Attachment]

    init(chat original: AppChat, liveContent: String = "", options: AppChatExportOptions = .init()) {
        var chat = original
        chat.contextSummary = nil
        chat.compactedMessageCount = 0
        chat.messages = original.messages.filter { $0.role == .user || $0.role == .assistant }.map {
            var message = $0
            message.reasoning = ""
            message.alternates = []
            if !options.includeToolDetails {
                message.toolCalls = []
                message.toolResults = []
            }
            return message
        }.filter { !$0.content.isEmpty || !$0.imagePaths.isEmpty || !$0.toolCalls.isEmpty }
        if !liveContent.isEmpty {
            chat.messages.append(AppChatMessage(role: .assistant, content: liveContent, stopReason: "partial"))
        }
        var attachments: [Attachment] = []
        for index in chat.messages.indices {
            for path in chat.messages[index].imagePaths {
                let remote = URL(string: path).map { ["http", "https"].contains($0.scheme ?? "") } ?? false
                let name = remote ? (URL(string: path)?.lastPathComponent ?? path) : (path as NSString).lastPathComponent
                let png = remote ? nil : Self.localPNG(path)
                attachments.append(Attachment(name: name, reference: path, png: png))
                chat.messages[index].content += "\n\nAttachment: \(name)\(png == nil ? " (unavailable locally)" : "")\n\(path)"
            }
        }
        self.chat = chat
        self.attachments = attachments
    }

    var markdown: String { AppChatExport.markdown(for: chat, modelAlias: nil) }

    var html: String {
        var page = AppChatExport.html(for: chat, modelAlias: nil)
        let gallery = attachments.compactMap { attachment -> String? in
            guard let png = attachment.png else { return nil }
            let label = AppChatExport.escapeHTML(attachment.name)
            return "<figure><img alt=\"\(label)\" src=\"data:image/png;base64,\(png.base64EncodedString())\" style=\"max-width:100%;height:auto\"><figcaption>\(label)</figcaption></figure>"
        }.joined(separator: "\n")
        page = page.replacingOccurrences(of: "</body>", with: gallery + "\n</body>")
        return page.replacingOccurrences(of: "<meta charset=\"utf-8\">", with: """
        <meta charset="utf-8">
        <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
        """)
    }

    func data(format: AppChatExportFormat) throws -> Data {
        switch format {
        case .markdown: return Data(markdown.utf8)
        case .html: return Data(html.utf8)
        case .json: return try AppChatExport.jsonData(for: chat)
        case .docx: return try wordData()
        }
    }

    private func wordData() throws -> Data {
        let rendered = ResponseMarkdownRenderer().render(markdown).attributedString
        let text = NSMutableAttributedString(attributedString: rendered)
        let range = NSRange(location: 0, length: text.length)
        rendered.enumerateAttribute(.font, in: range) { value, run, _ in
            guard let font = value as? NSFont else { return }
            let traits = NSFontManager.shared.traits(of: font)
            let family = font.isFixedPitch ? "Courier New" : "Arial"
            guard let base = NSFont(name: family, size: font.pointSize) else { return }
            let portable = NSFontManager.shared.convert(base, toHaveTrait: traits.intersection([.boldFontMask, .italicFontMask]))
            text.addAttribute(.font, value: portable, range: run)
        }
        // Document colors must remain readable regardless of app appearance.
        text.addAttribute(.foregroundColor, value: NSColor.black, range: range)
        text.removeAttribute(.backgroundColor, range: range)
        for attachment in attachments {
            guard let png = attachment.png, let image = NSImage(data: png) else { continue }
            let cell = NSTextAttachment()
            cell.attachmentCell = NSTextAttachmentCell(imageCell: image)
            let scale = min(1, 480 / max(image.size.width, 1))
            image.size = NSSize(width: image.size.width * scale, height: image.size.height * scale)
            text.append(NSAttributedString(string: "\n\(attachment.name)\n"))
            text.append(NSAttributedString(attachment: cell))
        }
        return try text.data(from: NSRange(location: 0, length: text.length), documentAttributes: [
            .documentType: NSAttributedString.DocumentType.officeOpenXML,
            .paperSize: NSSize(width: 612, height: 792),
            .leftMargin: 54, .rightMargin: 54, .topMargin: 54, .bottomMargin: 54,
        ])
    }

    private static func localPNG(_ path: String) -> Data? {
        guard path.hasPrefix("/"),
              let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size]) as? NSNumber,
              size.intValue <= 20 * 1024 * 1024,
              let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: 1600,
              ] as CFDictionary) else { return nil }
        return NSBitmapImageRep(cgImage: thumbnail).representation(using: .png, properties: [:])
    }
}

extension AppModel {
    func shareSelectedChat(format: AppChatExportFormat, options: AppChatExportOptions) {
        guard !selectedChat.isGhost else {
            showToast("Temporary chats cannot be exported.", style: .warning)
            return
        }
        var chat = selectedChat
        chat.messages = selectedTurnMessages
        let document = AppChatShareDocument(chat: chat, liveContent: outputText, options: options)
        guard !document.chat.messages.isEmpty else {
            showToast("This chat has no conversation to export yet.", style: .warning)
            return
        }
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.allowedContentTypes = [format.contentType]
        let invalid = CharacterSet(charactersIn: "/:\\?%*|\"<>")
        let title = chat.title.components(separatedBy: invalid).joined(separator: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        panel.nameFieldStringValue = String((title.isEmpty ? "chat" : title).prefix(60)) + "." + format.fileExtension
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try document.data(format: format).write(to: url, options: .atomic)
            showToast("Exported to \(url.lastPathComponent)", style: .success)
        } catch {
            showToast("Export failed: \(error.localizedDescription)", style: .error)
        }
    }
}
