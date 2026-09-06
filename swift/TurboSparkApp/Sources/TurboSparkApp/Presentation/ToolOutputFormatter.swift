import Foundation

/// Utilities for cleaning ANSI escape sequences and tailing large tool outputs.
public enum ToolOutputFormatter {
    private static let esc: UInt8 = 0x1B
    private static let c1Csi: UInt8 = 0x9B
    private static let c1Osc: UInt8 = 0x9D
    private static let c1St: UInt8 = 0x9C
    private static let bel: UInt8 = 0x07

    public struct OutputTail: Equatable, Sendable {
        public let fullText: String
        public let visibleText: String
        public let hiddenLineCount: Int
        public let hiddenCharCount: Int

        public var isTruncated: Bool {
            hiddenLineCount > 0 || hiddenCharCount > 0
        }
    }

    /// Strips ANSI CSI, OSC, and related terminal escape sequences from the string.
    public static func stripAnsi(_ text: String) -> String {
        guard text.contains("\u{1B}") || text.unicodeScalars.contains(where: { $0.value == 0x9B || $0.value == 0x9D }) else {
            return text
        }

        var outBytes = [UInt8]()
        outBytes.reserveCapacity(text.utf8.count)

        let bytes = Array(text.utf8)
        var i = 0
        let count = bytes.count

        while i < count {
            let byte = bytes[i]

            // Check for ESC [ (CSI) or single-byte C1 CSI (0x9B)
            if byte == esc && i + 1 < count && bytes[i + 1] == 0x5B {
                i += 2
                // Consume parameters and intermediate bytes (0x20..0x3F)
                while i < count && bytes[i] >= 0x20 && bytes[i] <= 0x3F {
                    i += 1
                }
                // Consume final byte (0x40..0x7E)
                if i < count && bytes[i] >= 0x40 && bytes[i] <= 0x7E {
                    i += 1
                }
                continue
            } else if byte == c1Csi {
                i += 1
                while i < count && bytes[i] >= 0x20 && bytes[i] <= 0x3F {
                    i += 1
                }
                if i < count && bytes[i] >= 0x40 && bytes[i] <= 0x7E {
                    i += 1
                }
                continue
            }

            // Check for ESC ] (OSC) or single-byte C1 OSC (0x9D)
            if (byte == esc && i + 1 < count && bytes[i + 1] == 0x5D) || byte == c1Osc {
                i += (byte == esc ? 2 : 1)
                // Consume until BEL (0x07) or ST (ESC \ or 0x9C)
                while i < count {
                    if bytes[i] == bel || bytes[i] == c1St {
                        i += 1
                        break
                    }
                    if bytes[i] == esc && i + 1 < count && bytes[i + 1] == 0x5C {
                        i += 2
                        break
                    }
                    i += 1
                }
                continue
            }

            // Other two-byte escape sequence (ESC + character in 0x40..0x5F)
            if byte == esc && i + 1 < count && bytes[i + 1] >= 0x40 && bytes[i + 1] <= 0x5F {
                i += 2
                continue
            }

            // Standalone control character (ASCII 0x1B by itself without matching)
            if byte == esc {
                i += 1
                continue
            }

            // Append standard UTF-8 sequence
            let scalarLength = utf8SequenceLength(byte)
            if i + scalarLength <= count {
                outBytes.append(contentsOf: bytes[i..<(i + scalarLength)])
                i += scalarLength
            } else {
                i += 1
            }
        }

        return String(decoding: outBytes, as: UTF8.self)
    }

    /// Computes the tail view of output when output exceeds limits.
    public static func tailOutput(
        _ text: String,
        maxLines: Int = 80,
        maxChars: Int = 4000
    ) -> OutputTail {
        let cleaned = stripAnsi(text)
        let lines = cleaned.components(separatedBy: "\n")

        var visible = cleaned
        var hiddenLines = 0
        var hiddenChars = 0

        if lines.count > maxLines {
            hiddenLines = lines.count - maxLines
            visible = lines.suffix(maxLines).joined(separator: "\n")
        }

        if visible.count > maxChars {
            hiddenChars = visible.count - maxChars
            visible = String(visible.suffix(maxChars))
        }

        return OutputTail(
            fullText: cleaned,
            visibleText: visible,
            hiddenLineCount: hiddenLines,
            hiddenCharCount: hiddenChars
        )
    }

    private static func utf8SequenceLength(_ byte: UInt8) -> Int {
        if (byte & 0x80) == 0 { return 1 }
        if (byte & 0xE0) == 0xC0 { return 2 }
        if (byte & 0xF0) == 0xE0 { return 3 }
        if (byte & 0xF8) == 0xF0 { return 4 }
        return 1
    }
}
