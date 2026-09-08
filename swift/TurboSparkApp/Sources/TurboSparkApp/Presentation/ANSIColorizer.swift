import Foundation

/// ANSI SGR escape parser, the qwen-code `ansi.ts` parity: tool output that
/// arrives with color codes renders WITH them instead of showing raw
/// escape bytes or being flattened to one undifferentiated run.
///
/// The parse is pure and UI-free: it produces styled SEGMENTS over a small
/// palette enum, and the view maps those to SwiftUI colors. That keeps the
/// rules (which codes win, how resets scope) testable without rendering.
///
/// Supported: bold (1/21/22), faint (2), italic (3/23), underline (4/24),
/// foreground 8+bright-8 (30-37, 90-97), 256-color foreground (38;5;N with
/// N mapped onto the 16-color cube bands), background 8 (40-47, 100-107),
/// reset (0), default fg/bg (39/49). Every other CSI sequence and any OSC
/// string is stripped, not rendered.
enum ANSIColorizer {
    /// The 16-color palette a segment can carry. `.default` is the absence
    /// of a choice, which is also what an out-of-range 256-color index
    /// falls back to.
    enum Palette: UInt8, Equatable {
        case black = 0, red, green, yellow, blue, magenta, cyan, white
        case brightBlack = 8, brightRed, brightGreen, brightYellow, brightBlue
        case brightMagenta, brightCyan, brightWhite
        case `default` = 255
    }

    struct Segment: Equatable {
        var text: String
        var foreground: Palette = .default
        var background: Palette = .default
        var bold = false
        var faint = false
        var italic = false
        var underline = false

        /// Whether anything is styled; adjacent unstyled segments merge.
        var isPlain: Bool {
            foreground == .default && background == .default
                && !bold && !faint && !italic && !underline
        }
    }

    /// Splits `text` into styled runs. Consecutive segments carrying the
    /// identical style (including two plain ones) are merged, so typical
    /// output yields one segment per style change rather than per escape.
    static func segments(in text: String) -> [Segment] {
        guard text.contains("\u{1B}") else {
            return text.isEmpty ? [] : [Segment(text: text)]
        }
        var result: [Segment] = []
        var current = Segment(text: "")
        var scanner = Substring(text)
        while let escape = scanner.firstIndex(of: "\u{1B}") {
            if escape > scanner.startIndex {
                current.text = String(scanner[..<escape])
                append(&result, current)
                current.text = ""
            }
            scanner = scanner[escape...]
            let introducer = scanner.count > 1
                ? scanner[scanner.index(after: scanner.startIndex)] : nil
            if introducer == "]" {
                // OSC string: consumed whole, through its terminator (BEL,
                // ST = ESC backslash, or the single-byte 0x9C).
                var rest = scanner.dropFirst(2)
                var terminated = false
                while let c = rest.first {
                    if c == "\u{07}" { rest = rest.dropFirst(); terminated = true; break }
                    if c == "\u{1B}" {
                        let after = rest.index(after: rest.startIndex)
                        if after < rest.endIndex, rest[after] == "\\" {
                            rest = rest.dropFirst(2)
                            terminated = true
                            break
                        }
                    }
                    if c.unicodeScalars.first?.value == 0x9C {
                        rest = rest.dropFirst(); terminated = true; break
                    }
                    rest = rest.dropFirst()
                }
                scanner = terminated ? rest : Substring("")
                continue
            }
            // CSI ... final-byte; SGR is the one ending in 'm'. Everything
            // else (cursor moves, private modes) is consumed and dropped.
            guard scanner.count > 2, introducer == "[" else {
                // Not CSI: drop the ESC and one following byte (two-char
                // escape sequences) and keep going.
                scanner = scanner.dropFirst(2)
                continue
            }
            let bodyStart = scanner.index(scanner.startIndex, offsetBy: 2)
            // A CSI sequence ends at its first final byte, ASCII 0x40-0x7E
            // (@ through ~); the parameter and intermediate bytes before it
            // are 0x20-0x3F. Anything else is a truncated escape: drop the
            // introducer and keep scanning.
            guard let finalIndex = scanner[bodyStart...].firstIndex(where: { c in
                let ascii = c.asciiValue ?? 0
                return ascii >= 0x40 && ascii <= 0x7E
            }) else {
                scanner = scanner.dropFirst(2)
                continue
            }
            let final = scanner[finalIndex]
            let body = scanner[bodyStart..<finalIndex]
            if final == "m" {
                applySGR(String(body), to: &current)
            }
            scanner = scanner[scanner.index(after: finalIndex)...]
        }
        if !scanner.isEmpty {
            current.text = String(scanner)
            append(&result, current)
        } else if !current.text.isEmpty {
            append(&result, current)
        }
        return result
    }

    /// All escape sequences removed; what a plain-text consumer (clipboard,
    /// log tail) wants. Empty CSI bodies (`ESC[m`) are a bare reset and
    /// strip to nothing.
    static func stripped(_ text: String) -> String {
        segments(in: text).map(\.text).joined()
    }

    private static func append(_ result: inout [Segment], _ segment: Segment) {
        guard !segment.text.isEmpty else { return }
        if let last = result.last, last.isPlain && segment.isPlain {
            result[result.count - 1].text += segment.text
            return
        }
        if let last = result.last, !last.isPlain, !segment.isPlain,
            last.foreground == segment.foreground, last.background == segment.background,
            last.bold == segment.bold, last.faint == segment.faint,
            last.italic == segment.italic, last.underline == segment.underline
        {
            result[result.count - 1].text += segment.text
            return
        }
        result.append(segment)
    }

    private static func applySGR(_ body: String, to style: inout Segment) {
        let parts = body.isEmpty ? ["0"] : body.split(separator: ";").map(String.init)
        var index = 0
        while index < parts.count {
            guard let code = Int(parts[index]) else { index += 1; continue }
            switch code {
            case 0:
                style = Segment(text: style.text)
            case 1: style.bold = true
            case 2: style.faint = true
            case 3: style.italic = true
            case 4: style.underline = true
            case 21, 22: style.bold = false; style.faint = false
            case 23: style.italic = false
            case 24: style.underline = false
            case 30...37: style.foreground = Palette(rawValue: UInt8(code - 30)) ?? .default
            case 39: style.foreground = .default
            case 40...47: style.background = Palette(rawValue: UInt8(code - 40)) ?? .default
            case 49: style.background = .default
            case 90...97:
                style.foreground = Palette(rawValue: UInt8(code - 90 + 8)) ?? .default
            case 100...107:
                style.background = Palette(rawValue: UInt8(code - 100 + 8)) ?? .default
            case 38, 48:
                // Extended color for fg (38) or bg (48): `38;5;N` (256) or
                // `38;2;R;G;B`. The 24-bit form maps to the nearest band;
                // an unknown form just consumes its parameters.
                let target = code == 38 ? \Segment.foreground : \Segment.background
                if index + 1 < parts.count, parts[index + 1] == "5", index + 2 < parts.count,
                    let n = Int(parts[index + 2])
                {
                    style[keyPath: target] = paletteFor256(n)
                    index += 2
                } else if index + 4 < parts.count, parts[index + 1] == "2" {
                    style[keyPath: target] = paletteForRGB(
                        Int(parts[index + 2]) ?? 0, Int(parts[index + 3]) ?? 0,
                        Int(parts[index + 4]) ?? 0)
                    index += 4
                }
            default:
                break
            }
            index += 1
        }
    }

    /// 256-color index onto the 16-entry palette: the first 16 pass
    /// through, the 6x6x6 cube bands pick the nearest channel level per
    /// component, and grayscale goes to black/white by midpoint.
    static func paletteFor256(_ index: Int) -> Palette {
        switch index {
        case 0...15:
            return Palette(rawValue: UInt8(index)) ?? .default
        case 16...231:
            let cell = index - 16
            let r = (cell / 36) % 6, g = (cell / 6) % 6, b = cell % 6
            return nearestRGB(r * 51, g * 51, b * 51)
        case 232...255:
            let level = index - 232
            return level < 12 ? .black : (level < 21 ? .brightBlack : .white)
        default:
            return .default
        }
    }

    static func paletteForRGB(_ r: Int, _ g: Int, _ b: Int) -> Palette {
        nearestRGB(clamp255(r), clamp255(g), clamp255(b))
    }

    private static func clamp255(_ value: Int) -> Int { max(0, min(255, value)) }

    private static func nearestRGB(_ r: Int, _ g: Int, _ b: Int) -> Palette {
        // The 16 entries' sRGB approximations; distance in squared RGB.
        let table: [(Palette, (Int, Int, Int))] = [
            (.black, (0, 0, 0)), (.red, (205, 0, 0)), (.green, (0, 205, 0)),
            (.yellow, (205, 205, 0)), (.blue, (0, 0, 238)), (.magenta, (205, 0, 205)),
            (.cyan, (0, 205, 205)), (.white, (229, 229, 229)),
            (.brightBlack, (127, 127, 127)), (.brightRed, (255, 0, 0)),
            (.brightGreen, (0, 255, 0)), (.brightYellow, (255, 255, 0)),
            (.brightBlue, (92, 92, 255)), (.brightMagenta, (255, 0, 255)),
            (.brightCyan, (0, 255, 255)), (.brightWhite, (255, 255, 255)),
        ]
        var best: Palette = .default
        var bestDistance = Int.max
        for (candidate, (cr, cg, cb)) in table {
            let dr = r - cr, dg = g - cg, db = b - cb
            let distance = dr * dr + dg * dg + db * db
            if distance < bestDistance {
                bestDistance = distance
                best = candidate
            }
        }
        return best
    }
}
