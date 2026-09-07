import Foundation

/// Strips invisible Unicode characters from model-bound prompt content.
///
/// Port of Claude Code's `src/utils/sanitization.ts` (hidden-character attack
/// mitigation). Tag characters (U+E0000 block), format controls, directional
/// overrides, and private-use characters are invisible in every renderer a
/// user will read the text in, but a model reads them as ordinary token
/// content -- which is exactly the property a hidden prompt injection needs.
/// The demonstrated attack (HackerOne #3086545) hid instructions in Unicode
/// tag characters; those are category Cf and fall to the property-class strip
/// below.
///
/// **THE WHOLE PIPELINE IS GATED BEHIND A PROBE, AND THAT IS A DELIBERATE
/// DEPARTURE FROM THE REFERENCE.** The reference NFKC-normalizes every
/// prompt unconditionally; NFKC visibly rewrites compatibility characters,
/// including the full-width punctuation ordinary CJK input is made of
/// (U+FF0C becomes U+002C), so unconditional normalization would corrupt
/// text the user chose deliberately. The probe here (`hasInvisibleCharacters`)
/// is character-class arithmetic with no normalization: ordinary text --
/// any script, any width -- returns unchanged after one cheap scan, and
/// only a string that already carries an invisible character pays for
/// NFKC. The attack payload needs no normalization to be caught: tag
/// characters and bidi overrides arrive already in the stripped classes.
///
/// **SCOPE IS USER PROMPT CONTENT ONLY, BY DECISION.** The same property
/// class that carries the attack (Cf) also carries ZWJ (U+200D), which is
/// legitimate inside emoji ZWJ sequences and several Indic and Arabic
/// shaping rules. Claude Code accepted that tradeoff for prompts and so does
/// this port; applying the strip to TOOL OUTPUT as well would corrupt real
/// content (a directory listing containing an emoji name, a document
/// excerpt in Devanagari) to guard a channel where the user can see the raw
/// text through the tool card anyway.
///
/// Applied where the model-bound content is finalized in `run()`; the
/// sanitized text is what gets stored, so the transcript and what the model
/// saw never diverge.
enum UnicodeSanitization {
    /// The fixed-point loop cannot spin forever: NFKC plus the two strips
    /// is stable after one change for any real input, and the cap exists so
    /// a pathological string degrades to "returned as reduced so far"
    /// instead of throwing mid-turn. The reference throws at its cap; a
    /// throw here would kill a whole generation over text the loop has
    /// already reduced.
    static let maximumIterations = 10

    /// ICU property classes, the primary defence. Cf is format controls
    /// (bidi marks, ZWJ/ZWNJ, joiners, the tag characters), Co is private
    /// use, Cn is unassigned.
    private static let propertyClassPattern = "[\\p{Cf}\\p{Co}\\p{Cn}]"

    /// Explicit ranges, the reference's fallback for engines whose regexes
    /// lack property-class support. ICU (NSRegularExpression) supports the
    /// classes, but the ranges are kept identical to the reference so the
    /// two implementations are checkable against each other: zero-width
    /// spaces and marks, directional formatting, directional isolates, the
    /// byte-order mark, and the BMP private-use area.
    private static let explicitRangePattern =
        "[\\u200B-\\u200F\\u202A-\\u202E\\u2066-\\u2069\\uFEFF\\uE000-\\uF8FF]"

    private static let propertyClassRegex = try! NSRegularExpression(
        pattern: propertyClassPattern)
    private static let explicitRangeRegex = try! NSRegularExpression(
        pattern: explicitRangePattern)

    /// Returns `input` with dangerous invisible characters removed, or
    /// unchanged when it carries none.
    static func sanitize(_ input: String) -> String {
        guard input.hasInvisibleCharacters else { return input }
        var current = input
        var iterations = 0
        while iterations < maximumIterations {
            var next = current.precomposedStringWithCompatibilityMapping
            next = replacingAll(in: next, with: propertyClassRegex)
            next = replacingAll(in: next, with: explicitRangeRegex)
            if next == current { break }
            current = next
            iterations += 1
        }
        return current
    }

    private static func replacingAll(in text: String, with regex: NSRegularExpression) -> String {
        let range = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(in: text, range: range, withTemplate: "")
    }
}

extension String {
    /// Whether any character falls in a class the sanitizer strips.
    ///
    /// Pure scalar inspection with no normalization, so it is safe to run
    /// on every prompt: it is the gate that keeps NFKC away from ordinary
    /// text. The ranges mirror `UnicodeSanitization`'s patterns plus the
    /// supplementary-plane blocks the property classes would catch (the
    /// tag characters U+E0000-E007F, and private use planes 15-16).
    var hasInvisibleCharacters: Bool {
        for scalar in unicodeScalars {
            if scalar.properties.isDefaultIgnorableCodePoint { return true }
            switch scalar.value {
            case 0x200B...0x200F, 0x202A...0x202E, 0x2066...0x2069,
                0xFEFF, 0xE000...0xF8FF, 0xE0000...0xE007F, 0xF0000...0x10FFFF:
                return true
            default:
                continue
            }
        }
        return false
    }
}
