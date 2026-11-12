import Foundation

struct WorkflowParseFailure: Error {
    let diagnostic: WorkflowScriptDiagnostic
}

enum WorkflowSourceMap {
    static func location(in bytes: [UInt8], offset requestedOffset: Int) -> WorkflowSourceLocation {
        let end = min(max(requestedOffset, 0), bytes.count)
        var index = 0
        var line = 1
        var column = 1
        var previousWasCarriageReturn = false

        while index < end {
            let byte = bytes[index]
            index += 1
            if byte == 0x0d {
                line += 1
                column = 1
                previousWasCarriageReturn = true
            } else if byte == 0x0a {
                if !previousWasCarriageReturn {
                    line += 1
                }
                column = 1
                previousWasCarriageReturn = false
            } else {
                previousWasCarriageReturn = false
                if byte & 0xc0 != 0x80 {
                    column += 1
                }
            }
        }

        return WorkflowSourceLocation(byteOffset: end, line: line, column: column)
    }
}

struct WorkflowToken: Equatable {
    enum Kind: Equatable {
        case identifier(String)
        case number(Double)
        case string(String)
        case symbol(WorkflowSymbol)
        case end
    }

    let kind: Kind
    let range: WorkflowSourceRange
}

enum WorkflowSymbol: Equatable {
    case leftParen
    case rightParen
    case leftBrace
    case rightBrace
    case leftBracket
    case rightBracket
    case dot
    case comma
    case colon
    case semicolon
    case assign
    case bang
    case strictEqual
    case strictNotEqual
    case andAnd
    case orOr
    case lessThan
    case lessThanOrEqual
    case greaterThan
    case greaterThanOrEqual
    case unsupportedOperator(String)
}

struct WorkflowLexer {
    private let bytes: [UInt8]
    private let maximumTokens: Int
    private var offset = 0
    private var line = 1
    private var column = 1
    private var previousWasCarriageReturn = false
    private var tokens: [WorkflowToken] = []

    init(bytes: [UInt8], maximumTokens: Int) {
        self.bytes = bytes
        self.maximumTokens = maximumTokens
    }

    mutating func lex() throws -> [WorkflowToken] {
        while true {
            try skipTrivia()
            guard !isAtEnd else { break }
            guard tokens.count < maximumTokens else {
                throw failure(
                    .tokenLimitExceeded,
                    "Source exceeds the \(maximumTokens)-token limit.",
                    at: currentLocation)
            }
            tokens.append(try nextToken())
        }

        let end = currentLocation
        tokens.append(WorkflowToken(
            kind: .end,
            range: WorkflowSourceRange(start: end, end: end)))
        return tokens
    }

    private var isAtEnd: Bool {
        offset >= bytes.count
    }

    private var currentByte: UInt8? {
        isAtEnd ? nil : bytes[offset]
    }

    private func peekByte(_ distance: Int = 1) -> UInt8? {
        let index = offset + distance
        return index < bytes.count ? bytes[index] : nil
    }

    private var currentLocation: WorkflowSourceLocation {
        WorkflowSourceLocation(byteOffset: offset, line: line, column: column)
    }

    private mutating func advance() -> UInt8? {
        guard !isAtEnd else { return nil }
        let byte = bytes[offset]
        offset += 1

        if byte == 0x0d {
            line += 1
            column = 1
            previousWasCarriageReturn = true
        } else if byte == 0x0a {
            if !previousWasCarriageReturn {
                line += 1
            }
            column = 1
            previousWasCarriageReturn = false
        } else {
            previousWasCarriageReturn = false
            if byte & 0xc0 != 0x80 {
                column += 1
            }
        }
        return byte
    }

    private mutating func skipTrivia() throws {
        while let byte = currentByte {
            if isWhitespace(byte) {
                _ = advance()
                continue
            }

            guard byte == 0x2f, let next = peekByte() else { return }
            if next == 0x2f {
                _ = advance()
                _ = advance()
                while let commentByte = currentByte, commentByte != 0x0a, commentByte != 0x0d {
                    _ = advance()
                }
                continue
            }

            if next == 0x2a {
                let start = currentLocation
                _ = advance()
                _ = advance()
                var closed = false
                while let commentByte = currentByte {
                    if commentByte == 0x2a, peekByte() == 0x2f {
                        _ = advance()
                        _ = advance()
                        closed = true
                        break
                    }
                    _ = advance()
                }
                guard closed else {
                    throw failure(
                        .unterminatedComment,
                        "Block comment is not terminated.",
                        at: start)
                }
                continue
            }
            return
        }
    }

    private mutating func nextToken() throws -> WorkflowToken {
        let start = currentLocation
        guard let byte = currentByte else {
            return WorkflowToken(
                kind: .end,
                range: WorkflowSourceRange(start: start, end: start))
        }

        if isIdentifierStart(byte) {
            _ = advance()
            while let next = currentByte, isIdentifierContinue(next) {
                _ = advance()
            }
            let value = String(decoding: bytes[start.byteOffset..<offset], as: UTF8.self)
            return token(.identifier(value), start: start)
        }

        if isDigit(byte) || (byte == 0x2d && peekByte().map(isDigit) == true) {
            return try numberToken(start: start)
        }

        if byte == 0x22 || byte == 0x27 {
            return try stringToken(start: start, quote: byte)
        }

        switch byte {
        case 0x28: _ = advance(); return token(.symbol(.leftParen), start: start)
        case 0x29: _ = advance(); return token(.symbol(.rightParen), start: start)
        case 0x7b: _ = advance(); return token(.symbol(.leftBrace), start: start)
        case 0x7d: _ = advance(); return token(.symbol(.rightBrace), start: start)
        case 0x5b: _ = advance(); return token(.symbol(.leftBracket), start: start)
        case 0x5d: _ = advance(); return token(.symbol(.rightBracket), start: start)
        case 0x2e: _ = advance(); return token(.symbol(.dot), start: start)
        case 0x2c: _ = advance(); return token(.symbol(.comma), start: start)
        case 0x3a: _ = advance(); return token(.symbol(.colon), start: start)
        case 0x3b: _ = advance(); return token(.symbol(.semicolon), start: start)
        case 0x3d:
            if peekByte() == 0x3d, peekByte(2) == 0x3d {
                _ = advance(); _ = advance(); _ = advance()
                return token(.symbol(.strictEqual), start: start)
            }
            if peekByte() == 0x3e {
                throw failure(.unsupportedOperator, "Arrow functions are not supported.", at: start)
            }
            _ = advance()
            return token(.symbol(.assign), start: start)
        case 0x21:
            if peekByte() == 0x3d, peekByte(2) == 0x3d {
                _ = advance(); _ = advance(); _ = advance()
                return token(.symbol(.strictNotEqual), start: start)
            }
            if peekByte() == 0x3d {
                throw failure(.unsupportedOperator, "Only strict inequality is supported.", at: start)
            }
            _ = advance()
            return token(.symbol(.bang), start: start)
        case 0x26:
            guard peekByte() == 0x26 else {
                throw failure(.unsupportedOperator, "Bitwise operators are not supported.", at: start)
            }
            _ = advance(); _ = advance()
            return token(.symbol(.andAnd), start: start)
        case 0x7c:
            guard peekByte() == 0x7c else {
                throw failure(.unsupportedOperator, "Bitwise operators are not supported.", at: start)
            }
            _ = advance(); _ = advance()
            return token(.symbol(.orOr), start: start)
        case 0x3c:
            _ = advance()
            if currentByte == 0x3d {
                _ = advance()
                return token(.symbol(.lessThanOrEqual), start: start)
            }
            return token(.symbol(.lessThan), start: start)
        case 0x3e:
            _ = advance()
            if currentByte == 0x3d {
                _ = advance()
                return token(.symbol(.greaterThanOrEqual), start: start)
            }
            return token(.symbol(.greaterThan), start: start)
        case 0x2b, 0x2a, 0x25, 0x2f, 0x7e:
            _ = advance()
            return token(.symbol(.unsupportedOperator(String(UnicodeScalar(byte)))), start: start)
        case 0x3f:
            throw failure(.unsupportedExpression, "Optional chaining and conditional expressions are not supported.", at: start)
        case 0x60:
            throw failure(.unsupportedExpression, "Template strings are not supported.", at: start)
        default:
            throw failure(.unsupportedExpression, "This source character is not supported.", at: start)
        }
    }

    private mutating func numberToken(start: WorkflowSourceLocation) throws -> WorkflowToken {
        let startOffset = offset
        if currentByte == 0x2d {
            _ = advance()
        }

        if currentByte == 0x30 {
            _ = advance()
            if currentByte.map(isDigit) == true {
                throw failure(.invalidLiteral, "Numeric literals cannot have leading zeroes.", at: start)
            }
        } else {
            guard currentByte.map(isDigit) == true else {
                throw failure(.invalidLiteral, "A number must contain an integer part.", at: start)
            }
            while currentByte.map(isDigit) == true {
                _ = advance()
            }
        }

        if currentByte == 0x2e {
            _ = advance()
            guard currentByte.map(isDigit) == true else {
                throw failure(.invalidLiteral, "A decimal point must be followed by digits.", at: start)
            }
            while currentByte.map(isDigit) == true {
                _ = advance()
            }
        }

        if currentByte == 0x65 || currentByte == 0x45 {
            _ = advance()
            if currentByte == 0x2b || currentByte == 0x2d {
                _ = advance()
            }
            guard currentByte.map(isDigit) == true else {
                throw failure(.invalidLiteral, "An exponent must contain digits.", at: start)
            }
            while currentByte.map(isDigit) == true {
                _ = advance()
            }
        }

        let text = String(decoding: bytes[startOffset..<offset], as: UTF8.self)
        guard let value = Double(text), value.isFinite else {
            throw failure(.invalidLiteral, "Numeric literals must be finite.", at: start)
        }
        return token(.number(value), start: start)
    }

    private mutating func stringToken(start: WorkflowSourceLocation, quote: UInt8) throws -> WorkflowToken {
        _ = advance()
        var decoded: [UInt8] = []

        while let byte = currentByte {
            if byte == quote {
                _ = advance()
                let value = String(decoding: decoded, as: UTF8.self)
                return token(.string(value), start: start)
            }
            if byte == 0x0a || byte == 0x0d {
                throw failure(.unterminatedString, "String literals cannot contain raw line breaks.", at: start)
            }
            if byte != 0x5c {
                decoded.append(advance()!)
                continue
            }

            let escapeLocation = currentLocation
            _ = advance()
            guard let escaped = currentByte else {
                throw failure(.unterminatedString, "String literal ends after an escape marker.", at: start)
            }
            _ = advance()
            switch escaped {
            case 0x22: decoded.append(0x22)
            case 0x27: decoded.append(0x27)
            case 0x5c: decoded.append(0x5c)
            case 0x2f: decoded.append(0x2f)
            case 0x62: decoded.append(0x08)
            case 0x66: decoded.append(0x0c)
            case 0x6e: decoded.append(0x0a)
            case 0x72: decoded.append(0x0d)
            case 0x74: decoded.append(0x09)
            case 0x76: decoded.append(0x0b)
            case 0x30:
                guard currentByte.map(isDigit) != true else {
                    throw failure(.invalidEscape, "Octal string escapes are not supported.", at: escapeLocation)
                }
                decoded.append(0)
            case 0x75:
                let first = try unicodeEscape(at: escapeLocation)
                let scalarValue: UInt32
                if (0xd800...0xdbff).contains(first) {
                    guard currentByte == 0x5c, peekByte() == 0x75 else {
                        throw failure(.invalidEscape, "A high surrogate needs a following low surrogate.", at: escapeLocation)
                    }
                    _ = advance()
                    _ = advance()
                    let second = try unicodeEscape(at: escapeLocation)
                    guard (0xdc00...0xdfff).contains(second) else {
                        throw failure(.invalidEscape, "A high surrogate must be followed by a low surrogate.", at: escapeLocation)
                    }
                    scalarValue = 0x10000 + ((UInt32(first) - 0xd800) << 10) + (UInt32(second) - 0xdc00)
                } else {
                    guard !(0xdc00...0xdfff).contains(first) else {
                        throw failure(.invalidEscape, "A low surrogate cannot appear by itself.", at: escapeLocation)
                    }
                    scalarValue = UInt32(first)
                }
                guard let scalar = UnicodeScalar(scalarValue) else {
                    throw failure(.invalidEscape, "Unicode escape is outside the scalar range.", at: escapeLocation)
                }
                decoded.append(contentsOf: String(scalar).utf8)
            default:
                throw failure(.invalidEscape, "This string escape is not supported.", at: escapeLocation)
            }
        }

        throw failure(.unterminatedString, "String literal is not terminated.", at: start)
    }

    private mutating func unicodeEscape(at location: WorkflowSourceLocation) throws -> UInt16 {
        var value: UInt16 = 0
        for _ in 0..<4 {
            guard let byte = currentByte, let digit = hexValue(byte) else {
                throw failure(.invalidEscape, "Unicode escapes need four hexadecimal digits.", at: location)
            }
            _ = advance()
            value = (value << 4) | UInt16(digit)
        }
        return value
    }

    private func hexValue(_ byte: UInt8) -> UInt8? {
        switch byte {
        case 0x30...0x39: byte - 0x30
        case 0x41...0x46: byte - 0x41 + 10
        case 0x61...0x66: byte - 0x61 + 10
        default: nil
        }
    }

    private func token(_ kind: WorkflowToken.Kind, start: WorkflowSourceLocation) -> WorkflowToken {
        WorkflowToken(kind: kind, range: WorkflowSourceRange(start: start, end: currentLocation))
    }

    private func failure(
        _ rule: WorkflowScriptDiagnosticRule,
        _ message: String,
        at location: WorkflowSourceLocation
    ) -> WorkflowParseFailure {
        WorkflowParseFailure(diagnostic: WorkflowScriptDiagnostic(
            rule: rule,
            message: message,
            location: location))
    }

    private func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x09 || byte == 0x0a || byte == 0x0d || byte == 0x0c
    }

    private func isDigit(_ byte: UInt8) -> Bool {
        (0x30...0x39).contains(byte)
    }

    private func isIdentifierStart(_ byte: UInt8) -> Bool {
        (0x41...0x5a).contains(byte) || (0x61...0x7a).contains(byte) || byte == 0x5f || byte == 0x24
    }

    private func isIdentifierContinue(_ byte: UInt8) -> Bool {
        isIdentifierStart(byte) || isDigit(byte)
    }
}

