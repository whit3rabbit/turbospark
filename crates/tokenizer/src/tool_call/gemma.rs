//! Gemma tool-call DSL parser: `call:NAME{key:value,...}`. Ported from
//! `Tokenization/GemmaToolCallParser.swift`.

use std::collections::BTreeMap;

use super::{ParsedToolCall, MAXIMUM_BYTES};
use crate::error::ToolCallParserError;
use crate::json_value::{JsonValue, MAXIMUM_DEPTH};

pub struct GemmaToolCallParser;

impl GemmaToolCallParser {
    pub fn new() -> Self {
        Self
    }

    pub fn is_representable_object_key(key: &str) -> bool {
        !key.is_empty()
            && key
                .chars()
                .all(|c| c.is_alphanumeric() || "_-.$".contains(c))
    }

    pub fn parse(
        &self,
        text: &str,
        allowed_tools: &std::collections::HashSet<String>,
        id: &str,
    ) -> Result<ParsedToolCall, ToolCallParserError> {
        if text.len() > MAXIMUM_BYTES {
            return Err(ToolCallParserError::Oversized);
        }
        let mut parser = Parser::new(text);
        parser.consume("call:")?;
        let name = parser.identifier()?;
        if !allowed_tools.contains(&name) {
            return Err(ToolCallParserError::UnknownTool(name));
        }
        let arguments = parser.object(0)?;
        parser.skip_whitespace();
        if !parser.is_at_end() {
            return Err(ToolCallParserError::Malformed);
        }
        let arguments_json = arguments.encoded();
        Ok(ParsedToolCall {
            id: id.to_string(),
            name,
            arguments,
            arguments_json,
        })
    }
}

impl Default for GemmaToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

struct Parser {
    chars: Vec<char>,
    index: usize,
}

type PResult<T> = Result<T, ToolCallParserError>;

impl Parser {
    fn new(text: &str) -> Self {
        Self {
            chars: text.chars().collect(),
            index: 0,
        }
    }

    fn is_at_end(&self) -> bool {
        self.index == self.chars.len()
    }

    fn skip_whitespace(&mut self) {
        while self.index < self.chars.len() && self.chars[self.index].is_whitespace() {
            self.index += 1;
        }
    }

    fn consume(&mut self, literal: &str) -> PResult<()> {
        self.skip_whitespace();
        let value: Vec<char> = literal.chars().collect();
        if self.index + value.len() > self.chars.len()
            || self.chars[self.index..self.index + value.len()] != value[..]
        {
            return Err(ToolCallParserError::Malformed);
        }
        self.index += value.len();
        Ok(())
    }

    fn identifier(&mut self) -> PResult<String> {
        self.skip_whitespace();
        let start = self.index;
        while self.index < self.chars.len() {
            let c = self.chars[self.index];
            if !(c.is_alphanumeric() || c == '_') {
                break;
            }
            self.index += 1;
        }
        if self.index == start {
            return Err(ToolCallParserError::Malformed);
        }
        Ok(self.chars[start..self.index].iter().collect())
    }

    fn object(&mut self, depth: usize) -> PResult<JsonValue> {
        self.consume("{")?;
        let mut result = BTreeMap::new();
        self.skip_whitespace();
        if self.take('}') {
            return Ok(JsonValue::Object(result));
        }
        loop {
            let key = self.object_key()?;
            self.consume(":")?;
            result.insert(key, self.value(depth + 1)?);
            self.skip_whitespace();
            if self.take('}') {
                return Ok(JsonValue::Object(result));
            }
            self.consume(",")?;
        }
    }

    fn value(&mut self, depth: usize) -> PResult<JsonValue> {
        if depth > MAXIMUM_DEPTH {
            return Err(ToolCallParserError::Malformed);
        }
        self.skip_whitespace();
        if self.starts_with("<|\"|>") {
            return Ok(JsonValue::String(self.gemma_string()?));
        }
        if self.starts_with("\"") {
            return Ok(JsonValue::String(self.json_string()?));
        }
        if self.starts_with("{") {
            return self.object(depth);
        }
        if self.starts_with("[") {
            return self.array(depth);
        }
        if self.take_word("true") {
            return Ok(JsonValue::Bool(true));
        }
        if self.take_word("false") {
            return Ok(JsonValue::Bool(false));
        }
        if self.take_word("null") {
            return Ok(JsonValue::Null);
        }
        self.number()
    }

    fn object_key(&mut self) -> PResult<String> {
        self.skip_whitespace();
        let start = self.index;
        while self.index < self.chars.len() {
            let c = self.chars[self.index];
            if !(c.is_alphanumeric() || "_-.$".contains(c)) {
                break;
            }
            self.index += 1;
        }
        if self.index == start {
            return Err(ToolCallParserError::Malformed);
        }
        Ok(self.chars[start..self.index].iter().collect())
    }

    fn array(&mut self, depth: usize) -> PResult<JsonValue> {
        self.consume("[")?;
        let mut result = Vec::new();
        self.skip_whitespace();
        if self.take(']') {
            return Ok(JsonValue::Array(result));
        }
        loop {
            result.push(self.value(depth + 1)?);
            self.skip_whitespace();
            if self.take(']') {
                return Ok(JsonValue::Array(result));
            }
            self.consume(",")?;
        }
    }

    fn gemma_string(&mut self) -> PResult<String> {
        self.consume("<|\"|>")?;
        let mut result = String::new();
        while !self.is_at_end() {
            if self.starts_with("<|\"|>") {
                self.consume("<|\"|>")?;
                return Ok(result);
            }
            if self.chars[self.index] == '\\' && self.index + 1 < self.chars.len() {
                self.index += 1;
                result.push_str(&self.escaped_fragment()?);
            } else {
                result.push(self.chars[self.index]);
                self.index += 1;
            }
        }
        Err(ToolCallParserError::Malformed)
    }

    fn json_string(&mut self) -> PResult<String> {
        self.consume("\"")?;
        let mut result = String::new();
        while !self.is_at_end() {
            if self.take('"') {
                return Ok(result);
            }
            if self.take('\\') {
                result.push_str(&self.escaped_fragment()?);
            } else {
                result.push(self.chars[self.index]);
                self.index += 1;
            }
        }
        Err(ToolCallParserError::Malformed)
    }

    fn escaped_fragment(&mut self) -> PResult<String> {
        if self.index >= self.chars.len() {
            return Err(ToolCallParserError::Malformed);
        }
        let escape = self.chars[self.index];
        self.index += 1;
        Ok(match escape {
            '"' => "\"".to_string(),
            '\\' => "\\".to_string(),
            '/' => "/".to_string(),
            'b' => "\u{8}".to_string(),
            'f' => "\u{c}".to_string(),
            'n' => "\n".to_string(),
            'r' => "\r".to_string(),
            't' => "\t".to_string(),
            'u' => {
                let first = self.unicode_code_unit()?;
                let scalar = if (0xD800..=0xDBFF).contains(&first) {
                    if self.index + 2 > self.chars.len()
                        || self.chars[self.index] != '\\'
                        || self.chars[self.index + 1] != 'u'
                    {
                        return Err(ToolCallParserError::Malformed);
                    }
                    self.index += 2;
                    let second = self.unicode_code_unit()?;
                    if !(0xDC00..=0xDFFF).contains(&second) {
                        return Err(ToolCallParserError::Malformed);
                    }
                    0x10000 + ((first as u32 - 0xD800) << 10) + (second as u32 - 0xDC00)
                } else {
                    if (0xDC00..=0xDFFF).contains(&first) {
                        return Err(ToolCallParserError::Malformed);
                    }
                    first as u32
                };
                char::from_u32(scalar)
                    .ok_or(ToolCallParserError::Malformed)?
                    .to_string()
            }
            _ => return Err(ToolCallParserError::Malformed),
        })
    }

    fn unicode_code_unit(&mut self) -> PResult<u16> {
        if self.index + 4 > self.chars.len() {
            return Err(ToolCallParserError::Malformed);
        }
        let mut value: u16 = 0;
        for _ in 0..4 {
            let digit = self.chars[self.index]
                .to_digit(16)
                .ok_or(ToolCallParserError::Malformed)?;
            value = value * 16 + digit as u16;
            self.index += 1;
        }
        Ok(value)
    }

    fn number(&mut self) -> PResult<JsonValue> {
        let start = self.index;
        while self.index < self.chars.len() && "-+0123456789.eE".contains(self.chars[self.index]) {
            self.index += 1;
        }
        if self.index == start {
            return Err(ToolCallParserError::Malformed);
        }
        let literal: String = self.chars[start..self.index].iter().collect();
        if !is_valid_json_number(&literal) {
            return Err(ToolCallParserError::Malformed);
        }
        if !literal.contains(['.', 'e', 'E']) {
            if let Ok(v) = literal.parse::<i64>() {
                return Ok(JsonValue::Integer(v));
            }
            if let Ok(v) = literal.parse::<u64>() {
                return Ok(JsonValue::UnsignedInteger(v));
            }
        }
        literal
            .parse::<f64>()
            .map(JsonValue::Number)
            .map_err(|_| ToolCallParserError::Malformed)
    }

    fn starts_with(&self, literal: &str) -> bool {
        let value: Vec<char> = literal.chars().collect();
        self.index + value.len() <= self.chars.len()
            && self.chars[self.index..self.index + value.len()] == value[..]
    }

    fn take(&mut self, literal: char) -> bool {
        self.skip_whitespace();
        if self.index < self.chars.len() && self.chars[self.index] == literal {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn take_word(&mut self, literal: &str) -> bool {
        if self.starts_with(literal) {
            self.index += literal.chars().count();
            true
        } else {
            false
        }
    }
}

fn is_valid_json_number(literal: &str) -> bool {
    let bytes = literal.as_bytes();
    let mut i = 0;
    if i < bytes.len() && bytes[i] == b'-' {
        i += 1;
    }
    let int_start = i;
    if i < bytes.len() && bytes[i] == b'0' {
        i += 1;
    } else {
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == int_start {
            return false;
        }
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == frac_start {
            return false;
        }
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return false;
        }
    }
    i == bytes.len()
}
