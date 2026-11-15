//! Newline-delimited frame reader with a hard size bound.
//!
//! ACP over stdio is one JSON message per line. `BufRead::read_line` would
//! buffer an unbounded line, so a hostile or broken client could make the
//! adapter allocate without limit. This reader stops accumulating at `max`,
//! discards the rest of the line, and reports the frame as oversized so the
//! caller can answer with an error and keep the stream aligned.

use std::io::{self, BufRead};

#[derive(Debug, PartialEq, Eq)]
pub enum Frame {
    /// One complete line, without its terminator. Never empty.
    Line(Vec<u8>),
    /// A line longer than the bound. Its bytes were discarded.
    Oversized,
    /// End of input.
    Eof,
}

pub fn read_frame<R: BufRead>(reader: &mut R, max: usize) -> io::Result<Frame> {
    let mut buf: Vec<u8> = Vec::new();
    let mut oversized = false;
    loop {
        let chunk = match reader.fill_buf() {
            Ok(c) => c,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if chunk.is_empty() {
            // EOF. A final line without a newline still counts.
            return Ok(if oversized {
                Frame::Oversized
            } else if buf.is_empty() {
                Frame::Eof
            } else {
                Frame::Line(buf)
            });
        }
        let (take, found_newline) = match chunk.iter().position(|&b| b == b'\n') {
            Some(i) => (i, true),
            None => (chunk.len(), false),
        };
        if !oversized {
            if buf.len() + take > max {
                oversized = true;
                buf = Vec::new();
            } else {
                buf.extend_from_slice(&chunk[..take]);
            }
        }
        let consume = take + usize::from(found_newline);
        reader.consume(consume);
        if found_newline {
            if oversized {
                return Ok(Frame::Oversized);
            }
            if buf.last() == Some(&b'\r') {
                buf.pop();
            }
            if buf.is_empty() {
                // Blank line: keep reading.
                continue;
            }
            return Ok(Frame::Line(buf));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn splits_lines_and_skips_blanks() {
        let mut r = Cursor::new(b"{\"a\":1}\n\n\r\n{\"b\":2}\r\n".to_vec());
        assert_eq!(
            read_frame(&mut r, 100).unwrap(),
            Frame::Line(b"{\"a\":1}".to_vec())
        );
        assert_eq!(
            read_frame(&mut r, 100).unwrap(),
            Frame::Line(b"{\"b\":2}".to_vec())
        );
        assert_eq!(read_frame(&mut r, 100).unwrap(), Frame::Eof);
    }

    #[test]
    fn oversized_line_is_discarded_and_stream_realigns() {
        let mut data = vec![b'x'; 50];
        data.push(b'\n');
        data.extend_from_slice(b"ok\n");
        let mut r = Cursor::new(data);
        assert_eq!(read_frame(&mut r, 10).unwrap(), Frame::Oversized);
        assert_eq!(read_frame(&mut r, 10).unwrap(), Frame::Line(b"ok".to_vec()));
    }

    #[test]
    fn exactly_at_bound_is_accepted() {
        let mut data = vec![b'x'; 10];
        data.push(b'\n');
        let mut r = Cursor::new(data);
        assert_eq!(read_frame(&mut r, 10).unwrap(), Frame::Line(vec![b'x'; 10]));
    }

    #[test]
    fn oversized_across_small_buffer_chunks() {
        // A 4-byte BufReader forces many fill_buf calls per line.
        let mut data = vec![b'y'; 100];
        data.push(b'\n');
        data.extend_from_slice(b"z\n");
        let mut r = std::io::BufReader::with_capacity(4, Cursor::new(data));
        assert_eq!(read_frame(&mut r, 10).unwrap(), Frame::Oversized);
        assert_eq!(read_frame(&mut r, 10).unwrap(), Frame::Line(b"z".to_vec()));
    }

    #[test]
    fn unterminated_final_line_counts() {
        let mut r = Cursor::new(b"tail".to_vec());
        assert_eq!(
            read_frame(&mut r, 100).unwrap(),
            Frame::Line(b"tail".to_vec())
        );
        assert_eq!(read_frame(&mut r, 100).unwrap(), Frame::Eof);
    }
}
