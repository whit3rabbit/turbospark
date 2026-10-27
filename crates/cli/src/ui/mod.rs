//! Terminal presentation shared by the `turbospark` front door and the user
//! views in `turbospark-model` (`ls`, `search`, `show`, `doctor`, `home`).
//!
//! ASCII only, like every other byte this crate prints (root AGENTS.md): the
//! marks are `+`, `>`, `!` and `*`, and color carries the emphasis. Color is
//! decided per stream, so `turbospark ls | less` stays plain while its
//! stderr notes stay styled, and it is off under `NO_COLOR` and `TERM=dumb`.
//! `CLICOLOR_FORCE` turns it on for a pager that understands it.
//!
//! Nothing here decides anything. A view computes its rows and hands them
//! over; a `--json` view never comes through this module at all.

#![allow(dead_code)] // Each binary that includes this uses a different part.

use std::io::IsTerminal;

use anstyle::{AnsiColor, Style};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Out,
    Err,
}

/// Whether `stream` gets ANSI styling, from the environment and the stream.
pub fn color_enabled(stream: Stream) -> bool {
    let var = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty());
    if var("CLICOLOR_FORCE").is_some_and(|v| v != "0") {
        return true;
    }
    if var("NO_COLOR").is_some() {
        return false;
    }
    if std::env::var("TERM").is_ok_and(|t| t == "dumb") {
        return false;
    }
    match stream {
        Stream::Out => std::io::stdout().is_terminal(),
        Stream::Err => std::io::stderr().is_terminal(),
    }
}

const BOLD: Style = Style::new().bold();
const DIM: Style = Style::new().dimmed();
const ACCENT: Style = AnsiColor::Green.on_default();
const LINK: Style = AnsiColor::Cyan.on_default();
const WARN: Style = AnsiColor::Yellow.on_default();
const ERROR: Style = AnsiColor::Red.on_default().bold();

/// Styles text for one stream, or passes it through when that stream has
/// no color.
#[derive(Clone, Copy, Debug)]
pub struct Paint {
    on: bool,
}

impl Paint {
    pub fn new(stream: Stream) -> Self {
        Paint {
            on: color_enabled(stream),
        }
    }

    pub fn out() -> Self {
        Self::new(Stream::Out)
    }

    pub fn err() -> Self {
        Self::new(Stream::Err)
    }

    /// Plain, whatever the terminal. For tests and `--json` neighbours.
    pub fn plain() -> Self {
        Paint { on: false }
    }

    fn style(self, style: Style, text: &str) -> String {
        if self.on && !text.is_empty() {
            format!("{style}{text}{style:#}")
        } else {
            text.to_string()
        }
    }

    pub fn bold(self, text: &str) -> String {
        self.style(BOLD, text)
    }
    pub fn dim(self, text: &str) -> String {
        self.style(DIM, text)
    }
    pub fn accent(self, text: &str) -> String {
        self.style(ACCENT, text)
    }
    pub fn link(self, text: &str) -> String {
        self.style(LINK, text)
    }
    pub fn warn(self, text: &str) -> String {
        self.style(WARN, text)
    }
    pub fn error(self, text: &str) -> String {
        self.style(ERROR, text)
    }
}

/// `  + text`, on stderr: something finished.
pub fn ok(text: &str) {
    let p = Paint::err();
    eprintln!("  {} {text}", p.accent("+"));
}

/// `  > text`, dim, on stderr: what is about to happen, or a hint.
pub fn note(text: &str) {
    let p = Paint::err();
    eprintln!("  {}", p.dim(&format!("> {text}")));
}

/// `  ! text`, on stderr.
pub fn warn(text: &str) {
    let p = Paint::err();
    eprintln!("  {} {text}", p.warn("!"));
}

/// `turbospark: error: text`, on stderr. The caller picks the exit code.
pub fn error(text: &str) {
    let p = Paint::err();
    eprintln!("{} {text}", p.error("error:"));
}

/// The terminal's width in columns, capped at 120 (wider lines read worse),
/// or 100 when stdout is not a terminal and `COLUMNS` says nothing.
pub fn columns() -> usize {
    const CAP: usize = 120;
    if let Some(n) = std::env::var("COLUMNS")
        .ok()
        .and_then(|c| c.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n.min(CAP);
    }
    terminal_columns().unwrap_or(100).min(CAP)
}

/// Through `console` (already here for indicatif) rather than an ioctl, so
/// this module carries no unsafe code.
fn terminal_columns() -> Option<usize> {
    [console::Term::stdout(), console::Term::stderr()]
        .iter()
        .find_map(|term| term.size_checked().map(|(_, cols)| cols as usize))
        .filter(|&cols| cols > 0)
}

/// Cuts `text` to `width` characters, ending in `...` when it had to.
pub fn clip(text: &str, width: usize) -> String {
    let count = text.chars().count();
    if count <= width {
        return text.to_string();
    }
    if width <= 3 {
        return text.chars().take(width).collect();
    }
    let mut out: String = text.chars().take(width - 3).collect();
    out.push_str("...");
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    Left,
    Right,
}

/// One table cell: the text that is measured, and how it is styled.
#[derive(Clone, Debug)]
pub struct Cell {
    text: String,
    style: CellStyle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellStyle {
    Plain,
    Bold,
    Dim,
    Accent,
    Warn,
}

impl Cell {
    pub fn new(text: impl Into<String>) -> Self {
        Cell {
            text: text.into(),
            style: CellStyle::Plain,
        }
    }
    pub fn styled(text: impl Into<String>, style: CellStyle) -> Self {
        Cell {
            text: text.into(),
            style,
        }
    }
}

impl From<String> for Cell {
    fn from(text: String) -> Self {
        Cell::new(text)
    }
}

impl From<&str> for Cell {
    fn from(text: &str) -> Self {
        Cell::new(text)
    }
}

/// Aligned columns, two spaces apart, indented two. Widths are measured on
/// the plain text so styling never shifts a column, and the LAST column is
/// clipped to the terminal so a long description wraps nothing.
pub struct Table {
    headers: Vec<String>,
    align: Vec<Align>,
    rows: Vec<Vec<Cell>>,
}

impl Table {
    pub fn new(headers: &[&str]) -> Self {
        Table {
            headers: headers.iter().map(|h| h.to_string()).collect(),
            align: vec![Align::Left; headers.len()],
            rows: Vec::new(),
        }
    }

    pub fn right(mut self, column: usize) -> Self {
        self.align[column] = Align::Right;
        self
    }

    pub fn row(&mut self, cells: Vec<Cell>) {
        debug_assert_eq!(cells.len(), self.headers.len());
        self.rows.push(cells);
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn render(&self, paint: Paint, width: usize) -> String {
        let columns = self.headers.len();
        let mut widths: Vec<usize> = self.headers.iter().map(|h| h.len()).collect();
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(cell.text.chars().count());
            }
        }
        // What the last column may use once the others and their gaps are
        // laid out. Never below its header.
        let used: usize = 2 + widths[..columns - 1].iter().map(|w| w + 2).sum::<usize>();
        let last_room = width
            .saturating_sub(used)
            .max(self.headers[columns - 1].len());
        widths[columns - 1] = widths[columns - 1].min(last_room);

        let mut out = String::new();
        let header: Vec<Cell> = self
            .headers
            .iter()
            .map(|h| Cell::styled(h.clone(), CellStyle::Dim))
            .collect();
        for (r, row) in std::iter::once(&header).chain(&self.rows).enumerate() {
            let mut line = String::from("  ");
            for (i, cell) in row.iter().enumerate() {
                let last = i == columns - 1;
                let text = if last {
                    clip(&cell.text, widths[i])
                } else {
                    cell.text.clone()
                };
                let pad = widths[i].saturating_sub(text.chars().count());
                let styled = match cell.style {
                    CellStyle::Plain => text,
                    CellStyle::Bold => paint.bold(&text),
                    CellStyle::Dim => paint.dim(&text),
                    CellStyle::Accent => paint.accent(&text),
                    CellStyle::Warn => paint.warn(&text),
                };
                match self.align[i] {
                    Align::Left if last => line.push_str(&styled),
                    Align::Left => {
                        line.push_str(&styled);
                        line.push_str(&" ".repeat(pad));
                    }
                    Align::Right => {
                        line.push_str(&" ".repeat(pad));
                        line.push_str(&styled);
                    }
                }
                if !last {
                    line.push_str("  ");
                }
            }
            let _ = r;
            out.push_str(line.trim_end());
            out.push('\n');
        }
        out
    }
}

/// `  Title`, bold, for a section of a screen.
pub fn section(paint: Paint, title: &str) -> String {
    format!("  {}", paint.bold(title))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_paint_adds_no_escapes() {
        let p = Paint::plain();
        assert_eq!(p.bold("x"), "x");
        assert_eq!(p.accent("ok"), "ok");
    }

    #[test]
    fn styled_paint_wraps_and_resets() {
        let p = Paint { on: true };
        let s = p.bold("x");
        assert!(s.starts_with('\u{1b}') && s.ends_with('m') && s.contains('x'));
    }

    #[test]
    fn clip_marks_what_it_cut() {
        assert_eq!(clip("abcdef", 6), "abcdef");
        assert_eq!(clip("abcdefgh", 6), "abc...");
    }

    #[test]
    fn table_aligns_on_plain_width_and_clips_the_last_column() {
        let mut t = Table::new(&["NAME", "SIZE", "ABOUT"]).right(1);
        t.row(vec![
            Cell::styled("gemma4", CellStyle::Bold),
            "14.3 GiB".into(),
            "a long description that will not fit".into(),
        ]);
        t.row(vec!["tinyllama".into(), "862 MiB".into(), "small".into()]);
        let text = t.render(Paint::plain(), 40);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "  NAME           SIZE  ABOUT");
        assert_eq!(lines[1], "  gemma4     14.3 GiB  a long descrip...");
        assert_eq!(lines[2], "  tinyllama   862 MiB  small");
        assert!(lines.iter().all(|l| l.chars().count() <= 40));
    }
}
