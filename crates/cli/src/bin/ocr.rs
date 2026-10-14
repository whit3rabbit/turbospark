//! Document OCR with the Core ML Unlimited OCR export (macOS 15+).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "\
usage: turbospark-ocr --model DIR [OPTIONS] PAGE [PAGE ...]

Reads page images (PNG or JPEG) as one document with the Core ML parts of
FluidInference unlimited-ocr-mlx-coreml and prints the text. One page uses
the single-page prompt; several pages use the multi-page prompt. Needs
macOS 15 or newer. The first run compiles the decoder for this machine.

Required:
    --model DIR               directory with UnlimitedOcrPageEncoder.mlpackage,
                              UnlimitedOcrDecoder.mlpackage, embeddings.f16,
                              tokenizer.json and config.json

Options:
    --raw                     keep the layout markup (<PAGE>, <|det|> boxes)
    --max-new-tokens N        cap the generated tokens
    -h, --help                show this help
";

struct Args {
    model: PathBuf,
    pages: Vec<PathBuf>,
    raw: bool,
    max_new_tokens: Option<usize>,
}

fn parse(args: &[String]) -> Result<Option<Args>, String> {
    let mut model = None;
    let mut pages = Vec::new();
    let mut raw = false;
    let mut max_new_tokens = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => return Ok(None),
            "--model" => model = Some(PathBuf::from(it.next().ok_or("--model needs a value")?)),
            "--raw" => raw = true,
            "--max-new-tokens" => {
                let v = it.next().ok_or("--max-new-tokens needs a value")?;
                max_new_tokens =
                    Some(v.parse::<usize>().ok().filter(|n| *n > 0).ok_or_else(|| {
                        format!("--max-new-tokens {v} is not a positive integer")
                    })?);
            }
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            page => pages.push(PathBuf::from(page)),
        }
    }
    let model = model.ok_or("--model is required")?;
    if pages.is_empty() {
        return Err("give at least one page image".into());
    }
    Ok(Some(Args {
        model,
        pages,
        raw,
        max_new_tokens,
    }))
}

#[cfg(target_os = "macos")]
fn run(a: Args) -> Result<(), String> {
    use runtime::coreml::{OcrEvent, OcrOptions, OcrPage, UnlimitedOcrCoreMl};
    let opts = OcrOptions {
        max_new_tokens: a.max_new_tokens,
        ..OcrOptions::default()
    };
    let ocr = UnlimitedOcrCoreMl::open(&a.model, &opts)?;
    let pages = a
        .pages
        .iter()
        .map(|p| OcrPage::from_file(Path::new(p)))
        .collect::<Result<Vec<_>, _>>()?;
    let prompt = if pages.len() == 1 { "single" } else { "multi" };
    let result = ocr.read(&pages, prompt, |e| {
        match e {
            OcrEvent::PageEncoded { index } => eprintln!("encoded page {}", index + 1),
            OcrEvent::Prefilled { tokens, .. } => eprintln!("reading ({tokens} prompt tokens)"),
            OcrEvent::Token { .. } => {}
        }
        true
    })?;
    if !result.stats.stopped_at_eos {
        eprintln!(
            "warning: stopped at the token limit after {} tokens; the text may be cut off",
            result.stats.output_tokens
        );
    }
    if a.raw {
        println!("{}", result.text);
    } else {
        println!("{}", result.plain_text());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn run(_a: Args) -> Result<(), String> {
    Err("OCR runs on Core ML and needs macOS 15 or newer".into())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse(&args) {
        Ok(None) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(Some(a)) => match run(a) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn parses_pages_and_flags() {
        let a = parse(&s(&[
            "--model",
            "/m",
            "a.png",
            "--raw",
            "b.png",
            "--max-new-tokens",
            "9",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(a.model, PathBuf::from("/m"));
        assert_eq!(
            a.pages,
            vec![PathBuf::from("a.png"), PathBuf::from("b.png")]
        );
        assert!(a.raw);
        assert_eq!(a.max_new_tokens, Some(9));
    }

    #[test]
    fn refuses_missing_and_bad_arguments() {
        assert!(parse(&s(&["a.png"])).is_err(), "no model");
        assert!(parse(&s(&["--model", "/m"])).is_err(), "no pages");
        assert!(parse(&s(&["--model", "/m", "--max-new-tokens", "0", "a.png"])).is_err());
        assert!(parse(&s(&["--model", "/m", "--bogus", "a.png"])).is_err());
        assert!(parse(&s(&["--help"])).unwrap().is_none());
    }
}
