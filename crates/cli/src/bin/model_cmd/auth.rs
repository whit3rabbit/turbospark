//! `turbospark-model auth`: Hugging Face token inspect, set, and clear.

use std::io::{self, IsTerminal, Read, Write};

use catalog::{resolve_hf_token_with_source, validate_hf_token, HfTokenValidationStatus, Store};

use crate::{Error, Options};

pub fn auth(store: &Store, positionals: &[String], options: &Options) -> Result<(), Error> {
    if options.clear {
        store.clear_hf_token().map_err(Error::Failed)?;
        println!(
            "Hugging Face token cleared from {}",
            store.hf_token_path().display()
        );
        return Ok(());
    }

    if !positionals.is_empty() {
        return Err(Error::Usage(
            "auth takes no arguments; use --set to read a token from stdin".to_string(),
        ));
    }

    let token_to_set = options.set.then(read_token).transpose()?;

    if let Some(token) = token_to_set {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return Err(Error::Usage("token cannot be empty".to_string()));
        }

        println!("Validating token with Hugging Face...");
        let status = validate_hf_token(trimmed);
        match &status {
            HfTokenValidationStatus::Valid { name, .. } => {
                store.set_hf_token(trimmed).map_err(Error::Failed)?;
                println!("Token saved to {}", store.hf_token_path().display());
                if let Some(user) = name {
                    println!("Authenticated as @{user}");
                }
                Ok(())
            }
            HfTokenValidationStatus::Invalid { message } => {
                let msg = message
                    .as_deref()
                    .unwrap_or("token was rejected by Hugging Face (401)");
                Err(Error::Failed(format!("Invalid token: {msg}")))
            }
            HfTokenValidationStatus::RateLimited {
                retry_after_seconds,
            } => {
                let wait = retry_after_seconds
                    .map(|s| format!(" (retry after {s}s)"))
                    .unwrap_or_default();
                Err(Error::Failed(format!(
                    "Hugging Face API rate limit reached{wait}"
                )))
            }
            HfTokenValidationStatus::Unavailable { message } => {
                store.set_hf_token(trimmed).map_err(Error::Failed)?;
                println!(
                    "Token saved to {} (warning: could not reach Hugging Face to verify: {message})",
                    store.hf_token_path().display()
                );
                Ok(())
            }
            HfTokenValidationStatus::Missing => Err(Error::Usage("no token provided".to_string())),
        }
    } else {
        let resolved = resolve_hf_token_with_source(None);
        match resolved {
            Some((token, source)) => {
                let masked = mask_token(&token);
                println!("Hugging Face token: {masked}");
                println!("Source:             {}", source.label());
                print!("Status:             ");
                let status = validate_hf_token(&token);
                match status {
                    HfTokenValidationStatus::Valid { name, fullname, .. } => {
                        let who = match (name, fullname) {
                            (Some(n), Some(f)) => format!("@{n} ({f})"),
                            (Some(n), None) => format!("@{n}"),
                            _ => "valid".to_string(),
                        };
                        println!("valid ({who})");
                    }
                    HfTokenValidationStatus::Invalid { message } => {
                        let why = message.as_deref().unwrap_or("invalid or expired");
                        println!("invalid ({why})");
                    }
                    HfTokenValidationStatus::RateLimited { .. } => {
                        println!("rate limited by Hugging Face");
                    }
                    HfTokenValidationStatus::Unavailable { message } => {
                        println!("offline / unable to verify ({message})");
                    }
                    HfTokenValidationStatus::Missing => {
                        println!("missing");
                    }
                }
            }
            None => {
                println!("No Hugging Face token found.");
                println!("Use `turbospark-model auth --set` to save one, or export HF_TOKEN.");
            }
        }
        Ok(())
    }
}

fn read_token() -> Result<String, Error> {
    let stdin = io::stdin();
    if stdin.is_terminal() {
        read_token_from_terminal(&stdin)
    } else {
        let mut token = String::new();
        stdin
            .lock()
            .read_to_string(&mut token)
            .map_err(|error| Error::Failed(format!("could not read token from stdin: {error}")))?;
        Ok(token)
    }
}

#[cfg(unix)]
fn read_token_from_terminal(stdin: &io::Stdin) -> Result<String, Error> {
    use std::os::fd::AsRawFd;

    let fd = stdin.as_raw_fd();
    let mut original = std::mem::MaybeUninit::<libc::termios>::uninit();
    // SAFETY: `original` is writable termios storage and `fd` is live stdin.
    if unsafe { libc::tcgetattr(fd, original.as_mut_ptr()) } != 0 {
        return Err(Error::Failed(format!(
            "could not configure the token prompt: {}",
            io::Error::last_os_error()
        )));
    }
    // SAFETY: tcgetattr initialized `original` after returning success.
    let original = unsafe { original.assume_init() };
    let mut hidden = original;
    hidden.c_lflag &= !libc::ECHO;
    // SAFETY: the call receives a valid stdin descriptor and termios value.
    if unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, &hidden) } != 0 {
        return Err(Error::Failed(format!(
            "could not hide token input: {}",
            io::Error::last_os_error()
        )));
    }
    let echo = TerminalEcho { fd, original };

    eprint!("Hugging Face token: ");
    io::stderr()
        .flush()
        .map_err(|error| Error::Failed(format!("could not display the token prompt: {error}")))?;
    let mut token = String::new();
    let read_result = stdin.read_line(&mut token);
    let restore_result = echo.restore();
    eprintln!();
    if let Err(error) = restore_result {
        return Err(Error::Failed(format!(
            "could not restore terminal echo: {error}"
        )));
    }
    read_result
        .map_err(|error| Error::Failed(format!("could not read token from stdin: {error}")))?;
    Ok(token)
}

#[cfg(unix)]
struct TerminalEcho {
    fd: std::os::fd::RawFd,
    original: libc::termios,
}

#[cfg(unix)]
impl TerminalEcho {
    fn restore(self) -> io::Result<()> {
        // SAFETY: `original` came from tcgetattr for this same live descriptor.
        let result = unsafe { libc::tcsetattr(self.fd, libc::TCSAFLUSH, &self.original) };
        std::mem::forget(self);
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

#[cfg(unix)]
impl Drop for TerminalEcho {
    fn drop(&mut self) {
        // SAFETY: `original` came from tcgetattr for this same live descriptor.
        unsafe {
            libc::tcsetattr(self.fd, libc::TCSAFLUSH, &self.original);
        }
    }
}

#[cfg(not(unix))]
fn read_token_from_terminal(_stdin: &io::Stdin) -> Result<String, Error> {
    Err(Error::Usage(
        "interactive token entry is unavailable; pipe the token to `auth --set`".to_string(),
    ))
}

fn mask_token(token: &str) -> String {
    let chars: Vec<char> = token.chars().collect();
    if chars.len() <= 8 {
        "***".to_string()
    } else {
        let prefix: String = chars[..4].iter().collect();
        let suffix: String = chars[chars.len() - 4..].iter().collect();
        format!("{prefix}...{suffix}")
    }
}
