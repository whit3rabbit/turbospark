//! `turbospark-model auth`: Hugging Face token inspect, set, and clear.

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

    let token_to_set = positionals
        .first()
        .cloned()
        .or_else(|| options.hf_token.clone());

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
                println!("Use `turbospark-model auth <TOKEN>` to save one, or export HF_TOKEN.");
            }
        }
        Ok(())
    }
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
