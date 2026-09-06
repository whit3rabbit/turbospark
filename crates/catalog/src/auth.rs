//! Hugging Face authentication token resolution, storage, and validation.

use crate::store::Store;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Validation verdict from Hugging Face whoami-v2.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum HfTokenValidationStatus {
    Missing,
    Valid {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        fullname: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        email: Option<String>,
    },
    Invalid {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
    RateLimited {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        retry_after_seconds: Option<u64>,
    },
    Unavailable {
        message: String,
    },
}

impl HfTokenValidationStatus {
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid { .. })
    }
}

/// The origin of a resolved Hugging Face token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfTokenSource {
    Explicit,
    Environment(&'static str),
    Store,
    Cache(&'static str),
}

impl HfTokenSource {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Explicit => "cli flag / argument",
            Self::Environment(v) => v,
            Self::Store => "store file (~/.turbospark/hf_token)",
            Self::Cache(p) => p,
        }
    }
}

/// Read candidate paths for standard huggingface-cli tokens.
fn cache_token_paths() -> Vec<(&'static str, PathBuf)> {
    let mut paths = Vec::new();
    if let Some(hf_home) = std::env::var_os("HF_HOME") {
        paths.push(("$HF_HOME/token", PathBuf::from(hf_home).join("token")));
    } else if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        paths.push((
            "~/.cache/huggingface/token",
            home.join(".cache").join("huggingface").join("token"),
        ));
        paths.push((
            "~/.huggingface/token",
            home.join(".huggingface").join("token"),
        ));
    }
    paths
}

/// Resolve a Hugging Face token in priority order, returning the token and its source:
/// 1. Explicit token override (if non-empty)
/// 2. `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` environment variables
/// 3. Store `hf_token` file (`~/.turbospark/hf_token`)
/// 4. Standard Hugging Face cache files (`~/.cache/huggingface/token` or `~/.huggingface/token`)
pub fn resolve_hf_token_with_source(explicit: Option<&str>) -> Option<(String, HfTokenSource)> {
    if let Some(token) = explicit {
        let trimmed = token.trim();
        if !trimmed.is_empty() {
            return Some((trimmed.to_string(), HfTokenSource::Explicit));
        }
    }

    for env_var in &["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(val) = std::env::var(env_var) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return Some((trimmed.to_string(), HfTokenSource::Environment(env_var)));
            }
        }
    }

    if let Ok(store) = Store::default_store() {
        if let Some(token) = store.get_hf_token() {
            return Some((token, HfTokenSource::Store));
        }
    }

    for (label, path) in cache_token_paths() {
        if let Ok(content) = std::fs::read_to_string(&path) {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                return Some((trimmed.to_string(), HfTokenSource::Cache(label)));
            }
        }
    }

    None
}

/// Resolve a Hugging Face token, returning only the token string.
pub fn resolve_hf_token(explicit: Option<&str>) -> Option<String> {
    resolve_hf_token_with_source(explicit).map(|(token, _)| token)
}

/// Validate a Hugging Face token against the whoami-v2 endpoint:
/// `GET https://huggingface.co/api/whoami-v2` with `Authorization: Bearer <token>`.
pub fn validate_hf_token(token: &str) -> HfTokenValidationStatus {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return HfTokenValidationStatus::Missing;
    }

    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            return HfTokenValidationStatus::Unavailable {
                message: e.to_string(),
            }
        }
    };

    let whoami_url = format!("{}/api/whoami-v2", crate::hf::hf_endpoint());
    let response = match client
        .get(&whoami_url)
        .bearer_auth(trimmed)
        .header(reqwest::header::USER_AGENT, "turbospark")
        .send()
    {
        Ok(r) => r,
        Err(e) => {
            return HfTokenValidationStatus::Unavailable {
                message: e.to_string(),
            }
        }
    };

    let status = response.status().as_u16();
    match status {
        200 => {
            #[derive(Deserialize)]
            struct WhoamiResponse {
                name: Option<String>,
                fullname: Option<String>,
                email: Option<String>,
            }
            let bytes = match response.bytes() {
                Ok(b) => b,
                Err(e) => {
                    return HfTokenValidationStatus::Unavailable {
                        message: format!("reading whoami response bytes: {e}"),
                    }
                }
            };
            match serde_json::from_slice::<WhoamiResponse>(&bytes) {
                Ok(info) => HfTokenValidationStatus::Valid {
                    name: info.name,
                    fullname: info.fullname,
                    email: info.email,
                },
                Err(e) => HfTokenValidationStatus::Unavailable {
                    message: format!("decoding whoami response: {e}"),
                },
            }
        }
        401 | 403 => HfTokenValidationStatus::Invalid {
            message: Some("invalid or expired token".to_string()),
        },
        429 => {
            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.trim().parse::<u64>().ok());
            HfTokenValidationStatus::RateLimited {
                retry_after_seconds: retry_after,
            }
        }
        other => HfTokenValidationStatus::Unavailable {
            message: format!("HTTP {other}"),
        },
    }
}
