//! Hand-rolled argument parsing, as `crates/bench/src/main.rs` does: the
//! workspace carries no CLI parser and this binary does not need one.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::knobs;
use crate::workload::{Arm, OpenParams, PromptSource, Residency, Slots};

pub struct Args {
    rest: Vec<String>,
}

impl Args {
    pub fn new(rest: Vec<String>) -> Self {
        Self { rest }
    }

    /// The value after `flag`, removed from the list. Last occurrence wins.
    pub fn take(&mut self, flag: &str) -> Result<Option<String>, String> {
        let mut found = None;
        while let Some(i) = self.rest.iter().position(|a| a == flag) {
            if i + 1 >= self.rest.len() {
                return Err(format!("{flag} needs a value"));
            }
            found = Some(self.rest.remove(i + 1));
            self.rest.remove(i);
        }
        Ok(found)
    }

    /// Every value after every `flag`, in order.
    pub fn take_all(&mut self, flag: &str) -> Result<Vec<String>, String> {
        let mut out = Vec::new();
        while let Some(i) = self.rest.iter().position(|a| a == flag) {
            if i + 1 >= self.rest.len() {
                return Err(format!("{flag} needs a value"));
            }
            out.push(self.rest.remove(i + 1));
            self.rest.remove(i);
        }
        Ok(out)
    }

    pub fn flag(&mut self, flag: &str) -> bool {
        let before = self.rest.len();
        self.rest.retain(|a| a != flag);
        self.rest.len() != before
    }

    pub fn required(&mut self, flag: &str) -> Result<String, String> {
        self.take(flag)?
            .ok_or_else(|| format!("{flag} is required"))
    }

    /// Refuse anything left over: a mistyped flag must not be ignored.
    pub fn finish(self) -> Result<(), String> {
        if self.rest.is_empty() {
            Ok(())
        } else {
            Err(format!("unrecognized arguments: {}", self.rest.join(" ")))
        }
    }

    pub fn parsed<T: std::str::FromStr>(&mut self, flag: &str, default: T) -> Result<T, String> {
        match self.take(flag)? {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("{flag}: cannot parse {v:?}")),
        }
    }
}

/// `protocol:ID`, `messages:PATH` or `text:PATH`.
pub fn parse_prompt(s: &str) -> Result<PromptSource, String> {
    match s.split_once(':') {
        Some(("protocol", id)) => Ok(PromptSource::Protocol(id.to_string())),
        Some(("messages", p)) => Ok(PromptSource::MessagesFile(PathBuf::from(p))),
        Some(("text", p)) => Ok(PromptSource::TextFile(PathBuf::from(p))),
        _ => Err(format!(
            "prompt {s:?}: use protocol:<case>, messages:<path> or text:<path>"
        )),
    }
}

pub fn parse_slots(s: &str) -> Result<Slots, String> {
    if s == "auto" {
        return Ok(Slots::Auto);
    }
    let n: u32 = s.parse().map_err(|_| format!("slots {s:?}"))?;
    if !crate::knobs::ALLOWED_SLOTS.contains(&n) {
        return Err(format!(
            "slots {n}: allowed {:?}",
            crate::knobs::ALLOWED_SLOTS
        ));
    }
    Ok(Slots::Fixed(n as usize))
}

pub fn parse_residency(s: &str) -> Result<Residency, String> {
    match s {
        "auto" => Ok(Residency::Auto),
        "streamed" => Ok(Residency::Streamed),
        "mapped" => Ok(Residency::Mapped),
        _ => Err(format!("residency {s:?}: auto, streamed or mapped")),
    }
}

/// `KEY=VALUE`, where KEY is a full `TURBOSPARK_*` name or a knob's short
/// name from the registry.
pub fn parse_env(s: &str) -> Result<(String, String), String> {
    let (k, v) = s
        .split_once('=')
        .ok_or_else(|| format!("{s:?}: expected KEY=VALUE"))?;
    let key = if k.starts_with("TURBOSPARK_") {
        k.to_string()
    } else if let Some(knob) = knobs::by_name(k) {
        knob.env.to_string()
    } else {
        return Err(format!(
            "{k:?} is neither a TURBOSPARK_* variable nor a known knob name"
        ));
    };
    Ok((key, v.to_string()))
}

/// `NAME` or `NAME:KEY=VALUE,KEY=VALUE`, plus open parameters shared by the
/// command and the common env every arm gets.
pub fn parse_arm(
    s: &str,
    open: OpenParams,
    common: &BTreeMap<String, String>,
) -> Result<Arm, String> {
    let (name, rest) = s.split_once(':').unwrap_or((s, ""));
    crate::paths::validate_name(name)?;
    let mut env = common.clone();
    for kv in rest.split(',').filter(|kv| !kv.is_empty()) {
        let (k, v) = parse_env(kv)?;
        env.insert(k, v);
    }
    Ok(Arm {
        name: name.to_string(),
        env,
        open,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_and_finish() {
        let mut a = Args::new(
            ["--model", "m", "--env", "A=1", "--env", "B=2", "--x"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        assert_eq!(a.take("--model").unwrap().as_deref(), Some("m"));
        assert_eq!(a.take_all("--env").unwrap(), vec!["A=1", "B=2"]);
        assert!(a.finish().is_err(), "leftover --x must be refused");
    }

    #[test]
    fn arms_accept_short_knob_names() {
        let arm = parse_arm(
            "off:prefill_steel=0",
            OpenParams::default(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(arm.env["TURBOSPARK_PREFILL_STEEL"], "0");
        assert!(parse_arm("x:PATH=/bin", OpenParams::default(), &BTreeMap::new()).is_err());
        assert!(parse_arm("bad name", OpenParams::default(), &BTreeMap::new()).is_err());
    }

    #[test]
    fn slots_and_prompts() {
        assert_eq!(parse_slots("auto").unwrap(), Slots::Auto);
        assert_eq!(parse_slots("32").unwrap(), Slots::Fixed(32));
        assert!(parse_slots("33").is_err());
        assert_eq!(
            parse_prompt("protocol:long-synthesis").unwrap(),
            PromptSource::Protocol("long-synthesis".into())
        );
        assert!(parse_prompt("long-synthesis").is_err());
    }
}
