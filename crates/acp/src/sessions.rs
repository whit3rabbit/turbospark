//! Session registry state and the on-disk session record.
//!
//! Records are written, never read back, in v1: `session/load` is not
//! advertised, so there is nothing to quarantine on a corrupt file (the
//! `AppJSONStore` quarantine rule applies the day a loader exists). They are
//! separate from the app's chat archives by design. A record holds the
//! conversation text, which can contain workspace source, so the directory is
//! 0700 and files are 0600.

use crate::engine::Message;
use serde::Serialize;
use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

/// In-memory and persisted history are both capped; the engine is
/// responsible for fitting what it is given into its context window.
pub const MAX_HISTORY_MESSAGES: usize = 400;

pub struct SessionState {
    pub history: Vec<Message>,
    /// Session-scoped decisions by permission name: true = always allow,
    /// false = always reject. Set only by the editor's Always options.
    pub grants: HashMap<String, bool>,
}

pub struct Session {
    pub id: String,
    pub root: PathBuf,
    pub created_unix: u64,
    pub cancel: AtomicBool,
    pub busy: AtomicBool,
    pub state: Mutex<SessionState>,
}

impl Session {
    pub fn new(id: String, root: PathBuf) -> Self {
        Session {
            id,
            root,
            created_unix: now_unix(),
            cancel: AtomicBool::new(false),
            busy: AtomicBool::new(false),
            state: Mutex::new(SessionState {
                history: Vec::new(),
                grants: HashMap::new(),
            }),
        }
    }
}

pub fn push_message(state: &mut SessionState, m: Message) {
    state.history.push(m);
    if state.history.len() > MAX_HISTORY_MESSAGES {
        let excess = state.history.len() - MAX_HISTORY_MESSAGES;
        state.history.drain(..excess);
    }
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 128 random-per-process bits as hex. Not a secret: it names a record and
/// scopes a stdio conversation. It must only be filename-safe, which hex is.
pub fn new_session_id() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut out = String::from("sess_");
    for salt in 0..2u8 {
        let mut h = RandomState::new().build_hasher();
        h.write_u64(n);
        h.write_u128(nanos);
        h.write_u8(salt);
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out
}

#[derive(Serialize)]
struct Record<'a> {
    version: u32,
    id: &'a str,
    cwd: &'a str,
    model: &'a str,
    created_unix: u64,
    updated_unix: u64,
    history: &'a [Message],
}

pub struct Store {
    dir: Option<PathBuf>,
}

impl Store {
    /// `None` keeps sessions in memory only.
    pub fn new(dir: Option<PathBuf>) -> Self {
        Store { dir }
    }

    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    pub fn save(&self, s: &Session, model: &str) -> io::Result<()> {
        let Some(dir) = &self.dir else { return Ok(()) };
        // The id is ours and hex, but the filename is built from it, so
        // refuse anything else rather than trust the invariant.
        if !s.id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "bad session id",
            ));
        }
        create_private_dir(dir)?;
        let history = s
            .state
            .lock()
            .map(|g| g.history.clone())
            .unwrap_or_default();
        let rec = Record {
            version: 1,
            id: &s.id,
            cwd: &s.root.to_string_lossy(),
            model,
            created_unix: s.created_unix,
            updated_unix: now_unix(),
            history: &history,
        };
        let bytes = serde_json::to_vec_pretty(&rec).map_err(io::Error::other)?;
        let tmp = dir.join(format!(".{}.tmp", s.id));
        let dst = dir.join(format!("{}.json", s.id));
        write_private(&tmp, &bytes)?;
        std::fs::rename(&tmp, &dst)
    }
}

fn create_private_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir)
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(dir)
    }
}

fn write_private(path: &Path, bytes: &[u8]) -> io::Result<()> {
    use std::io::Write;
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    let mut f = o.open(path)?;
    f.write_all(bytes)?;
    f.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Role;

    #[test]
    fn ids_are_unique_and_filename_safe() {
        let a = new_session_id();
        let b = new_session_id();
        assert_ne!(a, b);
        assert!(a.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_'));
    }

    #[test]
    fn history_is_capped() {
        let mut st = SessionState {
            history: vec![],
            grants: HashMap::new(),
        };
        for i in 0..(MAX_HISTORY_MESSAGES + 10) {
            push_message(
                &mut st,
                Message {
                    role: Role::User,
                    text: i.to_string(),
                },
            );
        }
        assert_eq!(st.history.len(), MAX_HISTORY_MESSAGES);
        assert_eq!(st.history[0].text, "10");
    }

    #[test]
    fn save_writes_private_record() {
        let dir = std::env::temp_dir().join(format!("tsacp-store-{}", std::process::id()));
        let store = Store::new(Some(dir.join("acp")));
        let s = Session::new("sess_abc123".into(), PathBuf::from("/tmp/x"));
        s.state.lock().unwrap().history.push(Message {
            role: Role::User,
            text: "hi".into(),
        });
        store.save(&s, "fake").unwrap();
        let p = dir.join("acp/sess_abc123.json");
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        assert_eq!(v["history"][0]["text"], "hi");
        assert_eq!(v["history"][0]["role"], "user");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
            let dmode = std::fs::metadata(dir.join("acp"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(dmode, 0o700);
        }
        // No temp file left behind.
        assert!(!dir.join("acp/.sess_abc123.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memory_only_store_writes_nothing() {
        let store = Store::new(None);
        let s = Session::new("sess_x".into(), PathBuf::from("/tmp"));
        store.save(&s, "m").unwrap();
    }

    #[test]
    fn bad_id_is_refused() {
        let dir = std::env::temp_dir().join(format!("tsacp-store-bad-{}", std::process::id()));
        let store = Store::new(Some(dir.clone()));
        let s = Session::new("../evil".into(), PathBuf::from("/tmp"));
        assert!(store.save(&s, "m").is_err());
        assert!(!dir.exists());
    }
}
