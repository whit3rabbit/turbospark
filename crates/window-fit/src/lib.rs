//! Pure, deterministic conversation-window fitting.
//!
//! Drops the oldest eligible turns from a conversation, oldest first, until
//! a caller-supplied measurement of the whole remaining conversation is
//! under a caller-supplied bound or nothing eligible remains. An optional
//! leading instruction turn and the newest turn are never removed. The
//! operation performs no input or output and holds no state between calls:
//! the same inputs always produce the same outcome.

pub mod fit;
pub mod outcome;

pub use fit::fit_conversation_window;
pub use outcome::WindowFitOutcome;
