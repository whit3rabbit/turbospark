//! Smoke test: the compute skeleton constructs and the core primitives flow
//! across the dependency edge.
use compute::{ComputeStrategy, TokenId};

#[test]
fn strategy_constructs() {
    let a = ComputeStrategy::new();
    let b = ComputeStrategy::default();
    assert_eq!(a, b);
}

#[test]
fn token_id_width_is_four_bytes() {
    let id: TokenId = 0;
    let _ = id;
    assert_eq!(std::mem::size_of::<TokenId>(), 4);
}
