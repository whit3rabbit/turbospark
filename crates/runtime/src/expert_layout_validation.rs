/// Whether a mapped layout covers exactly the expert range emitted by its router.
pub(crate) fn mapped_expert_count_matches(
    architecture_experts: i64,
    layout_experts: usize,
) -> bool {
    usize::try_from(architecture_experts).ok() == Some(layout_experts)
}

#[cfg(test)]
mod tests {
    use super::mapped_expert_count_matches;

    #[test]
    fn mapped_layout_must_cover_the_routers_expert_range() {
        assert!(mapped_expert_count_matches(8, 8));
        assert!(!mapped_expert_count_matches(8, 1));
        assert!(!mapped_expert_count_matches(-1, usize::MAX));
    }
}
