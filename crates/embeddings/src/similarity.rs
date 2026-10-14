//! Similarity and retrieval scoring.
//!
//! Single-vector scores (`dot`, `cosine_similarity`, the row-major
//! `similarity_matrix`) plus the ColVision late-interaction scorers from
//! mlx-embeddings `colvision_processor.py`: `score_single_vector` is a
//! dot-product matrix and `score_multi_vector` is MaxSim (for each query
//! token, the best-matching passage token, summed over query tokens).
//! Sums accumulate in f64.

use crate::error::{input_error, EmbeddingError};
use crate::normalize::l2_norm;

/// Dot product. Panics on a length mismatch, which is a caller bug.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot length mismatch");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum::<f64>() as f32
}

/// Cosine similarity with `torch.cosine_similarity`'s `eps = 1e-8` floor
/// on each norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    const EPS: f32 = 1e-8;
    dot(a, b) / (l2_norm(a).max(EPS) * l2_norm(b).max(EPS))
}

fn rows(x: &[f32], dim: usize, what: &str) -> Result<usize, EmbeddingError> {
    if dim == 0 || x.len() % dim != 0 {
        return Err(input_error(format!(
            "{what}: length {} is not a multiple of dim {dim}",
            x.len()
        )));
    }
    Ok(x.len() / dim)
}

/// `[n, m]` matrix of `score(a_i, b_j)` over row-major `a [n, dim]`,
/// `b [m, dim]`.
fn matrix(
    a: &[f32],
    b: &[f32],
    dim: usize,
    score: impl Fn(&[f32], &[f32]) -> f32,
) -> Result<Vec<f32>, EmbeddingError> {
    let n = rows(a, dim, "lhs")?;
    let m = rows(b, dim, "rhs")?;
    let mut out = Vec::with_capacity(n * m);
    for i in 0..n {
        for j in 0..m {
            out.push(score(
                &a[i * dim..(i + 1) * dim],
                &b[j * dim..(j + 1) * dim],
            ));
        }
    }
    Ok(out)
}

/// Cosine similarity matrix `[n, m]`.
pub fn similarity_matrix(a: &[f32], b: &[f32], dim: usize) -> Result<Vec<f32>, EmbeddingError> {
    matrix(a, b, dim, cosine_similarity)
}

/// ColVision single-vector score: `qs [n, dim] @ ps [m, dim].T`.
pub fn score_single_vector(qs: &[f32], ps: &[f32], dim: usize) -> Result<Vec<f32>, EmbeddingError> {
    matrix(qs, ps, dim, dot)
}

/// ColVision MaxSim. Each query and passage is a row-major `[tokens, dim]`
/// multi-vector; the result is `[queries, passages]` with
/// `sum_q max_p dot(q, p)`.
///
/// Each pair is scored unpadded. The Python reference zero-pads passages
/// to the batch maximum, so a pad row (dot 0) can win the max when every
/// real token scores negative, which makes a score depend on its batch
/// neighbors. This follows mlxrs, which masks pads out instead. A
/// zero-token query or passage, or an empty list, is an error because a
/// max over no tokens has no defined value.
pub fn score_multi_vector(
    queries: &[Vec<f32>],
    passages: &[Vec<f32>],
    dim: usize,
) -> Result<Vec<f32>, EmbeddingError> {
    if queries.is_empty() {
        return Err(input_error("No queries provided"));
    }
    if passages.is_empty() {
        return Err(input_error("No passages provided"));
    }
    for (what, set) in [("queries", queries), ("passages", passages)] {
        for (i, x) in set.iter().enumerate() {
            if rows(x, dim, what)? == 0 {
                return Err(input_error(format!("{what}[{i}] has zero tokens")));
            }
        }
    }
    let mut out = Vec::with_capacity(queries.len() * passages.len());
    for q in queries {
        let nq = q.len() / dim;
        for p in passages {
            let np = p.len() / dim;
            let mut total = 0.0f64;
            for i in 0..nq {
                let qi = &q[i * dim..(i + 1) * dim];
                let best = (0..np)
                    .map(|j| dot(qi, &p[j * dim..(j + 1) * dim]))
                    .fold(f32::NEG_INFINITY, f32::max);
                total += f64::from(best);
            }
            out.push(total as f32);
        }
    }
    Ok(out)
}

/// Indices of `scores` sorted by descending score, ties keeping input
/// order. NaN sorts last.
pub fn rank_descending(scores: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (scores[a], scores[b]);
        match (x.is_nan(), y.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => y.partial_cmp(&x).unwrap(),
        }
    });
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_parallel_and_orthogonal() {
        assert!((cosine_similarity(&[1.0, 2.0], &[2.0, 4.0]) - 1.0).abs() < 1e-6);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 3.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn matrix_is_row_major() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 0.0, 0.0, 2.0, 1.0, 1.0];
        let m = score_single_vector(&a, &b, 2).unwrap();
        assert_eq!(m, vec![1.0, 0.0, 1.0, 0.0, 2.0, 1.0]);
    }

    #[test]
    fn maxsim_sums_best_matches() {
        // q tokens (1,0), (0,1); passage tokens (2,0), (0,3), (1,1)
        let q = vec![vec![1.0, 0.0, 0.0, 1.0]];
        let p = vec![vec![2.0, 0.0, 0.0, 3.0, 1.0, 1.0], vec![-1.0, -1.0]];
        let s = score_multi_vector(&q, &p, 2).unwrap();
        // The second passage scores -2: no zero pad row may win the max.
        assert_eq!(s, vec![5.0, -2.0]);
    }

    #[test]
    fn maxsim_rejects_empty_inputs() {
        let q = vec![vec![1.0, 0.0]];
        assert!(score_multi_vector(&q, &[vec![]], 2).is_err());
        assert!(score_multi_vector(&[vec![]], &q, 2).is_err());
        assert!(score_multi_vector(&[], &q, 2).is_err());
        assert!(score_multi_vector(&q, &[], 2).is_err());
    }

    #[test]
    fn rank_is_stable_and_nan_last() {
        assert_eq!(
            rank_descending(&[0.1, f32::NAN, 0.5, 0.5]),
            vec![2, 3, 0, 1]
        );
    }

    #[test]
    fn ragged_input_is_an_error() {
        assert!(similarity_matrix(&[1.0, 2.0, 3.0], &[1.0, 2.0], 2).is_err());
        assert!(score_multi_vector(&[vec![1.0]], &[vec![1.0, 2.0]], 2).is_err());
    }
}
