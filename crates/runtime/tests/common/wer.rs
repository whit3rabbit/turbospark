//! Word error rate, shared by the Parakeet real-model gates and the Drama 3
//! WER example. Included with `#[path = ".../common/wer.rs"] mod wer;` so a
//! test target, an example and a second test file read one copy.

#![allow(dead_code)]

/// Lowercased words with everything but letters, digits and apostrophes
/// stripped, empty words dropped.
pub fn normalize(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Word error rate of `hyp` against `reference` (Levenshtein over words,
/// case and punctuation ignored). An empty reference divides by one, so a
/// transcript of nothing against nothing is 0 and any word against nothing
/// counts whole.
pub fn wer(reference: &str, hyp: &str) -> f64 {
    let (r, h) = (normalize(reference), normalize(hyp));
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rw) in r.iter().enumerate() {
        let mut cur = vec![i + 1; h.len() + 1];
        for (j, hw) in h.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(rw != hw))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[h.len()] as f64 / r.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_text_modulo_case_and_punctuation_is_zero() {
        assert_eq!(wer("The quick brown fox.", "the QUICK, brown fox"), 0.0);
        assert_eq!(wer("", ""), 0.0);
    }

    #[test]
    fn each_edit_kind_costs_one_word() {
        // One substitution, one deletion and one insertion over four words.
        assert_eq!(wer("a b c d", "a x c d"), 0.25);
        assert_eq!(wer("a b c d", "a c d"), 0.25);
        assert_eq!(wer("a b c d", "a b x c d"), 0.25);
    }

    #[test]
    fn a_missing_transcript_is_total_and_an_empty_reference_does_not_divide_by_zero() {
        assert_eq!(wer("one two three", ""), 1.0);
        assert_eq!(wer("", "extra words"), 2.0);
    }

    #[test]
    fn apostrophes_stay_inside_words() {
        assert_eq!(normalize("It's -- don't!"), ["it's", "don't"]);
    }
}
