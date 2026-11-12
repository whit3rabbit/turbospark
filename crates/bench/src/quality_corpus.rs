//! The distribution-probe corpus: the first 12 `docs/*.md` files, sorted,
//! joined with blank lines, tokenized without a chat template, and sliced at
//! offset `(i * 997) % (len - n)` per length so the prompts are not prefixes
//! of one another. The prefill arm probes built this inline; it lives here so
//! `tslab`'s quality gate prefills the same token ids.

use std::path::Path;

/// The concatenated corpus text, read from `<repo>/docs`.
pub fn corpus_text(repo_root: &Path) -> Result<String, String> {
    let docs = repo_root.join("docs");
    let mut files: Vec<_> = std::fs::read_dir(&docs)
        .map_err(|e| format!("{}: {e}", docs.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    files.sort();
    let mut text = String::new();
    for f in files.iter().take(12) {
        text.push_str(&std::fs::read_to_string(f).unwrap_or_default());
        text.push_str("\n\n");
    }
    Ok(text)
}

/// One slice per requested length, from already-tokenized corpus ids.
pub fn slices(ids: &[i32], lengths: &[usize]) -> Result<Vec<Vec<i32>>, String> {
    let longest = lengths.iter().copied().max().unwrap_or(0);
    if ids.len() <= longest {
        return Err(format!(
            "corpus has {} tokens, needs more than {longest}",
            ids.len()
        ));
    }
    Ok(lengths
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let start = (i * 997) % (ids.len() - n);
            ids[start..start + n].to_vec()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_follow_the_probe_offsets() {
        let ids: Vec<i32> = (0..5000).collect();
        let s = slices(&ids, &[64, 200]).unwrap();
        assert_eq!(s[0][0], 0);
        assert_eq!(s[1][0], 997);
        assert_eq!(s[1].len(), 200);
        assert!(slices(&ids, &[5000]).is_err());
    }
}
