//! Filename-derived GGUF quantization labels and logical file-set grouping.
//!
//! This module reads names only. A recognized [`QuantLabel`] says what a
//! filename advertises, not whether its block types are executable here or
//! whether a selected set has passed the probe and install gates.

use crate::hf::RepoFile;
use std::collections::BTreeMap;
use std::fmt;

/// Canonical uppercase quantization token parsed from a GGUF filename.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QuantLabel(pub String);

impl fmt::Display for QuantLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Validated GGUF sibling files belonging to one filename-derived variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantFiles {
    pub label: QuantLabel,
    pub files: Vec<RepoFile>,
    pub shard_set: ShardSetStatus,
}

/// Shard topology only. This does not compute size, fit, or installability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardSetStatus {
    SingleFile,
    Complete {
        expected_count: usize,
    },
    Incomplete {
        expected_count: usize,
        present_indices: Vec<usize>,
    },
    Inconsistent {
        issues: Vec<ShardSetIssue>,
    },
}

/// Filename-level reasons a shard set cannot be treated as one consistent set.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShardSetIssue {
    DuplicateIndex { index: usize },
    ConflictingCounts { declared_counts: Vec<usize> },
    UnexpectedIndex { index: usize, declared_count: usize },
    MixedVariantKey,
}

/// Parses one supported quantization token from a GGUF file stem.
///
/// The result is canonical uppercase and independent of the model family.
/// A token must be separated from neighboring filename text by the usual
/// hyphen, underscore, or dot boundaries. If a stem names multiple distinct
/// quantizations, it is ambiguous and returns `None`.
pub fn quant_label(file_stem: &str) -> Option<QuantLabel> {
    find_label_span(strip_gguf_extension(file_stem)).map(|(label, _, _)| label)
}

/// Groups recognized `.gguf` siblings into deterministic logical variants.
///
/// Shard suffixes such as `-00001-of-00003` are excluded from the logical key
/// and checked as a set. Unrecognized GGUF names and non-GGUF siblings are
/// omitted. The result carries no size, support, or fit conclusions.
pub fn group_variants(files: &[RepoFile]) -> Vec<VariantFiles> {
    let mut groups = BTreeMap::<GroupKey, Vec<ParsedFile>>::new();
    for file in files {
        let Some(parsed) = parse_file(file) else {
            continue;
        };
        groups
            .entry(GroupKey {
                label: parsed.label.clone(),
                base_key: parsed.base_key.clone(),
            })
            .or_default()
            .push(parsed);
    }

    let mut variants = Vec::<(String, VariantFiles)>::with_capacity(groups.len());
    for (key, mut group) in groups {
        group.sort_by(|left, right| left.file.name.cmp(&right.file.name));
        let shard_set = shard_status(&group);
        variants.push((
            key.base_key,
            VariantFiles {
                label: key.label,
                files: group.into_iter().map(|parsed| parsed.file).collect(),
                shard_set,
            },
        ));
    }

    mark_mixed_variant_shards(&mut variants);
    variants.sort_by(|left, right| {
        left.1
            .label
            .cmp(&right.1.label)
            .then_with(|| left.1.files[0].name.cmp(&right.1.files[0].name))
    });
    variants.into_iter().map(|(_, variant)| variant).collect()
}

const QUANT_TOKENS: &[&str] = &[
    "Q8_0_8_8", "Q8_0_4_8", "Q8_0_4_4", "Q4_0_8_8", "Q4_0_4_8", "Q4_0_4_4", "Q8_K_XL", "Q6_K_XL",
    "Q5_K_XL", "Q4_K_XL", "Q3_K_XL", "Q2_K_XL", "IQ3_XXS", "IQ2_XXS", "IQ2_XS", "IQ4_XS", "IQ4_NL",
    "IQ3_XS", "IQ3_M", "IQ3_S", "IQ2_M", "IQ2_S", "IQ1_M", "IQ1_S", "Q5_K_M", "Q5_K_S", "Q4_K_M",
    "Q4_K_S", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K_M", "Q2_K_S", "TQ2_0", "TQ1_0", "MXFP4", "BF16",
    "Q8_K", "Q8_1", "Q8_0", "Q6_K", "Q5_K", "Q5_1", "Q5_0", "Q4_K", "Q4_1", "Q4_0", "Q3_K", "Q2_K",
    "Q2_0", "Q1_0", "F32", "F16",
];

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey {
    label: QuantLabel,
    base_key: String,
}

#[derive(Debug, Clone)]
struct ParsedFile {
    file: RepoFile,
    label: QuantLabel,
    base_key: String,
    shard: Option<ShardPart>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ShardPart {
    index: usize,
    count: usize,
}

fn parse_file(file: &RepoFile) -> Option<ParsedFile> {
    let (directory, basename) = file
        .name
        .rsplit_once('/')
        .unwrap_or(("", file.name.as_str()));
    let stem = gguf_file_stem(basename)?;
    let (variant_stem, shard) = strip_shard_suffix(stem).ok()?;
    let (label, token_start, token_end) = find_label_span(variant_stem)?;
    let base_key = format!(
        "{}\0{}\0{}",
        directory.to_ascii_lowercase(),
        variant_stem[..token_start].to_ascii_lowercase(),
        variant_stem[token_end..].to_ascii_lowercase()
    );
    Some(ParsedFile {
        file: file.clone(),
        label: label.clone(),
        base_key,
        shard,
    })
}

fn strip_gguf_extension(file_name: &str) -> &str {
    if file_name
        .get(file_name.len().saturating_sub(5)..)
        .is_some_and(|extension| extension.eq_ignore_ascii_case(".gguf"))
    {
        &file_name[..file_name.len() - 5]
    } else {
        file_name
    }
}

fn gguf_file_stem(file_name: &str) -> Option<&str> {
    let extension_start = file_name.len().checked_sub(5)?;
    let extension = file_name.get(extension_start..)?;
    extension
        .eq_ignore_ascii_case(".gguf")
        .then(|| &file_name[..extension_start])
}

fn find_label_span(stem: &str) -> Option<(QuantLabel, usize, usize)> {
    let uppercase = stem.to_ascii_uppercase();
    let mut matches = Vec::<(usize, usize, &str)>::new();
    for token in QUANT_TOKENS {
        for (start, _) in uppercase.match_indices(token) {
            let end = start + token.len();
            if token_boundaries(&uppercase, start, end) {
                matches.push((start, end, token));
            }
        }
    }
    matches.sort_by(|left, right| {
        (right.1 - right.0)
            .cmp(&(left.1 - left.0))
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.2.cmp(right.2))
    });
    let &(start, end, token) = matches.first()?;
    if matches
        .iter()
        .skip(1)
        .any(|(other_start, other_end, _)| *other_start >= end || *other_end <= start)
    {
        return None;
    }
    Some((QuantLabel(token.to_string()), start, end))
}

fn token_boundaries(stem: &str, start: usize, end: usize) -> bool {
    let bytes = stem.as_bytes();
    (start == 0 || filename_separator(bytes[start - 1]))
        && (end == bytes.len() || filename_separator(bytes[end]))
}

fn filename_separator(byte: u8) -> bool {
    matches!(byte, b'-' | b'_' | b'.')
}

fn strip_shard_suffix(stem: &str) -> Result<(&str, Option<ShardPart>), ()> {
    let uppercase = stem.to_ascii_uppercase();
    let Some(of_position) = uppercase.rfind("-OF-") else {
        return Ok((stem, None));
    };
    let count_text = &stem[of_position + 4..];
    let count_is_numeric =
        !count_text.is_empty() && count_text.bytes().all(|byte| byte.is_ascii_digit());
    let before_of = &stem[..of_position];
    let mut index_start = before_of.len();
    while index_start > 0 && before_of.as_bytes()[index_start - 1].is_ascii_digit() {
        index_start -= 1;
    }
    let has_numeric_index = index_start < before_of.len();
    if !has_numeric_index && !count_is_numeric {
        return Ok((stem, None));
    }
    if !count_is_numeric || !has_numeric_index || index_start == 0 {
        return Err(());
    }
    let separator_position = index_start - 1;
    if !filename_separator(before_of.as_bytes()[separator_position]) {
        return Err(());
    }
    let index = before_of[index_start..].parse::<usize>().map_err(|_| ())?;
    let count = count_text.parse::<usize>().map_err(|_| ())?;
    Ok((
        &before_of[..separator_position],
        Some(ShardPart { index, count }),
    ))
}

fn shard_status(files: &[ParsedFile]) -> ShardSetStatus {
    let shards: Vec<ShardPart> = files.iter().filter_map(|file| file.shard).collect();
    if shards.is_empty() {
        return if files.len() == 1 {
            ShardSetStatus::SingleFile
        } else {
            ShardSetStatus::Inconsistent {
                issues: vec![ShardSetIssue::MixedVariantKey],
            }
        };
    }
    if shards.len() != files.len() {
        return ShardSetStatus::Inconsistent {
            issues: vec![ShardSetIssue::MixedVariantKey],
        };
    }

    let mut declared_counts: Vec<usize> = shards.iter().map(|shard| shard.count).collect();
    declared_counts.sort_unstable();
    declared_counts.dedup();
    let expected_count = declared_counts.first().copied().unwrap_or(0);

    let mut present_indices: Vec<usize> = shards.iter().map(|shard| shard.index).collect();
    present_indices.sort_unstable();
    let mut issues = Vec::new();
    for pair in present_indices.windows(2) {
        if pair[0] == pair[1] {
            issues.push(ShardSetIssue::DuplicateIndex { index: pair[0] });
        }
    }
    present_indices.dedup();

    if declared_counts.len() > 1 {
        issues.push(ShardSetIssue::ConflictingCounts { declared_counts });
    }
    for shard in &shards {
        if shard.index == 0 || shard.index > shard.count {
            issues.push(ShardSetIssue::UnexpectedIndex {
                index: shard.index,
                declared_count: shard.count,
            });
        }
    }
    issues.sort_unstable();
    issues.dedup();
    if !issues.is_empty() {
        return ShardSetStatus::Inconsistent { issues };
    }

    let complete = expected_count == present_indices.len()
        && present_indices
            .iter()
            .enumerate()
            .all(|(offset, index)| *index == offset + 1);
    if complete {
        ShardSetStatus::Complete { expected_count }
    } else {
        ShardSetStatus::Incomplete {
            expected_count,
            present_indices,
        }
    }
}

fn mark_mixed_variant_shards(variants: &mut [(String, VariantFiles)]) {
    let mut by_base = BTreeMap::<String, Vec<usize>>::new();
    for (index, (base_key, variant)) in variants.iter().enumerate() {
        if matches!(variant.shard_set, ShardSetStatus::Incomplete { .. }) {
            by_base.entry(base_key.clone()).or_default().push(index);
        }
    }

    for indices in by_base.values() {
        let mut by_count = BTreeMap::<usize, Vec<usize>>::new();
        for index in indices {
            if let ShardSetStatus::Incomplete {
                expected_count,
                present_indices: _,
            } = &variants[*index].1.shard_set
            {
                by_count.entry(*expected_count).or_default().push(*index);
            }
        }
        for (expected_count, group_indices) in by_count {
            if group_indices.len() < 2 {
                continue;
            }
            let mut combined = Vec::new();
            let mut each_incomplete = true;
            for index in &group_indices {
                let ShardSetStatus::Incomplete {
                    present_indices, ..
                } = &variants[*index].1.shard_set
                else {
                    each_incomplete = false;
                    break;
                };
                each_incomplete &= present_indices.len() < expected_count;
                combined.extend(present_indices.iter().copied());
            }
            combined.sort_unstable();
            combined.dedup();
            let combined_complete = combined.len() == expected_count
                && combined
                    .iter()
                    .enumerate()
                    .all(|(offset, index)| *index == offset + 1);
            if each_incomplete && combined_complete {
                for index in group_indices {
                    variants[index].1.shard_set = ShardSetStatus::Inconsistent {
                        issues: vec![ShardSetIssue::MixedVariantKey],
                    };
                }
            }
        }
    }
}
