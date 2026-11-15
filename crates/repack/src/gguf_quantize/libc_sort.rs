//! A port of Apple's libc `qsort` (Libc `stdlib/FreeBSD/qsort.c`: median-of-3
//! or 9 quicksort, an insertion-sort fast path for small or already-sorted
//! partitions, three-way partitioning of equal keys).
//!
//! Why this exists: ggml sorts each IQ1 block with libc `qsort` on `(value,
//! index)` pairs, comparing only the value. `qsort` is not stable, so the
//! order of exactly equal values depends on the platform's implementation, and
//! that order changes the block's per-half sums. Real weights converted from
//! BF16 (8 mantissa bits) have many exact ties, so this matters: a build of
//! ggml on macOS and one on Linux can produce different IQ1 bytes.
//!
//! The heapsort fallback for degenerate recursion depth is not ported (it
//! cannot trigger for the 16 and 32 element arrays IQ1 sorts); that case falls
//! back to an index-stable sort.

type Item = (f32, u32);

fn cmp(a: &Item, b: &Item) -> i32 {
    if a.0 < b.0 {
        -1
    } else if a.0 > b.0 {
        1
    } else {
        0
    }
}

fn isort(v: &mut [Item], swap_limit: i32) -> bool {
    let mut swap_cnt = 0;
    for pm in 1..v.len() {
        let mut pl = pm;
        while pl > 0 && cmp(&v[pl - 1], &v[pl]) > 0 {
            v.swap(pl, pl - 1);
            if swap_limit != 0 {
                swap_cnt += 1;
                if swap_cnt > swap_limit {
                    return false;
                }
            }
            pl -= 1;
        }
    }
    true
}

fn med3(v: &[Item], a: usize, b: usize, c: usize) -> usize {
    if cmp(&v[a], &v[b]) < 0 {
        if cmp(&v[b], &v[c]) < 0 {
            b
        } else if cmp(&v[a], &v[c]) < 0 {
            c
        } else {
            a
        }
    } else if cmp(&v[b], &v[c]) > 0 {
        b
    } else if cmp(&v[a], &v[c]) < 0 {
        a
    } else {
        c
    }
}

fn vecswap(v: &mut [Item], a: usize, b: usize, n: usize) {
    for k in 0..n {
        v.swap(a + k, b + k);
    }
}

fn qs(v: &mut [Item], mut a: usize, mut n: usize, mut depth: i32) {
    loop {
        let mut swap_cnt = 0;
        let d = depth;
        depth -= 1;
        if d <= 0 {
            // Heapsort in libc; an index-stable sort here (see module docs).
            v[a..a + n].sort_by(|x, y| cmp(x, y).cmp(&0).then(x.1.cmp(&y.1)));
            return;
        }
        if n <= 7 {
            isort(&mut v[a..a + n], 0);
            return;
        }
        let mut pl = a;
        let mut pm = a + n / 2;
        let mut pn = a + n - 1;
        if n > 40 {
            let d = n / 8;
            pl = med3(v, pl, pl + d, pl + 2 * d);
            pm = med3(v, pm - d, pm, pm + d);
            pn = med3(v, pn - 2 * d, pn - d, pn);
        }
        pm = med3(v, pl, pm, pn);
        v.swap(a, pm);
        let (mut pa, mut pb) = (a + 1, a + 1);
        let (mut pc, mut pd) = (a + n - 1, a + n - 1);
        loop {
            while pb <= pc {
                let r = cmp(&v[pb], &v[a]);
                if r > 0 {
                    break;
                }
                if r == 0 {
                    swap_cnt = 1;
                    v.swap(pa, pb);
                    pa += 1;
                }
                pb += 1;
            }
            while pb <= pc {
                let r = cmp(&v[pc], &v[a]);
                if r < 0 {
                    break;
                }
                if r == 0 {
                    swap_cnt = 1;
                    v.swap(pc, pd);
                    pd -= 1;
                }
                pc -= 1;
            }
            if pb > pc {
                break;
            }
            v.swap(pb, pc);
            swap_cnt = 1;
            pb += 1;
            pc -= 1;
        }
        let pn = a + n;
        let d1 = (pa - a).min(pb - pa);
        vecswap(v, a, pb - d1, d1);
        let d1 = (pd - pc).min(pn - pd - 1);
        vecswap(v, pb, pn - d1, d1);

        if swap_cnt == 0 {
            let r = 1 + (n / 4) as i32;
            if isort(&mut v[a..a + n], r) {
                return;
            }
        }
        let d1 = pb - pa;
        let d2 = pd - pc;
        if d1 <= d2 {
            if d1 > 1 {
                qs(v, a, d1, depth);
            }
            if d2 > 1 {
                a = pn - d2;
                n = d2;
                continue;
            }
        } else {
            if d2 > 1 {
                qs(v, pn - d2, d2, depth);
            }
            if d1 > 1 {
                n = d1;
                continue;
            }
        }
        return;
    }
}

/// Sort `v` by key as Apple's `qsort` would.
pub fn apple_qsort(v: &mut [Item]) {
    let n = v.len();
    if n < 2 {
        return;
    }
    // DEPTH(n) = 2 * (floor(log2(n)))
    let depth = 2 * (usize::BITS - 1 - n.leading_zeros()) as i32;
    qs(v, 0, n, depth);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Recorded output of macOS's libc `qsort` (60 tied arrays of 16 and 32
    /// `(value, index)` pairs, made with a small C helper), so the port is
    /// checked on any platform.
    #[test]
    fn matches_recorded_macos_qsort() {
        let text = include_str!("../../tests/fixtures/gguf_quantize/qsort_apple_cases.txt");
        for line in text.lines() {
            let (l, r) = line.split_once('|').unwrap();
            let vals: Vec<f32> = l
                .split_whitespace()
                .skip(1)
                .map(|v| v.parse().unwrap())
                .collect();
            let want: Vec<u32> = r.split_whitespace().map(|v| v.parse().unwrap()).collect();
            let mut v: Vec<Item> = vals
                .iter()
                .enumerate()
                .map(|(i, &x)| (x, i as u32))
                .collect();
            apple_qsort(&mut v);
            assert_eq!(v.iter().map(|x| x.1).collect::<Vec<_>>(), want, "{vals:?}");
        }
    }

    /// Compare against the system `qsort` on this machine: `QSORT_CASES` is a
    /// file of lines `n v0 .. v(n-1) | i0 .. i(n-1)` made by a C program that
    /// sorts `(value, index)` pairs with libc `qsort`.
    #[test]
    #[ignore = "needs QSORT_CASES from a C helper on the machine under test"]
    fn matches_system_qsort() {
        let Ok(path) = std::env::var("QSORT_CASES") else {
            return;
        };
        let text = std::fs::read_to_string(path).unwrap();
        let (mut bad, mut total) = (0, 0);
        for line in text.lines() {
            let (l, r) = line.split_once('|').unwrap();
            let vals: Vec<f32> = l
                .split_whitespace()
                .skip(1)
                .map(|v| v.parse().unwrap())
                .collect();
            let want: Vec<u32> = r.split_whitespace().map(|v| v.parse().unwrap()).collect();
            let mut v: Vec<Item> = vals
                .iter()
                .enumerate()
                .map(|(i, &x)| (x, i as u32))
                .collect();
            apple_qsort(&mut v);
            let got: Vec<u32> = v.iter().map(|x| x.1).collect();
            total += 1;
            if got != want {
                bad += 1;
                if bad <= 2 {
                    println!(
                        "n={} vals={vals:?}\n  libc={want:?}\n  port={got:?}",
                        vals.len()
                    );
                }
            }
        }
        println!("{bad} of {total} arrays ordered differently from the system qsort");
        assert_eq!(bad, 0);
    }
}
