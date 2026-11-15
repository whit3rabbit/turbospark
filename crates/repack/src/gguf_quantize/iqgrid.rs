//! Search grids for the IQ quantizers: ggml's `iq2xs_init_impl` and
//! `iq3xs_init_impl`, rebuilt at first use.
//!
//! Each grid is a set of points (8 values for IQ2/IQ1, 4 for IQ3) on an odd
//! integer lattice. `map` takes a point's packed coordinates to a grid index,
//! or, for an off-grid point, to `-(offset + 1)` into `neighbours`, where
//! `neighbours[offset]` is the count and the next `count` entries are the
//! nearest grid points.
//!
//! The C code sorts all grid distances per off-grid point. Here the same list
//! is produced without a full sort: take the `nwant` smallest distinct
//! distances, keep every grid point at or below the largest, and order them by
//! `(distance, index)`, which is exactly what ggml's `qsort` comparator yields.

// Index loops mirror iq2xs_init_impl / iq3xs_init_impl in ggml-quants.c.
#![allow(clippy::needless_range_loop)]

use std::sync::OnceLock;

use super::iq_kgrid_tables::*;

pub struct IqGrid {
    /// Values per grid point (8 or 4).
    pub dim: usize,
    /// `grid_size * dim` values, each `2 * l + 1`.
    pub points: Vec<i8>,
    pub map: Vec<i32>,
    pub neighbours: Vec<u16>,
}

#[derive(Clone, Copy)]
pub enum Kind {
    Iq2Xxs,
    Iq2Xs,
    /// IQ1_S and IQ1_M share one grid.
    Iq1,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    /// Unsloth fork IQ1_XS, IQ1_XXS and IQ1_XXXS (ggml ids 64 to 66): the
    /// decode tables ARE the search grids (`iq2xs_init_impl`'s `narrow` path).
    Iq1Xs,
    Iq1Xxs,
    Iq1Xxxs,
}

impl IqGrid {
    pub fn point(&self, j: usize) -> &[i8] {
        &self.points[j * self.dim..(j + 1) * self.dim]
    }

    /// The neighbour list for an off-grid map entry `m < 0`.
    pub fn neighbours_of(&self, m: i32) -> &[u16] {
        &self.neighbours[(-(m + 1)) as usize..]
    }
}

fn build(kgrid: &[u16], dim: usize, bits: usize, kmap_size: usize, nwant: usize) -> IqGrid {
    let grid_size = kgrid.len();
    let mask = (1usize << bits) - 1;
    let mut points = vec![0i8; grid_size * dim];
    for (k, &g) in kgrid.iter().enumerate() {
        for i in 0..dim {
            points[k * dim + i] = (2 * ((g as usize >> (bits * i)) & mask) + 1) as i8;
        }
    }
    let mut map = vec![-1i32; kmap_size];
    for j in 0..grid_size {
        let mut index = 0usize;
        for k in 0..dim {
            index |= (((points[j * dim + k] as usize) - 1) / 2) << (bits * k);
        }
        map[index] = j as i32;
    }

    // Neighbour list for one off-grid point, as (count, indices).
    let neighbours_for = |i: usize| -> Vec<u16> {
        let mut pos = [0i32; 8];
        for k in 0..dim {
            pos[k] = 2 * ((i >> (bits * k)) & mask) as i32 + 1;
        }
        let d2: Vec<i32> = (0..grid_size)
            .map(|j| {
                (0..dim)
                    .map(|k| {
                        let d = i32::from(points[j * dim + k]) - pos[k];
                        d * d
                    })
                    .sum()
            })
            .collect();
        // The nwant smallest distinct distances.
        let mut limit = *d2.iter().min().unwrap();
        for _ in 1..nwant {
            match d2.iter().filter(|&&d| d > limit).min() {
                Some(&next) => limit = next,
                None => break,
            }
        }
        let mut sel: Vec<(i32, u16)> = d2
            .iter()
            .enumerate()
            .filter(|(_, &d)| d <= limit)
            .map(|(j, &d)| (d, j as u16))
            .collect();
        sel.sort_unstable();
        sel.into_iter().map(|(_, j)| j).collect()
    };

    let todo: Vec<usize> = (0..kmap_size).filter(|&i| map[i] < 0).collect();
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(16);
    let per = todo.len().div_ceil(threads).max(1);
    let lists: Vec<Vec<Vec<u16>>> = std::thread::scope(|s| {
        let handles: Vec<_> = todo
            .chunks(per)
            .map(|chunk| s.spawn(|| chunk.iter().map(|&i| neighbours_for(i)).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("grid worker"))
            .collect()
    });

    let mut neighbours = Vec::new();
    for (&i, list) in todo.iter().zip(lists.iter().flatten()) {
        map[i] = -(neighbours.len() as i32 + 1);
        neighbours.push(list.len() as u16);
        neighbours.extend_from_slice(list);
    }
    IqGrid {
        dim,
        points,
        map,
        neighbours,
    }
}

static G_XXS: OnceLock<IqGrid> = OnceLock::new();
static G_XS: OnceLock<IqGrid> = OnceLock::new();
static G_1: OnceLock<IqGrid> = OnceLock::new();
static G_S: OnceLock<IqGrid> = OnceLock::new();
static G_3XXS: OnceLock<IqGrid> = OnceLock::new();
static G_3S: OnceLock<IqGrid> = OnceLock::new();
static G_1XS: OnceLock<IqGrid> = OnceLock::new();
static G_1XXS: OnceLock<IqGrid> = OnceLock::new();
static G_1XXXS: OnceLock<IqGrid> = OnceLock::new();

/// The compressed (2 bits per value, level `l = g + 1`) form of one of the
/// fork's `uint64_t` decode grids, which is what `build` consumes. The fork
/// derives its search points the same way: `2 * ((int8)g + 1) + 1`.
fn compress_narrow(decode: &[u64]) -> Vec<u16> {
    decode
        .iter()
        .map(|&w| {
            (0..8).fold(0u16, |acc, i| {
                let g = ((w >> (8 * i)) & 0xff) as u8 as i8;
                acc | (((g + 1) as u16) << (2 * i))
            })
        })
        .collect()
}

/// The grid for `kind`, built on first use (a second or two in release).
pub fn grid(kind: Kind) -> &'static IqGrid {
    match kind {
        Kind::Iq2Xxs => G_XXS.get_or_init(|| build(&KGRID_2BIT_256, 8, 2, 43692, 2)),
        Kind::Iq2Xs => G_XS.get_or_init(|| build(&KGRID_2BIT_512, 8, 2, 43692, 2)),
        Kind::Iq1 => G_1.get_or_init(|| build(&KGRID_1BIT_2048, 8, 2, 43692, 3)),
        Kind::Iq2S => G_S.get_or_init(|| build(&KGRID_2BIT_1024, 8, 2, 43692, 1)),
        Kind::Iq3Xxs => G_3XXS.get_or_init(|| build(&KGRID_3BIT_256, 4, 3, 4096, 2)),
        Kind::Iq3S => G_3S.get_or_init(|| build(&KGRID_3BIT_512, 4, 3, 4096, 3)),
        Kind::Iq1Xs => G_1XS.get_or_init(|| {
            build(
                &compress_narrow(&compute::quant_gguf_iq1_narrow::IQ1_XS_GRID),
                8,
                2,
                43692,
                3,
            )
        }),
        Kind::Iq1Xxs => G_1XXS.get_or_init(|| {
            build(
                &compress_narrow(&compute::quant_gguf_iq1_narrow::IQ1_XXS_GRID),
                8,
                2,
                43692,
                3,
            )
        }),
        Kind::Iq1Xxxs => G_1XXXS.get_or_init(|| {
            build(
                &compress_narrow(&compute::quant_gguf_iq1_narrow::IQ1_XXXS_GRID),
                8,
                2,
                43692,
                3,
            )
        }),
    }
}
