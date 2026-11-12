#![cfg(target_os = "macos")]
//! `gather_rows_fp16` (`dflash_conv.metal`) against the per-row
//! `encode_copy_rows_fp16_strided` loop it replaces in a tree verify's
//! compaction: bit-equal, out of place and in place, including the
//! overlapping path a flat one-dispatch gather would race on.

use half::f16;
use turbospark_gpu::{
    encode_copy_rows_fp16_strided, encode_gather_rows_fp16, read_buffer_f16, MetalContext,
};

const WIDTH: u32 = 1300;
const STRIDE: u32 = 1536;
const SLOTS: usize = 9;

fn data(seed: u32) -> Vec<u8> {
    (0..SLOTS * STRIDE as usize)
        .flat_map(|i| {
            let v = ((i as u32).wrapping_mul(2654435761) ^ seed) % 4000;
            f16::from_f32(v as f32 / 1000.0 - 2.0)
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

fn bits(buffer: &metal::Buffer) -> Vec<u16> {
    read_buffer_f16(buffer, 0, SLOTS * STRIDE as usize)
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// The reference: one strided copy per row, each in its own pass so no
/// dispatch can see another's half-written row.
fn reference(context: &mut MetalContext, src: &[u8], map: &[u32], in_place: bool) -> Vec<u16> {
    let src_buf = context.new_buffer_with_data(src);
    let snapshot = context.new_buffer_with_data(src);
    let dst = if in_place {
        context.new_buffer_with_data(src)
    } else {
        context.new_output_buffer((SLOTS * STRIDE as usize * 2) as u64)
    };
    for (r, &m) in map.iter().enumerate() {
        // In place, row r reads the slot as it stood before any write: the
        // compaction's contract, read here from an untouched copy.
        let from = if in_place { &snapshot } else { &src_buf };
        let pass = context.begin_pass();
        encode_copy_rows_fp16_strided(
            context,
            &pass,
            (from, u64::from(m * STRIDE) * 2),
            (&dst, (r as u64) * u64::from(STRIDE) * 2),
            1,
            WIDTH,
            STRIDE,
            STRIDE,
        )
        .unwrap();
        pass.commit_and_wait();
    }
    bits(&dst)
}

fn gathered(context: &mut MetalContext, src: &[u8], map: &[u32], in_place: bool) -> Vec<u16> {
    let src_buf = context.new_buffer_with_data(src);
    let dst = if in_place {
        src_buf.clone()
    } else {
        context.new_output_buffer((SLOTS * STRIDE as usize * 2) as u64)
    };
    let pass = context.begin_pass();
    encode_gather_rows_fp16(
        context,
        &pass,
        (&src_buf, 0),
        (&dst, 0),
        map,
        WIDTH,
        STRIDE,
        STRIDE,
    )
    .unwrap();
    pass.commit_and_wait();
    bits(&dst)
}

#[test]
fn the_gather_is_bit_equal_to_the_per_row_copy_loop() {
    let mut context = MetalContext::new().expect("Metal device");
    let src = data(0x5EED);
    for map in [
        vec![0u32, 2, 3],
        vec![0, 1, 2, 3, 4, 5, 6, 7],
        vec![0, 5, 6, 7, 8],
        vec![3],
        vec![1, 2, 4, 8],
    ] {
        for in_place in [false, true] {
            let want = reference(&mut context, &src, &map, in_place);
            let got = gathered(&mut context, &src, &map, in_place);
            assert!(got == want, "map {map:?}, in place {in_place}");
        }
    }
    // Out of place, any order and repeats are fine.
    let map = [7u32, 0, 7, 3];
    assert!(
        gathered(&mut context, &src, &map, false) == reference(&mut context, &src, &map, false)
    );
}

/// In place, a map that is not a compaction toward row 0 is refused: its
/// sources could be rows the walk already overwrote.
#[test]
fn an_unsafe_in_place_map_is_refused() {
    let mut context = MetalContext::new().expect("Metal device");
    let buf = context.new_buffer_with_data(&data(1));
    for map in [vec![1u32, 0], vec![0, 0], vec![2, 1, 3]] {
        let pass = context.begin_pass();
        let err = encode_gather_rows_fp16(
            &mut context,
            &pass,
            (&buf, 0),
            (&buf, 0),
            &map,
            WIDTH,
            STRIDE,
            STRIDE,
        );
        assert!(err.is_err(), "map {map:?} must be refused in place");
        pass.commit_and_wait();
    }
}
