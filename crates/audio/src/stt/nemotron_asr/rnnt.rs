//! RNN-T prediction and joint networks for the pinned Nemotron checkpoint.
//!
//! The LSTM stack, joint network, and `sigmoid` are shared with Parakeet
//! TDT, whose predictor and joint compute the same arithmetic.

use turbospark_model_io::safetensors::SafetensorsFile;

use crate::nn::Linear;
use crate::ops;
use crate::{Result, SpeechError};

fn tensor(file: &SafetensorsFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let desc = file.descriptor(name).ok_or_else(|| SpeechError::Tensor {
        name: name.to_string(),
        why: "missing from safetensors".into(),
    })?;
    if desc.shape != shape {
        return Err(SpeechError::Tensor {
            name: name.to_string(),
            why: format!("expected shape {shape:?}, found {:?}", desc.shape),
        });
    }
    if !matches!(desc.dtype.as_str(), "F16" | "BF16" | "F32") {
        return Err(SpeechError::Tensor {
            name: name.to_string(),
            why: format!("expected floating point weights, found {}", desc.dtype),
        });
    }
    file.load_as_f32(name).map_err(|error| SpeechError::Tensor {
        name: name.to_string(),
        why: format!("load failed: {error}"),
    })
}

pub(crate) struct LstmLayer {
    pub(crate) wx: Vec<f32>,
    pub(crate) wh: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

/// One step through a stack of LSTM layers: returns the top layer output
/// and every layer's next hidden and cell state. `input` is the embedded
/// token; the arithmetic order is the one both transducer families used.
pub(crate) fn lstm_stack(
    layers: &[LstmLayer],
    width: usize,
    mut input: Vec<f32>,
    hidden: &[Vec<f32>],
    cell: &[Vec<f32>],
) -> (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut next_hidden = Vec::with_capacity(layers.len());
    let mut next_cell = Vec::with_capacity(layers.len());
    for (index, layer) in layers.iter().enumerate() {
        let mut gates = ops::linear(&input, &layer.wx, Some(&layer.bias), 1, width, 4 * width);
        let recurrent = ops::linear(&hidden[index], &layer.wh, None, 1, width, 4 * width);
        for (gate, recurrent) in gates.iter_mut().zip(recurrent) {
            *gate += recurrent;
        }

        let mut h = vec![0.0; width];
        let mut c = vec![0.0; width];
        for unit in 0..width {
            let input_gate = sigmoid(gates[unit]);
            let forget_gate = sigmoid(gates[width + unit]);
            let candidate = gates[2 * width + unit].tanh();
            let output_gate = sigmoid(gates[3 * width + unit]);
            c[unit] = forget_gate * cell[index][unit] + input_gate * candidate;
            h[unit] = output_gate * c[unit].tanh();
        }
        input.clone_from(&h);
        next_hidden.push(h);
        next_cell.push(c);
    }
    (input, next_hidden, next_cell)
}

pub(crate) struct NemotronPredictor {
    embedding: Vec<f32>,
    layers: Vec<LstmLayer>,
    hidden: usize,
    vocabulary: usize,
}

/// Joint network: `output(relu(enc(encoder) + pred(predictor)))`.
pub(crate) struct Joint {
    pub(crate) encoder: Linear,
    pub(crate) predictor: Linear,
    pub(crate) output: Linear,
}

impl NemotronPredictor {
    pub(crate) fn load(
        file: &SafetensorsFile,
        hidden: usize,
        layers: usize,
        vocabulary: usize,
    ) -> Result<Self> {
        let embedding = tensor(
            file,
            "decoder.prediction.embed.weight",
            &[vocabulary + 1, hidden],
        )?;
        let mut loaded_layers = Vec::with_capacity(layers);
        for index in 0..layers {
            let prefix = format!("decoder.prediction.dec_rnn.lstm.{index}");
            loaded_layers.push(LstmLayer {
                wx: tensor(file, &format!("{prefix}.Wx"), &[hidden * 4, hidden])?,
                wh: tensor(file, &format!("{prefix}.Wh"), &[hidden * 4, hidden])?,
                bias: tensor(file, &format!("{prefix}.bias"), &[hidden * 4])?,
            });
        }
        Ok(Self {
            embedding,
            layers: loaded_layers,
            hidden,
            vocabulary,
        })
    }

    /// Run one prediction step. `None` is the blank-as-pad zero embedding.
    pub(crate) fn step(
        &self,
        token: Option<usize>,
        hidden: &[Vec<f32>],
        cell: &[Vec<f32>],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        if hidden.len() != self.layers.len() || cell.len() != self.layers.len() {
            return Err(SpeechError::Input {
                why: "Nemotron predictor state has the wrong layer count".into(),
            });
        }
        let mut input = vec![0.0; self.hidden];
        if let Some(token) = token {
            if token > self.vocabulary {
                return Err(SpeechError::Input {
                    why: format!("Nemotron token {token} is outside the embedding table"),
                });
            }
            let start = token * self.hidden;
            input.copy_from_slice(&self.embedding[start..start + self.hidden]);
        }

        Ok(lstm_stack(&self.layers, self.hidden, input, hidden, cell))
    }
}

impl Joint {
    pub(crate) fn new(encoder: Linear, predictor: Linear, output: Linear) -> Self {
        Self {
            encoder,
            predictor,
            output,
        }
    }

    pub(crate) fn load(
        file: &SafetensorsFile,
        encoder_hidden: usize,
        predictor_hidden: usize,
        joint_hidden: usize,
        classes: usize,
    ) -> Result<Self> {
        let linear = |name: &str, out: usize, input: usize| -> Result<Linear> {
            Ok(Linear::new(
                tensor(file, &format!("{name}.weight"), &[out, input])?,
                Some(tensor(file, &format!("{name}.bias"), &[out])?),
                input,
                out,
            ))
        };
        Ok(Self {
            encoder: linear("joint.enc", joint_hidden, encoder_hidden)?,
            predictor: linear("joint.pred", joint_hidden, predictor_hidden)?,
            output: linear("joint.joint_net.2", classes, joint_hidden)?,
        })
    }

    /// Encoder-side projection of `frames` rows at once. Rows are
    /// independent, so each equals the projection of that row alone.
    pub(crate) fn project_encoder(&self, features: &[f32], frames: usize) -> Vec<f32> {
        self.encoder.forward(features, frames)
    }

    pub(crate) fn project_predictor(&self, predictor: &[f32]) -> Vec<f32> {
        self.predictor.forward(predictor, 1)
    }

    /// Class logits from one projected encoder row and a projected
    /// predictor output.
    pub(crate) fn logits(
        &self,
        encoder_projected: &[f32],
        predictor_projected: &[f32],
    ) -> Vec<f32> {
        let mut joined = encoder_projected.to_vec();
        for (value, pred) in joined.iter_mut().zip(predictor_projected) {
            *value = (*value + pred).max(0.0);
        }
        self.output.forward(&joined, 1)
    }
}

/// Projected prediction plus the hidden and cell state it would advance to.
type Proposal = (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>);

/// Greedy RNN-T decode: the non-blank token ids in emission order.
///
/// The predictor output depends only on the last emitted token and the LSTM
/// state, which change only when a non-blank token is emitted, so after a
/// blank the previous prediction (and its joint projection) is reused. The
/// encoder-side joint projection is computed for all frames in one call.
pub(crate) fn greedy_tokens(
    predictor: &NemotronPredictor,
    joint: &Joint,
    features: &[f32],
    frames: usize,
    encoder_hidden: usize,
    blank_id: usize,
    max_symbols: usize,
) -> Result<Vec<usize>> {
    let encoder_projected = joint.project_encoder(&features[..frames * encoder_hidden], frames);
    let joint_hidden = encoder_projected.len().checked_div(frames).unwrap_or(0);
    let mut last_token = blank_id;
    let mut hidden = vec![vec![0.0; predictor.hidden]; predictor.layers.len()];
    let mut cell = vec![vec![0.0; predictor.hidden]; predictor.layers.len()];
    // (projected prediction, proposed hidden, proposed cell) for the
    // current (last_token, hidden, cell).
    let mut proposal: Option<Proposal> = None;
    let mut tokens = Vec::new();
    let mut frame = 0;
    let mut symbols = 0;
    while frame < frames {
        if proposal.is_none() {
            let (prediction, proposed_hidden, proposed_cell) = predictor.step(
                (last_token != blank_id).then_some(last_token),
                &hidden,
                &cell,
            )?;
            proposal = Some((
                joint.project_predictor(&prediction),
                proposed_hidden,
                proposed_cell,
            ));
        }
        let (predictor_projected, _, _) = proposal.as_ref().expect("proposal just computed");
        let logits = joint.logits(
            &encoder_projected[frame * joint_hidden..(frame + 1) * joint_hidden],
            predictor_projected,
        );
        let token = super::argmax(&logits);
        if token == blank_id {
            frame += 1;
            symbols = 0;
            continue;
        }

        last_token = token;
        let (_, proposed_hidden, proposed_cell) = proposal.take().expect("proposal present");
        hidden = proposed_hidden;
        cell = proposed_cell;
        tokens.push(token);
        symbols += 1;
        if symbols >= max_symbols {
            frame += 1;
            symbols = 0;
        }
    }
    Ok(tokens)
}

pub(crate) fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    //! Retained-reference parity for the shared LSTM stack, the projected
    //! joint, and the cached greedy loop. The `reference_*` functions are the
    //! original per-iteration code, kept verbatim, so the new code must give
    //! the same logits bitwise and the same tokens.
    // The reference copies below are the original code verbatim, lints included.
    #![allow(clippy::type_complexity)]
    use super::*;

    pub(crate) struct Rng(pub u64);

    impl Rng {
        pub(crate) fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 ^= self.0 << 13;
                    self.0 ^= self.0 >> 7;
                    self.0 ^= self.0 << 17;
                    ((self.0 >> 8) % 20001) as f32 / 10000.0 * scale - scale
                })
                .collect()
        }

        pub(crate) fn linear(&mut self, input: usize, output: usize) -> Linear {
            Linear::new(
                self.vec(input * output, 2.5 / (input as f32).sqrt()),
                Some(self.vec(output, 0.5)),
                input,
                output,
            )
        }
    }

    const HIDDEN: usize = 6;
    const LAYERS: usize = 2;
    const VOCAB: usize = 5;
    const ENCODER: usize = 7;
    const JOINT: usize = 8;

    fn predictor(rng: &mut Rng) -> NemotronPredictor {
        NemotronPredictor {
            embedding: rng.vec((VOCAB + 1) * HIDDEN, 1.0),
            layers: (0..LAYERS)
                .map(|_| LstmLayer {
                    wx: rng.vec(4 * HIDDEN * HIDDEN, 1.0),
                    wh: rng.vec(4 * HIDDEN * HIDDEN, 1.0),
                    bias: rng.vec(4 * HIDDEN, 0.5),
                })
                .collect(),
            hidden: HIDDEN,
            vocabulary: VOCAB,
        }
    }

    /// `blank_bias` raises the blank logit so some iterations end on blank
    /// and the cached prediction is reused.
    fn joint(rng: &mut Rng, blank_bias: f32) -> Joint {
        let mut output = rng.linear(JOINT, VOCAB + 1);
        output.bias.as_mut().unwrap()[VOCAB] += blank_bias;
        Joint::new(
            rng.linear(ENCODER, JOINT),
            rng.linear(HIDDEN, JOINT),
            output,
        )
    }

    fn reference_step(
        this: &NemotronPredictor,
        token: Option<usize>,
        hidden: &[Vec<f32>],
        cell: &[Vec<f32>],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        if hidden.len() != this.layers.len() || cell.len() != this.layers.len() {
            return Err(SpeechError::Input {
                why: "Nemotron predictor state has the wrong layer count".into(),
            });
        }
        let mut input = vec![0.0; this.hidden];
        if let Some(token) = token {
            if token > this.vocabulary {
                return Err(SpeechError::Input {
                    why: format!("Nemotron token {token} is outside the embedding table"),
                });
            }
            let start = token * this.hidden;
            input.copy_from_slice(&this.embedding[start..start + this.hidden]);
        }

        let mut next_hidden = Vec::with_capacity(this.layers.len());
        let mut next_cell = Vec::with_capacity(this.layers.len());
        for (index, layer) in this.layers.iter().enumerate() {
            let mut gates = ops::linear(
                &input,
                &layer.wx,
                Some(&layer.bias),
                1,
                this.hidden,
                4 * this.hidden,
            );
            let recurrent = ops::linear(
                &hidden[index],
                &layer.wh,
                None,
                1,
                this.hidden,
                4 * this.hidden,
            );
            for (gate, recurrent) in gates.iter_mut().zip(recurrent) {
                *gate += recurrent;
            }

            let mut h = vec![0.0; this.hidden];
            let mut c = vec![0.0; this.hidden];
            for unit in 0..this.hidden {
                let input_gate = sigmoid(gates[unit]);
                let forget_gate = sigmoid(gates[this.hidden + unit]);
                let candidate = gates[2 * this.hidden + unit].tanh();
                let output_gate = sigmoid(gates[3 * this.hidden + unit]);
                c[unit] = forget_gate * cell[index][unit] + input_gate * candidate;
                h[unit] = output_gate * c[unit].tanh();
            }
            input.clone_from(&h);
            next_hidden.push(h);
            next_cell.push(c);
        }
        Ok((input, next_hidden, next_cell))
    }

    fn reference_logits(this: &Joint, encoder: &[f32], predictor: &[f32]) -> Vec<f32> {
        let mut joined = this.encoder.forward(encoder, 1);
        let pred = this.predictor.forward(predictor, 1);
        for (value, pred) in joined.iter_mut().zip(pred) {
            *value = (*value + pred).max(0.0);
        }
        this.output.forward(&joined, 1)
    }

    /// The original loop with the piece mapping replaced by collecting the
    /// emitted token ids.
    fn reference_greedy(
        predictor: &NemotronPredictor,
        joint: &Joint,
        features: &[f32],
        frames: usize,
        blank_id: usize,
        max_symbols: usize,
    ) -> Result<(Vec<usize>, usize)> {
        let mut blanks = 0;
        let mut last_token = blank_id;
        let mut hidden = vec![vec![0.0; HIDDEN]; LAYERS];
        let mut cell = vec![vec![0.0; HIDDEN]; LAYERS];
        let mut tokens = Vec::new();
        let mut frame = 0;
        let mut symbols = 0;
        while frame < frames {
            let start = frame * ENCODER;
            let encoder_frame = &features[start..start + ENCODER];
            let (prediction, proposed_hidden, proposed_cell) = reference_step(
                predictor,
                (last_token != blank_id).then_some(last_token),
                &hidden,
                &cell,
            )?;
            let logits = reference_logits(joint, encoder_frame, &prediction);
            let token = super::super::argmax(&logits);
            if token == blank_id {
                blanks += 1;
                frame += 1;
                symbols = 0;
                continue;
            }

            last_token = token;
            hidden = proposed_hidden;
            cell = proposed_cell;
            tokens.push(token);
            symbols += 1;
            if symbols >= max_symbols {
                frame += 1;
                symbols = 0;
            }
        }
        Ok((tokens, blanks))
    }

    fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "{what}[{i}]: got {g} want {w}");
        }
    }

    #[test]
    fn shared_lstm_stack_matches_the_original_step_bitwise() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let predictor = predictor(&mut rng);
        let mut hidden: Vec<Vec<f32>> = (0..LAYERS).map(|_| rng.vec(HIDDEN, 1.0)).collect();
        let mut cell: Vec<Vec<f32>> = (0..LAYERS).map(|_| rng.vec(HIDDEN, 1.0)).collect();
        for token in [None, Some(0), Some(3), Some(VOCAB), None, Some(2)] {
            let want = reference_step(&predictor, token, &hidden, &cell).unwrap();
            let got = predictor.step(token, &hidden, &cell).unwrap();
            assert_bits("prediction", &got.0, &want.0);
            for layer in 0..LAYERS {
                assert_bits("hidden", &got.1[layer], &want.1[layer]);
                assert_bits("cell", &got.2[layer], &want.2[layer]);
            }
            (hidden, cell) = (got.1, got.2);
        }
        assert!(predictor.step(Some(VOCAB + 1), &hidden, &cell).is_err());
    }

    #[test]
    fn projected_joint_matches_the_per_frame_joint_bitwise() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let joint = joint(&mut rng, 0.0);
        // Enough rows to take the multi-row `ops::linear` path.
        let frames = 9;
        let features = rng.vec(frames * ENCODER, 1.5);
        let projected = joint.project_encoder(&features, frames);
        for frame in 0..frames {
            let prediction = rng.vec(HIDDEN, 1.0);
            let want = reference_logits(
                &joint,
                &features[frame * ENCODER..(frame + 1) * ENCODER],
                &prediction,
            );
            let got = joint.logits(
                &projected[frame * JOINT..(frame + 1) * JOINT],
                &joint.project_predictor(&prediction),
            );
            assert_bits("logits", &got, &want);
        }
    }

    #[test]
    fn cached_greedy_loop_matches_the_original_tokens() {
        let (mut emitted, mut blanks) = (0, 0);
        for (seed, frames, max_symbols, blank_bias) in [
            (1u64, 12usize, 4usize, 1.0f32),
            (2, 25, 2, 2.0),
            (3, 40, 6, 3.0),
            (4, 17, 1, 0.5),
            (5, 33, 3, 2.5),
            (6, 9, 5, 4.0),
            (7, 30, 6, 6.0),
        ] {
            let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
            let predictor = predictor(&mut rng);
            let joint = joint(&mut rng, blank_bias);
            let features = rng.vec(frames * ENCODER, 1.5);
            let (want, want_blanks) =
                reference_greedy(&predictor, &joint, &features, frames, VOCAB, max_symbols)
                    .unwrap();
            let got = greedy_tokens(
                &predictor,
                &joint,
                &features,
                frames,
                ENCODER,
                VOCAB,
                max_symbols,
            )
            .unwrap();
            assert_eq!(got, want, "seed {seed}");
            emitted += want.len();
            blanks += want_blanks;
        }
        // Both decisions must occur, and blanks must follow tokens, or the
        // cache-reuse path is never exercised.
        assert!(emitted > 0 && blanks > 0, "vacuous test inputs");
    }
}
