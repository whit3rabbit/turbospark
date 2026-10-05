//! RNN-T prediction and joint networks for the pinned Nemotron checkpoint.

use turbospark_model_io::safetensors::SafetensorsFile;

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

struct LstmLayer {
    wx: Vec<f32>,
    wh: Vec<f32>,
    bias: Vec<f32>,
}

pub(crate) struct NemotronPredictor {
    embedding: Vec<f32>,
    layers: Vec<LstmLayer>,
    hidden: usize,
    vocabulary: usize,
}

pub(crate) struct NemotronJoint {
    encoder: Linear,
    predictor: Linear,
    output: Linear,
    hidden: usize,
    classes: usize,
}

struct Linear {
    weight: Vec<f32>,
    bias: Vec<f32>,
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

        let mut next_hidden = Vec::with_capacity(self.layers.len());
        let mut next_cell = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            let mut gates = ops::linear(
                &input,
                &layer.wx,
                Some(&layer.bias),
                1,
                self.hidden,
                4 * self.hidden,
            );
            let recurrent = ops::linear(
                &hidden[index],
                &layer.wh,
                None,
                1,
                self.hidden,
                4 * self.hidden,
            );
            for (gate, recurrent) in gates.iter_mut().zip(recurrent) {
                *gate += recurrent;
            }

            let mut h = vec![0.0; self.hidden];
            let mut c = vec![0.0; self.hidden];
            for unit in 0..self.hidden {
                let input_gate = sigmoid(gates[unit]);
                let forget_gate = sigmoid(gates[self.hidden + unit]);
                let candidate = gates[2 * self.hidden + unit].tanh();
                let output_gate = sigmoid(gates[3 * self.hidden + unit]);
                c[unit] = forget_gate * cell[index][unit] + input_gate * candidate;
                h[unit] = output_gate * c[unit].tanh();
            }
            input.clone_from(&h);
            next_hidden.push(h);
            next_cell.push(c);
        }
        Ok((input, next_hidden, next_cell))
    }
}

impl NemotronJoint {
    pub(crate) fn load(
        file: &SafetensorsFile,
        encoder_hidden: usize,
        predictor_hidden: usize,
        joint_hidden: usize,
        classes: usize,
    ) -> Result<Self> {
        let linear = |name: &str, out: usize, input: usize| -> Result<Linear> {
            Ok(Linear {
                weight: tensor(file, &format!("{name}.weight"), &[out, input])?,
                bias: tensor(file, &format!("{name}.bias"), &[out])?,
            })
        };
        Ok(Self {
            encoder: linear("joint.enc", joint_hidden, encoder_hidden)?,
            predictor: linear("joint.pred", joint_hidden, predictor_hidden)?,
            output: linear("joint.joint_net.2", classes, joint_hidden)?,
            hidden: joint_hidden,
            classes,
        })
    }

    pub(crate) fn logits(&self, encoder: &[f32], predictor: &[f32]) -> Vec<f32> {
        let mut joined = ops::linear(
            encoder,
            &self.encoder.weight,
            Some(&self.encoder.bias),
            1,
            encoder.len(),
            self.hidden,
        );
        let pred = ops::linear(
            predictor,
            &self.predictor.weight,
            Some(&self.predictor.bias),
            1,
            predictor.len(),
            self.hidden,
        );
        for (value, pred) in joined.iter_mut().zip(pred) {
            *value = (*value + pred).max(0.0);
        }
        ops::linear(
            &joined,
            &self.output.weight,
            Some(&self.output.bias),
            1,
            self.hidden,
            self.classes,
        )
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}
