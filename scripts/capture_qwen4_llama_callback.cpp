// Diagnostic collector for the frozen Qwen4 Swift first-token prompt.
// Keep the token IDs aligned with qwen4exp_swift_first_token_probe.rs.
// It dumps the last prompt token's layer boundaries and router tensors from
// the exact GGUF loaded by llama.cpp. Output is diagnostic, not a quality gate.
//
// Build against the same llama.cpp revision used for the reference run:
// c++ -std=c++17 -I"$(brew --prefix llama.cpp)/include" \
//   -I"$(brew --prefix ggml)/include" scripts/capture_qwen4_llama_callback.cpp \
//   -L"$(brew --prefix llama.cpp)/lib" -L"$(brew --prefix ggml)/lib" \
//   -Wl,-rpath,"$(brew --prefix llama.cpp)/lib" \
//   -Wl,-rpath,"$(brew --prefix ggml)/lib" \
//   -lllama -lggml -lggml-base -o /tmp/qwen4-llama-capture

#include <llama.h>
#include <ggml.h>
#include <ggml-backend.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

struct CaptureState {
    std::filesystem::path dir;
    int32_t token_count = 0;
    std::map<std::string, int> occurrences;
    std::set<std::string> listed_nodes;
};

static bool wide_activation(ggml_tensor * tensor, int32_t token_count) {
    bool has_token_axis = false;
    size_t row_width = 1;
    for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
        if (!has_token_axis && tensor->ne[axis] == token_count) {
            has_token_axis = true;
            continue;
        }
        row_width *= (size_t)tensor->ne[axis];
    }
    return has_token_axis && row_width == 10240;
}

static bool selected(const char * name, ggml_tensor * tensor, int32_t token_count) {
    const std::string value(name ? name : "");
    static const std::array<const char *, 18> names = {
        "model.input_embed", "hc_combine-0", "ffn_out-0", "l_last-0",
        "hc_combine-1", "ffn_out-1", "l_last-1", "result_norm",
        "hc_norm-0", "hc_gate-0", "hc_mixed-0", "linear_attn_out-0",
        "hc_inject-0", "hc_norm-1", "hc_gate-1", "hc_mixed-1",
        "linear_attn_out-1", "hc_inject-1"
    };
    for (const char * wanted : names) {
        if (value == wanted) return true;
    }
    const bool router_trace = value.rfind("ffn_moe_topk-", 0) == 0 ||
        value.rfind("ffn_moe_logits-", 0) == 0 ||
        value.rfind("ffn_moe_weights_norm-", 0) == 0;
    return router_trace || value == "result_output" || value == "hc_init" ||
        value.find("ple") != std::string::npos || wide_activation(tensor, token_count);
}

static float value_at(const std::vector<uint8_t> & data, ggml_type type, size_t offset) {
    if (type == GGML_TYPE_F32) {
        float value;
        std::memcpy(&value, data.data() + offset, sizeof(value));
        return value;
    }
    if (type == GGML_TYPE_I32) {
        int32_t value;
        std::memcpy(&value, data.data() + offset, sizeof(value));
        return (float)value;
    }
    if (type == GGML_TYPE_F16) {
        ggml_fp16_t value;
        std::memcpy(&value, data.data() + offset, sizeof(value));
        return ggml_fp16_to_fp32(value);
    }
    if (type == GGML_TYPE_BF16) {
        ggml_bf16_t value;
        std::memcpy(&value, data.data() + offset, sizeof(value));
        return ggml_bf16_to_fp32(value);
    }
    throw std::runtime_error(std::string("unsupported capture type: ") + ggml_type_name(type));
}

static bool capture_tensor(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * state = static_cast<CaptureState *>(user_data);
    const std::string name(ggml_get_name(tensor));
    if (ask) {
        const bool relevant = name.find("-0") != std::string::npos &&
            (name.find("hc_") != std::string::npos || name.find("linear") != std::string::npos ||
             name.find("gdn") != std::string::npos || name.find("attn") != std::string::npos ||
             name.find("ffn") != std::string::npos || name.find("ple") != std::string::npos);
        if (relevant && state->listed_nodes.insert(name).second) {
            std::ofstream nodes(state->dir / "tensor-nodes.tsv", std::ios::app);
            nodes << name << "\t" << ggml_op_name(tensor->op) << "\t"
                  << tensor->ne[0] << "," << tensor->ne[1] << ","
                  << tensor->ne[2] << "," << tensor->ne[3] << "\t"
                  << ggml_type_name(tensor->type) << "\n";
        }
        return selected(name.c_str(), tensor, state->token_count);
    }
    if (!selected(name.c_str(), tensor, state->token_count)) return true;

    const size_t nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> bytes(nbytes);
    ggml_backend_tensor_get(tensor, bytes.data(), 0, nbytes);

    int token_axis = -1;
    for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
        if (tensor->ne[axis] == state->token_count) {
            token_axis = axis;
            break;
        }
    }
    int64_t token_index = 0;
    if (token_axis >= 0) token_index = tensor->ne[token_axis] - 1;
    else if (tensor->ne[1] == 1) token_axis = 1;

    std::vector<float> row;
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const int64_t coordinates[4] = {i0, i1, i2, i3};
                    if (token_axis >= 0 && coordinates[token_axis] != token_index) continue;
                    const size_t offset = (size_t)i0 * tensor->nb[0] +
                        (size_t)i1 * tensor->nb[1] + (size_t)i2 * tensor->nb[2] +
                        (size_t)i3 * tensor->nb[3];
                    row.push_back(value_at(bytes, tensor->type, offset));
                }
            }
        }
    }

    const int occurrence = state->occurrences[name]++;
    const std::string file_name = name + "-" + std::to_string(occurrence);
    std::ofstream out(state->dir / (file_name + ".f32"), std::ios::binary);
    out.write(reinterpret_cast<const char *>(row.data()), (std::streamsize)(row.size() * sizeof(float)));
    std::ofstream meta(state->dir / "capture-meta.txt", std::ios::app);
    meta << "name=" << name << " occurrence=" << occurrence << " type=" << ggml_type_name(tensor->type)
         << " dims=" << tensor->ne[0] << "," << tensor->ne[1] << ","
         << tensor->ne[2] << "," << tensor->ne[3] << " token_axis=" << token_axis
         << " values=" << row.size() << " bytes=" << nbytes << "\n";
    std::cerr << "captured " << name << " occurrence=" << occurrence << " values=" << row.size() << "\n";
    if (name == "hc_norm-1" && occurrence == 0) {
        ggml_tensor * raw_input = tensor;
        for (int depth = 0; depth < 3; ++depth) {
            if (raw_input->src[0] == nullptr) throw std::runtime_error("hc_norm input graph is shorter than expected");
            raw_input = raw_input->src[0];
        }
        const size_t raw_bytes = ggml_nbytes(raw_input);
        std::vector<uint8_t> raw_data(raw_bytes);
        ggml_backend_tensor_get(raw_input, raw_data.data(), 0, raw_bytes);
        int raw_token_axis = -1;
        for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
            if (raw_input->ne[axis] == state->token_count) {
                raw_token_axis = axis;
                break;
            }
        }
        const int64_t raw_token_index = raw_token_axis >= 0 ? raw_input->ne[raw_token_axis] - 1 : 0;
        std::vector<float> raw_row;
        for (int64_t i3 = 0; i3 < raw_input->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < raw_input->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < raw_input->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < raw_input->ne[0]; ++i0) {
                        const int64_t coordinates[4] = {i0, i1, i2, i3};
                        if (raw_token_axis >= 0 && coordinates[raw_token_axis] != raw_token_index) continue;
                        const size_t offset = (size_t)i0 * raw_input->nb[0] +
                            (size_t)i1 * raw_input->nb[1] + (size_t)i2 * raw_input->nb[2] +
                            (size_t)i3 * raw_input->nb[3];
                        raw_row.push_back(value_at(raw_data, raw_input->type, offset));
                    }
                }
            }
        }
        std::ofstream raw_out(state->dir / "hc_input-1.f32", std::ios::binary);
        raw_out.write(reinterpret_cast<const char *>(raw_row.data()), (std::streamsize)(raw_row.size() * sizeof(float)));
        std::ofstream raw_meta(state->dir / "capture-meta.txt", std::ios::app);
        raw_meta << "name=hc_input-1 occurrence=0 type=" << ggml_type_name(raw_input->type)
                 << " dims=" << raw_input->ne[0] << "," << raw_input->ne[1] << ","
                 << raw_input->ne[2] << "," << raw_input->ne[3] << " token_axis=" << raw_token_axis
                 << " values=" << raw_row.size() << " bytes=" << raw_bytes << "\n";
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: capture MODEL.gguf OUTPUT_DIR\n";
        return 2;
    }
    const std::filesystem::path model_path(argv[1]);
    CaptureState state{std::filesystem::path(argv[2]), 0};
    if (std::filesystem::exists(state.dir) && !std::filesystem::is_empty(state.dir)) {
        std::cerr << "output directory must be empty: " << state.dir << "\n";
        return 2;
    }
    std::filesystem::create_directories(state.dir);

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) throw std::runtime_error("failed to load reference model");
    // Exact frozen prompt IDs already matched the TurboSpark tokenizer output
    // and llama.cpp's own tokenizer during the first comparison.
    std::vector<llama_token> tokens = {
        248045,846,198,814,20139,1204,33101,13988,7909,628,7698,17198,5383,2261,
        42046,13,58737,279,6745,23038,11,2873,1330,2894,13001,11,321,30982,5048,
        13550,494,4434,8862,13,9357,364,449,15515,6396,1973,23011,4706,321,2426,
        279,4087,1172,220,19,20,15,4105,13,248046,198,248045,74455,198,248068,271,248069,271
    };
    state.token_count = (int32_t)tokens.size();
    std::cerr << "prompt_tokens=" << state.token_count << "\n";
    {
        std::ofstream meta(state.dir / "capture-meta.txt", std::ios::app);
        meta << "source_model=" << std::filesystem::weakly_canonical(model_path).string() << "\n";
        meta << "prompt_tokens=" << state.token_count << "\n";
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 2048;
    context_params.n_batch = 2048;
    context_params.n_ubatch = 512;
    context_params.no_perf = true;
    context_params.cb_eval = capture_tensor;
    context_params.cb_eval_user_data = &state;
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) throw std::runtime_error("failed to create context");
    llama_batch batch = llama_batch_get_one(tokens.data(), state.token_count);
    const int decode_status = llama_decode(context, batch);
    if (decode_status != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(decode_status));

    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
