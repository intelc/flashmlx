// test_heterogeneous_attention.cpp
//
// Correctness test: forward_heterogeneous() must produce the same logits
// as running two separate forward() calls at different cache offsets.
//
// Usage: ./test_heterogeneous_attention <model_path>

#include "flashmlx/model.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>

namespace mx = mlx::core;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>\n";
        return 1;
    }
    std::string model_path = argv[1];

    std::cout << "Loading model from: " << model_path << "\n";
    flashmlx::LlamaModel model(model_path);
    const auto& cfg = model.config();

    int n_layers = cfg.num_hidden_layers;
    int n_kv = cfg.num_key_value_heads;
    int hd = cfg.head_dim > 0 ? cfg.head_dim : cfg.hidden_size / cfg.num_attention_heads;
    int max_ctx = 128;

    std::cout << "Model loaded: " << n_layers << " layers, "
              << n_kv << " kv heads, head_dim=" << hd << "\n";

    // -----------------------------------------------------------------------
    // Prompts
    // -----------------------------------------------------------------------
    std::vector<int> prompt_a_vec = {1, 2, 3, 4, 5};
    std::vector<int> prompt_b_vec = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int len_a = (int)prompt_a_vec.size();
    int len_b = (int)prompt_b_vec.size();
    int tok_a = 42;
    int tok_b = 99;

    auto prompt_a = mx::array(prompt_a_vec.data(), {1, len_a}, mx::int32);
    auto prompt_b = mx::array(prompt_b_vec.data(), {1, len_b}, mx::int32);

    // -----------------------------------------------------------------------
    // Step 1: Prefill each sequence separately using array-offset forward
    // Pre-allocate caches with shape [1, n_kv, max_ctx, hd]
    // -----------------------------------------------------------------------
    std::cout << "Prefilling sequence A (len=" << len_a << ")...\n";

    std::vector<mx::array> cache_k_a, cache_v_a;
    for (int i = 0; i < n_layers; i++) {
        cache_k_a.push_back(mx::zeros({1, n_kv, max_ctx, hd}, cfg.activation_dtype));
        cache_v_a.push_back(mx::zeros({1, n_kv, max_ctx, hd}, cfg.activation_dtype));
    }
    auto offset_zero = mx::array({0}, mx::int32);
    auto logits_a_prefill = model.forward(prompt_a, cache_k_a, cache_v_a, offset_zero);
    mx::eval(logits_a_prefill);
    // Eval caches too
    for (int i = 0; i < n_layers; i++) {
        mx::eval({cache_k_a[i], cache_v_a[i]});
    }

    std::cout << "Prefilling sequence B (len=" << len_b << ")...\n";

    std::vector<mx::array> cache_k_b, cache_v_b;
    for (int i = 0; i < n_layers; i++) {
        cache_k_b.push_back(mx::zeros({1, n_kv, max_ctx, hd}, cfg.activation_dtype));
        cache_v_b.push_back(mx::zeros({1, n_kv, max_ctx, hd}, cfg.activation_dtype));
    }
    auto logits_b_prefill = model.forward(prompt_b, cache_k_b, cache_v_b, offset_zero);
    mx::eval(logits_b_prefill);
    for (int i = 0; i < n_layers; i++) {
        mx::eval({cache_k_b[i], cache_v_b[i]});
    }

    // -----------------------------------------------------------------------
    // Step 2: Reference decode — one token per sequence using int-offset path
    // The int-offset path uses concat, so we need trimmed caches [1, n_kv, prompt_len, hd]
    // Clone caches first since forward() modifies them via concat
    // -----------------------------------------------------------------------
    std::cout << "Reference decode: tok_a=" << tok_a << " at offset=" << len_a << "\n";

    // Clone + trim caches for sequence A
    std::vector<mx::array> ref_k_a, ref_v_a;
    for (int i = 0; i < n_layers; i++) {
        auto ck = mx::copy(mx::slice(cache_k_a[i], {0, 0, 0, 0}, {1, n_kv, len_a, hd}));
        auto cv = mx::copy(mx::slice(cache_v_a[i], {0, 0, 0, 0}, {1, n_kv, len_a, hd}));
        mx::eval({ck, cv});
        ref_k_a.push_back(ck);
        ref_v_a.push_back(cv);
    }

    auto ids_a = mx::array({tok_a}, {1, 1}, mx::int32);
    auto ref_logits_a = model.forward(ids_a, ref_k_a, ref_v_a, len_a);
    mx::eval(ref_logits_a);

    std::cout << "Reference decode: tok_b=" << tok_b << " at offset=" << len_b << "\n";

    // Clone + trim caches for sequence B
    std::vector<mx::array> ref_k_b, ref_v_b;
    for (int i = 0; i < n_layers; i++) {
        auto ck = mx::copy(mx::slice(cache_k_b[i], {0, 0, 0, 0}, {1, n_kv, len_b, hd}));
        auto cv = mx::copy(mx::slice(cache_v_b[i], {0, 0, 0, 0}, {1, n_kv, len_b, hd}));
        mx::eval({ck, cv});
        ref_k_b.push_back(ck);
        ref_v_b.push_back(cv);
    }

    auto ids_b = mx::array({tok_b}, {1, 1}, mx::int32);
    auto ref_logits_b = model.forward(ids_b, ref_k_b, ref_v_b, len_b);
    mx::eval(ref_logits_b);

    // ref_logits_a: [1, 1, vocab_size], ref_logits_b: [1, 1, vocab_size]
    std::cout << "Reference logits shapes: A=" << ref_logits_a.shape(0) << "x"
              << ref_logits_a.shape(1) << "x" << ref_logits_a.shape(2)
              << ", B=" << ref_logits_b.shape(0) << "x"
              << ref_logits_b.shape(1) << "x" << ref_logits_b.shape(2) << "\n";

    // -----------------------------------------------------------------------
    // Step 3: Heterogeneous decode — both tokens in a single batched call
    // Build padded KV caches [2, n_kv, max_ctx, hd] from the prefilled caches
    // -----------------------------------------------------------------------
    std::cout << "Heterogeneous decode: batch of 2...\n";

    std::vector<mx::array> batch_k, batch_v;
    for (int i = 0; i < n_layers; i++) {
        // cache_k_a[i] and cache_k_b[i] are both [1, n_kv, max_ctx, hd]
        // Just concatenate along batch dim
        auto bk = mx::concatenate({cache_k_a[i], cache_k_b[i]}, 0);  // [2, n_kv, max_ctx, hd]
        auto bv = mx::concatenate({cache_v_a[i], cache_v_b[i]}, 0);
        mx::eval({bk, bv});
        batch_k.push_back(bk);
        batch_v.push_back(bv);
    }

    // Batch input: [2, 1]
    auto batch_ids = mx::array({tok_a, tok_b}, {2, 1}, mx::int32);
    auto offsets = mx::array({len_a, len_b}, mx::int32);  // {5, 10}

    auto het_logits = model.forward_heterogeneous(batch_ids, batch_k, batch_v, offsets, max_ctx);
    mx::eval(het_logits);

    // het_logits: [2, 1, vocab_size]
    std::cout << "Heterogeneous logits shape: " << het_logits.shape(0) << "x"
              << het_logits.shape(1) << "x" << het_logits.shape(2) << "\n";

    // -----------------------------------------------------------------------
    // Step 4: Compare
    // -----------------------------------------------------------------------
    // Extract het_logits[0] and het_logits[1]
    int vocab_size = het_logits.shape(2);
    auto het_a = mx::reshape(mx::slice(het_logits, {0, 0, 0}, {1, 1, vocab_size}), {vocab_size});
    auto het_b = mx::reshape(mx::slice(het_logits, {1, 0, 0}, {2, 1, vocab_size}), {vocab_size});
    auto ref_a = mx::reshape(ref_logits_a, {vocab_size});
    auto ref_b = mx::reshape(ref_logits_b, {vocab_size});

    // Cast to float32 for comparison
    het_a = mx::astype(het_a, mx::float32);
    het_b = mx::astype(het_b, mx::float32);
    ref_a = mx::astype(ref_a, mx::float32);
    ref_b = mx::astype(ref_b, mx::float32);

    auto diff_a = mx::max(mx::abs(mx::subtract(het_a, ref_a)));
    auto diff_b = mx::max(mx::abs(mx::subtract(het_b, ref_b)));
    mx::eval({diff_a, diff_b});

    float max_diff_a = diff_a.item<float>();
    float max_diff_b = diff_b.item<float>();

    std::cout << "\n=== RESULTS ===\n";
    std::cout << "Max abs diff (sequence A): " << max_diff_a << "\n";
    std::cout << "Max abs diff (sequence B): " << max_diff_b << "\n";

    // Also print top-5 logit indices for sanity
    auto top_a_het = mx::argmax(het_a);
    auto top_b_het = mx::argmax(het_b);
    auto top_a_ref = mx::argmax(ref_a);
    auto top_b_ref = mx::argmax(ref_b);
    mx::eval({top_a_het, top_b_het, top_a_ref, top_b_ref});

    std::cout << "Top token A: het=" << top_a_het.item<int32_t>()
              << ", ref=" << top_a_ref.item<int32_t>() << "\n";
    std::cout << "Top token B: het=" << top_b_het.item<int32_t>()
              << ", ref=" << top_b_ref.item<int32_t>() << "\n";

    float threshold = 0.05f;
    bool pass = (max_diff_a < threshold) && (max_diff_b < threshold);

    if (pass) {
        std::cout << "\n*** PASS *** (threshold=" << threshold << ")\n";
        return 0;
    } else {
        std::cerr << "\n*** FAIL *** (threshold=" << threshold << ")\n";

        // Debug: print first 10 logits for comparison
        auto het_a_f = mx::slice(het_a, {0}, {std::min(10, vocab_size)});
        auto ref_a_f = mx::slice(ref_a, {0}, {std::min(10, vocab_size)});
        auto het_b_f = mx::slice(het_b, {0}, {std::min(10, vocab_size)});
        auto ref_b_f = mx::slice(ref_b, {0}, {std::min(10, vocab_size)});
        mx::eval({het_a_f, ref_a_f, het_b_f, ref_b_f});

        std::cout << "\nFirst 10 logits (seq A):\n";
        std::cout << "  het: ";
        for (int i = 0; i < std::min(10, vocab_size); i++)
            std::cout << het_a_f.data<float>()[i] << " ";
        std::cout << "\n  ref: ";
        for (int i = 0; i < std::min(10, vocab_size); i++)
            std::cout << ref_a_f.data<float>()[i] << " ";

        std::cout << "\n\nFirst 10 logits (seq B):\n";
        std::cout << "  het: ";
        for (int i = 0; i < std::min(10, vocab_size); i++)
            std::cout << het_b_f.data<float>()[i] << " ";
        std::cout << "\n  ref: ";
        for (int i = 0; i < std::min(10, vocab_size); i++)
            std::cout << ref_b_f.data<float>()[i] << " ";
        std::cout << "\n";

        return 1;
    }
}
