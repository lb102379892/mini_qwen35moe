// Updated build_moe_ffn function

void build_moe_ffn(...) {
    // Previous code...

    // Gather weights using the correct `ggml_get_rows`
    // Assuming `procs_3d` is created and selected is appropriately defined.  
    // The weights tensor will now be gathered correctly.
    const auto gathered_weights = ggml_get_rows(probs_3d, selected);

    // The output weights mapping to the experts is now valid, and we can reshape it.
    auto weights = ggml_reshape_2d(gathered_weights, n_top_k, n_tokens);

    // Further calculations and processing...
}

// Other surrounding code remains unchanged...
