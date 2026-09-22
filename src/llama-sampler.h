#pragma once

#include "llama.h"

#include <string>
#include <vector>

struct llama_vocab;
struct llama_grammar;

// INTERNAL exact-build checkpoint helpers. Not part of the public llama.h ABI.
bool llama_sampler_checkpoint_supported(
        const llama_sampler * smpl,
        std::string & error);
bool llama_sampler_checkpoint_export_apply_state(
        const llama_sampler * smpl,
        std::vector<uint8_t> & out,
        std::string & error);
bool llama_sampler_checkpoint_import_apply_state(
        llama_sampler * smpl,
        const uint8_t * data,
        size_t size,
        std::string & error);

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    uint32_t n_nodes = 0;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

uint32_t llama_sampler_backend_n_nodes(const llama_sampler * sampler);
void llama_sampler_backend_begin(llama_sampler * sampler);

struct llama_sampler * llama_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);
