#pragma once

#include "server-common.h"
#include "server-task.h"

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

static constexpr uint32_t SERVER_CHECKPOINT_VERSION = 1;

// opaque generation checkpoint, encoded to/from raw bytes by the codec below
struct server_generation_checkpoint {
    uint32_t version = SERVER_CHECKPOINT_VERSION;
    json header;
    std::vector<uint8_t> seq_target;
    std::vector<uint8_t> prompt_state;
    std::vector<uint8_t> accept_history;
    std::vector<uint8_t> sampler_apply_state;
    std::vector<uint8_t> generated_text;
    std::vector<uint8_t> seq_draft;
    std::vector<uint8_t> speculative_state;
    std::vector<uint8_t> prompt_checkpoints;

    // Runtime-only validated parser state. It is derived from header["parser"]
    // on resume and is intentionally not serialized as an additional section.
    server_checkpoint_parser_state parser_state;
    bool parser_state_valid = false;
};

// shared between the inference loop (safe boundary) and the response reader
// (checkpoint barrier), guarded by mutex
struct server_checkpoint_exchange {
    std::mutex mutex;
    std::condition_variable cv;
    // Set when the checkpoint core is complete and can be encoded. Generation
    // checkpoints become ready only after the parser barrier has also run;
    // idle checkpoints have no parser state and become ready immediately.
    bool ready = false;
    bool parser_ready = false;
    bool failed = false;
    bool cancelled = false;
    bool committed = false;
    std::string error;
    server_checkpoint_parser_state parser;
    // core checkpoint captured at the safe boundary, published for the suspend handler
    std::shared_ptr<server_generation_checkpoint> core;
};

bool server_checkpoint_encode(
        const server_generation_checkpoint & checkpoint,
        std::vector<uint8_t> & out,
        std::string & error);
bool server_checkpoint_decode(
        const uint8_t * data,
        size_t size,
        server_generation_checkpoint & out,
        std::string & error);

// hard checkpoint codec / checkpoint-resume route limit
size_t server_checkpoint_max_size();
