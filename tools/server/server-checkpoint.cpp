#include "server-checkpoint.h"

#include <algorithm>
#include <cstring>
#include <limits>

//
// opaque checkpoint codec
//
// framing:
//   8 bytes   magic = "LLCKPT1\0"
//   u32 LE    format_version
//   u32 LE    header_json_bytes
//   u64 LE    section_count
//   ...       header JSON
//   ...       repeated length-prefixed sections
//
// every reader operation is bounds checked against a hard maximum blob size

namespace {

constexpr char CHECKPOINT_MAGIC[8] = { 'L', 'L', 'C', 'K', 'P', 'T', '1', '\0' };

enum server_checkpoint_section : uint32_t {
    SERVER_CHECKPOINT_SECTION_SEQ_TARGET        = 1,
    SERVER_CHECKPOINT_SECTION_PROMPT_STATE      = 2,
    SERVER_CHECKPOINT_SECTION_ACCEPT_HISTORY    = 3,
    SERVER_CHECKPOINT_SECTION_SAMPLER_APPLY     = 4,
    SERVER_CHECKPOINT_SECTION_GENERATED_TEXT    = 5,
    SERVER_CHECKPOINT_SECTION_SEQ_DRAFT         = 6,
    SERVER_CHECKPOINT_SECTION_SPECULATIVE_STATE = 7,
    SERVER_CHECKPOINT_SECTION_PROMPT_CHECKPOINTS = 8,
};

struct checkpoint_reader {
    const uint8_t * data;
    size_t size;
    size_t pos = 0;
    bool ok = true;

    bool read_bytes(uint8_t * dst, size_t len) {
        if (pos + len > size) {
            ok = false;
            return false;
        }
        std::memcpy(dst, data + pos, len);
        pos += len;
        return true;
    }

    bool read_u32(uint32_t & v) {
        uint8_t buf[4];
        if (!read_bytes(buf, 4)) {
            return false;
        }
        v = (uint32_t) buf[0] | ((uint32_t) buf[1] << 8) | ((uint32_t) buf[2] << 16) | ((uint32_t) buf[3] << 24);
        return true;
    }

    bool read_u64(uint64_t & v) {
        uint8_t buf[8];
        if (!read_bytes(buf, 8)) {
            return false;
        }
        v = 0;
        for (int i = 7; i >= 0; --i) {
            v = (v << 8) | buf[i];
        }
        return true;
    }
};

struct checkpoint_writer {
    std::vector<uint8_t> & out;

    void write_bytes(const uint8_t * data, size_t len) {
        out.insert(out.end(), data, data + len);
    }

    void write_u32(uint32_t v) {
        uint8_t buf[4] = { (uint8_t) (v & 0xff), (uint8_t) ((v >> 8) & 0xff), (uint8_t) ((v >> 16) & 0xff), (uint8_t) ((v >> 24) & 0xff) };
        write_bytes(buf, 4);
    }

    void write_u64(uint64_t v) {
        uint8_t buf[8];
        for (int i = 0; i < 8; ++i) {
            buf[i] = (uint8_t) ((v >> (8 * i)) & 0xff);
        }
        write_bytes(buf, 8);
    }

    void write_section(uint32_t type, const std::vector<uint8_t> & data) {
        write_u32(type);
        write_u32((uint32_t) data.size());
        if (!data.empty()) {
            write_bytes(data.data(), data.size());
        }
    }
};

} // anonymous namespace

size_t server_checkpoint_max_size() {
    // Individual wire sections are u32-sized, but large prompt-checkpoint
    // topology is emitted as repeated chunks. Keep a hard aggregate bound so a
    // corrupt or hostile blob cannot make the server buffer arbitrary amounts.
    constexpr uint64_t limit = 16ull << 30; // 16 GiB
    return (size_t) std::min<uint64_t>(limit, std::numeric_limits<size_t>::max());
}

bool server_checkpoint_encode(
        const server_generation_checkpoint & checkpoint,
        std::vector<uint8_t> & out,
        std::string & error) {
    const std::string header_str = checkpoint.header.dump();

    // Validate all u32 length casts and the final size before allocating or
    // appending potentially multi-gigabyte payloads.
    const uint64_t max_size = (uint64_t) server_checkpoint_max_size();
    if ((uint64_t) header_str.size() > std::numeric_limits<uint32_t>::max()) {
        error = "checkpoint header exceeds u32 framing limit";
        return false;
    }

    const std::string kind = json_value(checkpoint.header, "kind", std::string("generation"));
    if (kind != "generation" && kind != "idle") {
        error = "unknown checkpoint kind";
        return false;
    }

    const bool has_seq_draft = !checkpoint.seq_draft.empty();
    const bool has_spec_state = !checkpoint.speculative_state.empty();
    if (kind == "generation" && has_seq_draft != has_spec_state) {
        error = "generation checkpoint requires both draft sequence and speculative state";
        return false;
    }
    if (kind == "idle" && has_spec_state) {
        error = "idle checkpoint must not contain speculative cross-round state";
        return false;
    }

    std::vector<const std::vector<uint8_t> *> sections = {
        &checkpoint.seq_target,
        &checkpoint.prompt_state,
        &checkpoint.accept_history,
        &checkpoint.sampler_apply_state,
        &checkpoint.generated_text,
    };
    if (has_seq_draft) {
        sections.push_back(&checkpoint.seq_draft);
    }
    if (has_spec_state) {
        sections.push_back(&checkpoint.speculative_state);
    }
    const bool has_prompt_checkpoints = !checkpoint.prompt_checkpoints.empty();

    uint64_t total_size = 8 + 4 + 4 + 8 + (uint64_t) header_str.size();
    for (const auto * section : sections) {
        if ((uint64_t) section->size() > std::numeric_limits<uint32_t>::max()) {
            error = "checkpoint section exceeds u32 framing limit";
            return false;
        }
        if (total_size > max_size - 8 - (uint64_t) section->size()) {
            error = "checkpoint blob exceeds maximum size";
            return false;
        }
        total_size += 8 + (uint64_t) section->size();
    }

    // v1 keeps u32 per-section lengths for backward compatibility.  The new
    // topology section may be several GiB, so split it into repeated type-8
    // chunks and concatenate them while decoding.
    constexpr uint64_t chunk_max = (uint64_t) std::numeric_limits<uint32_t>::max();
    uint64_t prompt_chunks = 0;
    if (has_prompt_checkpoints) {
        prompt_chunks = ((uint64_t) checkpoint.prompt_checkpoints.size() + chunk_max - 1) / chunk_max;
        const uint64_t overhead = 8 * prompt_chunks;
        if (overhead > max_size || (uint64_t) checkpoint.prompt_checkpoints.size() > max_size - overhead ||
                total_size > max_size - overhead - (uint64_t) checkpoint.prompt_checkpoints.size()) {
            error = "checkpoint blob exceeds maximum size";
            return false;
        }
        total_size += overhead + (uint64_t) checkpoint.prompt_checkpoints.size();
    }

    if (total_size > max_size) {
        error = "checkpoint blob exceeds maximum size";
        return false;
    }

    checkpoint_writer w { out };
    out.clear();
    out.reserve((size_t) total_size);

    w.write_bytes((const uint8_t *) CHECKPOINT_MAGIC, 8);
    w.write_u32(checkpoint.version);
    w.write_u32((uint32_t) header_str.size());
    w.write_u64((uint64_t) sections.size() + prompt_chunks);
    w.write_bytes((const uint8_t *) header_str.data(), header_str.size());

    w.write_section(SERVER_CHECKPOINT_SECTION_SEQ_TARGET,        checkpoint.seq_target);
    w.write_section(SERVER_CHECKPOINT_SECTION_PROMPT_STATE,      checkpoint.prompt_state);
    w.write_section(SERVER_CHECKPOINT_SECTION_ACCEPT_HISTORY,    checkpoint.accept_history);
    w.write_section(SERVER_CHECKPOINT_SECTION_SAMPLER_APPLY,     checkpoint.sampler_apply_state);
    w.write_section(SERVER_CHECKPOINT_SECTION_GENERATED_TEXT,    checkpoint.generated_text);
    if (has_seq_draft) {
        w.write_section(SERVER_CHECKPOINT_SECTION_SEQ_DRAFT, checkpoint.seq_draft);
    }
    if (has_spec_state) {
        w.write_section(SERVER_CHECKPOINT_SECTION_SPECULATIVE_STATE, checkpoint.speculative_state);
    }
    if (has_prompt_checkpoints) {
        size_t off = 0;
        while (off < checkpoint.prompt_checkpoints.size()) {
            const size_t n = std::min<size_t>(
                    checkpoint.prompt_checkpoints.size() - off,
                    (size_t) std::numeric_limits<uint32_t>::max());
            w.write_u32(SERVER_CHECKPOINT_SECTION_PROMPT_CHECKPOINTS);
            w.write_u32((uint32_t) n);
            w.write_bytes(checkpoint.prompt_checkpoints.data() + off, n);
            off += n;
        }
    }

    return true;
}

bool server_checkpoint_decode(
        const uint8_t * data,
        size_t size,
        server_generation_checkpoint & out,
        std::string & error) {
    if (size < 8 + 4 + 4 + 8) {
        error = "checkpoint blob too small";
        return false;
    }
    if (size > server_checkpoint_max_size()) {
        error = "checkpoint blob exceeds maximum size";
        return false;
    }

    checkpoint_reader r { data, size };

    uint8_t magic[8];
    if (!r.read_bytes(magic, 8) || std::memcmp(magic, CHECKPOINT_MAGIC, 8) != 0) {
        error = "invalid checkpoint magic";
        return false;
    }

    uint32_t version = 0;
    if (!r.read_u32(version)) {
        error = "malformed checkpoint header";
        return false;
    }
    if (version != SERVER_CHECKPOINT_VERSION) {
        error = "unsupported checkpoint version";
        return false;
    }

    uint32_t header_bytes = 0;
    if (!r.read_u32(header_bytes)) {
        error = "malformed checkpoint header";
        return false;
    }
    if (r.pos + header_bytes > r.size) {
        error = "malformed checkpoint header";
        return false;
    }

    uint64_t section_count = 0;
    if (!r.read_u64(section_count)) {
        error = "malformed checkpoint header";
        return false;
    }
    if (section_count > 16) {
        error = "too many checkpoint sections";
        return false;
    }

    std::string header_str((const char *) r.data + r.pos, header_bytes);
    r.pos += header_bytes;

    try {
        out.header = json::parse(header_str);
    } catch (const std::exception & e) {
        error = std::string("invalid checkpoint header JSON: ") + e.what();
        return false;
    }
    if (!out.header.is_object()) {
        error = "checkpoint header JSON must be an object";
        return false;
    }

    out.version = version;
    out.seq_target.clear();
    out.prompt_state.clear();
    out.accept_history.clear();
    out.sampler_apply_state.clear();
    out.generated_text.clear();
    out.seq_draft.clear();
    out.speculative_state.clear();
    out.prompt_checkpoints.clear();

    // Five base sections are retained for format-v1 compatibility. An idle
    // checkpoint may additionally contain only the draft sequence (6 sections),
    // while a speculative generation checkpoint contains both draft sections.
    // Repeated type-8 sections are chunks of one prompt-checkpoint topology.
    if (section_count < 5 || section_count > 16) {
        error = "checkpoint v1 has an invalid section count";
        return false;
    }

    const std::string kind = json_value(out.header, "kind", std::string("generation"));
    if (kind != "generation" && kind != "idle") {
        error = "unknown checkpoint kind";
        return false;
    }

    uint32_t seen_sections = 0;
    constexpr uint32_t required_sections = 0x1f; // sections 1..5

    for (uint64_t i = 0; i < section_count; ++i) {
        uint32_t type = 0;
        uint32_t len = 0;
        if (!r.read_u32(type) || !r.read_u32(len)) {
            error = "malformed checkpoint section";
            return false;
        }
        if (r.pos + len > r.size) {
            error = "malformed checkpoint section";
            return false;
        }

        uint32_t section_bit = 0;
        switch (type) {
            case SERVER_CHECKPOINT_SECTION_SEQ_TARGET:       section_bit = 1u << 0; break;
            case SERVER_CHECKPOINT_SECTION_PROMPT_STATE:     section_bit = 1u << 1; break;
            case SERVER_CHECKPOINT_SECTION_ACCEPT_HISTORY:   section_bit = 1u << 2; break;
            case SERVER_CHECKPOINT_SECTION_SAMPLER_APPLY:    section_bit = 1u << 3; break;
            case SERVER_CHECKPOINT_SECTION_GENERATED_TEXT:   section_bit = 1u << 4; break;
            case SERVER_CHECKPOINT_SECTION_SEQ_DRAFT:        section_bit = 1u << 5; break;
            case SERVER_CHECKPOINT_SECTION_SPECULATIVE_STATE:section_bit = 1u << 6; break;
            case SERVER_CHECKPOINT_SECTION_PROMPT_CHECKPOINTS: section_bit = 1u << 7; break;
            default:
                error = "unknown checkpoint section";
                return false;
        }
        if ((seen_sections & section_bit) && type != SERVER_CHECKPOINT_SECTION_PROMPT_CHECKPOINTS) {
            error = "duplicate checkpoint section";
            return false;
        }
        seen_sections |= section_bit;

        if (type == SERVER_CHECKPOINT_SECTION_PROMPT_CHECKPOINTS &&
                out.prompt_checkpoints.size() > server_checkpoint_max_size() - len) {
            error = "prompt checkpoint topology exceeds maximum size";
            return false;
        }

        std::vector<uint8_t> section(r.data + r.pos, r.data + r.pos + len);
        r.pos += len;

        switch (type) {
            case SERVER_CHECKPOINT_SECTION_SEQ_TARGET:      out.seq_target        = std::move(section); break;
            case SERVER_CHECKPOINT_SECTION_PROMPT_STATE:    out.prompt_state      = std::move(section); break;
            case SERVER_CHECKPOINT_SECTION_ACCEPT_HISTORY:  out.accept_history    = std::move(section); break;
            case SERVER_CHECKPOINT_SECTION_SAMPLER_APPLY:   out.sampler_apply_state = std::move(section); break;
            case SERVER_CHECKPOINT_SECTION_GENERATED_TEXT:  out.generated_text    = std::move(section); break;
            case SERVER_CHECKPOINT_SECTION_SEQ_DRAFT:       out.seq_draft         = std::move(section); break;
            case SERVER_CHECKPOINT_SECTION_SPECULATIVE_STATE: out.speculative_state = std::move(section); break;
            case SERVER_CHECKPOINT_SECTION_PROMPT_CHECKPOINTS:
                out.prompt_checkpoints.insert(
                        out.prompt_checkpoints.end(), section.begin(), section.end());
                break;
            default: GGML_ABORT("unreachable checkpoint section type");
        }
    }

    if ((seen_sections & required_sections) != required_sections) {
        error = "checkpoint is missing a required section";
        return false;
    }
    const bool has_seq_draft = (seen_sections & (1u << 5)) != 0;
    const bool has_spec_state = (seen_sections & (1u << 6)) != 0;
    if (kind == "generation" && has_seq_draft != has_spec_state) {
        error = "generation checkpoint has incomplete speculative state";
        return false;
    }
    if (kind == "idle" && has_spec_state) {
        error = "idle checkpoint contains speculative cross-round state";
        return false;
    }

    if (r.pos != r.size) {
        error = "trailing bytes in checkpoint blob";
        return false;
    }

    return true;
}
