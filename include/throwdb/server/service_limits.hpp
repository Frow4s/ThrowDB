#pragma once

#include <cstdint>

namespace throwdb::server {

// Application message and batch bounds shared by server embedders and gRPC.
struct ServiceLimits {
    // 64 MiB accommodates a 10k x 1024 float32 batch while avoiding the
    // previous hundreds-of-MiB implicit per-call allocation allowance.
    std::uint64_t max_request_bytes = 64ULL << 20;
    std::uint64_t max_response_bytes = 64ULL << 20;
    std::uint64_t max_ids_per_request = 4'000'000;
    std::uint32_t max_scan_batch_rows = 1'000'000;
    std::uint32_t max_error_message_bytes = 4096;
};

}  // namespace throwdb::server
