#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "throwdb/storage.hpp"

namespace throwdb {

struct NativeBuilderOptions {
    std::uint32_t dimension = 1024;
    DType dtype = DType::Float16;
    // Target payload bytes per immutable base segment. Headers/ID sidecars are
    // additional. The builder also caps every segment at 2^32 rows.
    std::uint64_t segment_size_bytes = 16ULL << 30;
};

class NativeCollectionBuilder {
public:
    static std::unique_ptr<NativeCollectionBuilder> create(
        const std::filesystem::path& path,
        NativeBuilderOptions options = {});

    ~NativeCollectionBuilder();

    NativeCollectionBuilder(const NativeCollectionBuilder&) = delete;
    NativeCollectionBuilder& operator=(const NativeCollectionBuilder&) = delete;
    NativeCollectionBuilder(NativeCollectionBuilder&&) = delete;
    NativeCollectionBuilder& operator=(NativeCollectionBuilder&&) = delete;

    void append(
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);
    void finish();

    [[nodiscard]] std::uint64_t count() const;
    [[nodiscard]] std::uint32_t dimension() const;
    [[nodiscard]] DType dtype() const;
    [[nodiscard]] std::uint32_t format_version() const;
    [[nodiscard]] std::uint64_t record_size() const;
    [[nodiscard]] bool finished() const;

private:
    class Impl;
    explicit NativeCollectionBuilder(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class NativeCollection {
public:
    static std::unique_ptr<NativeCollection> open(
        const std::filesystem::path& path,
        ReaderOptions reader_options = {},
        bool writable = false);

    ~NativeCollection();

    NativeCollection(const NativeCollection&) = delete;
    NativeCollection& operator=(const NativeCollection&) = delete;
    NativeCollection(NativeCollection&&) = delete;
    NativeCollection& operator=(NativeCollection&&) = delete;

    void get_into(
        std::span<const std::uint64_t> ids,
        std::span<std::byte> output) const;
    [[nodiscard]] std::vector<std::byte> get_bytes(
        std::span<const std::uint64_t> ids) const;

    void insert(
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);
    void update(
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);
    void upsert(
        std::span<const std::uint64_t> ids,
        std::span<const std::byte> embeddings);
    [[nodiscard]] std::vector<std::uint8_t> erase(
        std::span<const std::uint64_t> ids);

    [[nodiscard]] std::uint64_t count() const;
    [[nodiscard]] std::uint32_t dimension() const;
    [[nodiscard]] DType dtype() const;
    [[nodiscard]] std::uint32_t format_version() const;
    [[nodiscard]] std::uint64_t record_size() const;
    [[nodiscard]] bool writable() const;

private:
    class Impl;
    explicit NativeCollection(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace throwdb
