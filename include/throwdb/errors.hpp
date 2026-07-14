#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace throwdb {

class StorageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class InvalidArgumentError : public StorageError {
public:
    using StorageError::StorageError;
};

class CorruptionError : public StorageError {
public:
    using StorageError::StorageError;
};

// Resource/runtime-policy failure while constructing a reader backend, or a
// backend operation unsupported by the current storage target. Collection
// metadata, ordinary I/O failures, and data corruption use distinct exception
// types so an advisory policy may safely try a conservative backend.
class ReaderBackendError : public StorageError {
public:
    using StorageError::StorageError;
};

class MissingIdError : public std::out_of_range {
public:
    explicit MissingIdError(std::uint64_t id);

    [[nodiscard]] std::uint64_t id() const noexcept { return id_; }

private:
    std::uint64_t id_;
};

class DuplicateIdError : public std::invalid_argument {
public:
    explicit DuplicateIdError(std::uint64_t id);

    [[nodiscard]] std::uint64_t id() const noexcept { return id_; }

private:
    std::uint64_t id_;
};

class ExistingIdError : public std::invalid_argument {
public:
    explicit ExistingIdError(std::uint64_t id);

    [[nodiscard]] std::uint64_t id() const noexcept { return id_; }

private:
    std::uint64_t id_;
};

}  // namespace throwdb
