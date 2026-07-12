#pragma once

#include <filesystem>

#include "internal.hpp"

namespace throwdb::detail {

enum class FileLockMode {
    Shared,
    Exclusive,
};

class AdvisoryLockFile {
public:
    // Lock inodes are part of the published collection layout. Only the
    // builder may create them; collection opens must never repair or mutate
    // metadata as a side effect, especially in read-only mode.
    [[nodiscard]] static AdvisoryLockFile create(
        const std::filesystem::path& path);
    [[nodiscard]] static AdvisoryLockFile open_existing(
        const std::filesystem::path& path,
        bool writable);

    AdvisoryLockFile() = default;
    ~AdvisoryLockFile() = default;

    AdvisoryLockFile(const AdvisoryLockFile&) = delete;
    AdvisoryLockFile& operator=(const AdvisoryLockFile&) = delete;
    AdvisoryLockFile(AdvisoryLockFile&&) noexcept = default;
    AdvisoryLockFile& operator=(AdvisoryLockFile&&) noexcept = default;

    void lock(FileLockMode mode, bool nonblocking = false);
    void unlock();
    // A child created by fork shares this descriptor's open-file description
    // with its parent. Close the child's reference without LOCK_UN, which
    // would otherwise drop the parent's flock as well.
    void abandon_after_fork() noexcept;

    [[nodiscard]] bool locked() const noexcept { return locked_; }
    [[nodiscard]] int fd() const noexcept { return fd_.get(); }
    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    AdvisoryLockFile(std::filesystem::path path, FileDescriptor fd)
        : path_(std::move(path)), fd_(std::move(fd)) {}

    std::filesystem::path path_;
    FileDescriptor fd_;
    bool locked_{false};
};

class ScopedAdvisoryLock {
public:
    ScopedAdvisoryLock(
        AdvisoryLockFile& file,
        FileLockMode mode,
        bool nonblocking = false);
    ~ScopedAdvisoryLock();

    ScopedAdvisoryLock(const ScopedAdvisoryLock&) = delete;
    ScopedAdvisoryLock& operator=(const ScopedAdvisoryLock&) = delete;

private:
    AdvisoryLockFile* file_;
};

}  // namespace throwdb::detail
