#include "file_lock.hpp"

#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace throwdb::detail {
namespace {

[[nodiscard]] int nofollow_flag() noexcept {
#if defined(O_NOFOLLOW)
    return O_NOFOLLOW;
#else
    return 0;
#endif
}

void require_regular_lock_file(
    const int fd, const std::filesystem::path& path) {
    struct stat status {};
    if (::fstat(fd, &status) < 0) {
        throw StorageError(system_error_message("cannot inspect lock file", path));
    }
    if (!S_ISREG(status.st_mode)) {
        throw StorageError(
            "collection lock path is not a regular file: '" + path.string()
            + "'");
    }
}

}  // namespace

AdvisoryLockFile AdvisoryLockFile::create(
    const std::filesystem::path& path) {
    const int fd = ::open(
        path.c_str(),
        O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | nofollow_flag(), 0644);
    if (fd < 0) {
        throw StorageError(system_error_message("cannot create lock file", path));
    }
    FileDescriptor descriptor(fd);
    require_regular_lock_file(descriptor.get(), path);
    return AdvisoryLockFile(path, std::move(descriptor));
}

AdvisoryLockFile AdvisoryLockFile::open_existing(
    const std::filesystem::path& path, const bool writable) {
    // A read-only handle never needs writable lock metadata. A writer uses
    // O_RDWR so exclusive locking does not depend on platform-specific
    // flock behavior for read-only descriptors. O_NONBLOCK prevents a FIFO
    // substituted for a lock file from hanging before the fstat check.
    const int fd = ::open(
        path.c_str(),
        (writable ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_NONBLOCK
            | nofollow_flag());
    if (fd < 0) {
        throw StorageError(system_error_message("cannot open lock file", path));
    }
    FileDescriptor descriptor(fd);
    require_regular_lock_file(descriptor.get(), path);
    return AdvisoryLockFile(path, std::move(descriptor));
}

void AdvisoryLockFile::abandon_after_fork() noexcept {
    locked_ = false;
    fd_.reset();
}

void AdvisoryLockFile::lock(
    const FileLockMode mode, const bool nonblocking) {
    if (locked_) {
        throw StorageError("lock file is already locked: '" + path_.string() + "'");
    }
    int operation = mode == FileLockMode::Shared ? LOCK_SH : LOCK_EX;
    if (nonblocking) {
        operation |= LOCK_NB;
    }
    while (::flock(fd_.get(), operation) < 0) {
        if (errno == EINTR) {
            continue;
        }
        if (nonblocking && (errno == EWOULDBLOCK || errno == EAGAIN)) {
            throw StorageError(
                "another writer already holds the collection lease: '"
                + path_.string() + "'");
        }
        throw StorageError(system_error_message("cannot acquire lock", path_));
    }
    locked_ = true;
}

void AdvisoryLockFile::unlock() {
    if (!locked_) {
        return;
    }
    while (::flock(fd_.get(), LOCK_UN) < 0) {
        if (errno == EINTR) {
            continue;
        }
        throw StorageError(system_error_message("cannot release lock", path_));
    }
    locked_ = false;
}

ScopedAdvisoryLock::ScopedAdvisoryLock(
    AdvisoryLockFile& file,
    const FileLockMode mode,
    const bool nonblocking)
    : file_(&file) {
    file_->lock(mode, nonblocking);
}

ScopedAdvisoryLock::~ScopedAdvisoryLock() {
    if (file_ == nullptr) {
        return;
    }
    try {
        file_->unlock();
    } catch (...) {
        // Destructors cannot report an unlock failure. Closing the descriptor
        // with AdvisoryLockFile still releases the kernel lock.
    }
}

}  // namespace throwdb::detail
