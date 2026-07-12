#include <cerrno>
#include <cstring>
#include <iomanip>
#include <iostream>

#include <linux/io_uring.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

const char* explanation(const int error_number) {
    switch (error_number) {
        case ENOSYS:
            return "the running kernel does not implement io_uring_setup";
        case EPERM:
        case EACCES:
            return "io_uring is blocked by seccomp, sysctl, or container policy";
        case EMFILE:
        case ENFILE:
            return "the process or system file-descriptor limit was reached";
        default:
            return "io_uring_setup failed for an unspecified environment reason";
    }
}

}  // namespace

int main() {
    io_uring_params parameters{};
    constexpr unsigned int entries = 8;
    const long result = ::syscall(__NR_io_uring_setup, entries, &parameters);
    if (result < 0) {
        const int error_number = errno;
        std::cerr << "io_uring unavailable: errno=" << error_number << " ("
                  << std::strerror(error_number) << ")\n"
                  << explanation(error_number) << '\n';
        return 1;
    }

    const int ring_fd = static_cast<int>(result);
    std::cout << "io_uring available\n"
              << "ring_fd=" << ring_fd << '\n'
              << "sq_entries=" << parameters.sq_entries << '\n'
              << "cq_entries=" << parameters.cq_entries << '\n'
              << "features=0x" << std::hex << parameters.features << std::dec
              << '\n';
    if (::close(ring_fd) != 0) {
        std::cerr << "warning: close(ring_fd) failed: " << std::strerror(errno)
                  << '\n';
        return 2;
    }
    return 0;
}
