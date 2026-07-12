#pragma once

#include <algorithm>
#include <cstddef>

namespace throwdb::detail {

struct RangeTaskBounds {
    std::size_t begin{0};
    std::size_t end{0};
};

struct RangeTaskPartition {
    std::size_t task_count{0};
    std::size_t base_ranges_per_task{0};
    std::size_t larger_task_count{0};

    [[nodiscard]] constexpr RangeTaskBounds bounds(
        const std::size_t task_index) const noexcept {
        const std::size_t begin =
            task_index * base_ranges_per_task
            + std::min(task_index, larger_task_count);
        const std::size_t task_size =
            base_ranges_per_task + (task_index < larger_task_count ? 1 : 0);
        return RangeTaskBounds{begin, begin + task_size};
    }
};

// Produce the smallest balanced task set that both exposes all useful workers
// and keeps every task at or below max_ranges_per_task.
[[nodiscard]] constexpr RangeTaskPartition plan_range_tasks(
    const std::size_t range_count,
    const std::size_t worker_count,
    const std::size_t max_ranges_per_task) noexcept {
    if (range_count == 0 || worker_count == 0 || max_ranges_per_task == 0) {
        return {};
    }

    const std::size_t useful_workers = std::min(range_count, worker_count);
    const std::size_t tasks_required_by_limit =
        range_count / max_ranges_per_task
        + (range_count % max_ranges_per_task != 0 ? 1 : 0);
    const std::size_t task_count =
        std::max(useful_workers, tasks_required_by_limit);
    return RangeTaskPartition{
        task_count,
        range_count / task_count,
        range_count % task_count,
    };
}

}  // namespace throwdb::detail
