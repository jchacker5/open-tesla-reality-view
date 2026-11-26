#pragma once

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace otrv::detail {

// Join every started worker even if starting a later thread fails. A body exception is delivered
// on the calling thread only after all workers have stopped touching the caller's data.
template <typename Body>
void parallel_for(std::size_t count, unsigned threads, Body&& body) {
  std::atomic<std::size_t> next{0};
  std::atomic<bool> failed{false};
  std::exception_ptr error;
  std::mutex error_mutex;
  const auto work = [&] {
    try {
      while (!failed.load(std::memory_order_relaxed)) {
        const auto i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= count) break;
        body(i);
      }
    } catch (...) {
      std::lock_guard lock(error_mutex);
      if (!error) error = std::current_exception();
      failed.store(true, std::memory_order_relaxed);
    }
  };
  {
    std::vector<std::jthread> pool;
    for (unsigned k = 1; k < std::min<std::size_t>(threads, count); ++k) pool.emplace_back(work);
    work();
  }
  if (error) std::rethrow_exception(error);
}

}  // namespace otrv::detail
