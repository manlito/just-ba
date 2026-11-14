#pragma once

// A small persistent thread pool with deterministic results:
// - parallel_for splits [0, n) into ranges; callers write disjoint outputs.
// - sum() adds fixed-size blocks, then adds the block sums in order, so the result
//   is the same for any thread count.

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace jba::cpu {

class ThreadPool {
 public:
  explicit ThreadPool(int threads) {
    if (threads <= 0) threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    for (int i = 1; i < threads; ++i) workers_.emplace_back([this] { work(); });
    size_ = threads;
  }
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++generation_;
    }
    wake_.notify_all();
    for (auto& w : workers_) w.join();
  }
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return size_; }

  // Calls fn(begin, end) over chunks of [0, n). Blocks until all chunks are done.
  void parallel_for(std::size_t n, const std::function<void(std::size_t, std::size_t)>& fn,
                    std::size_t min_chunk = 256) {
    if (n == 0) return;
    const std::size_t max_chunks = static_cast<std::size_t>(size_) * 4;
    std::size_t chunks = std::min(max_chunks, (n + min_chunk - 1) / min_chunk);
    if (chunks <= 1 || size_ == 1) {
      fn(0, n);
      return;
    }
    const std::size_t step = (n + chunks - 1) / chunks;
    chunks = (n + step - 1) / step;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      fn_ = &fn;
      n_ = n;
      step_ = step;
      chunks_ = chunks;
      next_ = 0;
      done_ = 0;
      ++generation_;
    }
    wake_.notify_all();
    run_chunks();
    std::unique_lock<std::mutex> lock(mutex_);
    finished_.wait(lock, [&] { return done_ == chunks_; });
    fn_ = nullptr;
  }

  // Deterministic sum of f(i) for i in [0, n).
  template <typename F>
  double sum(std::size_t n, F f) {
    constexpr std::size_t kBlock = 1024;
    const std::size_t blocks = (n + kBlock - 1) / kBlock;
    std::vector<double> partial(blocks, 0.0);
    parallel_for(
        blocks,
        [&](std::size_t b0, std::size_t b1) {
          for (std::size_t b = b0; b < b1; ++b) {
            const std::size_t end = std::min(n, (b + 1) * kBlock);
            double s = 0;
            for (std::size_t i = b * kBlock; i < end; ++i) s += f(i);
            partial[b] = s;
          }
        },
        1);
    double total = 0;
    for (double s : partial) total += s;
    return total;
  }

 private:
  void run_chunks() {
    for (;;) {
      std::size_t c;
      const std::function<void(std::size_t, std::size_t)>* fn;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (fn_ == nullptr || next_ >= chunks_) return;
        c = next_++;
        fn = fn_;
      }
      const std::size_t begin = c * step_;
      (*fn)(begin, std::min(n_, begin + step_));
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (++done_ == chunks_) finished_.notify_all();
      }
    }
  }

  void work() {
    std::size_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        wake_.wait(lock, [&] { return generation_ != seen; });
        seen = generation_;
        if (stop_) return;
      }
      run_chunks();
    }
  }

  std::vector<std::thread> workers_;
  int size_ = 1;
  std::mutex mutex_;
  std::condition_variable wake_, finished_;
  const std::function<void(std::size_t, std::size_t)>* fn_ = nullptr;
  std::size_t n_ = 0, step_ = 0, chunks_ = 0, next_ = 0, done_ = 0;
  std::size_t generation_ = 0;
  bool stop_ = false;
};

}  // namespace jba::cpu
