#pragma once

#include <chrono>

namespace airsteady {
namespace yolo {

class ScopedTimer {
 public:
  explicit ScopedTimer(double* out_ms)
      : out_ms_(out_ms),
        t0_(std::chrono::high_resolution_clock::now()) {}

  ~ScopedTimer() {
    if (!out_ms_) return;
    const auto t1 = std::chrono::high_resolution_clock::now();
    *out_ms_ = std::chrono::duration<double, std::milli>(t1 - t0_).count();
  }

 private:
  double* out_ms_ = nullptr;
  std::chrono::high_resolution_clock::time_point t0_;
};

}  // namespace yolo
}  // namespace airsteady
