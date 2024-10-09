#pragma once

#include <chrono>

class CPUTimer {
 public:
  CPUTimer() = default;

  CPUTimer(const CPUTimer &) = delete;
  CPUTimer &operator=(const CPUTimer &) = delete;

  // move-only
  CPUTimer(CPUTimer &&) = default;
  CPUTimer &operator=(CPUTimer &&) = default;

  void Start() { start_ = std::chrono::high_resolution_clock::now(); }

  void Stop() { stop_ = std::chrono::high_resolution_clock::now(); }

  // In micro seconds
  float GetDuration() { return std::chrono::duration<float, std::micro>(stop_ - start_).count(); }

 private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::high_resolution_clock::time_point stop_;
};
