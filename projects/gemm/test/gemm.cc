#include <cmath>
#include <cstdlib>
#include <iostream>

#include "mlas.h"
#include "timer.h"

// Get max abs diff
inline float GetMaxAbsDiff(float* x, float* y, int n) {
  float max_diff = -1.0f;

  for (int i = 0; i < n; ++i) {
    float diff = std::fabs(x[i] - y[i]);

    if (diff > max_diff) max_diff = diff;
  }

  return max_diff;
}

// get range
inline std::pair<float, float> GetArrayRange(float* x, int n) {
  float xmin = 1e20f;
  float xmax = -1e20f;

  for (int i = 0; i < n; ++i) {
    float a = x[i];

    if (a < xmin) xmin = a;
    if (a > xmax) xmax = a;
  }

  return {xmin, xmax};
}

constexpr int kAlignment = 32;

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " m, n, k" << std::endl;
    return 1;
  }

  int m = std::atoi(argv[1]);
  int n = std::atoi(argv[2]);
  int k = std::atoi(argv[3]);

  // print m, n, k
  std::cout << "m = " << m << std::endl;
  std::cout << "n = " << n << std::endl;
  std::cout << "k = " << k << std::endl;

  float* A = reinterpret_cast<float*>(
      std::aligned_alloc(kAlignment, sizeof(float) * m * k));
  float* B = reinterpret_cast<float*>(
      std::aligned_alloc(kAlignment, sizeof(float) * k * n));
  float* C0 = reinterpret_cast<float*>(
      std::aligned_alloc(kAlignment, sizeof(float) * m * n));
  float* C1 = reinterpret_cast<float*>(
      std::aligned_alloc(kAlignment, sizeof(float) * m * n));

  // init A, B, C
  for (int i = 0; i < m * k; ++i) {
    A[i] = (float)i / (m * k);
  }

  for (int i = 0; i < k * n; ++i) {
    B[i] = (float)i / (k * n);
  }

  size_t packb_size = MlasGemmPackBSize(n, k);

  // malloc packed B
  float* packedB =
      reinterpret_cast<float*>(std::aligned_alloc(kAlignment, packb_size));

  MlasGemmPackB(CblasNoTrans, n, k, B, n, packedB);

  // SGEMM: no pack weight
  {
    float alpha = 1.0f;
    float beta = 0.0f;

    int lda = k;
    int ldb = n;
    int ldc = n;

    MlasGemm(CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C0,
             ldc, nullptr);
  }

  // SGEMM: pre-pack weight
  {
    float alpha = 1.0f;
    float beta = 0.0f;

    int lda = k;
    int ldb = n;
    int ldc = n;

    MlasGemm(CblasNoTrans, m, n, k, alpha, A, lda, packedB, beta, C1, ldc, nullptr);
  }

  auto r0 = GetArrayRange(C0, m * n);
  std::cout << "fp32 no pack range: " << r0.first << ", " << r0.second << std::endl;

  auto r1 = GetArrayRange(C1, m * n);
  std::cout << "fp32    pack range: " << r1.first << ", " << r1.second << std::endl;

  // get max diff
  float max_diff = GetMaxAbsDiff(C0, C1, m * n);
  std::cout << "fp32      max diff: " << max_diff << std::endl;

  // benchmark
  CPUTimer timer;
  const int kRepeats = 10;
  float time_gemm[kRepeats];
  float time_pack[kRepeats];

  // no-pack B version
  for (int i = 0; i < kRepeats; ++i) {
    float alpha = 1.0f;
    float beta = 0.0f;

    int lda = k;
    int ldb = n;
    int ldc = n;

    timer.Start();

    MlasGemm(CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C0,
             ldc, nullptr);
    timer.Stop();

    time_gemm[i] = timer.GetDuration();
  }

  // pack B version
  for (int i = 0; i < kRepeats; ++i) {
    float alpha = 1.0f;
    float beta = 0.0f;

    int lda = k;
    int ldb = n;
    int ldc = n;

    timer.Start();
    MlasGemm(CblasNoTrans, m, n, k, alpha, A, lda, packedB, beta, C1, ldc, nullptr);
    timer.Stop();

    time_pack[i] = timer.GetDuration();
  }

  std::cout << std::endl << "benchmark GEMM (μs)" << std::endl;

  std::cout << "fp32 no pack: ";

  for (int i = 0; i < kRepeats; ++i) {
    std::cout << time_gemm[i] << ", ";
  }
  std::cout << std::endl;

  std::cout << "fp32    pack: ";
  for (int i = 0; i < kRepeats; ++i) {
    std::cout << time_pack[i] << ", ";
  }
  std::cout << std::endl;

  std::free(A);
  std::free(B);
  std::free(packedB);
  std::free(C0);
  std::free(C1);

  return 0;
}
