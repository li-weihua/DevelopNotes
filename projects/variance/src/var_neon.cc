#include "var_neon.h"

#include <arm_neon.h>

#include "cpu_arch.h"

namespace neon {

constexpr int kPack = 4;

float GetVarNaive(std::vector<float> v) {
  int n = v.size();

  float32x4_t pmean = vdupq_n_f32(0.0f);
  float32x4_t pvar = vdupq_n_f32(0.0f);

  float* input = v.data();

  for (int i = 0; i < n; i += kPack) {
    float32x4_t v1 = vld1q_f32(input + i);

    pmean = vaddq_f32(pmean, v1);
#if defined(ARCH_ARM64)
    pvar = vfmaq_f32(pvar, v1, v1);
#else
    pvar = vmlaq_f32(pvar, v1, v1);
#endif
  }

#if defined(ARCH_ARM64)
  float mean = vaddvq_f32(pmean);
  float var = vaddvq_f32(pvar);
#else
  float32x2_t sum1 = vget_low_f32(pmean) + vget_high_f32(pmean);
  float32x2_t sum2 = vpadd_f32(sum1, sum1);
  float mean = vget_lane_f32(sum2, 0);

  sum1 = vget_low_f32(pvar) + vget_high_f32(pvar);
  sum2 = vpadd_f32(sum1, sum1);
  float var = vget_lane_f32(sum2, 0);
#endif

  mean = mean / n;
  var = var / n - mean * mean;

  return var;
}

inline float32x4_t vdiv(float32x4_t a, float32x4_t b) {
  float32x4_t recip = vrecpeq_f32(b);

  // Use Newton-Raphson iteration two times to improve precision!
  recip = vmulq_f32(recip, vrecpsq_f32(recip, b));
  recip = vmulq_f32(recip, vrecpsq_f32(recip, b));

  return vmulq_f32(a, recip);
}

float GetVarWelford(std::vector<float> v) {
  int n = v.size();

  float32x4_t pa = vdupq_n_f32(0.0f);
  float32x4_t ps = vdupq_n_f32(0.0f);

  float* input = v.data();

  for (int i = 0; i < n; i += kPack) {
    float32x4_t px = vld1q_f32(input + i);

    float32x4_t pn = vdupq_n_f32((float)(i + 1));

#if defined(ARCH_ARM64)
    float32x4_t pb = vaddq_f32(pa, vdivq_f32(vsubq_f32(px, pa), pn));
#else
    float32x4_t pb = vaddq_f32(pa, vdiv(vsubq_f32(px, pa), pn));
#endif

#if defined(ARCH_ARM64)
    ps = vfmaq_f32(ps, vsubq_f32(px, pb), vsubq_f32(px, pa));
#else
    ps = vmlaq_f32(ps, vsubq_f32(px, pb), vsubq_f32(px, pa));
#endif

    pa = pb;
  }

#if defined(ARCH_ARM64)
  float s = vaddvq_f32(ps);
#else
  float32x2_t sum1 = vget_low_f32(ps) + vget_high_f32(ps);
  float32x2_t sum2 = vpadd_f32(sum1, sum1);
  float s = vget_lane_f32(sum2, 0);
#endif

  return s / n;
}

}  // namespace neon
