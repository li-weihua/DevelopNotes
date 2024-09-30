#include "var_sse.h"

#include <immintrin.h>

namespace sse {

constexpr int kPack = 4;

inline float hsum_ps(__m128 v) {
  __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));  // [ C D | A B ]
  __m128 sums = _mm_add_ps(v, shuf);  // [ D+C C+D | B+A A+B ]
  shuf = _mm_movehl_ps(shuf, sums);   // [   C   D | D+C C+D ]
  // shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(0, 1, 2, 3));
  sums = _mm_add_ss(sums, shuf);

  return _mm_cvtss_f32(sums);
}

float GetVarNaive(std::vector<float> v) {
  int n = v.size();

  __m128 pmean = _mm_setzero_ps();
  __m128 pvar = _mm_setzero_ps();

  float* input = v.data();

  for (int i = 0; i < n; i += kPack) {
    __m128 v1 = _mm_loadu_ps(input + i);

    pmean = _mm_add_ps(pmean, v1);
    pvar = _mm_add_ps(_mm_mul_ps(v1, v1), pvar);
  }

  float mean = hsum_ps(pmean);
  float var = hsum_ps(pvar);

  mean = mean / n;
  var = var / n - mean * mean;

  return var;
}

float GetVarWelford(std::vector<float> v) {
  int n = v.size();

  __m128 pa = _mm_setzero_ps();
  __m128 ps = _mm_setzero_ps();

  float* input = v.data();

  for (int i = 0; i < n; i += kPack) {
    __m128 px = _mm_loadu_ps(input + i);

    __m128 pn = _mm_set1_ps((float)(i + 1));

    __m128 pb = _mm_add_ps(pa, _mm_div_ps(_mm_sub_ps(px, pa), pn));

    ps = _mm_add_ps(ps, _mm_mul_ps(_mm_sub_ps(px, pb), _mm_sub_ps(px, pa)));

    pa = pb;
  }

  float var = hsum_ps(ps);

  return var / n;
}

}  // namespace sse
