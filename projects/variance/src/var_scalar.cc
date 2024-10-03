#include "var_scalar.h"

namespace scalar {

float GetVarNaive(std::vector<float> v) {
  float mean = 0.0f;
  float var = 0.0f;

  int n = v.size();

  for (int i = 0; i < n; ++i) {
    float x = v[i];

    mean += x;
    var += x * x;
  }

  mean /= n;

  return var / n - mean * mean;
}

float GetVarTwoPass(std::vector<float> v) {
  float mean = 0.0f;

  int n = v.size();

  for (auto x : v) {
    mean += x;
  }
  mean /= n;

  float var = 0.0f;
  for (auto x : v) {
    var += (x - mean) * (x - mean);
  }
  var /= n;

  return var;
}

float GetVarWelford(std::vector<float> v) {
  float a = 0.0f;
  float s = 0.0f;

  int n = v.size();

  for (int i = 0; i < n; ++i) {
    float x = v[i];

    float b = a + (x - a) / (i + 1);

    s += (x - b) * (x - a);

    a = b;
  }

  return s / n;
}

}  // namespace scalar
