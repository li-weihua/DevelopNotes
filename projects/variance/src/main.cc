#include <iostream>
#include <vector>

#include "cpu_arch.h"
#include "input_array.h"

#if defined(ARCH_ADM64_IX86)
#include "var_sse.h"
#elif defined(ARCH_ARM_ANY)
#include "var_neon.h"
#endif

float GetVarNaive(std::vector<float>);
float GetVarTwoPass(std::vector<float>);
float GetVarWelford(std::vector<float>);

std::vector<float> ReadRowData(const std::string file);

int main() {
  std::vector<float> v(input_array, input_array + kArraySize);

  std::cout << "       naive var = " << GetVarNaive(v) << std::endl;
  std::cout << "    two-pass var = " << GetVarTwoPass(v) << std::endl;
  std::cout << "     welford var = " << GetVarWelford(v) << std::endl;
#if defined(ARCH_ADM64_IX86)
  std::cout << "         sse var = " << sse::GetVarNaive(v) << std::endl;
  std::cout << " welford sse var = " << sse::GetVarWelford(v) << std::endl;
#elif defined(ARCH_ARM_ANY)
  std::cout << "        neon var = " << neon::GetVarNaive(v) << std::endl;
  std::cout << "welford neon var = " << neon::GetVarWelford(v) << std::endl;
#endif

  return 0;
}

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
