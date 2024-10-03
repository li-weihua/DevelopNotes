#include <iostream>
#include <vector>

#include "cpu_arch.h"
#include "input_array.h"
#include "var_scalar.h"

#if defined(ARCH_ADM64_IX86)
#include "var_sse.h"
#elif defined(ARCH_ARM_ANY)
#include "var_neon.h"
#endif

int main() {
  std::vector<float> v(input_array, input_array + kArraySize);

  std::cout << "       naive var = " << scalar::GetVarNaive(v) << std::endl;
  std::cout << "    two-pass var = " << scalar::GetVarTwoPass(v) << std::endl;
  std::cout << "     welford var = " << scalar::GetVarWelford(v) << std::endl;
#if defined(ARCH_ADM64_IX86)
  std::cout << "         sse var = " << sse::GetVarNaive(v) << std::endl;
  std::cout << " welford sse var = " << sse::GetVarWelford(v) << std::endl;
#elif defined(ARCH_ARM_ANY)
  std::cout << "        neon var = " << neon::GetVarNaive(v) << std::endl;
  std::cout << "welford neon var = " << neon::GetVarWelford(v) << std::endl;
#endif

  return 0;
}
