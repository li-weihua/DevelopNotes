#pragma once

#include <vector>

namespace scalar {

float GetVarNaive(std::vector<float>);

float GetVarTwoPass(std::vector<float>);

float GetVarWelford(std::vector<float>);

}  // namespace scalar
