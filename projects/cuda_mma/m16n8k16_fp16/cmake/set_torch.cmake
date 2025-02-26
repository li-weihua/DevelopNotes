set(PYTHON_PATH "python" CACHE STRING "Python path")
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import torch; print(torch.__version__,end='');"
                RESULT_VARIABLE _PYTHON_SUCCESS
                OUTPUT_VARIABLE TORCH_VERSION)
if(TORCH_VERSION VERSION_LESS "1.5.0")
  message(FATAL_ERROR "PyTorch >= 1.5.0 is needed for TorchScript mode.")
endif()

execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
      RESULT_VARIABLE _PYTHON_SUCCESS
      OUTPUT_VARIABLE TORCH_DIR)

if (NOT _PYTHON_SUCCESS MATCHES 0)
  message(FATAL_ERROR "Torch config Error.")
endif()

list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)

# cxx 11 abi
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; from distutils import sysconfig;
print(sysconfig.get_python_inc());"
                RESULT_VARIABLE _PYTHON_SUCCESS
                OUTPUT_VARIABLE PY_INCLUDE_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
  message(FATAL_ERROR "Python config Error.")
endif()

list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import torch;
print(torch._C._GLIBCXX_USE_CXX11_ABI,end='');"
                RESULT_VARIABLE _PYTHON_SUCCESS
                OUTPUT_VARIABLE USE_CXX11_ABI)
message("-- USE_CXX11_ABI=${USE_CXX11_ABI}")
if (USE_CXX11_ABI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()
