cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

string(
  APPEND
  CMAKE_CXX_FLAGS_DEBUG
  " -DCLUE_DEBUG -D_GLIBCXX_ASSERTIONS -Wall -Wextra -Wpedantic -Wpedantic -Werror"
)
string(
  APPEND
  CMAKE_CXX_FLAGS
  "-O2 -funroll-loops -funsafe-math-optimizations -ftree-vectorize -march=native"
)

# include pybind11 extern subfolder
set(PYBIND11_FINDPYTHON ON)
set(PYBIND11_PYTHON_VERSION ">=3.8")

# create lib directory in CLUEstering folder
file(MAKE_DIRECTORY ../lib)

# Convolutional Kernels
pybind11_add_module(CLUE_Convolutional_Kernels SHARED
                    ${CMAKE_CURRENT_SOURCE_DIR}/binding_kernels.cpp)
target_include_directories(CLUE_Convolutional_Kernels
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(CLUE_Convolutional_Kernels PRIVATE CLUEstering::CLUEstering)
target_compile_definitions(
  CLUE_Convolutional_Kernels PRIVATE alpaka_SELECT_CpuSerial)
set_target_properties(
  CLUE_Convolutional_Kernels
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY
             ${CMAKE_CURRENT_BINARY_DIR}/../../lib/CLUEstering/lib/)
clue_finalize(CLUE_Convolutional_Kernels)
# copy shared library for local testing
add_custom_command(
  TARGET CLUE_Convolutional_Kernels
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/../../lib/CLUEstering/lib/CLUE_Convolutional_Kernels.*
    ${CMAKE_CURRENT_SOURCE_DIR}/../lib/)
if(alpaka_EXEC_CpuSerial)
  add_subdirectory(serial)
endif()
if(alpaka_DEP_TBB)
  add_subdirectory(tbb)
endif()
if(alpaka_DEP_OMP)
  add_subdirectory(openmp)
endif()
if(alpaka_DEP_CUDA)
  add_subdirectory(cuda)
endif()

# GPU HIP
if(alpaka_DEP_HIP)
  add_subdirectory(hip)
endif()
