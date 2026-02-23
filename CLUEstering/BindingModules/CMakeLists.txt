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
target_link_libraries(CLUE_Convolutional_Kernels PUBLIC CLUEstering::CLUEstering)
target_compile_definitions(
  CLUE_Convolutional_Kernels PRIVATE alpaka_SELECT_CpuSerial)
set_target_properties(CLUE_Convolutional_Kernels
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/CLUEstering/lib
)
alpaka_finalize(CLUE_Convolutional_Kernels)

# --- CUDA detection ---
find_program(NVCC_EXECUTABLE nvcc)
set(cuda_found FALSE)
if(NVCC_EXECUTABLE)
  set(cuda_found TRUE)
endif()

# --- HIP detection (simple: hipcc on PATH) ---
find_program(HIPCC_EXECUTABLE hipcc HINTS /opt/rocm/bin)
set(hip_found FALSE)
if(HIPCC_EXECUTABLE)
  set(hip_found TRUE)
endif()

# --- Optional deps ---
find_package(TBB QUIET)
find_package(OpenMP QUIET)

## FORCE Alpaka executor toggles (overwrite prior state)
#set(alpaka_EXEC_CpuSerial ON CACHE BOOL "" FORCE)
#
## Add subdirs based on detection
#add_subdirectory(serial)
#set(alpaka_EXEC_CpuSerial OFF CACHE BOOL "" FORCE)
#if(TBB_FOUND)
#  set(alpaka_DEP_TBB ${TBB_FOUND} CACHE BOOL "" FORCE)
#  add_subdirectory(tbb)
#  set(alpaka_DEP_TBB OFF CACHE BOOL "" FORCE)
#
#endif()

#if(OpenMP_CXX_FOUND)
#  set(alpaka_DEP_OMP ${OpenMP_CXX_FOUND} CACHE BOOL "" FORCE)
#  add_subdirectory(openmp)
#  set(alpaka_DEP_OMP OFF CACHE BOOL "" FORCE)
#endif()
#
if(cuda_found)
  set(alpaka_DEP_CUDA ${cuda_found} CACHE BOOL "" FORCE)
  add_subdirectory(cuda)
  set(alpaka_DEP_CUDA OFF CACHE BOOL "" FORCE)
endif()
#
#if(hip_found)
#  set(alpaka_DEP_HIP ${hip_found} CACHE BOOL "" FORCE)
#  add_subdirectory(hip)
#  set(alpaka_DEP_HIP OFF CACHE BOOL "" FORCE)
#endif()