cmake_minimum_required(VERSION 3.10)

project(NativeCUDACLUEsteringExample LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT TARGET CLUEstering::CLUEstering)
  find_package(CLUEstering REQUIRED)
endif()

if(NOT TARGET alpaka::alpaka)
  find_package(alpaka REQUIRED)
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  add_executable(native_cuda.out main.cu)
  target_link_libraries(native_cuda.out PRIVATE CLUEstering::CLUEstering
                                                alpaka::alpaka)
  target_compile_definitions(native_cuda.out
                             PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)
  target_compile_options(native_cuda.out PRIVATE --expt-relaxed-constexpr)
  set_target_properties(
    native_cuda.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES
                                                             native)
else()
  message(FATAL_ERROR "CUDA not found. Please install it.")
endif()
