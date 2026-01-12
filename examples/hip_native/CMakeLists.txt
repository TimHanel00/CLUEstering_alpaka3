cmake_minimum_required(VERSION 3.10)

project(NativeHIPCLUEsteringExample LANGUAGES CXX HIP)

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
check_language(HIP)
if(CMAKE_HIP_COMPILER)
  add_executable(native_hip.out main.cpp)
  target_link_libraries(native_hip.out PRIVATE CLUEstering::CLUEstering
                                               alpaka::alpaka)
  target_compile_definitions(native_hip.out PRIVATE ALPAKA_ACC_GPU_HIP_ENABLED)
else()
  message(FATAL_ERROR "CUDA not found. Please install it.")
  alpaka_finalize(native_hip.out)
endif()
