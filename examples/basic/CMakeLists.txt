cmake_minimum_required(VERSION 3.10)

project(BasicCLUEsteringExample)

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
if(alpaka_EXEC_CpuSerial)
  add_executable(serial.out main.cpp)
  target_compile_definitions(serial.out
                             PRIVATE alpaka_EXEC_CpuSerial)
  target_link_libraries(serial.out PRIVATE CLUEstering::CLUEstering
                                           alpaka::alpaka)
  alpaka_finalize(serial.out)

endif()

if(alpaka_DEP_TBB)

  if(NOT TARGET TBB::tbb)
    find_package(TBB QUIET)
    message(" TBB not found. Please install it if you want to use the tbb executor!")
  else()
    add_executable(tbb.out main.cpp)
    target_compile_definitions(tbb.out PRIVATE alpaka_DEP_TBB)
    target_link_libraries(tbb.out PRIVATE CLUEstering::CLUEstering alpaka::alpaka
            TBB::tbb)
    alpaka_finalize(tbb.out)
  endif()

endif()

if(alpaka_DEP_OMP)
  find_package(OpenMP REQUIRED)
  if(NOT TARGET OpenMP::OpenMP_CXX)
    message(FATAL_ERROR "OpenMP not found. Please install it.")
  endif()
  add_executable(openmp.out main.cpp)
  target_compile_definitions(openmp.out
                             PRIVATE alpaka_DEP_OMP)
  target_link_libraries(openmp.out PRIVATE CLUEstering::CLUEstering
                                           alpaka::alpaka OpenMP::OpenMP_CXX)
  alpaka_finalize(openmp.out)
endif()

if(alpaka_DEP_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(cuda.out main.cpp)
    target_link_libraries(cuda.out PRIVATE CLUEstering::CLUEstering
                                           alpaka::alpaka)
    target_compile_definitions(cuda.out PRIVATE alpaka_DEP_CUDA)
    target_compile_options(cuda.out PRIVATE --expt-relaxed-constexpr)
    set_target_properties(cuda.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                              CUDA_ARCHITECTURES native)
    alpaka_finalize(cuda.out)
  else()
    message(FATAL_ERROR "CUDA not found. Please install it.")
  endif()
endif()
