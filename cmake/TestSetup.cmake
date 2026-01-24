
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

string(
        APPEND
        CMAKE_CXX_FLAGS_DEBUG
        " -DCLUE_DEBUG -D_GLIBCXX_ASSERTIONS -O0 -Wall -Wextra -Wpedantic -Wshadow -Wimplicit-fallthrough -Wextra-semi -Wold-style-cast -g -pg -fsanitize=address"
)
set(CMAKE_CXX_FLAGS_RELEASE
        " -O2 -funroll-loops -funsafe-math-optimizations -ftree-vectorize -march=native"
)

include(FetchContent)

if(NOT doctest_FOUND)
    FetchContent_Declare(
            doctest
            GIT_REPOSITORY https://github.com/doctest/doctest.git
            GIT_TAG dev)
    FetchContent_MakeAvailable(doctest)
endif()
if(NOT fmt_FOUND)
    FetchContent_Declare(
            fmt
            GIT_REPOSITORY https://github.com/fmtlib/fmt.git
            GIT_TAG 10.2.1 # or another stable version
    )
    FetchContent_MakeAvailable(fmt)
endif()

option(COVERAGE "Enable code coverage" OFF)

enable_testing()

add_subdirectory(cpu)

include(CheckLanguage)

execute_process(
        COMMAND nvidia-smi --query-gpu=name --format=csv,noheader
        OUTPUT_VARIABLE GPU_LIST
        ERROR_QUIET
        RESULT_VARIABLE SMI_STATUS)

if(SMI_STATUS EQUAL 0 AND NOT GPU_LIST STREQUAL "")
    set(NVIDIA_GPU_PRESENT TRUE)
else()
    set(NVIDIA_GPU_PRESENT FALSE)
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER AND NVIDIA_GPU_PRESENT)
    add_subdirectory(cuda)
endif()

execute_process(
        COMMAND rocm-smi --showproductname
        OUTPUT_VARIABLE AMD_GPU_LIST
        ERROR_QUIET
        RESULT_VARIABLE SMI_STATUS)

if(SMI_STATUS EQUAL 0 AND NOT AMD_GPU_LIST STREQUAL "")
    set(AMD_GPU_PRESENT TRUE)
else()
    set(AMD_GPU_PRESENT FALSE)
endif()

set(_sycl_search_dirs ${SYCL_ROOT_DIR} /usr/lib /usr/local/lib
        /opt/intel/oneapi/compiler/latest/linux)
find_program(
        SYCL_COMPILER
        NAMES icpx
        HINTS ${_sycl_search_dirs}
        PATH_SUFFIXES bin)
find_path(
        SYCL_INCLUDE_DIR
        NAMES sycl/sycl.hpp
        HINTS ${_sycl_search_dirs}
        PATH_SUFFIXES include)
find_path(
        SYCL_LIB_DIR
        NAMES libsycl.so
        HINTS ${_sycl_search_dirs}
        PATH_SUFFIXES lib)
