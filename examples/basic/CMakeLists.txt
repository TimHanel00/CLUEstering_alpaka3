cmake_minimum_required(VERSION 3.10)
project(BasicCLUEsteringExample LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#find_package(CLUEstering REQUIRED)
#find_package(alpaka REQUIRED)

# include the helper
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
include(AddAlpakaExecutors)

add_alpaka_executor_binaries(
        PREFIX basic
        SOURCES main.cpp
        LINK_LIBS CLUEstering::CLUEstering alpaka::alpaka
        OUT_TARGETS created_binaries
)

message(STATUS "Created targets: ${created_bins}")