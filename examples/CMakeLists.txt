cmake_minimum_required(VERSION 3.10)
project(BasicCLUEsteringExample LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_alpaka_executor_binaries(
        PREFIX basic
        SOURCES basic/main.cpp
        LINK_LIBS CLUEstering::CLUEstering
        OUT_TARGETS created_binaries
)
foreach(tgt IN LISTS created_binaries)
    clue_finalize("${tgt}")
endforeach()