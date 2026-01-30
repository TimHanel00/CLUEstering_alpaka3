cmake_minimum_required(VERSION 3.16)
project(CLUE_Tests LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake")
include(TestSetup)

#find_package(CLUEstering REQUIRED)
#find_package(alpaka REQUIRED)

file(GLOB sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

foreach(source IN LISTS sources)
  get_filename_component(test_name "${source}" NAME_WE)

  add_executable("${test_name}" "${source}")

  target_include_directories("${test_name}" PRIVATE
          "${CMAKE_SOURCE_DIR}/../include"
          "${doctest_SOURCE_DIR}/doctest"
  )

  target_link_libraries("${test_name}" PRIVATE
          CLUEstering::CLUEstering
          fmt::fmt
  )

  target_compile_definitions("${test_name}" PRIVATE
          CLUE_ENABLE_CACHING_ALLOCATOR
          TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data"
  )
  clue_finalize(${test_name})

  add_test(NAME "${test_name}" COMMAND "${test_name}")

  if(COVERAGE)
    target_compile_options("${test_name}" PRIVATE -O0 -g --coverage)
    target_link_options("${test_name}" PRIVATE --coverage)
    target_compile_definitions("${test_name}" PRIVATE COVERAGE)
  endif()
endforeach()
