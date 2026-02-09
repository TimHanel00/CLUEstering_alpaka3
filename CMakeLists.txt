cmake_minimum_required(VERSION 3.16.0)

project(
  CLUEstering
  LANGUAGES CXX
  VERSION 2.9.0)
find_package(alpaka CONFIG)
if(NOT alpaka_FOUND)
  include(FetchContent)
  FetchContent_Declare(
          alpaka
          GIT_REPOSITORY https://github.com/alpaka-group/alpaka3.git
          GIT_TAG 2d731a38e21aa544c8011306e581f844754187aa
  )
  FetchContent_MakeAvailable(alpaka)

endif()



if(NOT TARGET CLUEstering::CLUEstering)
  # generate version header
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/CLUEstering/version.hpp @ONLY)

  set(CLUE_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/CLUEstering")
  # old alias, to be removed in future versions
  add_library(CLUEstering INTERFACE)
  add_library(CLUEstering::CLUEstering ALIAS CLUEstering)

  target_include_directories(CLUEstering
        INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CLUE_GENERATED_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
  )

  # IMPORTANT: only mention alpaka in the BUILD interface (export-safe)
  target_link_libraries(CLUEstering
        INTERFACE
        $<BUILD_INTERFACE:alpaka::alpaka>)


  install(DIRECTORY include/ DESTINATION include)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CLUEstering/version.hpp
        DESTINATION include/CLUEstering)
  #@TODO Issue: current install/export conflicts with alpaka_finalize()
  #@TODO currently either add alpaka as a additional dependency to your target + alpaka_finalize(),
  #@TODO this way you can export CLUEstering with link $<BUILD_INTERFACE:alpaka::alpaka)
  #@TODO the current version only allows builds in tree (here you only need to link CLUEstering::CLUEstering to your targets)
  #
  ## export old namespace target
  install(TARGETS CLUEstering EXPORT CLUEsteringTargets-clue)
  install(
  EXPORT CLUEsteringTargets-clue
  FILE CLUEsteringTargets-clue.cmake
  NAMESPACE clue::
  DESTINATION lib/cmake/CLUEstering)
  # export new namespace target
  install(TARGETS CLUEstering EXPORT CLUEsteringTargets)
  install(
  EXPORT CLUEsteringTargets
  FILE CLUEsteringTargets.cmake
  NAMESPACE CLUEstering::
  DESTINATION lib/cmake/CLUEstering)
  file(GLOB _cmake_files "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*.cmake")
  foreach(_cmake_file ${_cmake_files})
  include(${_cmake_file})
  endforeach()
  set(CLUEstering_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
  set(CLUEstering_VERSION_MINOR ${PROJECT_VERSION_MINOR})
  set(CLUEstering_VERSION_PATCH ${PROJECT_VERSION_PATCH})

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/CLUEsteringConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)
  configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CLUEsteringConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/CLUEsteringConfig.cmake"
  INSTALL_DESTINATION lib/cmake/CLUEstering)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/CLUEsteringConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/CLUEsteringConfigVersion.cmake"
        DESTINATION lib/cmake/CLUEstering)
endif ()
option(BUILD_PYTHON "Build the Python bindings" OFF)
if(BUILD_PYTHON)
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/extern/pybind11/CMakeLists.txt")
    message(FATAL_ERROR
            "pybind11 submodule is missing.\n"
            "Run:\n"
            "  git submodule update --init --recursive\n"
            "or configure with -DBUILD_PYTHON=OFF.")
  endif()
  add_subdirectory(extern/pybind11)
  add_subdirectory(CLUEstering/BindingModules)
endif()
option(CLUE_Examples "Build CLUE examples" OFF)

if(CLUE_Examples)

  set(CLUEstering_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  option(CLUE_Examples_GENERATE_MULTIPLE "Generates a target for each selected backend (using alpaka cmake DEP flags) - OFF: generates one target for the first valid backend/executor combination (using alpaka cmake flags)" ON)
  add_subdirectory(examples)
endif()
option(CLUE_Testing "Build CLUE tests" OFF)
if(CLUE_Testing)

  set(CLUEstering_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  add_subdirectory(tests)
endif()
