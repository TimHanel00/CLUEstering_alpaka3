pybind11_add_module(CLUE_CPU_TBB SHARED binding_cpu_tbb.cpp)
target_include_directories(CLUE_CPU_TBB
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_compile_definitions(
  CLUE_CPU_TBB PRIVATE alpaka_SELECT_TBB)
target_link_libraries(CLUE_CPU_TBB PUBLIC CLUEstering::CLUEstering)
set_target_properties(CLUE_CPU_TBB
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/CLUEstering/lib
)
alpaka_finalize(CLUE_CPU_TBB)
# copy shared library for local testing
add_custom_command(
  TARGET CLUE_CPU_TBB
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/CLUE_CPU_TBB.*
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/)
