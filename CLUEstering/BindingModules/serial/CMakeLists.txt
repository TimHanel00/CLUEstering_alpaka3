pybind11_add_module(CLUE_CPU_Serial SHARED binding_cpu.cpp)
target_include_directories(CLUE_CPU_Serial
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(CLUE_CPU_Serial PUBLIC CLUEstering::CLUEstering)
target_compile_definitions(
  CLUE_CPU_Serial PRIVATE alpaka_SELECT_CpuSerial)
set_target_properties(CLUE_CPU_Serial
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/CLUEstering/lib
)
alpaka_finalize(CLUE_CPU_Serial)
# copy shared library for local testing
add_custom_command(
  TARGET CLUE_CPU_Serial
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/CLUE_CPU_Serial.*
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/)
