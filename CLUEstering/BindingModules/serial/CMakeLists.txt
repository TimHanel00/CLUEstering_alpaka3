pybind11_add_module(CLUE_CPU_Serial SHARED binding_cpu.cpp)
target_include_directories(CLUE_CPU_Serial
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(CLUE_CPU_Serial PUBLIC CLUEstering::CLUEstering)
target_compile_definitions(
  CLUE_CPU_Serial PRIVATE alpaka_SELECT_CpuSerial)
set_target_properties(CLUE_CPU_Serial
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/CLUEstering/lib
)
alpaka_finalize(CLUE_CPU_Serial)