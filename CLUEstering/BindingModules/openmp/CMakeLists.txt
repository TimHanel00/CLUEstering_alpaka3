pybind11_add_module(CLUE_CPU_OMP SHARED binding_cpu_omp.cpp)
target_include_directories(CLUE_CPU_OMP
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(CLUE_CPU_OMP PUBLIC CLUEstering::CLUEstering)
target_compile_definitions(
  CLUE_CPU_OMP PRIVATE alpaka_SELECT_OMP)
set_target_properties(CLUE_CPU_OMP
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/CLUEstering/lib
)
alpaka_finalize(CLUE_CPU_OMP)
