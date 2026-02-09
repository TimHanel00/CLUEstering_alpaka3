pybind11_add_module(CLUE_CPU_OMP SHARED binding_cpu_omp.cpp)
target_include_directories(CLUE_CPU_OMP
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(CLUE_CPU_OMP PRIVATE CLUEstering::CLUEstering)
target_compile_definitions(
  CLUE_CPU_OMP PRIVATE alpaka_SELECT_OMP)
set_target_properties(
  CLUE_CPU_OMP
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY
             ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/)
clue_finalize(CLUE_CPU_OMP)
# copy shared library for local testing
add_custom_command(
  TARGET CLUE_CPU_OMP
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/CLUEstering/lib/CLUE_CPU_OMP.*
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/)
