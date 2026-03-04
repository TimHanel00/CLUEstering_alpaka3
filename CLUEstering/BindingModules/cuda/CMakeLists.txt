
pybind11_add_module(CLUE_GPU_CUDA SHARED binding_gpu_cuda.cpp)
target_compile_definitions(CLUE_GPU_CUDA PRIVATE alpaka_SELECT_CUDA)
target_include_directories(CLUE_GPU_CUDA
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(CLUE_GPU_CUDA PUBLIC CLUEstering::CLUEstering)

set_target_properties(CLUE_GPU_CUDA
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_SOURCE_DIR}/CLUEstering/lib
)

alpaka_finalize(CLUE_GPU_CUDA)