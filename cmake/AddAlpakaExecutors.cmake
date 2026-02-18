# cmake/AddAlpakaExecutors.cmake
# Usage:
#   add_alpaka_executor_binaries(
#     PREFIX <name>
#     SOURCES <src...>
#     OUT_TARGETS <var>
#     [OUT_BACKENDS <var>]              # <-- optional
#     [LINK_LIBS <libs...>]
#     [CUDA_ARCHITECTURES <archs|native>]
#   )

macro(add_alpaka_executor_binaries)
    set(options)
    set(oneValueArgs PREFIX OUT_TARGETS OUT_BACKENDS CUDA_ARCHITECTURES)
    set(multiValueArgs SOURCES LINK_LIBS)
    cmake_parse_arguments(AEB "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT AEB_PREFIX)
        message(FATAL_ERROR "add_alpaka_executor_binaries: PREFIX is required")
    endif()
    if(NOT AEB_OUT_TARGETS)
        message(FATAL_ERROR "add_alpaka_executor_binaries: OUT_TARGETS is required")
    endif()
    if(NOT AEB_SOURCES)
        message(FATAL_ERROR "add_alpaka_executor_binaries: SOURCES is required")
    endif()

    set(created_targets "")
    set(created_backends "")
    if(NOT CLUE_Examples_GENERATE_MULTIPLE)
        set(tgt "${AEB_PREFIX}_select")
        add_executable("${tgt}" ${AEB_SOURCES})
        target_link_libraries("${tgt}" PUBLIC ${AEB_LINK_LIBS})
        list(APPEND created_targets "${tgt}")
        list(APPEND created_backends "select")
        alpaka_finalize("${tgt}")
        return()
    endif()
    if(alpaka_EXEC_CpuSerial)
        set(tgt "${AEB_PREFIX}_serial")
        add_executable("${tgt}" ${AEB_SOURCES})
        target_link_libraries("${tgt}" PUBLIC ${AEB_LINK_LIBS})
        target_compile_definitions("${tgt}" PRIVATE alpaka_SELECT_CpuSerial)
        list(APPEND created_targets "${tgt}")
        list(APPEND created_backends "serial")
        alpaka_finalize("${tgt}")
    endif()

    # ---- TBB ----
    if(alpaka_DEP_TBB)
        if(NOT TARGET TBB::tbb)
            find_package(TBB QUIET)
        endif()
        if(TARGET TBB::tbb)
            set(tgt "${AEB_PREFIX}_tbb")
            add_executable("${tgt}" ${AEB_SOURCES})
            target_link_libraries("${tgt}" PUBLIC ${AEB_LINK_LIBS})
            target_compile_definitions("${tgt}" PRIVATE alpaka_SELECT_TBB)
            list(APPEND created_targets "${tgt}")
            list(APPEND created_backends "tbb")
            alpaka_finalize("${tgt}")
        else()
            message(STATUS "TBB not found; skipping ${AEB_PREFIX}_tbb")
        endif()
    endif()

    # ---- OpenMP ----
    if(alpaka_DEP_OMP)
        set(tgt "${AEB_PREFIX}_openmp")
        add_executable("${tgt}" ${AEB_SOURCES})
        target_link_libraries("${tgt}" PUBLIC ${AEB_LINK_LIBS})
        target_compile_definitions("${tgt}" PRIVATE alpaka_SELECT_OMP)
        list(APPEND created_targets "${tgt}")
        list(APPEND created_backends "omp")
        alpaka_finalize("${tgt}")
    endif()

    # ---- CUDA ----
    if(alpaka_DEP_CUDA)
        set(tgt "${AEB_PREFIX}_cuda")
        add_executable("${tgt}" ${AEB_SOURCES})
        target_link_libraries("${tgt}" PUBLIC ${AEB_LINK_LIBS})
        target_compile_definitions("${tgt}" PRIVATE alpaka_SELECT_CUDA)
        list(APPEND created_targets "${tgt}")
        list(APPEND created_backends "cuda")
        alpaka_finalize("${tgt}")
    endif()

    # ---- HIP ----
    if(alpaka_DEP_HIP)
        set(tgt "${AEB_PREFIX}_hip")
        add_executable("${tgt}" ${AEB_SOURCES})
        target_link_libraries("${tgt}" PRIVATE ${AEB_LINK_LIBS})
        target_compile_definitions("${tgt}" PRIVATE alpaka_SELECT_HIP)
        list(APPEND created_targets "${tgt}")
        list(APPEND created_backends "hip")
        alpaka_finalize("${tgt}")
    endif()
        # ---- ONEAPI----
    if(alpaka_DEP_ONEAPI)
        message("TODO configure select Sycl")
    endif()

    set(${AEB_OUT_TARGETS} "${created_targets}" PARENT_SCOPE)

    # Only return backends if requested
    if(AEB_OUT_BACKENDS)
        list(REMOVE_DUPLICATES created_backends)
        set(${AEB_OUT_BACKENDS} "${created_backends}" PARENT_SCOPE)
    endif()
endmacro()
