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

include(CheckLanguage)

function(add_alpaka_executor_binaries)
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

    # Helper to create and finalize a single executable.
    function(_aeb_make_exe suffix def)
        set(tgt "${AEB_PREFIX}_${suffix}")
        add_executable("${tgt}" ${AEB_SOURCES})
        target_compile_definitions("${tgt}" PRIVATE "${def}")
        target_link_libraries("${tgt}" PRIVATE ${AEB_LINK_LIBS})
        alpaka_finalize("${tgt}")

        list(APPEND created_targets "${tgt}")
        list(APPEND created_backends "${suffix}")

        set(created_targets  "${created_targets}"  PARENT_SCOPE)
        set(created_backends "${created_backends}" PARENT_SCOPE)
    endfunction()

    # ---- CPU Serial ----
    if(alpaka_EXEC_CpuSerial)
        _aeb_make_exe("serial" "alpaka_SELECT_CpuSerial")
    endif()

    # ---- TBB ----
    if(alpaka_DEP_TBB)
        if(NOT TARGET TBB::tbb)
            find_package(TBB QUIET)
        endif()
        if(TARGET TBB::tbb)
            set(_saved_link_libs "${AEB_LINK_LIBS}")
            list(APPEND AEB_LINK_LIBS TBB::tbb)
            _aeb_make_exe("tbb" "alpaka_SELECT_TBB")
            set(AEB_LINK_LIBS "${_saved_link_libs}")
        else()
            message(STATUS "TBB not found; skipping ${AEB_PREFIX}_tbb")
        endif()
    endif()

    # ---- OpenMP ----
    if(alpaka_DEP_OMP)
        find_package(OpenMP QUIET)
        if(TARGET OpenMP::OpenMP_CXX)
            set(_saved_link_libs "${AEB_LINK_LIBS}")
            list(APPEND AEB_LINK_LIBS OpenMP::OpenMP_CXX)
            _aeb_make_exe("openmp" "alpaka_SELECT_OMP")
            set(AEB_LINK_LIBS "${_saved_link_libs}")
        else()
            message(STATUS "OpenMP not found; skipping ${AEB_PREFIX}_openmp")
        endif()
    endif()

    # ---- CUDA ----
    if(alpaka_DEP_CUDA)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            set(tgt "${AEB_PREFIX}_cuda")
            add_executable("${tgt}" ${AEB_SOURCES})
            target_link_libraries("${tgt}" PRIVATE ${AEB_LINK_LIBS})
            target_compile_definitions("${tgt}" PRIVATE alpaka_SELECT_CUDA)
            target_compile_options("${tgt}" PRIVATE --expt-relaxed-constexpr)
            set_target_properties("${tgt}" PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

            if(AEB_CUDA_ARCHITECTURES)
                set_target_properties("${tgt}" PROPERTIES CUDA_ARCHITECTURES "${AEB_CUDA_ARCHITECTURES}")
            else()
                set_target_properties("${tgt}" PROPERTIES CUDA_ARCHITECTURES native)
            endif()
            alpaka_finalize("${tgt}")
            list(APPEND created_targets "${tgt}")
            list(APPEND created_backends "cuda")
        else()
            message(STATUS "CUDA compiler not available; skipping ${AEB_PREFIX}_cuda")
        endif()
    endif()

    # ---- HIP ----
    if(alpaka_DEP_HIP)
        set(tgt "${AEB_PREFIX}_hip")
        add_executable("${tgt}" ${AEB_SOURCES})
        target_link_libraries("${tgt}" PRIVATE ${AEB_LINK_LIBS})
        target_compile_definitions("${tgt}" PRIVATE alpaka_SELECT_HIP)
        target_compile_options("${tgt}" PRIVATE --expt-relaxed-constexpr)
        alpaka_finalize("${tgt}")
        list(APPEND created_targets "${tgt}")
        list(APPEND created_backends "hip")
    endif()
        # ---- ONEAPI----
    if(alpaka_DEP_ONEAPI)
        message("TODO configure select Sycl")
#        if()
#        set(tgt "${AEB_PREFIX}_syclCPU")
#        add_executable("${tgt}" ${AEB_SOURCES})
#        target_link_libraries("${tgt}" PRIVATE ${AEB_LINK_LIBS})
#        target_compile_definitions("${tgt}" PRIVATE alpaka_SELECT_SYCL)
#        target_compile_options("${tgt}" PRIVATE --expt-relaxed-constexpr)
#        alpaka_finalize("${tgt}")
#        list(APPEND created_targets "${tgt}")
#        list(APPEND created_backends "syclCPU")
    endif()

    set(${AEB_OUT_TARGETS} "${created_targets}" PARENT_SCOPE)

    # Only return backends if requested
    if(AEB_OUT_BACKENDS)
        list(REMOVE_DUPLICATES created_backends)
        set(${AEB_OUT_BACKENDS} "${created_backends}" PARENT_SCOPE)
    endif()
endfunction()
