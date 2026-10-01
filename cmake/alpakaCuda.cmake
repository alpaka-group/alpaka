#
# Copyright 2025 René Widera
# SPDX-License-Identifier: MPL-2.0
#

# This file assumes that the alpaka target is available and alpaka_COUNT_API_DEPS is set
if(NOT TARGET alpaka)
    message(FATAL_ERROR "No alpaka target available.")
endif()

if(NOT DEFINED alpaka_COUNT_API_DEPS)
    message(FATAL_ERROR "internal variable 'alpaka_COUNT_API_DEPS' must be defined.")
endif()

# evaluating CMAKE_CUDA_HOST_COMPILER_ID must be performed before check_language() is called.
# check_language() is removing the variable CMAKE_CUDA_HOST_COMPILER and CMAKE_CUDA_HOST_COMPILER_ID
if(NOT CMAKE_CUDA_HOST_COMPILER_ID)
    set(_alpaka_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER_ID}")
else()
    if(${CMAKE_CUDA_HOST_COMPILER_ID} STREQUAL "GNU")
        set(_alpaka_CUDA_HOST_COMPILER "GNU")
    endif()
endif()

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

get_property(_alpaka_TARGETS_EXTENDED GLOBAL PROPERTY ALPAKA_TARGETS_EXTENDED)
if(_alpaka_TARGETS_EXTENDED)
    # returns control back to the calling include() scope - it does not stop the cmake configure
    return()
endif()

if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA language is available")
    checkcompilercxxsupport(CUDA ${alpaka_CXX_STANDARD})

    add_library(alpaka_target_cuda INTERFACE)
    add_library(alpaka::cuda ALIAS alpaka_target_cuda)
    target_link_libraries(alpaka_target_cuda INTERFACE alpaka::host)
    target_compile_features(alpaka_target_cuda INTERFACE cuda_std_${alpaka_CXX_STANDARD})

    alpaka_compiler_option(CUDA_SHOW_REGISTER "Show kernel registers and create device ASM" DEFAULT)
    alpaka_compiler_option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps 'CMakeFiles/<targetname>.dir'" DEFAULT)

    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>")

        option(alpaka_CUDA_EXPT_EXTENDED_LAMBDA "Enable CUDA extended lambda support " ON)
        if(alpaka_CUDA_EXPT_EXTENDED_LAMBDA)
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--extended-lambda>")
        endif()

        if(alpaka_CUDA_SHOW_REGISTER STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xptxas -v>")
        endif()

        if(alpaka_CUDA_KEEP_FILES STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--keep>")
        endif()

        option(
            alpaka_CUDA_SHOW_CODELINES
            "Show kernel lines in cuda-gdb and cuda-memcheck. If alpaka_CUDA_KEEP_FILES is enabled source code will be inlined in ptx."
            OFF
        )
        if(alpaka_CUDA_SHOW_CODELINES)
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--source-in-ptx -lineinfo>")

            # This is shaky - We currently don't have a way of checking for the host compiler ID.
            # See https://gitlab.kitware.com/cmake/cmake/-/issues/20901
            if(NOT MSVC)
                alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -rdynamic>")
            endif()
            set(alpaka_CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
        endif()

        if(alpaka_FAST_MATH STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--use_fast_math>")
        endif()

        if(alpaka_FTZ STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--ftz=true>")
        elseif(alpaka_FTZ STREQUAL OFF)
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--ftz=false>")
        endif()

        if(alpaka_DEP_OMP)
            if(NOT MSVC)
                alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -fopenmp>")
            else()
                alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /openmp>")
            endif()
        endif()
    elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
        if(alpaka_CUDA_SHOW_REGISTER STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcuda-ptxas=-v>")
        endif()

        if(alpaka_CUDA_KEEP_FILES STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-save-temps>")
        endif()

        if(alpaka_FAST_MATH STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-ffast-math -ffp-contract=fast>")
        endif()

        if(alpaka_FTZ STREQUAL ON)
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-fcuda-flush-denormals-to-zero>")
        endif()

        if(alpaka_DEP_OMP)
            alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-fopenmp>")

            # See https://github.com/alpaka-group/alpaka/issues/1755
            if((${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") AND (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL 13))
                message(STATUS "clang >= 13 detected. Force-setting OpenMP to version 4.5.")
                alpaka_set_compiler_options(DEVICE target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-fopenmp-version=45>")
            endif()
        endif()
    endif()

    target_compile_definitions(alpaka_target_cuda INTERFACE ALPAKA_CMAKE_TARGET_CUDA)

    option(
        alpaka_CUDA_NvidiaGpu
        "Enable support for api::cuda and deviceKind::nvidiaGpu in examples/benchmarks and tests"
        ON
    )
    if(NOT alpaka_CUDA_NvidiaGpu)
        target_compile_definitions(alpaka_target_cuda INTERFACE ALPAKA_DISABLE_Cuda_NvidiaGpu)
    endif()

    if(alpaka_COUNT_API_DEPS EQUAL 1)
        target_link_libraries(alpaka INTERFACE alpaka_target_cuda)
    endif()

    message(STATUS "cuda with host compiler: " ${_alpaka_CUDA_HOST_COMPILER})

    ## GCC compiler flag to show a longer stack for concept diagnostics
    if(${_alpaka_CUDA_HOST_COMPILER} STREQUAL "GNU")
        alpaka_set_compiler_options(HOST target alpaka_target_cuda "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-fconcepts-diagnostics-depth=${alpaka_GCC_CONCEPT_DEPTH}>")
    endif()
else()
    find_package(CUDAToolkit)
    if(NOT CUDAToolkit_FOUND)
        message(
            WARNING
            "CUDA support was requested, but no Cuda Compiler/Toolchain has been found. "
            "CUDA backends will be disabled. "
        )
    else()
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.26)
            set(_alpaka_CMAKE_CONFIGURE_LOG "${CMAKE_BINARY_DIR}/CMakeFiles/CMakeConfigureLog.yaml")
        else()
            set(_alpaka_CMAKE_CONFIGURE_LOG "${CMAKE_BINARY_DIR}/CMakeFiles/CMakeError.log")
        endif()
        message(
            FATAL_ERROR
            "CUDA support was requested, but CMake did not accept "
            "CUDA as a language. This usually means that there is a compatibility problem, "
            "for example an unsupported host compiler. Check "
            "'${_alpaka_CMAKE_CONFIGURE_LOG}' for further information."
        )
    endif()
endif()
