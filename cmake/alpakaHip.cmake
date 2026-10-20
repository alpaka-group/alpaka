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

check_language(HIP)

if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
endif()

get_property(_alpaka_TARGETS_EXTENDED GLOBAL PROPERTY ALPAKA_TARGETS_EXTENDED)
if(_alpaka_TARGETS_EXTENDED)
    # returns control back to the calling include() scope - it does not stop the cmake configure
    return()
endif()

if(CMAKE_HIP_COMPILER)
    message(STATUS "HIP language is available")
    find_package(hip REQUIRED)

    set(_alpaka_HIP_MIN_VER 6.0)
    set(_alpaka_HIP_MAX_VER 10.0)

    checkcompilercxxsupport(HIP ${alpaka_CXX_STANDARD})

    # construct hip version only with major and minor level
    # cannot use hip_VERSION because of the patch level
    # 6.0 is smaller than 6.0.1234, so _alpaka_HIP_MAX_VER would have to be defined with a large patch level or
    # the next minor level, e.g. 6.1, would have to be used.
    set(_hip_MAJOR_MINOR_VERSION "${hip_VERSION_MAJOR}.${hip_VERSION_MINOR}")

    if(
        ${_hip_MAJOR_MINOR_VERSION} VERSION_LESS ${_alpaka_HIP_MIN_VER}
        OR ${_hip_MAJOR_MINOR_VERSION} VERSION_GREATER ${_alpaka_HIP_MAX_VER}
        # pre-releases
        OR ${_hip_MAJOR_MINOR_VERSION} VERSION_EQUAL "7.13"
        OR ${_hip_MAJOR_MINOR_VERSION} VERSION_EQUAL "7.9"
    )
        message(
            WARNING
            "HIP ${_hip_MAJOR_MINOR_VERSION} is not official supported by alpaka. Supported versions: ${_alpaka_HIP_MIN_VER} - ${_alpaka_HIP_MAX_VER}"
        )
    endif()

    add_library(alpaka_target_hip INTERFACE)
    target_link_libraries(alpaka_target_hip INTERFACE alpaka::host)
    add_library(alpaka::hip ALIAS alpaka_target_hip)

    # let the compiler find the HIP headers also when building host-only code
    target_include_directories(alpaka_target_hip SYSTEM INTERFACE ${hip_INCLUDE_DIR})

    target_link_libraries(alpaka_target_hip INTERFACE "$<$<LINK_LANGUAGE:CXX>:hip::host>")
    alpaka_set_compiler_options(HOST_DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:CXX>:-D__HIP_PLATFORM_AMD__>")

    if(alpaka_DEP_OMP)
        # The CXX and HIP compilers must use compatible OpenMP runtimes.
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            message(
                FATAL_ERROR
                "OpenMP with HIP requires a Clang CXX compiler to avoid incompatible OpenMP runtimes. "
                "Select CMAKE_CXX_COMPILER from the HIP toolchain or set alpaka_DEP_OMP=OFF."
            )
        endif()
        alpaka_set_compiler_options(HOST target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-Xarch_host -fopenmp>")
    endif()

    alpaka_compiler_option(HIP_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps 'CMakeFiles/<targetname>.dir'" OFF)
    if(alpaka_HIP_KEEP_FILES)
        alpaka_set_compiler_options(HOST_DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-save-temps>")
    endif()

    if(alpaka_FAST_MATH STREQUAL ON)
        alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-ffast-math>")
    elseif(alpaka_FAST_MATH STREQUAL OFF)
        alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-fno-fast-math>")
    endif()

    if(alpaka_FTZ STREQUAL ON)
        alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-fgpu-flush-denormals-to-zero>")
    elseif(alpaka_FTZ STREQUAL OFF)
        alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-fno-gpu-flush-denormals-to-zero>")
    endif()

    if(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL ON)
        alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-fgpu-rdc>")
        target_link_options(alpaka_target_hip INTERFACE "$<$<LINK_LANGUAGE:HIP>:SHELL:-fgpu-rdc --hip-link>")
    elseif(alpaka_RELOCATABLE_DEVICE_CODE STREQUAL OFF)
        alpaka_set_compiler_options(DEVICE target alpaka_target_hip "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-fno-gpu-rdc>")
    endif()

    # workaround linker error with ROCm 7.14 due to the change of the ROCM directories, see https://rocm.docs.amd.com/en/latest/about/transition-guide-TheRock.html#paths-and-linking
    # The error is appears with the original AMD container rocm/dev-ubuntu-24.04:7.14.0-full
    # 'error while loading shared libraries: libamdhip64.so.7: cannot open shared object file: No such file or directory'
    if(_hip_MAJOR_MINOR_VERSION VERSION_GREATER_EQUAL "7.14")
        find_library(
            _alpaka_hip_runtime
            NAMES amdhip64
            PATHS ${CMAKE_HIP_IMPLICIT_LINK_DIRECTORIES}
            NO_DEFAULT_PATH
            REQUIRED
        )

        get_filename_component(_alpaka_hip_runtime_dir "${_alpaka_hip_runtime}" DIRECTORY)

        target_link_options(
            alpaka_target_hip
            INTERFACE "$<$<LINK_LANGUAGE:HIP>:SHELL:-Wl,-rpath,${_alpaka_hip_runtime_dir}>"
        )
    endif()

    target_compile_definitions(alpaka_target_hip INTERFACE ALPAKA_CMAKE_TARGET_HIP)
    target_link_libraries(alpaka_target_hip INTERFACE alpaka::headers)

    option(alpaka_HIP_AmdGpu "Enable support for api::hip and deviceKind::amdGpu in examples/benchmarks and tests" ON)
    if(NOT alpaka_HIP_AmdGpu)
        target_compile_definitions(alpaka_target_hip INTERFACE ALPAKA_DISABLE_Hip_AmdGpu)
    endif()

    if(alpaka_COUNT_API_DEPS EQUAL 1)
        target_link_libraries(alpaka INTERFACE alpaka_target_hip)
    endif()
else()
    find_package(hip)

    if(NOT hip_FOUND)
        message(
            WARNING
            "HIP support was requested, but no HIP/ROCm "
            "package was found. HIP backends will be disabled."
        )
    else()
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.26)
            set(_alpaka_CMAKE_CONFIGURE_LOG "${CMAKE_BINARY_DIR}/CMakeFiles/CMakeConfigureLog.yaml")
        else()
            set(_alpaka_CMAKE_CONFIGURE_LOG "${CMAKE_BINARY_DIR}/CMakeFiles/CMakeError.log")
        endif()
        message(
            FATAL_ERROR
            "HIP support was requested, but CMake did not accept "
            "HIP as a language. Check "
            "'${_alpaka_CMAKE_CONFIGURE_LOG}' for further information."
        )
    endif()
endif()
