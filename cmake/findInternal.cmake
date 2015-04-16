################################################################################
# Copyright 2015 Benjamin Worpitz
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
# RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE
# USE OR PERFORMANCE OF THIS SOFTWARE.
################################################################################

################################################################################
# Required cmake version.
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

################################################################################
# alpaka.
################################################################################

UNSET(alpaka_DEFINITIONS)
UNSET(alpaka_FOUND)
UNSET(alpaka_INCLUDE_DIRS)
UNSET(alpaka_LIBRARIES)
UNSET(alpaka_VERSION)

UNSET(ALPAKA_FOUND_INTERNAL)

IF(ALPAKA_ROOT)
    #-------------------------------------------------------------------------------
    # Set found to true initially and set it on false if a required dependency is missing.
    #-------------------------------------------------------------------------------
    SET(ALPAKA_FOUND_INTERNAL TRUE)

    #-------------------------------------------------------------------------------
    # Common.
    #-------------------------------------------------------------------------------
    # Set helper paths to find libraries and packages.
    #SET(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")

    # Add find modules.
    LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}/cmake/modules/")

    # Add common functions.
    SET(ALPAKA_COMMON_FILE "${ALPAKA_ROOT}/cmake/common.cmake")
    INCLUDE("${ALPAKA_COMMON_FILE}")

    #-------------------------------------------------------------------------------
    # Options.
    #-------------------------------------------------------------------------------
    OPTION(ALPAKA_THREADS_ENABLE "Enable the threads accelerator" ON)
    OPTION(ALPAKA_FIBERS_ENABLE "Enable the fibers accelerator" ON)
    OPTION(ALPAKA_OPENMP2_ENABLE "Enable the OpenMP accelerator" ON)
    OPTION(ALPAKA_CUDA_ENABLE "Enable the CUDA accelerator" ON)

    # Drop-down combo box in cmake-gui.
    SET(ALPAKA_DEBUG "0" CACHE STRING "Debug level")
    SET_PROPERTY(CACHE ALPAKA_DEBUG PROPERTY STRINGS "0;1;2")

    #-------------------------------------------------------------------------------
    # Find Boost.
    #-------------------------------------------------------------------------------
    IF(ALPAKA_FIBERS_ENABLE)
        FIND_PACKAGE(Boost COMPONENTS fiber coroutine context system thread atomic chrono date_time regex)
        IF(NOT Boost_FIBER_FOUND)
            MESSAGE(WARNING "Optional alpaka dependency Boost fiber could not be found! Fibers accelerator disabled!")
            SET(ALPAKA_FIBERS_ENABLE OFF CACHE BOOL "Enable the Fibers accelerator" FORCE)
            FIND_PACKAGE(Boost)
        ENDIF()

    ELSE()
        FIND_PACKAGE(Boost)
    ENDIF()

    IF(NOT Boost_FOUND)
        MESSAGE(WARNING "Required alpaka dependency Boost could not be found!")
        SET(ALPAKA_FOUND_INTERNAL FALSE)

    ELSE()
        LIST(APPEND alpaka_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})

        #LIST(FIND "${Boost_LIBRARIES}" "optimized" ALPAKA_BOOST_LIBRARY_LIST_OPTIMIZED_ATTRIBUTE_INDEX)
        #IF(NOT ALPAKA_BOOST_LIBRARY_LIST_OPTIMIZED_ATTRIBUTE_INDEX EQUAL -1)
        #    list_add_prefix("general;" Boost_LIBRARIES)
        #ENDIF()
        LIST(APPEND alpaka_LIBRARIES ${Boost_LIBRARIES})
    ENDIF()

    #-------------------------------------------------------------------------------
    # Find OpenMP.
    #-------------------------------------------------------------------------------
    IF(ALPAKA_OPENMP2_ENABLE)
        FIND_PACKAGE(OpenMP)
        IF(NOT OPENMP_FOUND)
            MESSAGE(WARNING "Optional alpaka dependency OpenMP could not be found! OpenMP accelerator disabled!")
            SET(ALPAKA_OPENMP2_ENABLE OFF CACHE BOOL "Enable the OpenMP accelerator" FORCE)

        ELSE()
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        ENDIF()
    ENDIF()

    #-------------------------------------------------------------------------------
    # Find CUDA.
    #-------------------------------------------------------------------------------
    IF(ALPAKA_CUDA_ENABLE)

        IF(NOT DEFINED ALPAKA_CUDA_VERSION)
            SET(ALPAKA_CUDA_VERSION 7.0)
        ENDIF()

        IF(ALPAKA_CUDA_VERSION VERSION_LESS 7.0)
            MESSAGE(WARNING "CUDA Toolkit < 7.0 is not supported! CUDA accelerator disabled!")
            SET(ALPAKA_CUDA_ENABLE OFF CACHE BOOL "Enable the CUDA accelerator" FORCE)

        ELSE()
            FIND_PACKAGE(CUDA "${ALPAKA_CUDA_VERSION}")
            IF(NOT CUDA_FOUND)
                MESSAGE(WARNING "Optional alpaka dependency CUDA could not be found! CUDA accelerator disabled!")
                SET(ALPAKA_CUDA_ENABLE OFF CACHE BOOL "Enable the CUDA accelerator" FORCE)

            ELSE()
                #SET(CUDA_VERBOSE_BUILD ON)
                SET(CUDA_PROPAGATE_HOST_FLAGS ON)

                SET(ALPAKA_CUDA_ARCH sm_20 CACHE STRING "Set GPU architecture")
                STRING(COMPARE EQUAL "${ALPAKA_CUDA_ARCH}" "sm_10" IS_CUDA_ARCH_UNSUPPORTED)
                STRING(COMPARE EQUAL "${ALPAKA_CUDA_ARCH}" "sm_11" IS_CUDA_ARCH_UNSUPPORTED)
                STRING(COMPARE EQUAL "${ALPAKA_CUDA_ARCH}" "sm_12" IS_CUDA_ARCH_UNSUPPORTED)
                STRING(COMPARE EQUAL "${ALPAKA_CUDA_ARCH}" "sm_13" IS_CUDA_ARCH_UNSUPPORTED)

                IF(IS_CUDA_ARCH_UNSUPPORTED)
                    MESSAGE(WARNING "Unsupported CUDA architecture ${ALPAKA_CUDA_ARCH} specified. SM 2.0 or higher is required for CUDA 7.0. Using sm_20 instead.")
                    SET(ALPAKA_CUDA_ARCH sm_20 CACHE STRING "Set GPU architecture" FORCE)
                ENDIF(IS_CUDA_ARCH_UNSUPPORTED)

                LIST(APPEND CUDA_NVCC_FLAGS "-arch=${ALPAKA_CUDA_ARCH}")

                IF(NOT MSVC)
                    LIST(APPEND CUDA_NVCC_FLAGS "-std=c++11")
                    SET(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
                ENDIF()

                IF(CMAKE_BUILD_TYPE MATCHES "Debug")
                    LIST(APPEND CUDA_NVCC_FLAGS "-g" "-G")
                ENDIF()

                OPTION(ALPAKA_CUDA_FAST_MATH "Enable fast-math" ON)
                IF(ALPAKA_CUDA_FAST_MATH)
                    LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
                ENDIF()

                OPTION(ALPAKA_CUDA_FAST_MATH "Set flush to zero for GPU" OFF)
                IF(ALPAKA_CUDA_FAST_MATH)
                    LIST(APPEND CUDA_NVCC_FLAGS "--ftz=true")
                ELSE()
                    LIST(APPEND CUDA_NVCC_FLAGS "--ftz=false")
                ENDIF()

                OPTION(ALPAKA_CUDA_SHOW_REGISTER "Show kernel registers and create PTX" OFF)
                IF(ALPAKA_CUDA_SHOW_REGISTER)
                    LIST(APPEND CUDA_NVCC_FLAGS "-Xptxas=-v")
                ENDIF()

                OPTION(ALPAKA_CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps (folder: nvcc_tmp)" OFF)
                IF(ALPAKA_CUDA_KEEP_FILES)
                    MAKE_DIRECTORY("${PROJECT_BINARY_DIR}/nvcc_tmp")
                    LIST(APPEND CUDA_NVCC_FLAGS "--keep" "--keep-dir" "${PROJECT_BINARY_DIR}/nvcc_tmp")
                ENDIF()

                OPTION(ALPAKA_CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)
                IF(ALPAKA_CUDA_SHOW_CODELINES)
                    LIST(APPEND CUDA_NVCC_FLAGS "--source-in-ptx" "-lineinfo")
                    IF(NOT MSVC)
                        LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-rdynamic")
                    ENDIF()
                    SET(ALPAKA_CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
                ENDIF()

                IF(${CUDA_CUDART_LIBRARY})
                    LIST(APPEND alpaka_LIBRARIES "general" ${CUDA_CUDART_LIBRARY})
                ENDIF()
            ENDIF()
        ENDIF()
    ENDIF()

    #-------------------------------------------------------------------------------
    # Find alpaka version.
    #-------------------------------------------------------------------------------
    FILE(STRINGS "${ALPAKA_ROOT}/include/alpaka/alpaka.hpp"
         ALPAKA_VERSION_DEFINE REGEX "#define ALPAKA_VERSION BOOST_VERSION_NUMBER")

    STRING(REGEX REPLACE "[^0-9]*([0-9]+), ([0-9]+), ([0-9]+).*" "\\1" ALPAKA_VER_MAJOR "${ALPAKA_VERSION_DEFINE}")
    STRING(REGEX REPLACE "[^0-9]*([0-9]+), ([0-9]+), ([0-9]+).*" "\\2" ALPAKA_VER_MINOR "${ALPAKA_VERSION_DEFINE}")
    STRING(REGEX REPLACE "[^0-9]*([0-9]+), ([0-9]+), ([0-9]+).*" "\\3" ALPAKA_VER_PATCH "${ALPAKA_VERSION_DEFINE}")

    SET(alpaka_VERSION "${ALPAKA_VER_MAJOR}.${ALPAKA_VER_MINOR}.${ALPAKA_VER_PATCH}")

    #-------------------------------------------------------------------------------
    # Compiler settings.
    #-------------------------------------------------------------------------------
    IF(MSVC)
        # Empty append to define it if it does not already exist.
        LIST(APPEND ALPAKA_COMPILE_OPTIONS)
    ELSE()
        # Select C++ standard version.
        LIST(APPEND ALPAKA_COMPILE_OPTIONS "-std=c++11")

        # Add linker options.
        IF(ALPAKA_THREADS_ENABLE)
            LIST(APPEND alpaka_LIBRARIES "general;pthread")
        ENDIF()
        # librt: undefined reference to `clock_gettime'
        LIST(APPEND alpaka_LIBRARIES "general;rt")
    ENDIF()

    #-------------------------------------------------------------------------------
    # alpaka.
    #-------------------------------------------------------------------------------
    SET(alpaka_INCLUDE_DIR "${ALPAKA_ROOT}include/")
    LIST(APPEND alpaka_INCLUDE_DIRS "${alpaka_INCLUDE_DIR}")
    SET(alpaka_LIBRARY)
    LIST(APPEND alpaka_LIBRARIES "${alpaka_LIBRARY}")

    MARK_AS_ADVANCED(
        alpaka_INCLUDE_DIR
        alpaka_LIBRARY)

    IF(ALPAKA_THREADS_ENABLE)
        LIST(APPEND alpaka_DEFINITIONS "ALPAKA_THREADS_ENABLED")
        MESSAGE(STATUS ALPAKA_THREADS_ENABLED)
    ENDIF()
    IF(ALPAKA_FIBERS_ENABLE)
        LIST(APPEND alpaka_DEFINITIONS "ALPAKA_FIBERS_ENABLED")
        MESSAGE(STATUS ALPAKA_FIBERS_ENABLED)
    ENDIF()
    IF(ALPAKA_OPENMP2_ENABLE)
        LIST(APPEND alpaka_DEFINITIONS "ALPAKA_OPENMP2_ENABLED")
        MESSAGE(STATUS ALPAKA_OPENMP2_ENABLED)
    ENDIF()
    IF(ALPAKA_CUDA_ENABLE)
        LIST(APPEND alpaka_DEFINITIONS "ALPAKA_CUDA_ENABLED")
        MESSAGE(STATUS ALPAKA_CUDA_ENABLED)
    ENDIF()

    IF("${ALPAKA_DEBUG}" GREATER 0)
        LIST(APPEND alpaka_DEFINITIONS "ALPAKA_DEBUG=${ALPAKA_DEBUG}")
    ENDIF()

    IF(ALPAKA_INTEGRATION_TEST)
        LIST(APPEND alpaka_DEFINITIONS "ALPAKA_INTEGRATION_TEST")
    ENDIF()

    #-------------------------------------------------------------------------------
    # Target.
    #-------------------------------------------------------------------------------
    IF(NOT TARGET "alpaka")
        SET(ALPAKA_SUFFIXED_INCLUDE_DIR "${alpaka_INCLUDE_DIR}alpaka/")

        # Add all the include files in all recursive subdirectories and group them accordingly.
        append_recursive_files_add_to_src_group("${ALPAKA_SUFFIXED_INCLUDE_DIR}" "${ALPAKA_SUFFIXED_INCLUDE_DIR}" "hpp" "ALPAKA_HEADER_FILES_ALL")

        # Add all the source files in all recursive subdirectories and group them accordingly.
        append_recursive_files_add_to_src_group("${ALPAKA_SUFFIXED_INCLUDE_DIR}" "${ALPAKA_SUFFIXED_INCLUDE_DIR}" "cpp" "ALPAKA_SOURCE_FILES_ALL")

        SET(ALPAKA_CMAKE_FILES_ALL "${CMAKE_PARENT_LIST_FILE}" "${ALPAKA_ROOT}/cmake/findInternal.cmake" "${ALPAKA_COMMON_FILE}")

        ADD_LIBRARY(
            alpaka
            ${ALPAKA_HEADER_FILES_ALL} ${ALPAKA_SOURCE_FILES_ALL} ${ALPAKA_CMAKE_FILES_ALL})

        # Compile options.
        SET("ALPAKA_COMPILE_OPTIONS_COPY" "${ALPAKA_COMPILE_OPTIONS}")
        list_add_prefix("PUBLIC;" ALPAKA_COMPILE_OPTIONS_COPY)
        IF("${ALPAKA_DEBUG}" GREATER 0)
            MESSAGE(STATUS "ALPAKA_COMPILE_OPTIONS: ${ALPAKA_COMPILE_OPTIONS_COPY}")
        ENDIF()
        LIST(
            LENGTH
            "ALPAKA_COMPILE_OPTIONS_COPY"
            "ALPAKA_COMPILE_OPTIONS_COPY_LENGTH")
        IF("${ALPAKA_COMPILE_OPTIONS_COPY_LENGTH}")
            TARGET_COMPILE_OPTIONS(
                "alpaka"
                ${ALPAKA_COMPILE_OPTIONS_COPY})
        ENDIF()

        # Compile definitions.
        SET("ALPAKA_COMPILE_DEFINITIONS_COPY" "${alpaka_DEFINITIONS}")
        list_add_prefix("PUBLIC;" "ALPAKA_COMPILE_DEFINITIONS_COPY")
        IF("${ALPAKA_DEBUG}" GREATER 0)
            MESSAGE(STATUS "alpaka_DEFINITIONS: ${ALPAKA_COMPILE_DEFINITIONS_COPY}")
        ENDIF()
        LIST(
            LENGTH
            "ALPAKA_COMPILE_DEFINITIONS_COPY"
            "ALPAKA_COMPILE_DEFINITIONS_COPY_LENGTH")
        IF("${ALPAKA_COMPILE_DEFINITIONS_COPY_LENGTH}")
            TARGET_COMPILE_DEFINITIONS(
                "alpaka"
                ${ALPAKA_COMPILE_DEFINITIONS_COPY})
        ENDIF()

        # Include directories.
        SET("ALPAKA_INCLUDE_DIRS_COPY" "${alpaka_INCLUDE_DIRS}")
        list_add_prefix("PUBLIC;" "ALPAKA_INCLUDE_DIRS_COPY")
        IF("${ALPAKA_DEBUG}" GREATER 0)
            MESSAGE(STATUS "alpaka_INCLUDE_DIRS: ${ALPAKA_INCLUDE_DIRS_COPY}")
        ENDIF()
        LIST(
            LENGTH
            "ALPAKA_INCLUDE_DIRS_COPY"
            "ALPAKA_INCLUDE_DIRS_COPY_LENGTH")
        IF("${ALPAKA_INCLUDE_DIRS_COPY_LENGTH}")
            TARGET_INCLUDE_DIRECTORIES(
                "alpaka"
                ${ALPAKA_INCLUDE_DIRS_COPY})
        ENDIF()

        # Link libraries.
        SET("ALPAKA_LINK_LIBS_COPY" "${alpaka_LIBRARIES}")
        # NOTE: All libraries are required to be prefixed with general, debug or optimized!
        # Add PUBLIC; to all link libraries.
        list_add_prefix_to("PUBLIC;" "optimized;" "ALPAKA_LINK_LIBS_COPY")
        list_add_prefix_to("PUBLIC;" "debug;" "ALPAKA_LINK_LIBS_COPY")
        list_add_prefix_to("PUBLIC;" "general;" "ALPAKA_LINK_LIBS_COPY")
        IF("${ALPAKA_DEBUG}" GREATER 0)
            MESSAGE(STATUS "alpaka_LIBRARIES: ${ALPAKA_LINK_LIBS_COPY}")
        ENDIF()
        LIST(
            LENGTH
            "ALPAKA_LINK_LIBS_COPY"
            "ALPAKA_LINK_LIBS_COPY_LENGTH")
        IF("${ALPAKA_LINK_LIBS_COPY_LENGTH}")
            TARGET_LINK_LIBRARIES(
                "alpaka"
                ${ALPAKA_LINK_LIBS_COPY})
        ENDIF()
    ENDIF()

    # Add '-D' to the definitions
    list_add_prefix("-D" alpaka_DEFINITIONS)
    # Add the compile options to the definitions.
    LIST(APPEND alpaka_DEFINITIONS ${ALPAKA_COMPILE_OPTIONS})
ENDIF()

# Unset already set variables if not found.
IF(NOT ALPAKA_FOUND_INTERNAL)
    UNSET(alpaka_INCLUDE_DIRS)
    UNSET(alpaka_LIBRARIES)
    UNSET(alpaka_VERSION)
    UNSET(alpaka_DEFINITIONS)
ENDIF()

###############################################################################
# FindPackage options
###############################################################################

# Handles the REQUIRED, QUIET and version-related arguments for FIND_PACKAGE.
# NOTE: We do not check for alpaka_LIBRARIES and alpaka_DEFINITIONS because they can be empty.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(
    "alpaka"
    FOUND_VAR alpaka_FOUND
    REQUIRED_VARS alpaka_INCLUDE_DIR
    VERSION_VAR alpaka_VERSION)
