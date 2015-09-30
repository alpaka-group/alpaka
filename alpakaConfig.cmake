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

# Return values.
UNSET(alpaka_FOUND)
UNSET(alpaka_VERSION)
UNSET(alpaka_COMPILE_OPTIONS)
UNSET(alpaka_COMPILE_DEFINITIONS)
UNSET(alpaka_DEFINITIONS)
UNSET(alpaka_INCLUDE_DIR)
UNSET(alpaka_INCLUDE_DIRS)
UNSET(alpaka_LIBRARY)
UNSET(alpaka_LIBRARIES)

# Internal usage.
UNSET(_ALPAKA_FOUND)
UNSET(_ALPAKA_COMPILE_OPTIONS_PUBLIC)
UNSET(_ALPAKA_COMPILE_DEFINITIONS_PUBLIC)
UNSET(_ALPAKA_INCLUDE_DIRECTORY)
UNSET(_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC)
UNSET(_ALPAKA_LINK_LIBRARIES_PUBLIC)
UNSET(_ALPAKA_LINK_FLAGS_PUBLIC)
UNSET(_ALPAKA_COMMON_FILE)
UNSET(_ALPAKA_ADD_EXECUTABLE_FILE)
UNSET(_ALPAKA_FILES_HEADER)
UNSET(_ALPAKA_FILES_SOURCE)
UNSET(_ALPAKA_FILES_OTHER)
UNSET(_ALPAKA_VERSION_DEFINE)
UNSET(_ALPAKA_VER_MAJOR)
UNSET(_ALPAKA_VER_MINOR)
UNSET(_ALPAKA_VER_PATCH)

#-------------------------------------------------------------------------------
# Directory of this file.
#-------------------------------------------------------------------------------
SET(_ALPAKA_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

# Normalize the path (e.g. remove ../)
GET_FILENAME_COMPONENT(_ALPAKA_ROOT_DIR "${_ALPAKA_ROOT_DIR}" ABSOLUTE)

#-------------------------------------------------------------------------------
# Set found to true initially and set it on false if a required dependency is missing.
#-------------------------------------------------------------------------------
SET(_ALPAKA_FOUND TRUE)

#-------------------------------------------------------------------------------
# Common.
#-------------------------------------------------------------------------------
# Add find modules.
LIST(APPEND CMAKE_MODULE_PATH "${_ALPAKA_ROOT_DIR}/cmake/modules")

# Add common functions.
SET(_ALPAKA_COMMON_FILE "${_ALPAKA_ROOT_DIR}/cmake/common.cmake")
INCLUDE("${_ALPAKA_COMMON_FILE}")

# Add ALPAKA_ADD_EXECUTABLE function.
SET(_ALPAKA_ADD_EXECUTABLE_FILE "${_ALPAKA_ROOT_DIR}/cmake/addExecutable.cmake")
INCLUDE("${_ALPAKA_ADD_EXECUTABLE_FILE}")

#-------------------------------------------------------------------------------
# Options.
#-------------------------------------------------------------------------------
OPTION(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE "Enable the serial CPU accelerator" ON)
OPTION(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE "Enable the threads CPU block thread accelerator" ON)
OPTION(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE "Enable the fibers CPU block thread accelerator" ON)
OPTION(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE "Enable the OpenMP 2.0 CPU grid block accelerator" ON)
OPTION(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE "Enable the OpenMP 2.0 CPU block thread accelerator" ON)
OPTION(ALPAKA_ACC_CPU_BT_OMP4_ENABLE "Enable the OpenMP 4.0 CPU block and block thread accelerator" OFF)
OPTION(ALPAKA_ACC_GPU_CUDA_ENABLE "Enable the CUDA GPU accelerator" ON)

# Drop-down combo box in cmake-gui.
SET(ALPAKA_DEBUG "0" CACHE STRING "Debug level")
SET_PROPERTY(CACHE ALPAKA_DEBUG PROPERTY STRINGS "0;1;2")

#-------------------------------------------------------------------------------
# Find Boost.
#-------------------------------------------------------------------------------
SET(_ALPAKA_BOOST_MIN_VER "1.56.0") # minimum version for basic features
IF(${ALPAKA_DEBUG} GREATER 1)
    SET(Boost_DEBUG ON)
    SET(Boost_DETAILED_FAILURE_MSG ON)
ENDIF()
IF(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)
    FIND_PACKAGE(Boost ${_ALPAKA_BOOST_MIN_VER} QUIET COMPONENTS fiber context system thread atomic chrono date_time)
    IF(NOT Boost_FIBER_FOUND)
        MESSAGE(WARNING "Optional alpaka dependency Boost fiber could not be found! Fibers accelerator disabled!")
        SET(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE OFF CACHE BOOL "Enable the Fibers CPU accelerator" FORCE)
        FIND_PACKAGE(Boost ${_ALPAKA_BOOST_MIN_VER} QUIET)
    ENDIF()

ELSE()
    FIND_PACKAGE(Boost ${_ALPAKA_BOOST_MIN_VER} QUIET)
ENDIF()

IF(${ALPAKA_DEBUG} GREATER 1)
    MESSAGE(STATUS "Boost in:")
    MESSAGE(STATUS "BOOST_ROOT : ${BOOST_ROOT}")
    MESSAGE(STATUS "BOOSTROOT : ${BOOSTROOT}")
    MESSAGE(STATUS "BOOST_INCLUDEDIR: ${BOOST_INCLUDEDIR}")
    MESSAGE(STATUS "BOOST_LIBRARYDIR: ${BOOST_LIBRARYDIR}")
    MESSAGE(STATUS "Boost_NO_SYSTEM_PATHS: ${Boost_NO_SYSTEM_PATHS}")
    MESSAGE(STATUS "Boost_ADDITIONAL_VERSIONS: ${Boost_ADDITIONAL_VERSIONS}")
    MESSAGE(STATUS "Boost_USE_MULTITHREADED: ${Boost_USE_MULTITHREADED}")
    MESSAGE(STATUS "Boost_USE_STATIC_LIBS: ${Boost_USE_STATIC_LIBS}")
    MESSAGE(STATUS "Boost_USE_STATIC_RUNTIME: ${Boost_USE_STATIC_RUNTIME}")
    MESSAGE(STATUS "Boost_USE_DEBUG_RUNTIME: ${Boost_USE_DEBUG_RUNTIME}")
    MESSAGE(STATUS "Boost_USE_DEBUG_PYTHON: ${Boost_USE_DEBUG_PYTHON}")
    MESSAGE(STATUS "Boost_USE_STLPORT: ${Boost_USE_STLPORT}")
    MESSAGE(STATUS "Boost_USE_STLPORT_DEPRECATED_NATIVE_IOSTREAMS: ${Boost_USE_STLPORT_DEPRECATED_NATIVE_IOSTREAMS}")
    MESSAGE(STATUS "Boost_COMPILER: ${Boost_COMPILER}")
    MESSAGE(STATUS "Boost_THREADAPI: ${Boost_THREADAPI}")
    MESSAGE(STATUS "Boost_NAMESPACE: ${Boost_NAMESPACE}")
    MESSAGE(STATUS "Boost_DEBUG: ${Boost_DEBUG}")
    MESSAGE(STATUS "Boost_DETAILED_FAILURE_MSG: ${Boost_DETAILED_FAILURE_MSG}")
    MESSAGE(STATUS "Boost_REALPATH: ${Boost_REALPATH}")
    MESSAGE(STATUS "Boost_NO_BOOST_CMAKE: ${Boost_NO_BOOST_CMAKE}")
    MESSAGE(STATUS "Boost out:")
    MESSAGE(STATUS "Boost_FOUND: ${Boost_FOUND}")
    MESSAGE(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    MESSAGE(STATUS "Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")
    MESSAGE(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    MESSAGE(STATUS "Boost_FIBER_FOUND: ${Boost_FIBER_FOUND}")
    MESSAGE(STATUS "Boost_FIBER_LIBRARY: ${Boost_FIBER_LIBRARY}")
    MESSAGE(STATUS "Boost_CONTEXT_FOUND: ${Boost_CONTEXT_FOUND}")
    MESSAGE(STATUS "Boost_CONTEXT_LIBRARY: ${Boost_CONTEXT_LIBRARY}")
    MESSAGE(STATUS "Boost_SYSTEM_FOUND: ${Boost_SYSTEM_FOUND}")
    MESSAGE(STATUS "Boost_SYSTEM_LIBRARY: ${Boost_SYSTEM_LIBRARY}")
    MESSAGE(STATUS "Boost_THREAD_FOUND: ${Boost_THREAD_FOUND}")
    MESSAGE(STATUS "Boost_THREAD_LIBRARY: ${Boost_THREAD_LIBRARY}")
    MESSAGE(STATUS "Boost_ATOMIC_FOUND: ${Boost_ATOMIC_FOUND}")
    MESSAGE(STATUS "Boost_ATOMIC_LIBRARY: ${Boost_ATOMIC_LIBRARY}")
    MESSAGE(STATUS "Boost_CHRONO_FOUND: ${Boost_CHRONO_FOUND}")
    MESSAGE(STATUS "Boost_CHRONO_LIBRARY: ${Boost_CHRONO_LIBRARY}")
    MESSAGE(STATUS "Boost_DATE_TIME_FOUND: ${Boost_DATE_TIME_FOUND}")
    MESSAGE(STATUS "Boost_DATE_TIME_LIBRARY: ${Boost_DATE_TIME_LIBRARY}")
    MESSAGE(STATUS "Boost_VERSION: ${Boost_VERSION}")
    MESSAGE(STATUS "Boost_LIB_VERSION: ${Boost_LIB_VERSION}")
    MESSAGE(STATUS "Boost_MAJOR_VERSION: ${Boost_MAJOR_VERSION}")
    MESSAGE(STATUS "Boost_MINOR_VERSION: ${Boost_MINOR_VERSION}")
    MESSAGE(STATUS "Boost_SUBMINOR_VERSION: ${Boost_SUBMINOR_VERSION}")
    MESSAGE(STATUS "Boost_LIB_DIAGNOSTIC_DEFINITIONS: ${Boost_LIB_DIAGNOSTIC_DEFINITIONS}")
    MESSAGE(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    MESSAGE(STATUS "Boost cached:")
    MESSAGE(STATUS "Boost_INCLUDE_DIR: ${Boost_INCLUDE_DIR}")
    MESSAGE(STATUS "Boost_LIBRARY_DIR: ${Boost_LIBRARY_DIR}")
ENDIF()

IF(NOT Boost_FOUND)
    MESSAGE(WARNING "Required alpaka dependency Boost (>=${_ALPAKA_BOOST_MIN_VER}) could not be found!")
    SET(_ALPAKA_FOUND FALSE)

ELSE()
    LIST(APPEND _ALPAKA_INCLUDE_DIRECTORIES_PUBLIC ${Boost_INCLUDE_DIRS})
    LIST(APPEND _ALPAKA_LINK_LIBRARIES_PUBLIC ${Boost_LIBRARIES})
ENDIF()

#-------------------------------------------------------------------------------
# Find OpenMP.
#-------------------------------------------------------------------------------
IF(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OR ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE OR ALPAKA_ACC_CPU_BT_OMP4_ENABLE)
    FIND_PACKAGE(OpenMP)
    IF(NOT OPENMP_FOUND)
        MESSAGE(WARNING "Optional alpaka dependency OpenMP could not be found! OpenMP accelerators disabled!")
        SET(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE OFF CACHE BOOL "Enable the OpenMP 2.0 CPU grid block accelerator" FORCE)
        SET(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE OFF CACHE BOOL "Enable the OpenMP 2.0 CPU block thread accelerator" FORCE)
        SET(ALPAKA_ACC_CPU_BT_OMP4_ENABLE OFF CACHE BOOL "Enable the OpenMP 4.0 CPU block and thread accelerator" FORCE)

    ELSE()
        SET(_ALPAKA_COMPILE_OPTIONS_PUBLIC ${OpenMP_CXX_FLAGS})
        IF(NOT MSVC)
            SET(_ALPAKA_LINK_FLAGS_PUBLIC ${OpenMP_CXX_FLAGS})
        ENDIF()
        # CUDA requires some special handling
        IF(ALPAKA_ACC_GPU_CUDA_ENABLE)
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        ENDIF()
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Find CUDA.
#-------------------------------------------------------------------------------
IF(ALPAKA_ACC_GPU_CUDA_ENABLE)

    IF(NOT DEFINED ALPAKA_CUDA_VERSION)
        SET(ALPAKA_CUDA_VERSION 7.0)
    ENDIF()

    IF(ALPAKA_CUDA_VERSION VERSION_LESS 7.0)
        MESSAGE(WARNING "CUDA Toolkit < 7.0 is not supported! CUDA accelerator disabled!")
        SET(ALPAKA_ACC_GPU_CUDA_ENABLE OFF CACHE BOOL "Enable the CUDA GPU accelerator" FORCE)

    ELSE()
        FIND_PACKAGE(CUDA "${ALPAKA_CUDA_VERSION}")
        IF(NOT CUDA_FOUND)
            MESSAGE(WARNING "Optional alpaka dependency CUDA could not be found! CUDA accelerator disabled!")
            SET(ALPAKA_ACC_GPU_CUDA_ENABLE OFF CACHE BOOL "Enable the CUDA GPU accelerator" FORCE)

        ELSE()
            IF(${ALPAKA_DEBUG} GREATER 1)
                SET(CUDA_VERBOSE_BUILD ON)
            ENDIF()
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

            OPTION(ALPAKA_CUDA_FTZ "Set flush to zero for GPU" OFF)
            IF(ALPAKA_CUDA_FTZ)
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

            LIST(APPEND _ALPAKA_LINK_LIBRARIES_PUBLIC "general;${CUDA_CUDART_LIBRARY}")
            LIST(APPEND _ALPAKA_INCLUDE_DIRECTORIES_PUBLIC ${CUDA_INCLUDE_DIRS})
        ENDIF()
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Compiler settings.
#-------------------------------------------------------------------------------
IF(MSVC)
    # Empty append to define it if it does not already exist.
    LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC)
ELSE()
    # Select C++ standard version.
    IF(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)
        LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "-std=c++14")
    ELSE()
        LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "-std=c++11")
    ENDIF()

    # Add linker options.
    IF(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE)
        LIST(APPEND _ALPAKA_LINK_LIBRARIES_PUBLIC "general;pthread")
    ENDIF()
    # librt: undefined reference to `clock_gettime'
    LIST(APPEND _ALPAKA_LINK_LIBRARIES_PUBLIC "general;rt")

    # GNU
    IF(CMAKE_COMPILER_IS_GNUCXX)
        LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "-ftemplate-depth-512")
    # Clang or AppleClang
    ELSEIF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        LIST(APPEND _ALPAKA_COMPILE_OPTIONS_PUBLIC "-ftemplate-depth=512")
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# alpaka.
#-------------------------------------------------------------------------------
IF(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
ENDIF()
IF(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
ENDIF()
IF(ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_CPU_B_SEQ_T_FIBERS_ENABLED)
ENDIF()
IF(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
ENDIF()
IF(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
ENDIF()
IF(ALPAKA_ACC_CPU_BT_OMP4_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_CPU_BT_OMP4_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_CPU_BT_OMP4_ENABLED)
ENDIF()
IF(ALPAKA_ACC_GPU_CUDA_ENABLE)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_ACC_GPU_CUDA_ENABLED")
    MESSAGE(STATUS ALPAKA_ACC_GPU_CUDA_ENABLED)
ENDIF()

LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_DEBUG=${ALPAKA_DEBUG}")

IF(ALPAKA_INTEGRATION_TEST)
    LIST(APPEND _ALPAKA_COMPILE_DEFINITIONS_PUBLIC "ALPAKA_INTEGRATION_TEST")
ENDIF()

SET(_ALPAKA_INCLUDE_DIRECTORY "${_ALPAKA_ROOT_DIR}/include")
LIST(APPEND _ALPAKA_INCLUDE_DIRECTORIES_PUBLIC "${_ALPAKA_INCLUDE_DIRECTORY}")
SET(_ALPAKA_SUFFIXED_INCLUDE_DIR "${_ALPAKA_INCLUDE_DIRECTORY}/alpaka")

SET(_ALPAKA_LINK_LIBRARY)
LIST(APPEND _ALPAKA_LINK_LIBRARIES_PUBLIC "${_ALPAKA_LINK_LIBRARY}")

SET(_ALPAKA_FILES_OTHER "${_ALPAKA_ROOT_DIR}/alpakaConfig.cmake" "${_ALPAKA_ADD_EXECUTABLE_FILE}" "${_ALPAKA_COMMON_FILE}" "${_ALPAKA_ROOT_DIR}/.travis.yml" "${_ALPAKA_ROOT_DIR}/README.md")

# Add all the source and include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_ALPAKA_SUFFIXED_INCLUDE_DIR}" "${_ALPAKA_SUFFIXED_INCLUDE_DIR}" "hpp" "_ALPAKA_FILES_HEADER")
append_recursive_files_add_to_src_group("${_ALPAKA_SUFFIXED_INCLUDE_DIR}" "${_ALPAKA_SUFFIXED_INCLUDE_DIR}" "cpp" "_ALPAKA_FILES_SOURCE")

#-------------------------------------------------------------------------------
# Target.
#-------------------------------------------------------------------------------
IF(NOT TARGET "alpaka")
    ADD_LIBRARY(
        "alpaka"
        ${_ALPAKA_FILES_HEADER} ${_ALPAKA_FILES_SOURCE} ${_ALPAKA_FILES_OTHER})

    # Compile options.
    IF(${ALPAKA_DEBUG} GREATER 1)
        MESSAGE(STATUS "_ALPAKA_COMPILE_OPTIONS_PUBLIC: ${_ALPAKA_COMPILE_OPTIONS_PUBLIC}")
    ENDIF()
    LIST(
        LENGTH
        _ALPAKA_COMPILE_OPTIONS_PUBLIC
        _ALPAKA_COMPILE_OPTIONS_PUBLIC_LENGTH)
    IF(${_ALPAKA_COMPILE_OPTIONS_PUBLIC_LENGTH} GREATER 0)
        TARGET_COMPILE_OPTIONS(
            "alpaka"
            PUBLIC ${_ALPAKA_COMPILE_OPTIONS_PUBLIC})
    ENDIF()

    # Compile definitions.
    IF(${ALPAKA_DEBUG} GREATER 1)
        MESSAGE(STATUS "_ALPAKA_COMPILE_DEFINITIONS_PUBLIC: ${_ALPAKA_COMPILE_DEFINITIONS_PUBLIC}")
    ENDIF()
    LIST(
        LENGTH
        _ALPAKA_COMPILE_DEFINITIONS_PUBLIC
        _ALPAKA_COMPILE_DEFINITIONS_PUBLIC_LENGTH)
    IF(${_ALPAKA_COMPILE_DEFINITIONS_PUBLIC_LENGTH} GREATER 0)
        TARGET_COMPILE_DEFINITIONS(
            "alpaka"
            PUBLIC ${_ALPAKA_COMPILE_DEFINITIONS_PUBLIC})
    ENDIF()

    # Include directories.
    IF(${ALPAKA_DEBUG} GREATER 1)
        MESSAGE(STATUS "_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC: ${_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC}")
    ENDIF()
    LIST(
        LENGTH
        _ALPAKA_INCLUDE_DIRECTORIES_PUBLIC
        _ALPAKA_INCLUDE_DIRECTORIES_PUBLIC_LENGTH)
    IF(${_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC_LENGTH} GREATER 0)
        TARGET_INCLUDE_DIRECTORIES(
            "alpaka"
            PUBLIC ${_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC})
    ENDIF()

    # Link libraries.
    # There are no PUBLIC_LINK_FLAGS in CMAKE:
    # http://stackoverflow.com/questions/26850889/cmake-keeping-link-flags-of-internal-libs
    IF(${ALPAKA_DEBUG} GREATER 1)
        MESSAGE(STATUS "_ALPAKA_LINK_LIBRARIES_PUBLIC: ${_ALPAKA_LINK_LIBRARIES_PUBLIC}")
    ENDIF()
    LIST(
        LENGTH
        _ALPAKA_LINK_LIBRARIES_PUBLIC
        _ALPAKA_LINK_LIBRARIES_PUBLIC_LENGTH)
    IF(${_ALPAKA_LINK_LIBRARIES_PUBLIC_LENGTH} GREATER 0)
        TARGET_LINK_LIBRARIES(
            "alpaka"
            PUBLIC ${_ALPAKA_LINK_LIBRARIES_PUBLIC} ${_ALPAKA_LINK_FLAGS_PUBLIC})
    ENDIF()
ENDIF()

#-------------------------------------------------------------------------------
# Find alpaka version.
#-------------------------------------------------------------------------------
FILE(STRINGS "${_ALPAKA_ROOT_DIR}/include/alpaka/alpaka.hpp"
     _ALPAKA_VERSION_DEFINE REGEX "#define ALPAKA_VERSION BOOST_VERSION_NUMBER")

STRING(REGEX REPLACE "[^0-9]*([0-9]+), ([0-9]+), ([0-9]+).*" "\\1" _ALPAKA_VER_MAJOR "${_ALPAKA_VERSION_DEFINE}")
STRING(REGEX REPLACE "[^0-9]*([0-9]+), ([0-9]+), ([0-9]+).*" "\\2" _ALPAKA_VER_MINOR "${_ALPAKA_VERSION_DEFINE}")
STRING(REGEX REPLACE "[^0-9]*([0-9]+), ([0-9]+), ([0-9]+).*" "\\3" _ALPAKA_VER_PATCH "${_ALPAKA_VERSION_DEFINE}")

#-------------------------------------------------------------------------------
# Set return values.
#-------------------------------------------------------------------------------
SET(alpaka_VERSION "${_ALPAKA_VER_MAJOR}.${_ALPAKA_VER_MINOR}.${_ALPAKA_VER_PATCH}")
SET(alpaka_COMPILE_OPTIONS ${_ALPAKA_COMPILE_OPTIONS_PUBLIC})
SET(alpaka_COMPILE_DEFINITIONS ${_ALPAKA_COMPILE_DEFINITIONS_PUBLIC})
# Add '-D' to the definitions
SET(alpaka_DEFINITIONS ${_ALPAKA_COMPILE_DEFINITIONS_PUBLIC})
list_add_prefix("-D" alpaka_DEFINITIONS)
# Add the compile options to the definitions.
LIST(APPEND alpaka_DEFINITIONS ${_ALPAKA_COMPILE_OPTIONS_PUBLIC})
SET(alpaka_INCLUDE_DIR ${_ALPAKA_INCLUDE_DIRECTORY})
SET(alpaka_INCLUDE_DIRS ${_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC})
SET(alpaka_LIBRARY ${_ALPAKA_LINK_LIBRARY})
SET(alpaka_LIBRARIES ${_ALPAKA_LINK_FLAGS_PUBLIC})
LIST(APPEND alpaka_LIBRARIES ${_ALPAKA_LINK_LIBRARIES_PUBLIC})

#-------------------------------------------------------------------------------
# Print the return values.
#-------------------------------------------------------------------------------
IF(${ALPAKA_DEBUG} GREATER 0)
    MESSAGE(STATUS "alpaka_FOUND: ${alpaka_FOUND}")
    MESSAGE(STATUS "alpaka_VERSION: ${alpaka_VERSION}")
    MESSAGE(STATUS "alpaka_COMPILE_OPTIONS: ${alpaka_COMPILE_OPTIONS}")
    MESSAGE(STATUS "alpaka_COMPILE_DEFINITIONS: ${alpaka_COMPILE_DEFINITIONS}")
    MESSAGE(STATUS "alpaka_DEFINITIONS: ${alpaka_DEFINITIONS}")
    MESSAGE(STATUS "alpaka_INCLUDE_DIR: ${alpaka_INCLUDE_DIR}")
    MESSAGE(STATUS "alpaka_INCLUDE_DIRS: ${alpaka_INCLUDE_DIRS}")
    MESSAGE(STATUS "alpaka_LIBRARY: ${alpaka_LIBRARY}")
    MESSAGE(STATUS "alpaka_LIBRARIES: ${alpaka_LIBRARIES}")
ENDIF()

# Unset already set variables if not found.
IF(NOT _ALPAKA_FOUND)
    UNSET(alpaka_FOUND)
    UNSET(alpaka_VERSION)
    UNSET(alpaka_COMPILE_OPTIONS)
    UNSET(alpaka_COMPILE_DEFINITIONS)
    UNSET(alpaka_DEFINITIONS)
    UNSET(alpaka_INCLUDE_DIR)
    UNSET(alpaka_INCLUDE_DIRS)
    UNSET(alpaka_LIBRARY)
    UNSET(alpaka_LIBRARIES)

    UNSET(_ALPAKA_FOUND)
    UNSET(_ALPAKA_COMPILE_OPTIONS_PUBLIC)
    UNSET(_ALPAKA_COMPILE_DEFINITIONS_PUBLIC)
    UNSET(_ALPAKA_INCLUDE_DIRECTORY)
    UNSET(_ALPAKA_INCLUDE_DIRECTORIES_PUBLIC)
    UNSET(_ALPAKA_LINK_LIBRARY)
    UNSET(_ALPAKA_LINK_LIBRARIES_PUBLIC)
    UNSET(_ALPAKA_LINK_FLAGS_PUBLIC)
    UNSET(_ALPAKA_COMMON_FILE)
    UNSET(_ALPAKA_ADD_EXECUTABLE_FILE)
    UNSET(_ALPAKA_FILES_HEADER)
    UNSET(_ALPAKA_FILES_SOURCE)
    UNSET(_ALPAKA_FILES_OTHER)
    UNSET(_ALPAKA_BOOST_MIN_VER)
    UNSET(_ALPAKA_VERSION_DEFINE)
    UNSET(_ALPAKA_VER_MAJOR)
    UNSET(_ALPAKA_VER_MINOR)
    UNSET(_ALPAKA_VER_PATCH)
ELSE()
    # Make internal variables advanced options in the GUI.
    MARK_AS_ADVANCED(
        alpaka_INCLUDE_DIR
        alpaka_LIBRARY
        _ALPAKA_COMPILE_OPTIONS_PUBLIC
        _ALPAKA_COMPILE_DEFINITIONS_PUBLIC
        _ALPAKA_INCLUDE_DIRECTORY
        _ALPAKA_INCLUDE_DIRECTORIES_PUBLIC
        _ALPAKA_LINK_LIBRARY
        _ALPAKA_LINK_LIBRARIES_PUBLIC
        _ALPAKA_LINK_FLAGS_PUBLIC
        _ALPAKA_COMMON_FILE
        _ALPAKA_ADD_EXECUTABLE_FILE
        _ALPAKA_FILES_HEADER
        _ALPAKA_FILES_SOURCE
        _ALPAKA_FILES_OTHER
        _ALPAKA_BOOST_MIN_VER
        _ALPAKA_VERSION_DEFINE
        _ALPAKA_VER_MAJOR
        _ALPAKA_VER_MINOR
        _ALPAKA_VER_PATCH)
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
