#
# Copyright 2014-2015 Benjamin Worpitz
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

# Options:
#   ALPAKA_SERIAL_ENABLE {ON, OFF}
#   ALPAKA_THREADS_ENABLE {ON, OFF}
#   ALPAKA_FIBERS_ENABLE {ON, OFF}
#   ALPAKA_OPENMP_ENABLE {ON, OFF}
#   ALPAKA_CUDA_ENABLE {ON, OFF}
#   ALPAKA_CUDA_VERSION {7.0, ...}
#   ALPAKA_CUDA_ARCH {sm_20, sm...}
#    ALPAKA_CUDA_FAST_MATH {ON, OFF}
#   ALPAKA_CUDA_SHOW_REGISTER {ON, OFF}
#   ALPAKA_CUDA_KEEP_FILES {ON, OFF}
#   ALPAKA_CUDA_SHOW_CODELINES {ON, OFF}
#   ALPAKA_DEBUG {0, 1, 2}
#   ALPAKA_INTEGRATION_TEST {ON, OFF}
#   BOOST_REQUIRED_COMPONENTS {program_options, ...}

################################################################################
# Required CMake version.
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.
################################################################################

MESSAGE(STATUS "### alpaka ###")

SET(ALPAKA_COMPILE_OPTIONS)
SET(ALPAKA_COMPILE_DEFINITIONS)
SET(ALPAKA_INCLUDE_DEPENDENCIES)
SET(ALPAKA_LIBRARY_DEPENDENCIES)

# Set helper paths to find libraries and packages.
SET(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")

# Add common functions.
SET(ALPAKA_RESURSIVE_FILE_FUNCTIONS_FILE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/common.cmake")
INCLUDE(${ALPAKA_RESURSIVE_FILE_FUNCTIONS_FILE_PATH})

OPTION(ALPAKA_SERIAL_ENABLE "Enable the serial accelerator" ON)
IF(ALPAKA_SERIAL_ENABLE)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS ALPAKA_SERIAL_ENABLED)
    MESSAGE(STATUS ALPAKA_SERIAL_ENABLED)
ENDIF()
OPTION(ALPAKA_THREADS_ENABLE "Enable the threads accelerator" ON)
IF(ALPAKA_THREADS_ENABLE)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS ALPAKA_THREADS_ENABLED)
    MESSAGE(STATUS ALPAKA_THREADS_ENABLED)
ENDIF()
OPTION(ALPAKA_FIBERS_ENABLE "Enable the fibers accelerator" OFF)
IF(ALPAKA_FIBERS_ENABLE)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS ALPAKA_FIBERS_ENABLED)
    MESSAGE(STATUS ALPAKA_FIBERS_ENABLED)
ENDIF()
OPTION(ALPAKA_OPENMP_ENABLE "Enable the OpenMP accelerator" ON)
IF(ALPAKA_OPENMP_ENABLE)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS ALPAKA_OPENMP_ENABLED)
    MESSAGE(STATUS ALPAKA_OPENMP_ENABLED)
ENDIF()
OPTION(ALPAKA_CUDA_ENABLE "Enable the CUDA accelerator" OFF)
IF(ALPAKA_CUDA_ENABLE)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS ALPAKA_CUDA_ENABLED)
    MESSAGE(STATUS ALPAKA_CUDA_ENABLED)
ENDIF()

################################################################################
# Debug options.
################################################################################
OPTION(ALPAKA_DEBUG "Debug level [0,1,2]" 0)
IF(${ALPAKA_DEBUG} GREATER 0)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS "ALPAKA_DEBUG=${ALPAKA_DEBUG}")
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS "NDEBUG")
ELSE()
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS "_DEBUG")
ENDIF()

IF(ALPAKA_INTEGRATION_TEST)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS "ALPAKA_INTEGRATION_TEST=1")
ENDIF()

################################################################################
# Find Boost.
################################################################################
SET(Boost_USE_STATIC_LIBS ON)
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)

IF(ALPAKA_FIBERS_ENABLE)
    LIST(APPEND BOOST_REQUIRED_COMPONENTS fiber coroutine context system thread atomic chrono date_time regex)
ELSE()
    # Empty addition to define the variable if it is not existing.
    LIST(APPEND BOOST_REQUIRED_COMPONENTS)
ENDIF()

FIND_PACKAGE(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})
LIST(APPEND ALPAKA_INCLUDE_DEPENDENCIES ${Boost_INCLUDE_DIRS})

LIST(FIND "${Boost_LIBRARIES}" "optimized" ALPAKA_BOOST_LIBRARY_LIST_OPTIMIZED_ATTRIBUTE_INDEX)
IF(NOT ALPAKA_BOOST_LIBRARY_LIST_OPTIMIZED_ATTRIBUTE_INDEX EQUAL -1)
    list_add_prefix("general;" Boost_LIBRARIES)
ENDIF()
LIST(APPEND ALPAKA_LIBRARY_DEPENDENCIES ${Boost_LIBRARIES})

################################################################################
# Find OpenMP.
################################################################################
IF(ALPAKA_OPENMP_ENABLE)
    FIND_PACKAGE(OpenMP REQUIRED)
    IF(OPENMP_FOUND)
        # FIXME: We have to set the global variables because it does not work with setting ALPAKA_COMPILE_OPTIONS.
        #LIST(APPEND ALPAKA_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})

        #SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        #SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    ENDIF()
ENDIF()

################################################################################
# Find CUDA.
################################################################################
IF(ALPAKA_CUDA_ENABLE)

    IF(NOT DEFINED ALPAKA_CUDA_VERSION)
        SET(ALPAKA_CUDA_VERSION 7.0)
    ENDIF()

    IF(ALPAKA_CUDA_VERSION VERSION_LESS 7.0)
        MESSAGE(FATAL_ERROR "CUDA Toolkit < 7.0 is not supported!")
    ENDIF()

    FIND_PACKAGE(CUDA ${ALPAKA_CUDA_VERSION} REQUIRED)
    #SET(CUDA_VERBOSE_BUILD ON)
    SET(CUDA_PROPAGATE_HOST_FLAGS ON)

    SET(ALPAKA_CUDA_ARCH sm_20 CACHE STRING "Set GPU architecture")
    STRING(COMPARE EQUAL ${ALPAKA_CUDA_ARCH} "sm_10" IS_CUDA_ARCH_UNSUPPORTED)
    STRING(COMPARE EQUAL ${ALPAKA_CUDA_ARCH} "sm_11" IS_CUDA_ARCH_UNSUPPORTED)
    STRING(COMPARE EQUAL ${ALPAKA_CUDA_ARCH} "sm_12" IS_CUDA_ARCH_UNSUPPORTED)
    STRING(COMPARE EQUAL ${ALPAKA_CUDA_ARCH} "sm_13" IS_CUDA_ARCH_UNSUPPORTED)

    IF(IS_CUDA_ARCH_UNSUPPORTED)
        MESSAGE(FATAL_ERROR "Unsupported CUDA architecture ${ALPAKA_CUDA_ARCH} specified. SM 2.0 or higher is required for CUDA 7.0.")
    ENDIF(IS_CUDA_ARCH_UNSUPPORTED)

    LIST(APPEND CUDA_NVCC_FLAGS -arch=${ALPAKA_CUDA_ARCH})
    
    IF(NOT MSVC)
        LIST(APPEND CUDA_NVCC_FLAGS -std=c++11)
        SET(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    ENDIF()
    
    IF(CMAKE_BUILD_TYPE MATCHES "Debug")
        LIST(APPEND CUDA_NVCC_FLAGS -g -G)
    ENDIF()

    OPTION(ALPAKA_CUDA_FAST_MATH "Enable fast-math" ON)
    IF(ALPAKA_CUDA_FAST_MATH)
        LIST(APPEND CUDA_NVCC_FLAGS --use_fast_math)
    ENDIF()

    OPTION(ALPAKA_CUDA_FAST_MATH "Set flush to zero for GPU" OFF)
    IF(ALPAKA_CUDA_FAST_MATH)
        LIST(APPEND CUDA_NVCC_FLAGS --ftz=true)
    ELSE()
        LIST(APPEND CUDA_NVCC_FLAGS --ftz=false)
    ENDIF()
    
    OPTION(ALPAKA_CUDA_SHOW_REGISTER "Show kernel registers and create PTX" OFF)
    IF(ALPAKA_CUDA_SHOW_REGISTER)
        LIST(APPEND CUDA_NVCC_FLAGS -Xptxas=-v)
    ENDIF()

    OPTION(ALPAKA_CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps (folder: nvcc_tmp)" OFF)
    IF(ALPAKA_CUDA_KEEP_FILES)
        MAKE_DIRECTORY("${PROJECT_BINARY_DIR}/nvcc_tmp")
        LIST(APPEND CUDA_NVCC_FLAGS --keep --keep-dir "${PROJECT_BINARY_DIR}/nvcc_tmp")
    ENDIF()

    OPTION(ALPAKA_CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)
    IF(ALPAKA_CUDA_SHOW_CODELINES)
        LIST(APPEND CUDA_NVCC_FLAGS --source-in-ptx -lineinfo)
        IF(NOT MSVC)
            LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler -rdynamic)
        ENDIF()
        SET(ALPAKA_CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
    ENDIF()

    IF(${CUDA_CUDART_LIBRARY})
        LIST(APPEND ALPAKA_LIBRARY_DEPENDENCIES general ${CUDA_CUDART_LIBRARY})
    ENDIF()
ENDIF()

################################################################################
# Compiler settings.
################################################################################

#MSVC
IF(MSVC)
    # Force to always compile with W4
    IF(ALPAKA_COMPILE_OPTIONS MATCHES "/W[0-4]")
        STRING(REGEX REPLACE "/W[0-4]" "/W4" ALPAKA_COMPILE_OPTIONS "${ALPAKA_COMPILE_OPTIONS}")
    ELSE()
        LIST(APPEND ALPAKA_COMPILE_OPTIONS "/W4")
    ENDIF()
ELSE()
    # GNU
    IF(CMAKE_COMPILER_IS_GNUCXX)
        LIST(APPEND ALPAKA_COMPILE_OPTIONS "-Wall")
        LIST(APPEND ALPAKA_COMPILE_OPTIONS "-pedantic")
        LIST(APPEND ALPAKA_COMPILE_OPTIONS "-Wextra")
    # Clang or AppleClang
    ELSEIF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        LIST(APPEND ALPAKA_COMPILE_OPTIONS "-Wall")
        #Clang 3.7 seems to require librt: undefined reference to `clock_gettime'
        # \TODO: Remove when using fiber develop.
        #LIST(APPEND ALPAKA_LIBRARY_DEPENDENCIES "general;lrt")
    # ICC
    ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        LIST(APPEND ALPAKA_COMPILE_OPTIONS "-Wall")
    # PGI
    ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "PGI")
        LIST(APPEND ALPAKA_COMPILE_OPTIONS "-Minform=inform")
    ENDIF()
    
    # Select C++ standard version.
    LIST(APPEND ALPAKA_COMPILE_OPTIONS "-std=c++11")
    
    # Add linker options.
    IF(ALPAKA_THREADS_ENABLE)
        LIST(APPEND ALPAKA_LIBRARY_DEPENDENCIES "general;pthread")
    ENDIF()
ENDIF()
  
################################################################################
# Compile & Link.
################################################################################

# alpaka
SET(ALPAKA_REALTIVE_INCLUDE_DIR "include/")
SET(ALPAKA_SUFFIXED_REALTIVE_INCLUDE_DIR "${ALPAKA_REALTIVE_INCLUDE_DIR}alpaka/")
SET(ALPAKA_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include/")

# Add all the include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_source_group(${ALPAKA_SUFFIXED_REALTIVE_INCLUDE_DIR} "hpp" ALPAKA_HEADER_FILES_ALL)

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_source_group(${ALPAKA_SUFFIXED_REALTIVE_INCLUDE_DIR} "cpp" ALPAKA_SOURCE_FILES_ALL)

SET(ALPAKA_CMAKE_FILES_ALL ${CMAKE_CURRENT_LIST_FILE} ${ALPAKA_RESURSIVE_FILE_FUNCTIONS_FILE_PATH})

ADD_LIBRARY(
    alpaka 
    ${ALPAKA_HEADER_FILES_ALL}
    ${ALPAKA_SOURCE_FILES_ALL}
    ${ALPAKA_CMAKE_FILES_ALL})

# CUDA does not work well with the much better target dependent TARGET_XXX commands: https://www.cmake.org/Bug/view.php?id=14201&nbn=1
# It requires the settings to be available globally. So we duplicate the settings from TARGET_INCLUDE_DIRECTORIES and TARGET_COMPILE_DEFINITIONS. Maybe this can be removed some time...
IF(ALPAKA_CUDA_ENABLE)
    INCLUDE_DIRECTORIES(${ALPAKA_INCLUDE_DEPENDENCIES} ${ALPAKA_INCLUDE_DIR})

    SET(ALPAKA_COMPILE_DEFINITIONS_COPY ${ALPAKA_COMPILE_DEFINITIONS})
    list_add_prefix("-D" ALPAKA_COMPILE_DEFINITIONS_COPY)
    ADD_DEFINITIONS(${ALPAKA_COMPILE_DEFINITIONS_COPY})
ENDIF()
   
list_add_prefix("PUBLIC;" ALPAKA_COMPILE_OPTIONS)
IF(${ALPAKA_DEBUG} GREATER 0)
    MESSAGE(STATUS "ALPAKA_COMPILE_OPTIONS: ${ALPAKA_COMPILE_OPTIONS}")
ENDIF()
TARGET_COMPILE_OPTIONS(alpaka ${ALPAKA_COMPILE_OPTIONS})

list_add_prefix("PUBLIC;" ALPAKA_COMPILE_DEFINITIONS)
IF(${ALPAKA_DEBUG} GREATER 0)
    MESSAGE(STATUS "ALPAKA_COMPILE_DEFINITIONS: ${ALPAKA_COMPILE_DEFINITIONS}")
ENDIF()
TARGET_COMPILE_DEFINITIONS(alpaka ${ALPAKA_COMPILE_DEFINITIONS})

list_add_prefix("PUBLIC;" ALPAKA_INCLUDE_DEPENDENCIES)
IF(${ALPAKA_DEBUG} GREATER 0)
    MESSAGE(STATUS "ALPAKA_INCLUDE_DEPENDENCIES: ${ALPAKA_INCLUDE_DEPENDENCIES}")
ENDIF()
TARGET_INCLUDE_DIRECTORIES(alpaka ${ALPAKA_INCLUDE_DEPENDENCIES} PUBLIC ${ALPAKA_INCLUDE_DIR})

# NOTE: All libraries are required to be prefixed with general, debug or optimized!
# Add PUBLIC; to all link libraries.
list_add_prefix_to("PUBLIC;" "optimized;" ALPAKA_LIBRARY_DEPENDENCIES)
list_add_prefix_to("PUBLIC;" "debug;" ALPAKA_LIBRARY_DEPENDENCIES)
list_add_prefix_to("PUBLIC;" "general;" ALPAKA_LIBRARY_DEPENDENCIES)
IF(${ALPAKA_DEBUG} GREATER 0)
    MESSAGE(STATUS "ALPAKA_LIBRARY_DEPENDENCIES: ${ALPAKA_LIBRARY_DEPENDENCIES}")
ENDIF()
TARGET_LINK_LIBRARIES(alpaka ${ALPAKA_LIBRARY_DEPENDENCIES})

MESSAGE(STATUS "### alpaka done ###")
