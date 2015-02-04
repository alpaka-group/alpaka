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

################################################################################
# Required CMake version.
################################################################################

#CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

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

OPTION(ALPAKA_SERIAL_ENABLE "Enable the serial accelerator" OFF)
IF(ALPAKA_SERIAL_ENABLE)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS ALPAKA_SERIAL_ENABLED)
ENDIF()
OPTION(ALPAKA_THREADS_ENABLE "Enable the threads accelerator" OFF)
IF(ALPAKA_THREADS_ENABLE)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS ALPAKA_THREADS_ENABLED)
ENDIF()
OPTION(ALPAKA_FIBERS_ENABLE "Enable the fibers accelerator" OFF)
IF(ALPAKA_FIBERS_ENABLE)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS ALPAKA_FIBERS_ENABLED)
ENDIF()
OPTION(ALPAKA_OPENMP_ENABLE "Enable the OpenMP accelerator" ON)
IF(ALPAKA_OPENMP_ENABLE)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS ALPAKA_OPENMP_ENABLED)
ENDIF()
OPTION(ALPAKA_CUDA_ENABLE "Enable the CUDA accelerator" ON)
IF(ALPAKA_CUDA_ENABLE)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS ALPAKA_CUDA_ENABLED)
ENDIF()

################################################################################
# Debug option.
################################################################################
OPTION(ALPAKA_DEBUG "Debug level [0,1,2]" 0)
IF(ALPAKA_DEBUG)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS "ALPAKA_DEBUG=${ALPAKA_DEBUG}")
ENDIF()

################################################################################
# Find Boost.
################################################################################
SET(Boost_USE_STATIC_LIBS ON)
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)

IF(ALPAKA_FIBERS_ENABLE)
    LIST(APPEND BOOST_REQUIRED_COMPONENTS fiber coroutine context system atomic chrono date_time regex thread)
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
    FIND_PACKAGE(CUDA 6.5 REQUIRED)
    #SET(CUDA_VERBOSE_BUILD ON)
    SET(CUDA_PROPAGATE_HOST_FLAGS ON)

    IF(CUDA_VERSION VERSION_LESS 6.5)
        MESSAGE(FATAL_ERROR "CUDA Toolkit < 6.5 detected!")
    ENDIF()

    SET(CUDA_ARCH sm_20 CACHE STRING "Set GPU architecture")
    STRING(COMPARE EQUAL ${CUDA_ARCH} "sm_10" IS_CUDA_ARCH_UNSUPPORTED)
    STRING(COMPARE EQUAL ${CUDA_ARCH} "sm_11" IS_CUDA_ARCH_UNSUPPORTED)
    STRING(COMPARE EQUAL ${CUDA_ARCH} "sm_12" IS_CUDA_ARCH_UNSUPPORTED)
    STRING(COMPARE EQUAL ${CUDA_ARCH} "sm_13" IS_CUDA_ARCH_UNSUPPORTED)

    IF(IS_CUDA_ARCH_UNSUPPORTED)
        MESSAGE(FATAL_ERROR "Unsupported CUDA architecture ${CUDA_ARCH} specified. SM 2.0 or higher is required.")
    ENDIF(IS_CUDA_ARCH_UNSUPPORTED)
    
    IF(NOT MSVC)
        LIST(APPEND CUDA_NVCC_FLAGS -std=c++11)
        SET(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    ENDIF()
    
    SET(CUDA_FTZ "--ftz=false" CACHE STRING "Set flush to zero for GPU")
    SET(CUDA_MATH --use_fast_math CACHE STRING "Enable fast-math")

    LIST(APPEND CUDA_NVCC_FLAGS -arch=${CUDA_ARCH} ${CUDA_MATH} ${CUDA_FTZ})

    OPTION(CUDA_SHOW_REGISTER "Show kernel registers and create PTX" OFF)
    IF(CUDA_SHOW_REGISTER)
        LIST(APPEND CUDA_NVCC_FLAGS -Xptxas=-v)
    ENDIF(CUDA_SHOW_REGISTER)

    OPTION(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps (folder: nvcc_tmp)" OFF)
    IF(CUDA_KEEP_FILES)
        MAKE_DIRECTORY("${PROJECT_BINARY_DIR}/nvcc_tmp")
        LIST(APPEND CUDA_NVCC_FLAGS --keep --keep-dir "${PROJECT_BINARY_DIR}/nvcc_tmp")
    ENDIF(CUDA_KEEP_FILES)

    OPTION(CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)
    IF(CUDA_SHOW_CODELINES)
        LIST(APPEND CUDA_NVCC_FLAGS --source-in-ptx -Xcompiler -rdynamic -lineinfo)
        SET(CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
    ENDIF(CUDA_SHOW_CODELINES)

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
    # -D_SCL_SECURE_NO_WARNINGS
    #LIST(APPEND ALPAKA_COMPILE_OPTIONS "$<$<CONFIG:RELEASE>:-D_CRT_NONSTDC_NO_DEPRECATE>")
    #LIST(APPEND ALPAKA_COMPILE_OPTIONS "$<$<CONFIG:RELEASE>:-D_CRT_SECURE_NO_DEPRECATE>")
    #LIST(APPEND ALPAKA_COMPILE_OPTIONS "$<$<CONFIG:RELEASE>:-D_SCL_SECURE_NO_DEPRECATE>")
# GNU
ELSEIF(CMAKE_COMPILER_IS_GNUCXX)
    LIST(APPEND ALPAKA_COMPILE_OPTIONS "-Wall")
    LIST(APPEND ALPAKA_COMPILE_OPTIONS "-pedantic")
    LIST(APPEND ALPAKA_COMPILE_OPTIONS "-Wextra")
    LIST(APPEND ALPAKA_COMPILE_OPTIONS "-std=c++11")
# ICC
ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    LIST(APPEND ALPAKA_COMPILE_OPTIONS "-Wall")
    LIST(APPEND ALPAKA_COMPILE_OPTIONS "-std=c++11")
# PGI
ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "PGI")
    LIST(APPEND ALPAKA_COMPILE_OPTIONS "-Minform=inform")
    LIST(APPEND ALPAKA_COMPILE_OPTIONS "-std=c++11")
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

# CUDA does not work well with the way better target dependent TARGET_XXX commands: https://www.cmake.org/Bug/view.php?id=14201&nbn=1
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
