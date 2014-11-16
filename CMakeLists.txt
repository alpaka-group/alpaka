#
# Copyright 2014 Benjamin Worpitz
#
# This file is part of alpaka.
#
# alpaka is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alpaka is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with alpaka.
# If not, see <http://www.gnu.org/licenses/>.
#

################################################################################
# Required CMake version
################################################################################

#CMAKE_MINIMUM_REQUIRED(VERSION 2.8.5)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project 
################################################################################

MESSAGE(STATUS "### alpaka ###")

SET(ALPAKA_LIBRARY_DEPENDENCIES)
SET(ALPAKA_INCLUDE_DEPENDENCIES)
SET(ALPAKA_COMPILE_DEFINITIONS)

# set helper paths to find libraries and packages
#SET(CMAKE_PREFIX_PATH "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")    # "/usr/lib/x86_64-linux-gnu/"

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
# Build type (debug, release)
################################################################################
OPTION(ALPAKA_RELEASE "Build release version, disables all runtime asserts" OFF)
IF(ALPAKA_RELEASE)
    MESSAGE(STATUS "Release version")

    SET(CMAKE_BUILD_TYPE Release)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS NDEBUG)
ELSE()
    MESSAGE(STATUS "Debug version")

    SET(CMAKE_BUILD_TYPE Debug)
    LIST(APPEND ALPAKA_COMPILE_DEFINITIONS _DEBUG)

    LIST(APPEND CUDA_NVCC_FLAGS -g "-Xcompiler=-g")
ENDIF()


################################################################################
# Find Boost
################################################################################
SET(Boost_USE_STATIC_LIBS ON)
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)

SET(BOOST_REQUIRED_COMPONENTS)

IF(ALPAKA_FIBERS_ENABLE)
    SET(BOOST_REQUIRED_COMPONENTS fiber coroutine context system atomic chrono date_time regex thread)
ENDIF()

FIND_PACKAGE(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})
LIST(APPEND ALPAKA_INCLUDE_DEPENDENCIES ${Boost_INCLUDE_DIRS})
LIST(APPEND ALPAKA_LIBRARY_DEPENDENCIES ${Boost_LIBRARIES})


################################################################################
# Find OpenMP
################################################################################
IF(ALPAKA_OPENMP_ENABLE)
    FIND_PACKAGE( OpenMP REQUIRED)
    IF(OPENMP_FOUND)
        LIST(APPEND CMAKE_C_FLAGS ${OpenMP_C_FLAGS})
        LIST(APPEND CMAKE_CXX_FLAGS ${OpenMP_CXX_FLAGS})
        LIST(APPEND CMAKE_EXE_LINKER_FLAGS ${OpenMP_EXE_LINKER_FLAGS})
    ENDIF()
ENDIF()


################################################################################
# Find CUDA 
################################################################################
IF(ALPAKA_CUDA_ENABLE)
    FIND_PACKAGE(CUDA 6.5 REQUIRED)

    IF(CUDA_VERSION VERSION_LESS 6.5)
        MESSAGE(FATAL_ERROR "CUDA Toolkit < 6.5 detected!")
    ENDIF()
    
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11")
    SET(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

    SET(CUDA_ARCH sm_20 CACHE STRING "Set GPU architecture")
    STRING(COMPARE EQUAL ${CUDA_ARCH} "sm_10" IS_CUDA_ARCH_UNSUPPORTED)
    STRING(COMPARE EQUAL ${CUDA_ARCH} "sm_11" IS_CUDA_ARCH_UNSUPPORTED)
    STRING(COMPARE EQUAL ${CUDA_ARCH} "sm_12" IS_CUDA_ARCH_UNSUPPORTED)
    STRING(COMPARE EQUAL ${CUDA_ARCH} "sm_13" IS_CUDA_ARCH_UNSUPPORTED)

    IF(IS_CUDA_ARCH_UNSUPPORTED)
        MESSAGE(FATAL_ERROR "Unsupported CUDA architecture ${CUDA_ARCH} specified. SM 2.0 or higher is required.")
    ENDIF(IS_CUDA_ARCH_UNSUPPORTED)

    SET(CUDA_FTZ "--ftz=false" CACHE STRING "Set flush to zero for GPU")

    SET(CUDA_MATH --use_fast_math CACHE STRING "Enable fast-math" )
    OPTION(CUDA_SHOW_REGISTER "Show kernel registers and create PTX" OFF)
    OPTION(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps (folder: nvcc_tmp)" OFF)
    OPTION(CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)

    IF(CUDA_SHOW_CODELINES)
        LIST(APPEND CUDA_NVCC_FLAGS --source-in-ptx -Xcompiler -rdynamic -lineinfo)
        SET(CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
    ENDIF(CUDA_SHOW_CODELINES)

    LIST(APPEND CUDA_NVCC_FLAGS ${nvcc_flags} -arch=${CUDA_ARCH} ${CUDA_MATH} ${CUDA_FTZ})
    IF(CUDA_SHOW_REGISTER)
        SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xptxas=-v)
    ENDIF(CUDA_SHOW_REGISTER)

    IF(CUDA_KEEP_FILES)
        MAKE_DIRECTORY("${PROJECT_BINARY_DIR}/nvcc_tmp")
        SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --keep --keep-dir "${PROJECT_BINARY_DIR}/nvcc_tmp")
    ENDIF(CUDA_KEEP_FILES)

    LIST(APPEND ALPAKA_LIBRARY_DEPENDENCIES ${CUDA_CUDART_LIBRARY})
ENDIF()


################################################################################
# Compiler settings
################################################################################

#MSVC
IF(MSVC)
    # Force to always compile with W4
    IF(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        STRING(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    ELSE()
        LIST(APPEND CMAKE_CXX_FLAGS /W4)
    ENDIF()
    #LIST(APPEND CMAKE_CXX_FLAGS_RELEASE -D_CRT_NONSTDC_NO_DEPRECATE -D_CRT_SECURE_NO_DEPRECATE -D_SCL_SECURE_NO_DEPRECATE ") # -D_SCL_SECURE_NO_WARNINGS
# GNU
ELSEIF(CMAKE_COMPILER_IS_GNUCXX)
    LIST(APPEND CMAKE_CXX_FLAGS -Wall)
    LIST(APPEND CMAKE_CXX_FLAGS -pedantic)
    LIST(APPEND CMAKE_CXX_FLAGS -Wno-unknown-pragmas)
    LIST(APPEND CMAKE_CXX_FLAGS -Wextra)
    LIST(APPEND CMAKE_CXX_FLAGS -Wno-unused-parameter)
    # new warning in gcc 4.8 (flag ignored in previous version)
    LIST(APPEND CMAKE_CXX_FLAGS -Wno-unused-local-typedefs)
    LIST(APPEND CMAKE_CXX_FLAGS -std=c++11)
# ICC
ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    LIST(APPEND CMAKE_CXX_FLAGS -Wall)
    LIST(APPEND CMAKE_CXX_FLAGS -std=c++11)
# PGI
ELSEIF(${CMAKE_CXX_COMPILER_ID} STREQUAL "PGI")
    LIST(APPEND CMAKE_CXX_FLAGS -Minform=inform)
    LIST(APPEND CMAKE_CXX_FLAGS -std=c++11)
ENDIF()

  
################################################################################
# Compile & Link
################################################################################

INCLUDE("cmake/recursive_file_functions.cmake")

# alpaka
SET(ALPAKA_INCLUDE_DIR "include/")
SET(ALPAKA_SUFFIXED_INCLUDE_DIR "${ALPAKA_INCLUDE_DIR}alpaka/")
        #MESSAGE( "ALPAKA_SUFFIXED_INCLUDE_DIR: ${ALPAKA_SUFFIXED_INCLUDE_DIR}" )

# Add all the include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_source_group(${ALPAKA_SUFFIXED_INCLUDE_DIR} "hpp" ALPAKA_HEADER_FILES_ALL)
        #MESSAGE( "ALPAKA_HEADER_FILES_ALL: ${ALPAKA_HEADER_FILES_ALL}" )

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_source_group(${ALPAKA_SUFFIXED_INCLUDE_DIR} "cpp" ALPAKA_SOURCE_FILES_ALL)
        #MESSAGE( "ALPAKA_SOURCE_FILES_ALL: ${ALPAKA_SOURCE_FILES_ALL}" )
        
ADD_LIBRARY(
    alpaka 
    ${ALPAKA_HEADER_FILES_ALL}
    ${ALPAKA_SOURCE_FILES_ALL})

TARGET_INCLUDE_DIRECTORIES(alpaka PUBLIC ${ALPAKA_INCLUDE_DEPENDENCIES} ${ALPAKA_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(alpaka PUBLIC ${ALPAKA_LIBRARY_DEPENDENCIES})
TARGET_COMPILE_DEFINITIONS(alpaka PUBLIC "${ALPAKA_COMPILE_DEFINITIONS}")

MESSAGE(STATUS "### alpaka done ###")