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
# Required cmake version
################################################################################

cmake_minimum_required(VERSION 2.8.5)

################################################################################
# Project 
################################################################################

project(test)

SET(TARGET_NAME matmul

# set helper paths to find libraries and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")

OPTION(ALPAKA_SERIAL_ENABLE "Enable the serial accelerator" OFF)
IF(ALPAKA_SERIAL_ENABLE)
    ADD_DEFINITIONS(-DALPAKA_SERIAL_ENABLED)
ENDIF()
OPTION(ALPAKA_THREADS_ENABLE "Enable the threas accelerator" OFF)
IF(ALPAKA_THREADS_ENABLE)
    ADD_DEFINITIONS(-DALPAKA_THREADS_ENABLED)
ENDIF()
OPTION(ALPAKA_FIBERS_ENABLE "Enable the fibers accelerator" OFF)
IF(ALPAKA_FIBERS_ENABLE)
    ADD_DEFINITIONS(-DALPAKA_FIBERS_ENABLED)
ENDIF()
OPTION(ALPAKA_OPENMP_ENABLE "Enable the OpenMP accelerator" ON)
IF(ALPAKA_OPENMP_ENABLE)
    ADD_DEFINITIONS(-DALPAKA_OPENMP_ENABLED)
ENDIF()
OPTION(ALPAKA_CUDA_ENABLE "Enable the CUDA accelerator" ON)
IF(ALPAKA_CUDA_ENABLE)
    ADD_DEFINITIONS(-DALPAKA_CUDA_ENABLED)
ENDIF()


################################################################################
# Build type (debug, release)
################################################################################
OPTION(ALPAKA_RELEASE "Build release version, disables all runtime asserts" OFF)
if(ALPAKA_RELEASE)
    message(STATUS "Release version")

    set(CMAKE_BUILD_TYPE Release)
    add_definitions(-DNDEBUG)
else()
    message(STATUS "Debug version")

    set(CMAKE_BUILD_TYPE Debug)
    add_definitions(-D_DEBUG)

    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -g "-Xcompiler=-g")
endif()


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
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
SET(LIBS ${LIBS} ${Boost_LIBRARIES})


################################################################################
# Find OpenMP
################################################################################
IF(ALPAKA_OPENMP_ENABLE)
    FIND_PACKAGE( OpenMP REQUIRED)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
ENDIF()


################################################################################
# Find CUDA 
################################################################################
IF(ALPAKA_CUDA_ENABLE)
    find_package(CUDA 6.5 REQUIRED)

    if(CUDA_VERSION VERSION_LESS 6.5)
        message(FATAL_ERROR "CUDA Toolkit < 6.5 detected!")
    endif()
    
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++11")
    SET(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

    set(CUDA_ARCH sm_20 CACHE STRING "Set GPU architecture")
    string(COMPARE EQUAL ${CUDA_ARCH} "sm_10" IS_CUDA_ARCH_UNSUPPORTED)
    string(COMPARE EQUAL ${CUDA_ARCH} "sm_11" IS_CUDA_ARCH_UNSUPPORTED)
    string(COMPARE EQUAL ${CUDA_ARCH} "sm_12" IS_CUDA_ARCH_UNSUPPORTED)
    string(COMPARE EQUAL ${CUDA_ARCH} "sm_13" IS_CUDA_ARCH_UNSUPPORTED)

    if(IS_CUDA_ARCH_UNSUPPORTED)
        message(FATAL_ERROR "Unsupported CUDA architecture ${CUDA_ARCH} specified. SM 2.0 or higher is required.")
    endif(IS_CUDA_ARCH_UNSUPPORTED)

    set(CUDA_FTZ "--ftz=false" CACHE STRING "Set flush to zero for GPU")

    set(CUDA_MATH --use_fast_math CACHE STRING "Enable fast-math" )
    option(CUDA_SHOW_REGISTER "Show kernel registers and create PTX" OFF)
    option(CUDA_KEEP_FILES "Keep all intermediate files that are generated during internal compilation steps (folder: nvcc_tmp)" OFF)
    option(CUDA_SHOW_CODELINES "Show kernel lines in cuda-gdb and cuda-memcheck" OFF)

    if(CUDA_SHOW_CODELINES)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --source-in-ptx -Xcompiler -rdynamic -lineinfo)
        set(CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
    endif(CUDA_SHOW_CODELINES)

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${nvcc_flags} -arch=${CUDA_ARCH} ${CUDA_MATH} ${CUDA_FTZ})
    if(CUDA_SHOW_REGISTER)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xptxas=-v)
    endif(CUDA_SHOW_REGISTER)

    if(CUDA_KEEP_FILES)
        make_directory("${PROJECT_BINARY_DIR}/nvcc_tmp")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --keep --keep-dir "${PROJECT_BINARY_DIR}/nvcc_tmp")
    endif(CUDA_KEEP_FILES)
endif()


################################################################################
# Compiler settings
################################################################################

# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
    # new warning in gcc 4.8 (flag ignored in previous version)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# ICC
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# PGI
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "PGI")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minform=inform")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()


################################################################################
# Configure include directories
################################################################################

# alpaka
find_path(ALPAKA_ROOT_DIR
  NAMES "include/alpaka/IAcc.hpp"
  PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../"
  DOC "alpaka root location"
  )
include_directories(${ALPAKA_ROOT_DIR}/include)


################################################################################
# Compile & Link
################################################################################

# alpaka
FILE(GLOB_RECURSE ALPAKA_HPP_FILES "${ALPAKA_ROOT_DIR}/include/*.hpp")
FILE(GLOB_RECURSE ALPAKA_CPP_FILES "${ALPAKA_ROOT_DIR}/include/*.cpp")

ADD_LIBRARY(
    alpaka 
    ${ALPAKA_HPP_FILES}
    ${ALPAKA_CPP_FILES})

# target

FILE(GLOB_RECURSE HPP_FILES "include/*.hpp")
FILE(GLOB_RECURSE CPP_FILES "src/*.cpp")
FILE(GLOB_RECURSE CU_FILES "src/*.cu")

IF(ALPAKA_CUDA_ENABLE)
    cuda_add_executable(${TARGET_NAME}
        ${HPP_FILES}
        ${CU_FILES}
        #OPTIONS -arch sm_20
    )
    SET(LIBS ${LIBS} ${CUDA_CUDART_LIBRARY})

ELSE()
    ADD_EXECUTABLE(${TARGET_NAME}
        ${HPP_FILES}
        ${CPP_FILES}
    )
ENDIF()

# z m
TARGET_LINK_LIBRARIES(${TARGET_NAME} ${LIBS})
