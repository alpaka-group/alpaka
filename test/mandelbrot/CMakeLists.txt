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
# Required CMake version
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.5)

################################################################################
# Settings
################################################################################

# Set project macros.
SET(TARGET_NAME mandelbrot)
SET(INCLUDE_DIR "include/")
SET(SUFFIXED_INCLUDE_DIR "${INCLUDE_DIR}${TARGET_NAME}/")
SET(SUFFIXED_SOURCE_DIR "src/")

################################################################################
# Project 
################################################################################

PROJECT(${TARGET_NAME})

# Include the alpaka library.
INCLUDE("${CMAKE_CURRENT_LIST_DIR}/../../CMakeLists.txt")

# Add all the include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_source_group(${SUFFIXED_INCLUDE_DIR} "hpp" HPP_FILES_ALL)

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_source_group(${SUFFIXED_SOURCE_DIR} "cpp" CPP_FILES_ALL)

# Add all the CUDA source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_source_group(${SUFFIXED_SOURCE_DIR} "cu" CU_FILES_ALL)

INCLUDE_DIRECTORIES(${INCLUDE_DIR})

# Builds the project as a static library and adds the headers and sources.
# Always add all files to the target executable build call to add them to the build project.
IF(ALPAKA_CUDA_ENABLE)
    # Force the .cpp files to be recognized as headers and not be compiled so there wont be a second entry point.
    SET_SOURCE_FILES_PROPERTIES(${CPP_FILES_ALL} PROPERTIES HEADER_FILE_ONLY TRUE)
    CMAKE_POLICY(SET CMP0023 OLD)   # CUDA_ADD_EXECUTABLE calls TARGET_LINK_LIBRARIES without keywords.
    CUDA_ADD_EXECUTABLE(${TARGET_NAME}
        ${HPP_FILES_ALL}
        ${CPP_FILES_ALL}
        ${CU_FILES_ALL}
    )
ELSE()
    # Force the .cu files to be recognized as headers and not be compiled so there wont be a second entry point.
    IF(NOT ALPAKA_CUDA_ENABLE)
        SET_SOURCE_FILES_PROPERTIES(${CU_FILES_ALL} PROPERTIES HEADER_FILE_ONLY TRUE)
    ENDIF()
    ADD_EXECUTABLE(${TARGET_NAME}
        ${HPP_FILES_ALL}
        ${CPP_FILES_ALL}
        ${CU_FILES_ALL}
    )
ENDIF()

# Set the link libraries for this library (adds libs and include directories).
TARGET_LINK_LIBRARIES(${TARGET_NAME} PUBLIC alpaka)
