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

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Project.
################################################################################

SET(TARGET_NAME "sharedMem")
SET(INCLUDE_DIR "include/")
SET(SUFFIXED_INCLUDE_DIR "${INCLUDE_DIR}${TARGET_NAME}/")
SET(SUFFIXED_SOURCE_DIR "src/")

PROJECT("${TARGET_NAME}")

#-------------------------------------------------------------------------------
# Find alpaka.
#-------------------------------------------------------------------------------

SET(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../" CACHE STRING  "The location of the alpaka library")

LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
FIND_PACKAGE("alpaka" REQUIRED)

#-------------------------------------------------------------------------------
# Common.
#-------------------------------------------------------------------------------

INCLUDE("${ALPAKA_ROOT}/cmake/common.cmake")
INCLUDE("${ALPAKA_ROOT}/cmake/dev.cmake")

#-------------------------------------------------------------------------------
# Add library.
#-------------------------------------------------------------------------------

# Add all the include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${SUFFIXED_INCLUDE_DIR}" "" "hpp" HPP_FILES_ALL)

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${SUFFIXED_SOURCE_DIR}" "" "cpp" CPP_FILES_ALL)

# Add all the CUDA source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${SUFFIXED_SOURCE_DIR}" "" "cu" CU_FILES_ALL)

# Always add all files to the target executable build call to add them to the build project.
IF(ALPAKA_GPU_CUDA_ENABLE)
    # CUDA does not work well with the much better target dependent TARGET_XXX commands but requires the settings to be available globally: https://www.cmake.org/Bug/view.php?id=14201&nbn=1
    INCLUDE_DIRECTORIES(
        ${INCLUDE_DIR}
        ${alpaka_INCLUDE_DIRS})
    ADD_DEFINITIONS(
        ${alpaka_DEFINITIONS} ${ALPAKA_DEV_COMPILE_OPTIONS})
    # Force the .cpp files to be recognized as headers and not be compiled so there wont be a second entry point.
    SET_SOURCE_FILES_PROPERTIES(
        ${CPP_FILES_ALL}
        PROPERTIES HEADER_FILE_ONLY TRUE)
    CMAKE_POLICY(SET CMP0023 OLD)   # CUDA_ADD_EXECUTABLE calls TARGET_LINK_LIBRARIES without keywords.
    CUDA_ADD_EXECUTABLE(
        "${TARGET_NAME}"
        ${HPP_FILES_ALL} ${CPP_FILES_ALL} ${CU_FILES_ALL})
ELSE()
    # Force the .cu files to be recognized as headers and not be compiled so there wont be a second entry point.
    SET_SOURCE_FILES_PROPERTIES(
        ${CU_FILES_ALL}
        PROPERTIES HEADER_FILE_ONLY TRUE)
    ADD_EXECUTABLE(
        "${TARGET_NAME}"
        ${HPP_FILES_ALL} ${CPP_FILES_ALL} ${CU_FILES_ALL})
    TARGET_INCLUDE_DIRECTORIES(
        "${TARGET_NAME}"
        PUBLIC ${INCLUDE_DIR})
    TARGET_COMPILE_OPTIONS(
        "${TARGET_NAME}"
        PUBLIC ${ALPAKA_DEV_COMPILE_OPTIONS})
ENDIF()

# Set the link libraries for this library (adds libs, include directories, defines and compile options).
TARGET_LINK_LIBRARIES(
    "${TARGET_NAME}"
    PUBLIC "alpaka")
    