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

SET(_INCLUDE_DIR "include/")
SET(_SUFFIXED_INCLUDE_DIR "${_INCLUDE_DIR}vectorAdd/")
SET(_SOURCE_DIR "src/")

PROJECT("vectorAdd")

#-------------------------------------------------------------------------------
# Find alpaka.
#-------------------------------------------------------------------------------

SET(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../" CACHE STRING  "The location of the alpaka library")

LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
FIND_PACKAGE("alpaka" REQUIRED)

#-------------------------------------------------------------------------------
# Common.
#-------------------------------------------------------------------------------

INCLUDE("${ALPAKA_ROOT}cmake/common.cmake")
INCLUDE("${ALPAKA_ROOT}cmake/dev.cmake")
SET(_INCLUDE_DIRECTORIES_PRIVATE ${_INCLUDE_DIR} "${ALPAKA_ROOT}examples/common/")

#-------------------------------------------------------------------------------
# Add library.
#-------------------------------------------------------------------------------

# Add all the include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_SUFFIXED_INCLUDE_DIR}" "" "hpp" _FILES_HEADER)

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_SOURCE_DIR}" "" "cpp" _FILES_SOURCE_CXX)

# Add all the CUDA source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_SOURCE_DIR}" "" "cu" _FILES_SOURCE_CU)

# Always add all files to the target executable build call to add them to the build project.
IF(ALPAKA_ACC_GPU_CUDA_ENABLE)
    # Force the .cpp files to be recognized as headers and not be compiled so there wont be a second entry point.
    SET_SOURCE_FILES_PROPERTIES(
        ${_FILES_SOURCE_CXX}
        PROPERTIES HEADER_FILE_ONLY TRUE)
    # CUDA does not work well with the much better target dependent TARGET_XXX commands but requires the settings to be available globally: https://www.cmake.org/Bug/view.php?id=14201&nbn=1
    INCLUDE_DIRECTORIES(
        ${_INCLUDE_DIRECTORIES_PRIVATE}
        ${alpaka_INCLUDE_DIRS})
    ADD_DEFINITIONS(
        ${alpaka_DEFINITIONS} ${ALPAKA_DEV_COMPILE_OPTIONS})
    CMAKE_POLICY(SET CMP0023 OLD)   # CUDA_ADD_EXECUTABLE calls TARGET_LINK_LIBRARIES without keywords.
    CUDA_ADD_EXECUTABLE(
        "vectorAdd"
        ${_FILES_HEADER} ${_FILES_SOURCE_CXX} ${_FILES_SOURCE_CU})
ELSE()
    # Force the .cu files to be recognized as headers and not be compiled so there wont be a second entry point.
    SET_SOURCE_FILES_PROPERTIES(
        ${_FILES_SOURCE_CU}
        PROPERTIES HEADER_FILE_ONLY TRUE)
    ADD_EXECUTABLE(
        "vectorAdd"
        ${_FILES_HEADER} ${_FILES_SOURCE_CXX} ${_FILES_SOURCE_CU})
    TARGET_INCLUDE_DIRECTORIES(
        "vectorAdd"
        PRIVATE ${_INCLUDE_DIRECTORIES_PRIVATE})
    TARGET_COMPILE_OPTIONS(
        "vectorAdd"
        PRIVATE ${ALPAKA_DEV_COMPILE_OPTIONS})
ENDIF()

# Set the link libraries for this library (adds libs, include directories, defines and compile options).
TARGET_LINK_LIBRARIES(
    "vectorAdd"
    PUBLIC "alpaka")
