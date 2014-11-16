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

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)


################################################################################
# Settings
################################################################################

# Set project macros.
SET( TARGET_NAME "basic" )
SET( INCLUDE_DIR "include/" )
SET( SUFFIXED_INCLUDE_DIR "${INCLUDE_DIR}/${TARGET_NAME}/" )
SET( SUFFIXED_SOURCE_DIR "src/" )

################################################################################
# Project 
################################################################################

PROJECT(${TARGET_NAME})

# Include the alpaka library.
ADD_SUBDIRECTORY("${CMAKE_CURRENT_SOURCE_DIR}/../../" "${CMAKE_CURRENT_BINARY_DIR}/alpaka")

# Add all the include files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_source_group(${SUFFIXED_INCLUDE_DIR} "hpp" HPP_FILES_ALL)
        #MESSAGE( "HPP_FILES_ALL: ${HPP_FILES_ALL}" )

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_source_group(${SUFFIXED_SOURCE_DIR} "cpp" CPP_FILES_ALL)
        #MESSAGE( "CPP_FILES_ALL: ${CPP_FILES_ALL}" )

# Add all the CUDA source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_source_group(${SUFFIXED_SOURCE_DIR} "cu" CU_FILES_ALL)
        #MESSAGE( "CU_FILES_ALL: ${CU_FILES_ALL}" )

# Builds the project as a static library and adds the headers and sources.
IF(ALPAKA_CUDA_ENABLE)
    CUDA_ADD_EXECUTABLE(${TARGET_NAME}
        ${HPP_FILES_ALL}
        ${CU_FILES_ALL}
        #OPTIONS -arch sm_20
    )
ELSE()
    ADD_EXECUTABLE(${TARGET_NAME}
        ${HEADER_FILES_ALL}
        ${CPP_FILES_ALL}
    )
ENDIF()

ADD_DEPENDENCIES(${TARGET_NAME} alpaka)

# Set the include directories for this library.
TARGET_INCLUDE_DIRECTORIES(${TARGET_NAME} PUBLIC ${INCLUDE_DIR})

# Set the include libraries for this library (adds libs and include directories).
TARGET_LINK_LIBRARIES(${TARGET_NAME} PUBLIC alpaka)