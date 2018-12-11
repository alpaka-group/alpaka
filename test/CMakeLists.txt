#
# Copyright 2015-2018 Benjamin Worpitz, Axel Huebl
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

CMAKE_MINIMUM_REQUIRED(VERSION 3.11.0)

SET(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../" CACHE STRING "The location of the alpaka library")
LIST(APPEND CMAKE_MODULE_PATH "${ALPAKA_ROOT}")
FIND_PACKAGE(alpaka REQUIRED)

ADD_SUBDIRECTORY("common/")

find_package(Catch2 2.4.0 CONFIG REQUIRED)
set_target_properties(Catch2::Catch2 PROPERTIES IMPORTED_GLOBAL TRUE)
message(STATUS "Catch2: Found version ${Catch2_VERSION}")

add_library(CatchMain CatchMain.cpp)
# target_compile_features(CatchMain PUBLIC cxx_std_11)  # min C++11
set_target_properties(CatchMain PROPERTIES
    CXX_STANDARD 11  # exactly C++11
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(CatchMain PUBLIC "/bigobj")
endif()
target_include_directories(CatchMain SYSTEM PUBLIC
    $<TARGET_PROPERTY:Catch2::Catch2,INTERFACE_INCLUDE_DIRECTORIES>)

# NVCC does not incorporate the COMPILE_OPTIONS of a target but only the CMAKE_CXX_FLAGS
IF(ALPAKA_ACC_GPU_CUDA_ENABLE AND NOT ALPAKA_CUDA_COMPILER MATCHES "clang")
    GET_TARGET_PROPERTY(_COMMON_COMPILE_OPTIONS common COMPILE_OPTIONS)
    STRING(REPLACE ";" " " _COMMON_COMPILE_OPTIONS_STRING "${_COMMON_COMPILE_OPTIONS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_COMMON_COMPILE_OPTIONS_STRING}")
ENDIF()

LIST(APPEND _ALPAKA_TEST_OPTIONS "--use-colour yes")

ADD_SUBDIRECTORY("analysis/")
ADD_SUBDIRECTORY("integ/")
ADD_SUBDIRECTORY("unit/")
