#
# Copyright 2015-2017 Benjamin Worpitz
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
# Preamble

cmake_minimum_required(VERSION 3.7.0)

project(alpaka VERSION 0.3.0
) # LANGUAGES CXX

################################################################################
# Project Structure

# install directories
set(CMAKE_INSTALL_BINDIR bin)
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_INCLUDEDIR include)
if(WIN32)
    set(CMAKE_INSTALL_CMAKEDIR)
else()
    set(CMAKE_INSTALL_CMAKEDIR)# lib/cmake/alpaka
endif()

################################################################################
# Options and Variants

option(alpaka_BUILD_EXAMPLES "Build the examples" ON)

include(CTest)
# automatically defines: BUILD_TESTING, default is ON

################################################################################
# Tests and Examples

if(alpaka_BUILD_EXAMPLES)
    add_subdirectory("example/")
endif()
if(BUILD_TESTING)
    add_subdirectory("test/")
endif()

################################################################################
# Generate Files with Configuration Options

#configure_file(
#    ${alpaka_SOURCE_DIR}/alpakaConfig.cmake.in
#    ${alpaka_BINARY_DIR}/alpakaConfig.cmake
#    @ONLY
#)
file(COPY
    ${alpaka_SOURCE_DIR}/alpakaConfig.cmake
    DESTINATION ${alpaka_BINARY_DIR}
)

include(CMakePackageConfigHelpers)

# alpaka is a header-only library and thus independ on the architecture.
# This removes the 32/64bit version specifier from the installed cmake package,
# allowing to use a 32bit install on a 64bit machine and vice versa.
set(_alpaka_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file("alpakaConfigVersion.cmake"
    VERSION ${alpaka_VERSION}
    COMPATIBILITY SameMajorVersion
)
set(CMAKE_SIZEOF_VOID_P ${_alpaka_CMAKE_SIZEOF_VOID_P})

################################################################################
# Installs

# headers
install(DIRECTORY "${alpaka_SOURCE_DIR}/include"
    DESTINATION include
)

# CMake package file for find_package(alpaka::alpaka) in depending projects
#install(EXPORT alpakaTargets
#    FILE alpakaTargets.cmake
#    NAMESPACE alpaka::
#    DESTINATION lib/cmake/alpaka
#)
install(
    FILES
        ${alpaka_BINARY_DIR}/alpakaConfig.cmake
        ${alpaka_BINARY_DIR}/alpakaConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_CMAKEDIR}
)

################################################################################
# Status Message for Build Options

message("")
message("alpaka build configuration:")
message("  library Version: ${alpaka_VERSION}")
message("  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} "
                        "${CMAKE_CXX_COMPILER_VERSION} "
                        "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("")
message("  Installation prefix: ${CMAKE_INSTALL_PREFIX}")
message("        bin: ${CMAKE_INSTALL_BINDIR}")
message("        lib: ${CMAKE_INSTALL_LIBDIR}")
message("    include: ${CMAKE_INSTALL_INCLUDEDIR}")
message("      cmake: ${CMAKE_INSTALL_CMAKEDIR}")
message("")
message("  Build Type: ${CMAKE_BUILD_TYPE}")
message("  Testing: ${BUILD_TESTING}")
message("  Examples: ${alpaka_BUILD_EXAMPLES}")
message("")
