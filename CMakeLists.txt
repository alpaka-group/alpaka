#
# Copyright 2023 Benjamin Worpitz, Erik Zenker, Axel Hübl, Jan Stephan, René Widera, Jeffrey Kelling, Andrea Bocci,
#                Bernhard Manfred Gruber, Aurora Perego, Tim Hanel
# SPDX-License-Identifier: MPL-2.0
#

cmake_minimum_required(VERSION 3.25)

# Find alpaka version.
file(
    STRINGS
    "${CMAKE_CURRENT_LIST_DIR}/include/alpaka/version.hpp"
    alpaka_VERSION_MAJOR_HPP
    REGEX "#define ALPAKA_VERSION_MAJOR "
)
file(
    STRINGS
    "${CMAKE_CURRENT_LIST_DIR}/include/alpaka/version.hpp"
    alpaka_VERSION_MINOR_HPP
    REGEX "#define ALPAKA_VERSION_MINOR "
)
file(
    STRINGS
    "${CMAKE_CURRENT_LIST_DIR}/include/alpaka/version.hpp"
    alpaka_VERSION_PATCH_HPP
    REGEX "#define ALPAKA_VERSION_PATCH "
)

string(REGEX MATCH "([0-9]+)" alpaka_VERSION_MAJOR ${alpaka_VERSION_MAJOR_HPP})
string(REGEX MATCH "([0-9]+)" alpaka_VERSION_MINOR ${alpaka_VERSION_MINOR_HPP})
string(REGEX MATCH "([0-9]+)" alpaka_VERSION_PATCH ${alpaka_VERSION_PATCH_HPP})

set(PACKAGE_VERSION "${alpaka_VERSION_MAJOR}.${alpaka_VERSION_MINOR}.${alpaka_VERSION_PATCH}")

project(
    alpaka
    VERSION ${alpaka_VERSION_MAJOR}.${alpaka_VERSION_MINOR}.${alpaka_VERSION_PATCH}
    DESCRIPTION "The alpaka library is a header-only C++20 abstraction library for accelerator development."
    HOMEPAGE_URL "https://github.com/alpaka-group/alpaka3"
    LANGUAGES NONE
)

# This file's directory.
set(_alpaka_ROOT_DIR ${alpaka_SOURCE_DIR})

# Compiler feature tests.
set(_alpaka_FEATURE_TESTS_DIR "${_alpaka_ROOT_DIR}/cmake/test")
set(_alpaka_CMAKE_DIR "${_alpaka_ROOT_DIR}/cmake")
set(_alpaka_TESTS_DIR "${_alpaka_ROOT_DIR}/test")
# Set include directories
set(_alpaka_INCLUDE_DIRECTORY "${_alpaka_ROOT_DIR}/include")

# CMake Options
option(alpaka_TESTS "Enable/Disable testing" OFF)
option(alpaka_DOCS "Enable/Disable documentation code snippet testing" OFF)
option(alpaka_BENCHMARKS "Enable/Disable benchmarks" OFF)
option(alpaka_EXAMPLES "Enable/Disable examples" OFF)
option(alpaka_HEADERCHECKS "Enable/Disable header check tests" OFF)
option(alpaka_CLANG_TIDY "Enable/Disable clang-tidy tests" OFF)

# only check for compiler support if this cmake is called with add_subdirectory or examples, benchmarks, ... should be compiled too
if(
    NOT PROJECT_NAME STREQUAL CMAKE_PROJECT_NAME
    OR (alpaka_TESTS OR alpaka_DOCS OR alpaka_BENCHMARKS OR alpaka_EXAMPLES OR alpaka_HEADERCHECKS OR alpaka_CLANG_TIDY)
)
    include(${_alpaka_CMAKE_DIR}/alpakaCommon.cmake)
endif()

if(alpaka_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory("${_alpaka_TESTS_DIR}")
endif()

if(alpaka_DOCS)
    include(CTest)
    enable_testing()
    add_subdirectory("${_alpaka_ROOT_DIR}/docs/snippets")
endif()

if(alpaka_BENCHMARKS)
    enable_testing()
    add_subdirectory("${_alpaka_ROOT_DIR}/benchmark")
endif()

if(alpaka_EXAMPLES)
    enable_testing()
    add_subdirectory("${_alpaka_ROOT_DIR}/example")
endif()

if(alpaka_HEADERCHECKS)
    enable_testing()
    add_subdirectory("${_alpaka_ROOT_DIR}/headerCheck")
endif()

if(alpaka_CLANG_TIDY)
    enable_testing()
    add_subdirectory("${_alpaka_ROOT_DIR}/clangTidy")
endif()

# Installation and config
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

write_basic_package_version_file("alpakaConfigVersion.cmake" VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

install(
    FILES
        "${_alpaka_ROOT_DIR}/cmake/finalize.cmake"
        "${_alpaka_ROOT_DIR}/cmake/alpakaCommon.cmake"
        "${_alpaka_ROOT_DIR}/cmake/alpakaCuda.cmake"
        "${_alpaka_ROOT_DIR}/cmake/alpakaHip.cmake"
        "${_alpaka_ROOT_DIR}/cmake/alpakaOneApi.cmake"
        "${_alpaka_ROOT_DIR}/cmake/alpakaHwloc.cmake"
        "${_alpaka_ROOT_DIR}/cmake/buildDependencies.cmake"
        "${_alpaka_ROOT_DIR}/cmake/common.cmake"
        "${_alpaka_ROOT_DIR}/cmake/alpakaCompilePedantic.cmake"
        "${PROJECT_BINARY_DIR}/alpakaConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpaka
)

configure_package_config_file(
    ${_alpaka_ROOT_DIR}/cmake/alpakaConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/alpakaConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpaka
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_DATAROOTDIR
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/alpakaConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpaka)

# copy tests to the installation
install(DIRECTORY ${_alpaka_TESTS_DIR} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR} USE_SOURCE_PERMISSIONS)

# Files required to validate optional features std::atomic_ref and std::simd
install(FILES "${_alpaka_FEATURE_TESTS_DIR}/StdAtomicRef.cpp" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpaka/test)
install(FILES "${_alpaka_FEATURE_TESTS_DIR}/StdSimd.cpp" DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpaka/test)
