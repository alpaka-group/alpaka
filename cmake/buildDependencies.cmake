#
# Copyright 2025 Simeon Ehrig
# SPDX-License-Identifier: ISC
#

option(alpaka_SYSTEM_CATCH2 "Use the system provided Catch2." OFF)

#------------------------------------------------------------------------------
# Install Catch2 if not already done.
function(alpaka_install_catch2)
    if(NOT TARGET Catch2::Catch2)
        if(alpaka_SYSTEM_CATCH2)
            message(STATUS "use System Catch2")
            message(
                WARNING
                "Your System version of CATCH2 V3 should have set '-DCATCH_CONFIG_NO_POSIX_SIGNALS=ON', else some tests e.g. with managed memory can fail."
            )
            find_package(Catch2 3.15.0 REQUIRED)
            include(Catch)
        else()
            message(STATUS "download and install Catch2 locally")
            # bug with SYCL alloc_shared and Catch2 V3 https://github.com/intel/llvm/issues/6720#issuecomment-1882988322
            set(CATCH_CONFIG_NO_POSIX_SIGNALS ON)
            # get Catch2 v3 and build it from source with the same C++ standard as the tests
            include(FetchContent)
            # SYSTEM: code from Catch2 headers and macros, e.g. TEST_CASE, produces no compiler warnings or clang-tidy findings
            FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.15.0 SYSTEM)
            FetchContent_MakeAvailable(Catch2)
            target_compile_features(Catch2 PUBLIC cxx_std_20)
            include(Catch)

            # hide Catch2 cmake variables by default in cmake gui
            get_cmake_property(variables VARIABLES)
            foreach(var ${variables})
                if(var MATCHES "^CATCH_")
                    mark_as_advanced(${var})
                endif()
            endforeach()
        endif()
    else()
        message(STATUS "Catch2 is already installed")
    endif()
endfunction()
