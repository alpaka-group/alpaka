#
# Copyright 2023 Benjamin Worpitz, Jeffrey Kelling, Bernhard Manfred Gruber, René Widera, Jan Stephan
# SPDX-License-Identifier: MPL-2.0
#

#-------------------------------------------------------------------------------
# Compiler settings.
#-------------------------------------------------------------------------------
if(alpaka_ACC_GPU_CUDA_ENABLE AND (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"))
    list(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-Wdefault-stream-launch -Werror=default-stream-launch>)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.3)
        # supress error in Catch: 'error #177-D: variable "<unnamed>::autoRegistrar1" was declared but never referenced'
        list(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=177>)
    endif()
endif()

if(MSVC)
    # Force to always compile with W4
    list(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
    if(alpaka_ENABLE_WERROR)
        # WX treats warnings as errors
        list(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/WX> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>)
    endif()
    # Improve debugging.
    list(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Zo>
                                           $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/Zo>)

    # Flags added in Visual Studio 2013
    list(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/Zc:throwingNew> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:throwingNew>)
    list(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/Zc:strictStrings> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:strictStrings>)

    # Flags added in Visual Studio 2015
    list(APPEND alpaka_DEV_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/permissive-> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/permissive->)
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wall")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wextra")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-pedantic")
    if(alpaka_ENABLE_WERROR)
        list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Werror")
    endif()
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wdouble-promotion")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wmissing-include-dirs")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wconversion")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wunknown-pragmas")
    # Higher levels (max is 5) produce some strange warnings
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wstrict-overflow=2")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wtrampolines")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wfloat-equal")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wundef")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wshadow")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wcast-qual")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wcast-align")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wwrite-strings")
    # Too noisy as it warns for every operation using numeric types smaller than int.
    # Such values are converted to int implicitly before the calculation is done.
    # E.g.: uint16_t = uint16_t * uint16_t will trigger the following warning:
    # conversion to ‘short unsigned int’ from ‘int’ may alter its value
    #list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wconversion")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wsign-conversion")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wvector-operation-performance")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wzero-as-null-pointer-constant")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wdate-time")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wuseless-cast")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wlogical-op")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-aggressive-loop-optimizations")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wmissing-declarations")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-multichar")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wopenmp-simd")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wpacked")
    # Too much noise
    #list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wpadded")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wredundant-decls")
    # Too much noise
    #list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Winline")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wdisabled-optimization")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wformat-nonliteral")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wformat-security")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wformat-y2k")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wctor-dtor-privacy")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wdelete-non-virtual-dtor")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wliteral-suffix")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wnon-virtual-dtor")
    # This warns about members that have not explicitly been listed in the constructor initializer list.
    # This could be useful even for members that have a default constructor.
    # However, it also issues this warning for defaulted constructurs.
    #list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Weffc++")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Woverloaded-virtual")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wsign-promo")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wconditionally-supported")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wnoexcept")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wold-style-cast")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wsuggest-final-types")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wsuggest-final-methods")
    # This does not work correctly as it suggests override to methods that are already marked with final.
    # Because final implies override, this is not useful.
    #list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wsuggest-override")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wnormalized")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wformat-signedness")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wnull-dereference")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wduplicated-cond")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wsubobject-linkage")
    # This warning might be useful but it is triggered by compile-time code where it does not make any sense:
    # E.g. "Vec<DimInt<(TidxDimOut < TidxDimIn) ? TidxDimIn : TidxDimOut>, TElem>" when both values are equal
    #list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wduplicated-branches")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Walloc-zero")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Walloca")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wcast-align=strict")
endif()

# Clang, AppleClang, ICPX
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    if(alpaka_ENABLE_WERROR)
        list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Werror")
    endif()
    # Weverything really means everything (including Wall, Wextra, pedantic, ...)
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Weverything")
    # We are not C++98 compatible
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-c++98-compat")
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-c++98-compat-pedantic")
    # Triggered by inline constants
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-global-constructors")
    # This padding warning is generated by the execution tasks depending on the argument types
    # as they are stored as members. Therefore, the padding warning is triggered by the calling code
    # and does not indicate a failure within alpaka.
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-padded")
    # Triggers for all instances of alpaka_DEBUG_MINIMAL_LOG_SCOPE and similar macros followed by semicolon
    list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-extra-semi-stmt")

    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
        list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-poison-system-directories")
    endif()

    if(alpaka_ACC_GPU_HIP_ENABLE)
        list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-unused-command-line-argument")
        if(HIP_VERSION VERSION_LESS_EQUAL 5.3)
            # avoid error:
            #  rocrand/rocrand_common.h:73:6: error: "Disabled inline asm, because
            #  the build target does not support it." [-Werror,-W#warnings]
            #  #warning "Disabled inline asm, because the build target does not support it."
            list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-error=#warnings")
        endif()
    endif()

    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM")
        # fast math is turned on by default with ICPX, which breaks our unit tests
        list(APPEND alpaka_DEV_COMPILE_OPTIONS "-ffp-model=precise")

        if (alpaka_ACC_SYCL_ENABLE)
            # avoid: warning: disabled expansion of recursive macro
            list(APPEND alpaka_DEV_COMPILE_OPTIONS "-Wno-disabled-macro-expansion")
        endif()
    endif()
endif()

