#
# Copyright 2014-2025 Benjamin Worpitz, René Widera, Mehmet Yusufoglu
# SPDX-License-Identifier: MPL-2.0
#

# enable CXX because this is not done in the CMakeLists.txt of alpaka
enable_language(CXX)

include(${_alpaka_CMAKE_DIR}/buildDependencies.cmake)
include(${_alpaka_CMAKE_DIR}/finalize.cmake)
include(${_alpaka_CMAKE_DIR}/common.cmake)

# Add append compiler flags to a variable or target
#
# This method is automatically documenting all compile flags added into the variables
# alpaka_COMPILER_OPTIONS_HOST, alpaka_COMPILER_OPTIONS_DEVICE.
#
# scope - which compiler is effected: DEVICE, HOST, or HOST_DEVICE
# type - type of 'name': var, list, or target
#        var: space separated list
#        list: is semicolon separated
# name - name of the variable or target
# ... - parameter to appended to the variable or target 'name'
function(alpaka_set_compiler_options scope type name)
    if(scope STREQUAL HOST)
        set(alpaka_COMPILER_OPTIONS_HOST ${alpaka_COMPILER_OPTIONS_HOST} ${ARGN} PARENT_SCOPE)
    elseif(scope STREQUAL DEVICE)
        set(alpaka_COMPILER_OPTIONS_DEVICE ${alpaka_COMPILER_OPTIONS_DEVICE} ${ARGN} PARENT_SCOPE)
    elseif(scope STREQUAL HOST_DEVICE)
        set(alpaka_COMPILER_OPTIONS_HOST ${alpaka_COMPILER_OPTIONS_HOST} ${ARGN} PARENT_SCOPE)
        set(alpaka_COMPILER_OPTIONS_DEVICE ${alpaka_COMPILER_OPTIONS_DEVICE} ${ARGN} PARENT_SCOPE)
    else()
        message(
            FATAL_ERROR
            "alpaka_set_compiler_option 'scope' unknown, value must be 'HOST', 'DEVICE', or 'HOST_DEVICE'."
        )
    endif()
    if(type STREQUAL "list")
        set(${name} ${${name}} ${ARGN} PARENT_SCOPE)
    elseif(type STREQUAL "var")
        foreach(arg IN LISTS ARGN)
            set(tmp "${tmp} ${arg}")
        endforeach()
        set(${name} "${${name}} ${tmp}" PARENT_SCOPE)
    elseif(type STREQUAL "target")
        foreach(arg IN LISTS ARGN)
            target_compile_options(${name} INTERFACE ${arg})
        endforeach()
    else()
        message(
            FATAL_ERROR
            "alpaka_set_compiler_option 'type=${type}' unknown, value must be 'list', 'var', or 'target'."
        )
    endif()
endfunction()

# Compiler options
macro(alpaka_compiler_option name description default)
    if(NOT DEFINED alpaka_${name})
        set(alpaka_${name} ${default} CACHE STRING "${description}")
        set_property(CACHE alpaka_${name} PROPERTY STRINGS "DEFAULT;ON;OFF")
    endif()
endmacro()

# Check if compiler supports required C++ standard.
#
# language - can be CXX, HIP or CUDA
# min_cxx_standard - C++ standard which is the minimum requirement
function(checkCompilerCXXSupport language min_cxx_standard)
    string(TOUPPER "${language}" language_upper_case)
    string(TOLOWER "${language}" language_lower_case)

    if(NOT "${language_lower_case}_std_${min_cxx_standard}" IN_LIST CMAKE_${language_upper_case}_COMPILE_FEATURES)
        message(
            FATAL_ERROR
            "The ${language_upper_case} compiler does not support C++ ${min_cxx_standard}. \
        Please upgrade your compiler or use alpaka 3.0 which supports C++20."
        )
    endif()
endfunction()

set(alpaka_CXX_STANDARD 20 CACHE STRING "C++ standard")

checkcompilercxxsupport(CXX ${alpaka_CXX_STANDARD})

# check for CUDA/HIP language support
include(CheckLanguage)

option(alpaka_DEP_CUDA "Enable the CUDA as dependency, allows the usage of api::Cuda and exec::gpuCuda." OFF)
option(alpaka_DEP_HIP "Enable the HIP as dependency, allows the usage of api::Hip and exec::gpuHip" OFF)
option(alpaka_DEP_OMP "Enable the OpenMP as dependency, allows the usage of exec::cpuOmpBlocks" ON)
option(alpaka_DEP_TBB "Enable the Intel oneTBB dependency, allows the usage of exec::cpuTbbBlocks" OFF)
option(alpaka_DEP_ONEAPI "Enable the Intel oneAPI SYCL dependency, allows using exec::oneApi" OFF)
option(
    alpaka_LANGUAGE_SEPARATION
    "Enable copying C++ source files used for CUDA/HIP compilation to a unique folder and compiled from there, instead of using the original file."
    ON
)

set(alpaka_DEP_HWLOC
    "AUTO"
    CACHE STRING
    "Enable the OpenMP as dependency, allows the usage of deviceKind::numaCpu (OFF, ON, AUTO)"
)
set_property(CACHE alpaka_DEP_HWLOC PROPERTY STRINGS OFF ON AUTO)

# Unified compiler options
alpaka_compiler_option(FAST_MATH "Enable fast-math" DEFAULT)
alpaka_compiler_option(FTZ "Set flush to zero" DEFAULT)

alpaka_compiler_option(RELOCATABLE_DEVICE_CODE "Enable relocatable device code for CUDA, HIP and SYCL devices" DEFAULT)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(alpaka_GCC_CONCEPT_DEPTH
        5
        CACHE STRING
        "Setup for GCC: How many nested concepts are displayed in the event of an error?"
    )
endif()

# Create core targets
if(NOT TARGET alpaka)
    ## target: alpaka::header
    add_library(alpaka_target_headers INTERFACE)
    add_library(alpaka::headers ALIAS alpaka_target_headers)
    target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_CMAKE_TARGET_HEADERS)

    add_library(alpaka INTERFACE)
    set_target_properties(alpaka PROPERTIES VERSION ${alpaka_VERSION})

    get_target_property(version alpaka VERSION)
    message(STATUS "Alpaka version: ${version}")

    add_library(alpaka::alpaka ALIAS alpaka)
    target_compile_definitions(alpaka INTERFACE ALPAKA_CMAKE_TARGET_ALPAKA)

    add_library(alpaka_target_host INTERFACE)
    add_library(alpaka::host ALIAS alpaka_target_host)
    target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_CMAKE_TARGET_HOST)
    target_link_libraries(alpaka_target_host INTERFACE alpaka_target_headers)
    target_link_libraries(alpaka INTERFACE alpaka::host)

    # the alpaka library itself
    # SYSTEM voids showing warnings produced by alpaka when used in user applications.
    if(alpaka_TESTS)
        target_include_directories(
            alpaka
            INTERFACE $<BUILD_INTERFACE:${_alpaka_INCLUDE_DIRECTORY}> $<INSTALL_INTERFACE:include>
        )
    else()
        target_include_directories(
            alpaka
            SYSTEM
            INTERFACE $<BUILD_INTERFACE:${_alpaka_INCLUDE_DIRECTORY}> $<INSTALL_INTERFACE:include>
        )
    endif()

    target_include_directories(
        alpaka_target_host
        INTERFACE $<BUILD_INTERFACE:${_alpaka_INCLUDE_DIRECTORY}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_compile_features(alpaka_target_headers INTERFACE cxx_std_${alpaka_CXX_STANDARD})

    target_link_libraries(alpaka INTERFACE alpaka_target_host)

    option(alpaka_HOST_Cpu "Enable support for api::host and deviceKind::cpu in examples/benchmarks and tests" ON)
    if(NOT alpaka_HOST_Cpu)
        target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_DISABLE_Host_Cpu)
    endif()
endif()

set(alpaka_COUNT_API_DEPS 0)
if(alpaka_DEP_CUDA)
    math(EXPR alpaka_COUNT_API_DEPS "${alpaka_COUNT_API_DEPS} + 1")
endif()
if(alpaka_DEP_HIP)
    math(EXPR alpaka_COUNT_API_DEPS "${alpaka_COUNT_API_DEPS} + 1")
endif()
if(alpaka_DEP_ONEAPI)
    math(EXPR alpaka_COUNT_API_DEPS "${alpaka_COUNT_API_DEPS} + 1")
endif()

if((NOT alpaka_SUPPRESS_TARGET_WARNING) AND alpaka_COUNT_API_DEPS GREATER 1)
    message(
        WARNING
        "More than one dependency of alpaka_DEP_CUDA, alpaka_DEP_HIP, or alpaka_DEP_ONEAPI is activated. \
    The cmake taget 'alpaka::alpaka' and 'alpaka' will therefore not linked against any of these. \
    Please link your app against alpaka::cuda, alpaka::hip, or alpaka::oneapi directly. \
    This warning can be suppressed by setting 'alpaka_SUPPRESS_TARGET_WARNING'."
    )
endif()

option(alpaka_COMPILE_PEDANTIC "Compile all code with strict compiler settings." OFF)

set(alpaka_SIMD
    "DEFAULT"
    CACHE STRING
    "Select the SIMD implementation. DEFAULT try using std::simd, if not found fall back to SIMD emulation (aka EMULATION)."
)
set_property(CACHE alpaka_SIMD PROPERTY STRINGS "DEFAULT;STDSIMD;EMULATION")

# compute device backends
if(alpaka_DEP_CUDA)
    include(${_alpaka_CMAKE_DIR}/alpakaCuda.cmake)
endif()
if(alpaka_DEP_HIP)
    include(${_alpaka_CMAKE_DIR}/alpakaHip.cmake)
endif()
if(alpaka_DEP_ONEAPI)
    include(${_alpaka_CMAKE_DIR}/alpakaOneApi.cmake)
endif()
# Avoid extending global alpaka targets more than once in the complete configure tree.
get_property(_alpaka_TARGETS_EXTENDED GLOBAL PROPERTY ALPAKA_TARGETS_EXTENDED)
if(NOT _alpaka_TARGETS_EXTENDED)
    ## GCC compiler flag to show a longer stack for concept diagnostics
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        alpaka_set_compiler_options(HOST target alpaka_target_host "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-fconcepts-diagnostics-depth=${alpaka_GCC_CONCEPT_DEPTH}>")
    endif()

    ## HWLOC
    include(${_alpaka_CMAKE_DIR}/alpakaHwloc.cmake)

    ## OpenMP
    # There is no way to get the correct flags for the language CUDA or HIP
    if(alpaka_DEP_OMP)
        find_package(OpenMP REQUIRED COMPONENTS CXX)
        target_link_libraries(alpaka_target_host INTERFACE OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    endif()

    # Check for optional TBB
    if(alpaka_DEP_TBB)
        find_package(TBB 2021.10 REQUIRED COMPONENTS tbb)
        target_link_libraries(alpaka_target_host INTERFACE TBB::tbb)
        message(STATUS "oneTBB found: ${TBB_VERSION}")
    else()
        # This will enforce config.hpp to not activate TBB even if thee headers are available.
        # If the headers are available but the linker flags are not set there will be an error during linking.
        target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_DISABLE_TBB)
    endif()

    if(alpaka_COMPILE_PEDANTIC)
        message(STATUS "Enable strict compiler settings.")
        include(${_alpaka_CMAKE_DIR}/alpakaCompilePedantic.cmake)
    endif()

    ## search for atomic ref
    # Check for C++20 std::atomic_ref first
    # we only check the CXX compiler, equal to OpenMP which is checked for the CXX compiler only too.
    try_compile(
        alpaka_HAS_STD_ATOMIC_REF # Result stored here
        "${PROJECT_BINARY_DIR}/alpakaFeatureTests" # Binary directory for output file
        SOURCES
            "${_alpaka_FEATURE_TESTS_DIR}/StdAtomicRef.cpp" # Source file
        CXX_STANDARD ${alpaka_CXX_STANDARD}
        CXX_STANDARD_REQUIRED TRUE
        CXX_EXTENSIONS FALSE
    )
    if(alpaka_HAS_STD_ATOMIC_REF)
        message(STATUS "std::atomic_ref<T> found")
    else()
        message(STATUS "std::atomic_ref<T> NOT found")
        target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_DISABLE_STD_ATOMIC_REF)
    endif()

    if(NOT alpaka_HAS_STD_ATOMIC_REF)
        if(Boost_ATOMIC_FOUND)
            message(STATUS "boost::atomic_ref<T> found")
            target_link_libraries(alpaka_target_host INTERFACE Boost::atomic)
        else()
            message(STATUS "boost::atomic_ref<T> NOT found")
            message(FATAL_ERROR "std::atomic_ref<T> OR boost::atomic_ref<T> is required")
        endif()
    endif()

    if(alpaka_SIMD STREQUAL "STDSIMD" OR alpaka_SIMD STREQUAL "DEFAULT")
        # Check for C++ std::simd
        # we only check the CXX compiler, equal to OpenMP which is checked for the CXX compiler only too.
        try_compile(
            alpaka_HAS_STD_SIMD # Result stored here
            "${PROJECT_BINARY_DIR}/alpakaFeatureTests" # Binary directory for output file
            SOURCES
                "${_alpaka_FEATURE_TESTS_DIR}/StdSimd.cpp" # Source file
            CXX_STANDARD ${alpaka_CXX_STANDARD}
            CXX_STANDARD_REQUIRED TRUE
            CXX_EXTENSIONS FALSE
        )

        if(alpaka_HAS_STD_SIMD)
            message(STATUS "std::simd found")
        else()
            if(alpaka_SIMD STREQUAL "STDSIMD")
                message(FATAL_ERROR "std::simd not found but requested via 'alpaka_SIMD=STDSIMD'")
            else()
                message(STATUS "std::simd NOT found, emulated SIMD is used")
                target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_DISABLE_STD_SIMD)
            endif()
        endif()
    else()
        message(STATUS "std::simd disabled, emulated SIMD is used")
        target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_DISABLE_STD_SIMD)
    endif()
    # These options are used in the alpaka_finalize call
    option(alpaka_ASAN "Enable/Disable linking the address sanitizer for cpu targets" OFF)
    option(alpaka_TSAN "Enable/Disable linking the thread sanitizer for cpu targets" OFF)
    option(alpaka_LSAN "Enable/Disable linking the memory leak sanitizer for cpu targets" OFF)
    option(alpaka_UBSAN "Enable/Disable linking the undefined behavior sanitizer for cpu targets" OFF)
    option(alpaka_COVERAGE "Enable/Disable coverage instrumentation for cpu targets" OFF)

    if(alpaka_TSAN)
        message(
            WARNING
            "Thread sanitizer enabled: You should reduce mmap rnd bits to avoid compile issues: call as root 'sysctl vm.mmap_rnd_bits=28'"
        )
    endif()

    set(alpaka_LOG "OFF" CACHE STRING "Set how the logging should be compiled into alpaka")
    set_property(CACHE alpaka_LOG PROPERTY STRINGS "static;dynamic;OFF")

    if((${alpaka_LOG} STREQUAL "OFF"))
        set(alpaka_LOG_ENABLED OFF)
    else()
        set(alpaka_LOG_ENABLED ON)
    endif()
    if(alpaka_LOG_ENABLED)
        if(${alpaka_LOG} STREQUAL "static")
            target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_LOG_STATIC)
        endif()

        if(${alpaka_LOG} STREQUAL "dynamic")
            target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_LOG_DYNAMIC)
        endif()

        option(alpaka_LOG_INDENT "Enable/Disable call stack indention for logging" ON)
        if(alpaka_LOG_INDENT)
            target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_LOG_INDENT)
        endif()

        set(alpaka_LOG_DETAIL "short" CACHE STRING "Set how the logging should be compiled into alpaka")
        set_property(CACHE alpaka_LOG_DETAIL PROPERTY STRINGS "short;long")
        if(${alpaka_LOG_DETAIL} STREQUAL "short")
            target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_LOG_DETAIL_SHORT)
        elseif(${alpaka_LOG_DETAIL} STREQUAL "long")
            target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_LOG_DETAIL_LONG)
        endif()

        option(alpaka_LOG_FUNCTIONS "Enable/Disable logging of function entry and exit" ON)
        if(alpaka_LOG_FUNCTIONS)
            target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_ENABLE_LOG_FUNCTIONS)
        endif()

        option(alpaka_LOG_INFO "Enable/Disable logging of additional information" ON)
        if(alpaka_LOG_INFO)
            target_compile_definitions(alpaka_target_host INTERFACE ALPAKA_ENABLE_LOG_INFO)
        endif()
        if(${alpaka_LOG} STREQUAL "static")
            # Set default options for each log level (Device, Event, Memory, Queue, Kernel)
            option(alpaka_LOG_STATIC_Device "Enable Device logging" ON)
            option(alpaka_LOG_STATIC_Event "Enable Event logging" ON)
            option(alpaka_LOG_STATIC_Memory "Enable Memory logging" ON)
            option(alpaka_LOG_STATIC_Queue "Enable Queue logging" ON)
            option(alpaka_LOG_STATIC_Kernel "Enable Kernel logging" ON)

            # Initialize bit mask to 0
            set(alpaka_LOG_STATIC_LVL_MASK 0)

            # Set bit mask based on user-selected options
            if(alpaka_LOG_STATIC_Device)
                math(EXPR alpaka_LOG_STATIC_LVL_MASK "${alpaka_LOG_STATIC_LVL_MASK} | 1")
            endif()

            if(alpaka_LOG_STATIC_Event)
                math(EXPR alpaka_LOG_STATIC_LVL_MASK "${alpaka_LOG_STATIC_LVL_MASK} | 2")
            endif()

            if(alpaka_LOG_STATIC_Memory)
                math(EXPR alpaka_LOG_STATIC_LVL_MASK "${alpaka_LOG_STATIC_LVL_MASK} | 4")
            endif()

            if(alpaka_LOG_STATIC_Queue)
                math(EXPR alpaka_LOG_STATIC_LVL_MASK "${alpaka_LOG_STATIC_LVL_MASK} | 8")
            endif()

            if(alpaka_LOG_STATIC_Kernel)
                math(EXPR alpaka_LOG_STATIC_LVL_MASK "${alpaka_LOG_STATIC_LVL_MASK} | 16")
            endif()

            # Convert bitmask string to a numeric value
            math(EXPR alpaka_LOG_STATIC_LVL_MASK_VALUE "${alpaka_LOG_STATIC_LVL_MASK}")

            # Pass the bit mask to the C++ code via target_compile_definitions
            target_compile_definitions(
                alpaka_target_host
                INTERFACE ALPAKA_LOG_STATIC_LVL_MASK=${alpaka_LOG_STATIC_LVL_MASK_VALUE}
            )

            # Print the final bit mask value (for debugging purposes)
            message(STATUS "define ALPAKA_LOG_STATIC_LVL_MASK is set to: ${alpaka_LOG_STATIC_LVL_MASK_VALUE}")
        endif()
    endif()

    # CMake executor options for tests/benchmarks and examples
    # The follwoing option influences the alpaka header target
    option(alpaka_EXEC_CpuSerial "Enable/Disable serial executor in examples/benchmarks and tests" ON)
    if(NOT alpaka_EXEC_CpuSerial)
        target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_DISABLE_EXEC_CpuSerial)
    endif()
    option(alpaka_EXEC_CpuOmpBlocks "Enable/Disable OpenMP blocks executor in examples/benchmarks and tests" ON)
    if(NOT alpaka_EXEC_CpuOmpBlocks)
        target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_DISABLE_EXEC_CpuOmpBlocks)
    endif()
    option(alpaka_EXEC_TbbBlocks "Enable/Disable TBB blocks executor in examples/benchmarks and tests" ON)
    if((NOT alpaka_EXEC_TbbBlocks) OR (NOT alpaka_DEP_TBB))
        target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_DISABLE_EXEC_CpuTbbBlocks)
    endif()
    option(alpaka_EXEC_GpuCuda "Enable/Disable CUDA executor in examples/benchmarks and tests" ON)
    if(NOT alpaka_EXEC_GpuCuda)
        target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_DISABLE_EXEC_GpuCuda)
    endif()
    option(alpaka_EXEC_GpuHip "Enable/Disable HIP executor in examples/benchmarks and tests" ON)
    if(NOT alpaka_EXEC_GpuHip)
        target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_DISABLE_EXEC_GpuHip)
    endif()
    option(alpaka_EXEC_OneApi "Enable/Disable Intel OneAPI SYCL executor in examples/benchmarks and tests" ON)
    if(NOT alpaka_EXEC_OneApi)
        target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_DISABLE_EXEC_OneApi)
    endif()
endif()
set_property(GLOBAL PROPERTY ALPAKA_TARGETS_EXTENDED ON)
