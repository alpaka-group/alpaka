#.rst:
# Findalpaka
# ----------
#
# Abstraction library for parallel kernel acceleration
# https://github.com/ComputationalRadiationPhysics/alpaka
#
# Finding and Using alpaka
# ^^^^^^^^^^^^^^^^^^^^^
#
# .. code-block:: cmake
#
#   FIND_PACKAGE(alpaka
#     [version] [EXACT]     # Minimum or EXACT version, e.g. 1.0.0
#     [REQUIRED]            # Fail with an error if alpaka or a required
#                           # component is not found
#     [QUIET]               # Do not warn if this module was not found
#     [COMPONENTS <...>]    # Compiled in components: ignored
#   )
#   TARGET_LINK_LIBRARIES(<target> PUBLIC alpaka)
#
# To provide a hint to this module where to find the alpaka installation,
# set the ALPAKA_ROOT variable.
#
# This module requires Boost. Make sure to provide a valid install of it
# under the environment variable BOOST_ROOT.
#
# ALPAKA_FIBERS_ENABLE will require Boost.Fiber to be built.
# ALPAKA_OPENMP2_ENABLE will require a OpenMP 2.0 capable compiler.
# ALPAKA_CUDA_ENABLE will require CUDA 7.0 to be installed.
#
# Set the following CMake variables BEFORE calling find_packages to
# change the behaviour of this module:
# - ``ALPAKA_SERIAL_CPU_ENABLE`` {ON, OFF}
# - ``ALPAKA_THREADS_CPU_ENABLE`` {ON, OFF}
# - ``ALPAKA_FIBERS_CPU_ENABLE`` {ON, OFF}
# - ``ALPAKA_OPENMP2_CPU_ENABLE`` {ON, OFF}
# - ``ALPAKA_OPENMP4_CPU_ENABLE`` {ON, OFF}
# - ``ALPAKA_CUDA_GPU_ENABLE`` {ON, OFF}
# - ``ALPAKA_CUDA_VERSION`` {7.0, ...}
# - ``ALPAKA_CUDA_ARCH`` {sm_20, sm...}
# - ``ALPAKA_CUDA_FAST_MATH`` {ON, OFF}
# - ``ALPAKA_CUDA_FTZ`` {ON, OFF}
# - ``ALPAKA_CUDA_SHOW_REGISTER`` {ON, OFF}
# - ``ALPAKA_CUDA_KEEP_FILES`` {ON, OFF}
# - ``ALPAKA_CUDA_SHOW_CODELINES`` {ON, OFF}
# - ``ALPAKA_DEBUG`` {0, 1, 2}
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# - ``alpaka_DEFINITIONS``
#   Compiler definitions.
# - ``alpaka_FOUND``
#   TRUE if alpaka found a working install.
# - ``alpaka_INCLUDE_DIRS``
#   Include directories for the alpaka headers.
# - ``alpaka_LIBRARIES``
#   alpaka libraries.
# - ``alpaka_VERSION``
#   Version in format Major.Minor.Patch
#
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the :prop_tgt:`IMPORTED` target ``alpaka``, if alpaka has
# been found.
#


################################################################################
# Copyright 2015 Benjamin Worpitz
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
# RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE
# USE OR PERFORMANCE OF THIS SOFTWARE.
################################################################################


################################################################################
# alpaka.
################################################################################

FIND_PATH(ALPAKA_ROOT
  NAMES "include/alpaka/alpaka.hpp"
  HINTS "${ALPAKA_ROOT}" ENV ALPAKA_ROOT
  DOC "alpaka ROOT location"
)

# Normalize the path (e.g. remove ../)
GET_FILENAME_COMPONENT(ALPAKA_ROOT "${ALPAKA_ROOT}" ABSOLUTE)
# Add the / at the end.
SET(ALPAKA_ROOT "${ALPAKA_ROOT}/")

# Use the internal find implementation.
INCLUDE("${ALPAKA_ROOT}cmake/findInternal.cmake")
    