"""Copyright 2023 Simeon Ehrig, Jan Stephan
SPDX-License-Identifier: MPL-2.0

Alpaka project specific filter rules.
"""

from typing import List

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.util import (
    row_check_name,
    row_check_version,
    is_in_row,
    row_check_backend_version,
)


def alpaka_post_filter(row: List) -> bool:
    # OpenMP is not supported for clang as cuda compiler
    # https://github.com/alpaka-group/alpaka/issues/639
    if row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA) and (
        row_check_backend_version(row, ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, "==", ON_VER)
        or row_check_backend_version(row, ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, "==", ON_VER)
    ):
        return False

    # cmake 3.24 and older does not support C++20 for nvcc
    if (
        row_check_name(row, DEVICE_COMPILER, "==", NVCC)
        and row_check_version(row, CXX_STANDARD, ">=", "20")
        and row_check_version(row, CMAKE, "<", "3.25")
    ):
        return False

    # Debug builds with HIP/ROCm 6.2 produce compiler errors
    if (
        is_in_row(row, BUILD_TYPE)
        and row[param_map[BUILD_TYPE]][VERSION] == CMAKE_DEBUG
        and row_check_name(row, DEVICE_COMPILER, "==", HIPCC)
        and row_check_version(row, DEVICE_COMPILER, "==", "6.2")
    ):
        return False

    # g++-12 is not available on the Ubuntu 20.04 ppa's
    if (
        row_check_name(row, HOST_COMPILER, "==", GCC)
        and row_check_version(row, HOST_COMPILER, "==", "12")
        and row_check_version(row, UBUNTU, "==", "20.04")
    ):
        return False

    # there is a bug with g++-13 and cuda 12.4 on Ubuntu 20.04
    if (
        row_check_name(row, DEVICE_COMPILER, "==", NVCC)
        and row_check_version(row, DEVICE_COMPILER, "==", "12.4")
        and row_check_name(row, HOST_COMPILER, "==", GCC)
        and row_check_version(row, HOST_COMPILER, "==", "13")
        and row_check_version(row, UBUNTU, "==", "20.04")
    ):
        return False

    # Clang-CUDA has three support levels, full support, partial support, and if newer, then it throws a warning and continues.
    # We only test unitl the partially supported version in the CI

    # Clang-CUDA 16 and below officially only partially support as a maximum up to CUDA SDK 11.8, so we disable CI tests for these
    if row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA) and row_check_version(row, DEVICE_COMPILER, "<=", "16"):
        return False

    # Clang-CUDA 17 fully supports up to CUDA SDK 11.8 and partially upto 12.1
    if (
        row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA)
        and row_check_version(row, DEVICE_COMPILER, "==", "17")
        and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "12.1")
    ):
        return False

    # Clang-CUDA 18 fully supports up to CUDA SDK 12.3 (unless it is 18.0 which fully supports 12.1 only partially supports 12.3)
    if (
        row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA)
        and row_check_version(row, DEVICE_COMPILER, "==", "18")
        and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "12.3")
    ):
        return False

    # Clang-CUDA 19 fully supports up to CUDA SDK 12.3 and partially upto 12.5
    if (
        row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA)
        and row_check_version(row, DEVICE_COMPILER, "==", "19")
        and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "12.5")
    ):
        return False

    # Clang-CUDA 20 fully supports up to CUDA SDK 12.3 and partially upto 12.8
    if (
        row_check_name(row, DEVICE_COMPILER, "==", CLANG_CUDA)
        and row_check_version(row, DEVICE_COMPILER, "==", "20")
        and row_check_backend_version(row, ALPAKA_ACC_GPU_CUDA_ENABLE, ">", "12.8")
    ):
        return False

    if row_check_name(row, DEVICE_COMPILER, "==", NVCC) and row_check_name(row, HOST_COMPILER, "==", CLANG):
        # nvcc 12.5 is the minimum requirement for host compiler Clang 18
        if row_check_version(row, HOST_COMPILER, "==", "18") and row_check_version(row, DEVICE_COMPILER, "<=", "12.5"):
            return False

        # nvcc 12.8 is the minimum requirement for host compiler Clang 19
        if row_check_version(row, HOST_COMPILER, "==", "19") and row_check_version(row, DEVICE_COMPILER, "<=", "12.6"):
            return False

        # nvcc 13.0 is the minimum requirement for host compiler Clang 20
        if row_check_version(row, HOST_COMPILER, "==", "20") and row_check_version(row, DEVICE_COMPILER, "<=", "12.9"):
            return False

    # the SYCL backends needs to be enabled if the icpx compiler is used
    if row_check_name(row, DEVICE_COMPILER, "==", ICPX):
        if is_in_row(row, BACKENDS) and ((ALPAKA_ACC_SYCL_ENABLE, ON_VER) not in row[param_map[BACKENDS]]):
            return False

    # we use the Ubuntu 22.04 for
    # - HIP 6.0 until 6.2
    # - If Clang 14 and older is used
    # - If CUDA 12.3 and older is used, because the CUDA versions does not support the libstdc++ 13
    # which is provided by the Ubuntu host compiler GCC 13
    # the ROCm Ubuntu support is handled by the alpaka-job-matrix-library
    if row_check_version(row, UBUNTU, "==", "22.04"):
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            for compiler in (GCC, ICPX):
                if row_check_name(row, compiler_type, "==", compiler):
                    return False

            for compiler in (CLANG, CLANG_CUDA):
                if row_check_name(row, compiler_type, "==", compiler) and row_check_version(
                    row, compiler_type, ">", "14"
                ):
                    return False

    if row_check_version(row, UBUNTU, "==", "24.04"):
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            for compiler in (CLANG, CLANG_CUDA):
                if row_check_name(row, compiler_type, "==", compiler) and row_check_version(
                    row, compiler_type, "<", "15"
                ):
                    return False
            if (
                row_check_name(row, DEVICE_COMPILER, "==", NVCC)
                and row_check_name(row, HOST_COMPILER, "==", CLANG)
                and row_check_version(row, DEVICE_COMPILER, "<=", "12.3")
            ):
                return False
    return True
