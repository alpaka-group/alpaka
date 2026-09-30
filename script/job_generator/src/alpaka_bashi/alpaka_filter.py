"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import bashi
import packaging.version
from bashi.globals import (
    ALPAKA_ACC_GPU_CUDA_ENABLE,
    CLANG,
    CLANG_CUDA,
    CMAKE,
    DEVICE_COMPILER,
    GCC,
    HOST_COMPILER,
    NVCC,
)
from bashi.results import OFF_VER

from alpaka_bashi.versions import get_allowed_backend_combinations, get_used_backends


def check_only_valid_backend_combinations_a1(row: bashi.BashiRow, alpaka_filter: "AlpakaFilter") -> bool:
    """
    Check if still possible valid backend combinations exist.

    Args:
        row (bashi.BashiRow): parameter-value-tuple to verify.
        alpaka_filter (AlpakaFilter): alpaka filter

    Returns:
        bool: True if passed.
    """
    if (
        len(bashi.get_valid_compiler_backend_combinations(row, get_allowed_backend_combinations(), get_used_backends()))
        == 0
    ):
        alpaka_filter.reason("No valid backend combination available.")
        return False
    return True


def check_clang_host_compiler_supported_cuda_sdk_a2(row: bashi.BashiRow, alpaka_filter: "AlpakaFilter") -> bool:
    """
    Clang as nvcc host compiler is only working since CUDA 13.3.

    Args:
        row (bashi.BashiRow): parameter-value-tuple to verify.
        alpaka_filter (AlpakaFilter): alpaka filter

    Returns:
        bool: True if passed.
    """
    if row[HOST_COMPILER].name == CLANG and OFF_VER < row[ALPAKA_ACC_GPU_CUDA_ENABLE].version < packaging.version.parse(
        "13.3"
    ):
        alpaka_filter.reason("Clang as nvcc host compiler is only working since CUDA 13.3.")
        return False

    return True


def check_clang_host_compiler_supported_nvcc_a3(row: bashi.BashiRow, alpaka_filter: "AlpakaFilter") -> bool:
    """
    Clang as nvcc host compiler is only working since CUDA 13.3.

    Args:
        row (bashi.BashiRow): parameter-value-tuple to verify.
        alpaka_filter (AlpakaFilter): alpaka filter

    Returns:
        bool: True if passed.
    """
    if (
        row[HOST_COMPILER].name == CLANG
        and row[DEVICE_COMPILER].name == NVCC
        and row[DEVICE_COMPILER].version < packaging.version.parse("13.3")
    ):
        alpaka_filter.reason("The Clang host compiler is only working since nvcc 13.3.")
        return False

    return True


def _pretty_name_compiler(constant: str) -> str:
    """Returns the string representation of the constants HOST_COMPILER and DEVICE_COMPILER in a
    human-readable version.

    Args:
        constant (str): Ether HOST_COMPILER or DEVICE_COMPILER

    Returns:
        str: human-readable string representation of HOST_COMPILER or DEVICE_COMPILER
    """
    if constant == HOST_COMPILER:
        return "host compiler"
    if constant == DEVICE_COMPILER:
        return "device compiler"
    return "unknown compiler type"


def check_clang_cuda_cmake_support_a4(row: bashi.BashiRow, alpaka_filter: "AlpakaFilter") -> bool:
    """
    Clang-CUDA requires at least CMake 3.31

    Args:
        row (bashi.BashiRow): parameter-value-tuple to verify.
        alpaka_filter (AlpakaFilter): alpaka filter

    Returns:
        bool: True if passed.
    """
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        if (
            row[compiler_type].name == CLANG_CUDA
            and row[compiler_type].version >= packaging.version.parse("23")
            and row[CMAKE].version < packaging.version.parse("3.31")
        ):
            alpaka_filter.reason(
                f"CMAKE {row[CMAKE].version} does not support "
                f"{_pretty_name_compiler(compiler_type)} Clang-Cuda {row[compiler_type].version}",
            )
            return False
    return True


def check_if_nvcc_supports_the_host_compiler_a5(row: bashi.BashiRow, alpaka_filter: "AlpakaFilter"):
    """
    If GCC or Clang is a host compiler but not supported by any available nvcc version, disallow
    the combination if there is no possibility to use the GCC or Clang as CPU compiler.

    Args:
        row (bashi.BashiRow): parameter-value-tuple to verify.
        alpaka_filter (AlpakaFilter): alpaka filter

    Returns:
        bool: True if passed.
    """
    for host_compiler, max_host_compiler_version in (
        (GCC, alpaka_filter.version.get_nvcc_gcc_max_supported_host_compiler_version()),
        (CLANG, alpaka_filter.version.get_nvcc_clang_max_supported_host_compiler_version()),
    ):
        if row[HOST_COMPILER].name == host_compiler:
            # Rule a1 checks before, that the list can be never empty. Therefore gcc/clang is not used as
            # device compiler, it can be only used as host compiler for nvcc.
            compiler_as_device_compiler = [
                compier_backend
                for compier_backend in bashi.get_valid_compiler_backend_combinations(
                    row, get_allowed_backend_combinations(), get_used_backends()
                )
                if compier_backend.device == host_compiler
            ]
            if len(compiler_as_device_compiler) == 0 and row[HOST_COMPILER].version > max_host_compiler_version:
                alpaka_filter.reason(
                    f"There is nvcc which supports {host_compiler} {row[HOST_COMPILER].version} as host compiler"
                )
                return False

    return True


# pylint: disable=too-few-public-methods
class AlpakaFilter(bashi.FilterBase):
    """Alpaka specific filter rules."""

    def __call__(
        self,
        row: bashi.BashiRow,
    ) -> bool:
        """Check if given parameter-value-tuple is valid

        Args:
            row (bashi.BashiRow): parameter-value-tuple to verify.

        Returns:
            bool: True, if parameter-value-tuple is valid.
        """

        return (
            check_only_valid_backend_combinations_a1(row, self)
            and check_clang_host_compiler_supported_cuda_sdk_a2(row, self)
            and check_clang_host_compiler_supported_nvcc_a3(row, self)
            and check_clang_cuda_cmake_support_a4(row, self)
            and check_if_nvcc_supports_the_host_compiler_a5(row, self)
        )
