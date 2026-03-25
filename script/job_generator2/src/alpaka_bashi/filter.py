"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

from typing import Dict, Callable, IO, List
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import


def all_backends_fine(
    row: bashi.ParameterValueTuple,
    backends: List[ValueName],
    all_available_backends: List[ValueName],
) -> bool:
    """Check if the combination of backends in a row is corresponding to at least one valid
    combination of backends.

    Args:
        row (bashi.ParameterValueTuple): row with backends
        backends (List[ValueName]): Backends which needs to be enabled.
        all_available_backends (List[ValueName]): All available backends. If a backend is not in the
            backends list, but in this list, it needs to be disabled.

    Returns:
        bool: True if all enabled backends of the `row` are defined in `backends` and all disabled
            backends are defined in `all_available_backends`.
    """
    for backend in all_available_backends:
        if backend in row:
            if backend in backends:
                if row[backend].version == OFF_VER:
                    return False
            else:
                if row[backend].version != OFF_VER:
                    return False

    return True


def get_valid_compiler_backend_combinations(
    row: bashi.ParameterValueTuple,
) -> List[CompilerBackendComb]:
    """Return a list of all possible compiler and backend combinations, which are still possible
    for the given row.

    Args:
        row (bashi.ParameterValueTuple): parameter-value-tuple

    Returns:
        List[CompilerBackendComb]: List if possible backends combinations
    """
    valid_combs: List[CompilerBackendComb] = []
    for comb in ALLOWED_BACKEND_COMBINATIONS:
        host_compiler, device_compiler, backends = comb
        if HOST_COMPILER in row and row[HOST_COMPILER].name != host_compiler:
            continue
        if DEVICE_COMPILER in row and row[DEVICE_COMPILER].name != device_compiler:
            continue
        if all_backends_fine(row, backends, BACKENDS):
            valid_combs.append(comb)

    return valid_combs


def only_cuda_compiler_backends(combinations: List[CompilerBackendComb]) -> bool:
    """Return True, if there are only CompilerBackendComb in the list, which contains the CUDA
    compilers and backend."""
    for comb in combinations:
        if ALPAKA_ACC_GPU_CUDA_ENABLE not in comb.backends:
            return False
    return True


def only_clang_cuda_compiler_backends(combinations: List[CompilerBackendComb]) -> bool:
    """Return True, if only compiler backend combinations with the Clang-CUDA compiler exist."""
    for comb in combinations:
        host_compiler = comb[0]
        if host_compiler != CLANG_CUDA:
            return False
    return True


def no_hip_compiler_backends(combinations: List[CompilerBackendComb]) -> bool:
    """Return True, if there is no CompilerBackendComb in the list, which contains the HIP compiler
    and backend."""
    for comb in combinations:
        if ALPAKA_ACC_GPU_HIP_ENABLE in comb.backends:
            return False
    return True


class AlpakaFilter(bashi.FilterBase):
    """Alpaka specific filter rules."""

    def __init__(
        self,
        runtime_infos: Dict[str, Callable[..., bool]] | None = None,
        version_relation: bashi.VersionRelation = bashi.VersionRelation(),
        output: IO[bashi.Parameter] | None = None,
    ):
        super().__init__(runtime_infos, version_relation, output)

    def __call__(
        self,
        row: bashi.ParameterValueTuple,
    ) -> bool:
        """Check if given parameter-value-tuple is valid

        Args:
            row (ParameterValueTuple): parameter-value-tuple to verify.

        Returns:
            bool: True, if parameter-value-tuple is valid.
        """
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-return-statements

        # Rule: a1
        valid_combs = get_valid_compiler_backend_combinations(row)
        if len(valid_combs) == 0:
            self.reason("No valid backend combination available.")
            return False

        if only_cuda_compiler_backends(valid_combs):
            # Rule: a2
            if (
                HOST_COMPILER in row
                and row[HOST_COMPILER].name in (GCC, CLANG)
                and RT_HOST_COMPILER_CUDA_SUPPORT in self.runtime_infos
                and not self.runtime_infos[RT_HOST_COMPILER_CUDA_SUPPORT](
                    row[HOST_COMPILER].name, row[HOST_COMPILER].version
                )
            ):
                self.reason(
                    "Only backend combinations with CUDA backend possible. There is no CUDA SDK "
                    f"version, which supports the host compiler {row[HOST_COMPILER].name}-"
                    f"{row[HOST_COMPILER].version}"
                )
                return False

            # Rule: a6
            if (
                RT_CLANG_CUDA_MAX_CUDA_SUPPORT in self.runtime_infos
                and only_clang_cuda_compiler_backends(valid_combs)
                and ALPAKA_ACC_GPU_CUDA_ENABLE in row
                and not self.runtime_infos[RT_CLANG_CUDA_MAX_CUDA_SUPPORT](
                    row[ALPAKA_ACC_GPU_CUDA_ENABLE].version
                )
            ):
                self.reason(
                    "There is no Clang-CUDA version in the combination list, which supports the "
                    f"CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version} SDK."
                )
                return False

        # Rule: a3
        if BUILD_TYPE in row and row[BUILD_TYPE].version == CMAKE_DEBUG_VER:
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                if (
                    compiler_type in row
                    and row[compiler_type].name == HIPCC
                    and row[compiler_type].version == packaging.version.parse("6.2")
                ):
                    self.reason("Debug builds with HIP/ROCm 6.2 produce compiler errors.")
                    return False

        # Rule: a4
        # OVERWORK: remove/overwork me, if bashi supports standard library
        if UBUNTU in row and row[UBUNTU].version < packaging.version.parse("24.04"):
            for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
                if compiler_type in row and row[compiler_type].name not in (CLANG, HIPCC):
                    self.reason("Only the HIPCC and Clang can be used on Ubuntu 24.04")
                    return False

                if (
                    compiler_type in row
                    and row[compiler_type].name == CLANG
                    and row[compiler_type].version > packaging.version.parse("16")
                ):
                    self.reason("Clang 16 and later will be tested on Ubuntu 24.04 and later.")
                    return False

            for backend in ONE_API_BACKENDS + [ALPAKA_ACC_GPU_CUDA_ENABLE]:
                if backend in row and row[backend].version != OFF_VER:
                    self.reason(
                        f"The backend {row[backend].name} will be not used on Ubuntu 22.04 and "
                        "older."
                    )
                    return False

        # Rule: a5
        # OVERWORK: remove/overwork me, if bashi supports standard library
        for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
            if (
                compiler_type in row
                and row[compiler_type].name == CLANG
                and row[compiler_type].version <= packaging.version.parse("16")
            ):
                if UBUNTU in row and row[UBUNTU].version >= packaging.version.parse("24.04"):
                    self.reason(
                        f"Clang {row[compiler_type].version} does not support libc++-13 and later "
                        f"of the host compiler of Ubuntu {row[UBUNTU].version}"
                    )
                    return False

                if (
                    DEVICE_COMPILER in row
                    and row[DEVICE_COMPILER].name == NVCC
                    and row[DEVICE_COMPILER].version >= packaging.version.parse("12.0")
                ):
                    self.reason(
                        f"NVCC {row[DEVICE_COMPILER].version} is only available on UBUNTU 24.04 "
                        f"and later but Clang {row[HOST_COMPILER].version} does not support 24.04 "
                        "and later."
                    )
                    return False

                if ALPAKA_ACC_GPU_CUDA_ENABLE in row and row[
                    ALPAKA_ACC_GPU_CUDA_ENABLE
                ].version >= packaging.version.parse("12.0"):
                    self.reason(
                        f"CUDA {row[ALPAKA_ACC_GPU_CUDA_ENABLE].version} is only available on "
                        f"UBUNTU 24.04 and later but Clang {row[HOST_COMPILER].version} does not "
                        "support 24.04 and later."
                    )
                    return False

                for cpu_backend in CPU_BACKENDS:
                    if cpu_backend in row and row[cpu_backend].version == OFF_VER:
                        self.reason(
                            f"Clang {row[compiler_type].version} works only together with CPU "
                            "backends."
                        )
                        return False

        if self.debug_print != bashi.FilterDebugMode.OFF:
            print("passed")
        return True
