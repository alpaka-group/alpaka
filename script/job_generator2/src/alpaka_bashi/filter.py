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


def get_valid_backend_combinations(row: bashi.ParameterValueTuple) -> List[CompilerBackendComb]:
    """Return a list of all possible backend combinations, which are still possible for the given
    row.

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


def only_cuda_backend(combinations: List[CompilerBackendComb]) -> bool:
    """Return True, if CompilerBackendComb in the list, which contains the CUDA backend."""
    for comb in combinations:
        if ALPAKA_ACC_GPU_CUDA_ENABLE not in comb.backends:
            return False
    return True


class AlpakaFilter(bashi.FilterBase):
    """Alpaka specific filter rules."""

    def __init__(
        self,
        runtime_infos: Dict[str, Callable[..., bool]] | None = None,
        output: IO[bashi.Parameter] | None = None,
    ):
        super().__init__(runtime_infos, output)

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
        # Rule: a1
        valid_combs = get_valid_backend_combinations(row)
        if len(valid_combs) == 0:
            self.reason("No valid backend combination available.")
            return False

        # Rule: a2
        if only_cuda_backend(valid_combs):
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

        if self.debug_print != bashi.FilterDebugMode.OFF:
            print("passed")
        return True
