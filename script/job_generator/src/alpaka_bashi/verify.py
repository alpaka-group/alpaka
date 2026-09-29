"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Verify generated combinations.
"""

from collections.abc import Callable

import bashi
from bashi.globals import (
    ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
    ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
    ALPAKA_ACC_GPU_CUDA_ENABLE,
    CLANG,
    CLANG_CUDA,
    CMAKE,
    DEVICE_COMPILER,
    GCC,
    HOST_COMPILER,
    NVCC,
    OFF,
    ON,
)

from alpaka_bashi.versions import get_allowed_backend_combinations, get_used_backends, get_used_compiler_versions


def remove_disabled_serial_backend_for_gcc_and_clang(
    parameter_value_pairs: list[bashi.ParameterValuePair],
    removed_parameter_value_pairs: list[bashi.ParameterValuePair],
):
    """GCC and Clang as device compiler uses the serial backend all the time."""
    for compiler_name in (GCC, CLANG):
        bashi.remove_parameter_value_pairs(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=DEVICE_COMPILER,
            value_name1=compiler_name,
            parameter2=ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
            value_version2=OFF,
        )


def serial_backend_is_always_on_for_unsupported_nvcc_host_compiler(
    parameter_value_pairs: list[bashi.ParameterValuePair],
    removed_parameter_value_pairs: list[bashi.ParameterValuePair],
    version_relation: bashi.VersionRelation,
):
    """If there is no nvcc version, which supports specific GCC and Clang version remove all pairs
    of the host compiler version with an disable serial cpu backend. The serial backend can be
    never disabled, because the host compiler is used as CPU backend compiler all the time."""
    for compiler_name, max_host_compiler_version in (
        (GCC, version_relation.get_nvcc_gcc_max_supported_host_compiler_version()),
        (CLANG, version_relation.get_nvcc_clang_max_supported_host_compiler_version()),
    ):
        bashi.remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=HOST_COMPILER,
            value_name1=compiler_name,
            value_min_version1=str(max_host_compiler_version),
            value_min_version1_inclusive=False,
            parameter2=ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
            value_min_version2=OFF,
            value_max_version2=OFF,
        )


def remove_disabled_serial_and_openmp_backend(
    parameter_value_pairs: list[bashi.ParameterValuePair],
    removed_parameter_value_pairs: list[bashi.ParameterValuePair],
):
    """The serial backend is tested with the openmp backend all the time."""

    bashi.remove_parameter_value_pairs(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
        value_version1=OFF,
        parameter2=ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE,
        value_version2=ON,
    )


def remove_unsupported_cuda_sdk_for_clang_host_compiler(
    parameter_value_pairs: list[bashi.ParameterValuePair],
    removed_parameter_value_pairs: list[bashi.ParameterValuePair],
):
    """Since CUDA 13.3 Clang as host compiler is working.
    Remove all unsupported combinations of clang + CUDA backend and clang and nvcc."""
    bashi.remove_parameter_value_pairs_ranges(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=CLANG,
        parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
        value_min_version2=OFF,
        value_min_version2_inclusive=False,
        value_max_version2=13.3,
        value_max_version2_inclusive=False,
    )
    bashi.remove_parameter_value_pairs_ranges(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=HOST_COMPILER,
        value_name1=CLANG,
        parameter2=DEVICE_COMPILER,
        value_name2=NVCC,
        value_min_version2=OFF,
        value_min_version2_inclusive=False,
        value_max_version2=13.3,
        value_max_version2_inclusive=False,
    )


def remove_unsupported_cmake_versions_for_clang_host_compiler(
    parameter_value_pairs: list[bashi.ParameterValuePair],
    removed_parameter_value_pairs: list[bashi.ParameterValuePair],
):
    """CMake 3.30 and older does not support Clang-CUDA 23."""
    for compiler_type in (HOST_COMPILER, DEVICE_COMPILER):
        bashi.remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=CMAKE,
            value_max_version1=3.31,
            value_max_version1_inclusive=False,
            parameter2=compiler_type,
            value_name2=CLANG_CUDA,
            value_min_version2=23,
        )


def verify(
    combination_list: bashi.CombinationList,
    param_value_matrix: bashi.ParameterValueMatrix,
    version_relation: bashi.VersionRelation,
    run_infos: dict[str, Callable[..., bool]],
) -> bool:
    """Check if all expected parameter-value-pairs exists in the combination-list.

    Args:
        combination_list (CombinationList): The generated combination list.
        param_value_matrix (ParameterValueMatrix): The expected parameter-values-pairs are generated
            from the parameter-value-list.

    Returns:
        bool: True if it found all pairs
    """

    expected_param_val_tuple, unexpected_param_val_tuple = bashi.get_expected_bashi_parameter_value_pairs(
        param_value_matrix, version_relation, run_infos
    )

    bashi.remove_unsupported_compiler_backend_combinations(
        expected_param_val_tuple,
        unexpected_param_val_tuple,
        list(get_used_compiler_versions().keys()),
        get_used_backends(),
        get_allowed_backend_combinations(),
    )
    bashi.remove_unsupported_backend_combinations(
        expected_param_val_tuple,
        unexpected_param_val_tuple,
        get_used_backends(),
        get_allowed_backend_combinations(),
    )

    remove_disabled_serial_backend_for_gcc_and_clang(expected_param_val_tuple, unexpected_param_val_tuple)
    serial_backend_is_always_on_for_unsupported_nvcc_host_compiler(
        expected_param_val_tuple, unexpected_param_val_tuple, version_relation
    )
    remove_disabled_serial_and_openmp_backend(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_unsupported_cuda_sdk_for_clang_host_compiler(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_unsupported_cmake_versions_for_clang_host_compiler(expected_param_val_tuple, unexpected_param_val_tuple)

    expected_param_val_okay = bashi.check_parameter_value_pair_in_combination_list(
        combination_list, expected_param_val_tuple
    )
    unexpected_param_val_okay = bashi.check_unexpected_parameter_value_pair_in_combination_list(
        combination_list, unexpected_param_val_tuple
    )

    return expected_param_val_okay and unexpected_param_val_okay
