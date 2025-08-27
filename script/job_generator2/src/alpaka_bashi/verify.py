"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Verify generated combinations.
"""

from typing import Dict, Callable, List, cast
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
import alpaka_bashi.runtime_info
import alpaka_bashi.globals


def remove_disabled_serial_backend(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
):
    """All combinations requires an enabled serial backend. Therefore remove all pairs with disabled
    serial backend.
    """
    bashi.remove_parameter_value_pairs(
        parameter_value_pairs,
        removed_parameter_value_pairs,
        parameter1=ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
        value_version1=OFF,
    )


def remove_disabled_backend_for_compiler(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
):
    """Remove all combination where a specific backend cannot be disabled for a given host or device
    compiler"""
    backend_compilers = {
        ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE: [
            (HOST_COMPILER, GCC),
            (DEVICE_COMPILER, GCC),
            (HOST_COMPILER, CLANG),
            (DEVICE_COMPILER, CLANG),
            (DEVICE_COMPILER, NVCC),
        ],
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: [
            (HOST_COMPILER, GCC),
            (DEVICE_COMPILER, GCC),
            (HOST_COMPILER, CLANG),
            (DEVICE_COMPILER, CLANG),
            (DEVICE_COMPILER, NVCC),
        ],
        ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE: [
            (HOST_COMPILER, GCC),
            (DEVICE_COMPILER, GCC),
            (HOST_COMPILER, CLANG),
            (DEVICE_COMPILER, CLANG),
            (HOST_COMPILER, CLANG_CUDA),
            (DEVICE_COMPILER, CLANG_CUDA),
            (DEVICE_COMPILER, NVCC),
        ],
        ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE: [
            # only device compiler GCC and Clang because depending if GCC can be used as NVCC
            # compiler or not it is possible that the this compilers are tested without TBB
            (DEVICE_COMPILER, GCC),
            (DEVICE_COMPILER, CLANG),
            (HOST_COMPILER, ICPX),
            (DEVICE_COMPILER, ICPX),
        ],
    }

    for backend, compilers in backend_compilers.items():
        for parameter, value_name in compilers:
            bashi.remove_parameter_value_pairs_ranges(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                parameter1=backend,
                value_min_version1=OFF,
                value_max_version1=OFF,
                parameter2=parameter,
                value_name2=value_name,
            )


def remove_enabled_backend_for_compiler(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
):
    """Remove all combination where a specific backend cannot be enabled for a given host or device
    compiler"""
    backend_compilers = {
        ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE: [
            (HOST_COMPILER, HIPCC),
            (DEVICE_COMPILER, HIPCC),
            (HOST_COMPILER, CLANG_CUDA),
            (DEVICE_COMPILER, CLANG_CUDA),
            (HOST_COMPILER, ICPX),
            (DEVICE_COMPILER, ICPX),
        ],
        ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE: [
            (HOST_COMPILER, HIPCC),
            (DEVICE_COMPILER, HIPCC),
            (HOST_COMPILER, CLANG_CUDA),
            (DEVICE_COMPILER, CLANG_CUDA),
            (HOST_COMPILER, ICPX),
            (DEVICE_COMPILER, ICPX),
        ],
        ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE: [
            (HOST_COMPILER, HIPCC),
            (DEVICE_COMPILER, HIPCC),
            (HOST_COMPILER, ICPX),
            (DEVICE_COMPILER, ICPX),
        ],
        ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE: [
            (HOST_COMPILER, HIPCC),
            (DEVICE_COMPILER, HIPCC),
            (HOST_COMPILER, CLANG_CUDA),
            (DEVICE_COMPILER, CLANG_CUDA),
            (DEVICE_COMPILER, NVCC),
        ],
    }

    for backend, compilers in backend_compilers.items():
        for parameter, value_name in compilers:
            bashi.remove_parameter_value_pairs_ranges(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                parameter1=backend,
                value_min_version1=ON,
                value_max_version1=ON,
                parameter2=parameter,
                value_name2=value_name,
            )


def remove_simple_backend_backend_combinations(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
):
    """Remove all invalid combinations of the enabled/disabled backends."""
    all_one_api_backends = "all_one_api_backends"
    for (
        backend1,
        state1,
        backend2,
        state2,
    ) in [
        (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, OFF, ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON),
        (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, OFF, ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
        (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON, all_one_api_backends, ON),
        (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON, ALPAKA_ACC_GPU_HIP_ENABLE, ON),
        (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, OFF, all_one_api_backends, ON),
        (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF, ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
        (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON, ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, OFF),
        (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON, all_one_api_backends, ON),
        (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON, all_one_api_backends, ON),
        (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON, ALPAKA_ACC_GPU_HIP_ENABLE, ON),
        (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON, ALPAKA_ACC_GPU_HIP_ENABLE, ON),
        (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON, ALPAKA_ACC_GPU_HIP_ENABLE, ON),
    ]:
        if backend2 == all_one_api_backends:
            second_backends = ONE_API_BACKENDS
        else:
            second_backends = [backend2]
        for second_backend in second_backends:
            bashi.remove_parameter_value_pairs(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                parameter1=backend1,
                value_version1=state1,
                parameter2=second_backend,
                value_version2=state2,
            )


def remove_cuda_backend_backend_combinations(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
):
    """Remove all invalid combinations of an enabled CUDA backend and a enabled/disabled other
    backends. Handles the special case, the CUDA backend has a version number instead is simply
    enabled."""
    for backend, state in [
        (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, OFF),
        (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON),
    ]:
        bashi.remove_parameter_value_pairs_ranges(
            parameter_value_pairs,
            removed_parameter_value_pairs,
            parameter1=backend,
            value_min_version1=state,
            value_max_version1=state,
            parameter2=ALPAKA_ACC_GPU_CUDA_ENABLE,
            value_min_version2=OFF,
            value_min_version2_inclusive=False,
        )


def remove_non_used_nvcc_device_compiler(
    parameter_value_pairs: List[bashi.ParameterValuePair],
    removed_parameter_value_pairs: List[bashi.ParameterValuePair],
    run_infos: Dict[str, Callable[..., bool]],
):
    """Removes the the combination of disabled TBB backend and host compiler GCC or Clang.
    The TBB backend can be only disabled, if the GCC or Clang host compiler can be used as host
    compiler for nvcc. The enabled TBB backend is tested with all Clang and GCC versions when GCC or
    Clang is the device compiler and therefore also the host compiler."""
    if alpaka_bashi.globals.RT_HOST_COMPILER_CUDA_SUPPORT in run_infos:
        host_compiler_supports_cuda = cast(
            alpaka_bashi.runtime_info.HostCompilerSupportsCuda,
            run_infos[alpaka_bashi.globals.RT_HOST_COMPILER_CUDA_SUPPORT],
        )

        for compiler in (GCC, CLANG):
            bashi.remove_parameter_value_pairs_ranges(
                parameter_value_pairs,
                removed_parameter_value_pairs,
                parameter1=HOST_COMPILER,
                value_name1=compiler,
                value_min_version1=str(host_compiler_supports_cuda.get_max_version(compiler)),
                value_min_version1_inclusive=False,
                parameter2=ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
                value_min_version2=OFF,
                value_max_version2=OFF,
            )


def verify(
    combination_list: bashi.CombinationList,
    param_value_matrix: bashi.ParameterValueMatrix,
    run_infos: Dict[str, Callable[..., bool]],
) -> bool:
    """Check if all expected parameter-value-pairs exists in the combination-list.

    Args:
        combination_list (CombinationList): The generated combination list.
        param_value_matrix (ParameterValueMatrix): The expected parameter-values-pairs are generated
            from the parameter-value-list.

    Returns:
        bool: True if it found all pairs
    """

    expected_param_val_tuple, unexpected_param_val_tuple = (
        bashi.get_expected_bashi_parameter_value_pairs(param_value_matrix, run_infos)
    )

    remove_disabled_serial_backend(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_disabled_backend_for_compiler(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_enabled_backend_for_compiler(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_simple_backend_backend_combinations(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_cuda_backend_backend_combinations(expected_param_val_tuple, unexpected_param_val_tuple)
    remove_non_used_nvcc_device_compiler(
        expected_param_val_tuple, unexpected_param_val_tuple, run_infos
    )

    expected_param_val_okay = bashi.check_parameter_value_pair_in_combination_list(
        combination_list, expected_param_val_tuple
    )
    unexpected_param_val_okay = bashi.check_unexpected_parameter_value_pair_in_combination_list(
        combination_list, unexpected_param_val_tuple
    )

    return expected_param_val_okay and unexpected_param_val_okay
