# pylint: disable=missing-docstring

"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import unittest
import io
from typing import Optional, IO, Dict, Callable, cast
from typeguard import typechecked
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.types import ParameterValueTuple
import alpaka_bashi.filter
import alpaka_bashi.runtime_info
from alpaka_bashi.globals import (
    RT_HOST_COMPILER_CUDA_SUPPORT,
    BUILD_TYPE,
    CMAKE_RELEASE,
    CMAKE_DEBUG,
)
from utils import parse_param_value_tuples


@typechecked
def alpaka_filter_typechecked(
    row: ParameterValueTuple,
    output: Optional[IO[str]] = None,
    runtime_info: Dict[str, Callable[..., bool]] | None = None,
) -> bool:
    """Type-checked version of SoftwareDependencyFilter()(). Type checking has a big performance
    cost, which is why the non type-checked version is used for the pairwise generator.
    """
    return alpaka_bashi.filter.AlpakaFilter(output=output, runtime_infos=runtime_info)(row)


class TestAlpakaFilter(unittest.TestCase):
    def test_valid_backend_combinations_a1(self):
        for row in [
            [(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)],
            [(DEVICE_COMPILER, GCC, 8), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)],
            [
                (DEVICE_COMPILER, GCC, 8),
                (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                (HOST_COMPILER, GCC, 8),
            ],
            [(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.4)],
            [
                (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                (HOST_COMPILER, GCC, 11),
                (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
                (DEVICE_COMPILER, GCC, 11),
                (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON),
            ],
        ]:
            self.assertTrue(alpaka_filter_typechecked(parse_param_value_tuples(row)), f"{row}")

    def test_invalid_backend_combinations_a1(self):
        for row in [
            [(ALPAKA_ACC_GPU_HIP_ENABLE, ON), (ALPAKA_ACC_ONEAPI_GPU_ENABLE, ON)],
            [(DEVICE_COMPILER, CLANG, 12), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)],
            [(ALPAKA_ACC_ONEAPI_GPU_ENABLE, ON), (HOST_COMPILER, HIPCC, 6.3)],
            [
                (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                (HOST_COMPILER, GCC, 11),
                (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
                (DEVICE_COMPILER, GCC, 11),
                (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON),
                (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
            ],
        ]:
            reason_msg = io.StringIO()
            self.assertFalse(
                alpaka_filter_typechecked(parse_param_value_tuples(row), reason_msg), f"{row}"
            )
            self.assertEqual(
                reason_msg.getvalue(), "No valid backend combination available.", f"{row}"
            )

    def test_valid_only_cuda_backend_a2(self):
        input_versions = {
            GCC: [10, 11, 12, 13],
            CLANG: [15, 16, 17, 18],
            NVCC: ["12.0", "12.1", "12.2", "12.3", "12.4", "12.5", "12.6"],
        }

        runtime_info = {
            RT_HOST_COMPILER_CUDA_SUPPORT: alpaka_bashi.runtime_info.get_rt_func_host_compiler_supports_cuda(
                input_versions
            )
        }

        rt_host_compiler_cuda_support = cast(
            alpaka_bashi.runtime_info.HostCompilerSupportsCuda,
            runtime_info[RT_HOST_COMPILER_CUDA_SUPPORT],
        )

        self.assertEqual(
            rt_host_compiler_cuda_support.get_max_version(GCC), packaging.version.parse("13")
        )
        self.assertEqual(
            rt_host_compiler_cuda_support.get_max_version(CLANG), packaging.version.parse("18")
        )

        for row in [
            [
                (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
                (HOST_COMPILER, GCC, 11),
                (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON),
                (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
                (DEVICE_COMPILER, GCC, 11),
                (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON),
            ],
            [(ALPAKA_ACC_GPU_CUDA_ENABLE, 12.4)],
            [(ALPAKA_ACC_GPU_CUDA_ENABLE, 12.4), (HOST_COMPILER, GCC, 13)],
            [
                (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.4),
                (HOST_COMPILER, GCC, 13),
                (DEVICE_COMPILER, NVCC, 12.4),
            ],
            [(HOST_COMPILER, CLANG, 18), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.4)],
            [(ALPAKA_ACC_GPU_CUDA_ENABLE, 12.4), (HOST_COMPILER, GCC, 9)],
        ]:
            self.assertTrue(
                alpaka_filter_typechecked(
                    row=parse_param_value_tuples(row), runtime_info=runtime_info
                ),
                f"{row}",
            )

    def test_invalid_only_cuda_backend_a2(self):
        input_versions = {
            GCC: [10, 11, 12, 13],
            CLANG: [15, 16, 17, 18],
            NVCC: ["12.0", "12.1", "12.2", "12.3", "12.4", "12.5", "12.6"],
        }

        runtime_info = {
            RT_HOST_COMPILER_CUDA_SUPPORT: alpaka_bashi.runtime_info.get_rt_func_host_compiler_supports_cuda(
                input_versions
            )
        }

        rt_host_compiler_cuda_support = cast(
            alpaka_bashi.runtime_info.HostCompilerSupportsCuda,
            runtime_info[RT_HOST_COMPILER_CUDA_SUPPORT],
        )

        self.assertEqual(
            rt_host_compiler_cuda_support.get_max_version(GCC), packaging.version.parse("13")
        )
        self.assertEqual(
            rt_host_compiler_cuda_support.get_max_version(CLANG), packaging.version.parse("18")
        )

        for untyped_row in [
            [(HOST_COMPILER, CLANG, 19), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.4)],
            [(ALPAKA_ACC_GPU_CUDA_ENABLE, 12.4), (HOST_COMPILER, GCC, 14)],
            [(ALPAKA_ACC_GPU_CUDA_ENABLE, 12.7), (HOST_COMPILER, GCC, 99)],
        ]:
            row = parse_param_value_tuples(untyped_row)
            reason_msg = io.StringIO()
            self.assertFalse(
                alpaka_filter_typechecked(row=row, output=reason_msg, runtime_info=runtime_info),
            )
            self.assertEqual(
                reason_msg.getvalue(),
                "Only backend combinations with CUDA backend possible. There is no CUDA SDK "
                f"version, which supports the host compiler {row[HOST_COMPILER].name}-"
                f"{row[HOST_COMPILER].version}",
                f"{row}",
            )

    def test_valid_hipcc62_debug_build_a1(self):
        for row in [
            [(DEVICE_COMPILER, HIPCC, 6.2), (BUILD_TYPE, CMAKE_RELEASE)],
            [(HOST_COMPILER, HIPCC, 6.2), (BUILD_TYPE, CMAKE_RELEASE)],
            [(HOST_COMPILER, HIPCC, 6.1), (BUILD_TYPE, CMAKE_DEBUG)],
            [(HOST_COMPILER, HIPCC, 6.3), (BUILD_TYPE, CMAKE_DEBUG)],
            [(HOST_COMPILER, HIPCC, 6.3), (BUILD_TYPE, CMAKE_RELEASE)],
        ]:
            self.assertTrue(alpaka_filter_typechecked(parse_param_value_tuples(row)), f"{row}")

    def test_invalid_hipcc62_debug_build_a1(self):
        for row in [
            [(DEVICE_COMPILER, HIPCC, 6.2), (BUILD_TYPE, CMAKE_DEBUG)],
            [(BUILD_TYPE, CMAKE_DEBUG), (HOST_COMPILER, HIPCC, 6.2)],
        ]:
            self.assertFalse(alpaka_filter_typechecked(parse_param_value_tuples(row)), f"{row}")
