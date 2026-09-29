# pylint: disable=missing-docstring

"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import unittest

from bashi import ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF, ON, VersionRelation
from bashi.globals import ALPAKA_ACC_GPU_CUDA_ENABLE, CLANG, CMAKE, DEVICE_COMPILER, GCC, HOST_COMPILER, NVCC
from bashi.types import ParameterValuePair
from bashi.version.dependencies.nvcc import NvccHostSupport
from utils import default_remove_test, parse_expected_val_pairs

from alpaka_bashi.verify import (
    remove_unsupported_cuda_sdk_for_clang_host_compiler,
    serial_backend_is_always_on_for_unsupported_nvcc_host_compiler,
)


class TestClangHostCompilerForNvcc(unittest.TestCase):
    def test_remove_invalid_combinations(self):

        test_param_value_pairs: list[ParameterValuePair] = parse_expected_val_pairs(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((HOST_COMPILER, CLANG, 9), (ALPAKA_ACC_GPU_CUDA_ENABLE, "13.2")),
                ((HOST_COMPILER, CLANG, 10), (ALPAKA_ACC_GPU_CUDA_ENABLE, "13.3")),
                ((HOST_COMPILER, CLANG, 12), (ALPAKA_ACC_GPU_CUDA_ENABLE, "13.4")),
                ((DEVICE_COMPILER, NVCC, 10.0), (HOST_COMPILER, CLANG, 12)),
                ((DEVICE_COMPILER, NVCC, 13.2), (HOST_COMPILER, CLANG, 14)),
                ((DEVICE_COMPILER, NVCC, 13.3), (HOST_COMPILER, CLANG, 16)),
                ((HOST_COMPILER, CLANG, 16), (DEVICE_COMPILER, NVCC, 13.4)),
                ((DEVICE_COMPILER, NVCC, 12.3), (HOST_COMPILER, GCC, 16)),
            ]
        )

        expected_results: list[ParameterValuePair] = parse_expected_val_pairs(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((HOST_COMPILER, CLANG, 10), (ALPAKA_ACC_GPU_CUDA_ENABLE, "13.3")),
                ((HOST_COMPILER, CLANG, 12), (ALPAKA_ACC_GPU_CUDA_ENABLE, "13.4")),
                ((DEVICE_COMPILER, NVCC, 13.3), (HOST_COMPILER, CLANG, 16)),
                ((HOST_COMPILER, CLANG, 16), (DEVICE_COMPILER, NVCC, 13.4)),
                ((DEVICE_COMPILER, NVCC, 12.3), (HOST_COMPILER, GCC, 16)),
            ]
        )

        default_remove_test(
            remove_unsupported_cuda_sdk_for_clang_host_compiler,
            test_param_value_pairs,
            expected_results,
            self,
        )


class TestSerialBackendIsAlwaysOnForUnsupportedNvccHostCompiler(unittest.TestCase):
    def test_remove_invalid_combinations(self):
        nvcc_gcc_support: list[NvccHostSupport] = [
            NvccHostSupport("13.4", "16"),
            NvccHostSupport("13.0", "15"),
            NvccHostSupport("12.9", "14"),
            NvccHostSupport("12.6", "13"),
        ]

        nvcc_clang_support: list[NvccHostSupport] = [
            NvccHostSupport("13.4", "22"),
            NvccHostSupport("13.0", "20"),
            NvccHostSupport("12.9", "19"),
            NvccHostSupport("12.6", "18"),
            NvccHostSupport("12.4", "17"),
        ]

        test_param_value_pairs: list[ParameterValuePair] = parse_expected_val_pairs(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((HOST_COMPILER, GCC, 14), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                ((HOST_COMPILER, GCC, 14), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, GCC, 16), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                ((HOST_COMPILER, GCC, 16), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, GCC, 17), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                ((HOST_COMPILER, GCC, 17), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, GCC, 20), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                ((HOST_COMPILER, GCC, 20), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, CLANG, 14), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                ((HOST_COMPILER, CLANG, 14), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, CLANG, 22), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                ((HOST_COMPILER, CLANG, 22), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, CLANG, 23), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                ((HOST_COMPILER, CLANG, 23), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, CLANG, 30), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                ((HOST_COMPILER, CLANG, 30), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
            ]
        )

        expected_results: list[ParameterValuePair] = parse_expected_val_pairs(
            [
                ((HOST_COMPILER, GCC, 6), (CMAKE, "3.30.2")),
                ((HOST_COMPILER, GCC, 14), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                ((HOST_COMPILER, GCC, 14), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, GCC, 16), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                ((HOST_COMPILER, GCC, 16), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, GCC, 17), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                # ((HOST_COMPILER, GCC, 17), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, GCC, 20), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                # ((HOST_COMPILER, GCC, 20), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, CLANG, 14), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                ((HOST_COMPILER, CLANG, 14), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, CLANG, 22), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                ((HOST_COMPILER, CLANG, 22), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, CLANG, 23), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                # ((HOST_COMPILER, CLANG, 23), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
                ((HOST_COMPILER, CLANG, 30), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)),
                # ((HOST_COMPILER, CLANG, 30), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, OFF)),
            ]
        )

        version_relation = VersionRelation(
            nvcc_gcc_max_version=nvcc_gcc_support, nvcc_clang_max_version=nvcc_clang_support
        )

        default_remove_test(
            serial_backend_is_always_on_for_unsupported_nvcc_host_compiler,
            test_param_value_pairs,
            expected_results,
            self,
            version_relation,
        )
