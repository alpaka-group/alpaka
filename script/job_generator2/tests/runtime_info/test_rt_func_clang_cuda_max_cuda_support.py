# pylint: disable=missing-docstring

"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import unittest
import packaging
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.version.dependencies.clang_cuda import ClangCudaSDKSupport
from alpaka_bashi.runtime_info import ClangCUDAMaxSupportsCuda


class TestRtClangCUDAMaxSupportsCuda(unittest.TestCase):
    def test_rt_clang_cuda_max_cuda_support_case1(self):
        clang_cuda_versions = {CLANG_CUDA: [14, 15, 16, 17, 18, 19]}
        clang_cuda_sdk_support = [
            ClangCudaSDKSupport("7", "9.2"),
            ClangCudaSDKSupport("8", "10.0"),
            ClangCudaSDKSupport("10", "10.1"),
            ClangCudaSDKSupport("12", "11.0"),
            ClangCudaSDKSupport("13", "11.2"),
            ClangCudaSDKSupport("14", "11.5"),
            ClangCudaSDKSupport("16", "11.8"),
            ClangCudaSDKSupport("17", "12.1"),
            ClangCudaSDKSupport("18", "12.3"),
            ClangCudaSDKSupport("22", "13.0"),
        ]

        rt = ClangCUDAMaxSupportsCuda(
            bashi.VersionRelation(clang_cuda_max_cuda_version=clang_cuda_sdk_support),
            clang_cuda_versions,
        )
        self.assertEqual(rt.max_cuda_sdk_version, packaging.version.parse("12.3"))
        self.assertTrue(rt(packaging.version.parse("12.0")))
        self.assertTrue(rt(packaging.version.parse("12.3")))
        self.assertFalse(rt(packaging.version.parse("12.4")))
        self.assertFalse(rt(packaging.version.parse("13.0")))

    def test_rt_clang_cuda_max_cuda_support_case2(self):
        clang_cuda_versions = {CLANG_CUDA: [14, 15, 16, 17, 18]}
        clang_cuda_sdk_support = [
            ClangCudaSDKSupport("7", "9.2"),
            ClangCudaSDKSupport("8", "10.0"),
            ClangCudaSDKSupport("10", "10.1"),
            ClangCudaSDKSupport("12", "11.0"),
            ClangCudaSDKSupport("13", "11.2"),
            ClangCudaSDKSupport("14", "11.5"),
            ClangCudaSDKSupport("16", "11.8"),
            ClangCudaSDKSupport("17", "12.1"),
            ClangCudaSDKSupport("18", "12.3"),
            ClangCudaSDKSupport("22", "13.0"),
        ]

        rt = ClangCUDAMaxSupportsCuda(
            bashi.VersionRelation(clang_cuda_max_cuda_version=clang_cuda_sdk_support),
            clang_cuda_versions,
        )
        self.assertEqual(rt.max_cuda_sdk_version, packaging.version.parse("12.3"))
        self.assertTrue(rt(packaging.version.parse("12.0")))
        self.assertTrue(rt(packaging.version.parse("12.3")))
        self.assertFalse(rt(packaging.version.parse("12.4")))
        self.assertFalse(rt(packaging.version.parse("13.0")))

    def test_rt_clang_cuda_max_cuda_support_case3(self):
        clang_cuda_versions = {CLANG_CUDA: [14, 15, 16, 17, 18, 23]}
        clang_cuda_sdk_support = [
            ClangCudaSDKSupport("7", "9.2"),
            ClangCudaSDKSupport("8", "10.0"),
            ClangCudaSDKSupport("10", "10.1"),
            ClangCudaSDKSupport("12", "11.0"),
            ClangCudaSDKSupport("13", "11.2"),
            ClangCudaSDKSupport("14", "11.5"),
            ClangCudaSDKSupport("16", "11.8"),
            ClangCudaSDKSupport("17", "12.1"),
            ClangCudaSDKSupport("18", "12.3"),
            ClangCudaSDKSupport("22", "13.0"),
        ]

        rt = ClangCUDAMaxSupportsCuda(
            bashi.VersionRelation(clang_cuda_max_cuda_version=clang_cuda_sdk_support),
            clang_cuda_versions,
        )
        self.assertEqual(rt.max_cuda_sdk_version, packaging.version.parse("13.0"))
        self.assertTrue(rt(packaging.version.parse("12.0")))
        self.assertTrue(rt(packaging.version.parse("12.3")))
        self.assertTrue(rt(packaging.version.parse("12.4")))
        self.assertTrue(rt(packaging.version.parse("13.0")))
        self.assertFalse(rt(packaging.version.parse("13.1")))
