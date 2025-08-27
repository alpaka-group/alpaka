"""Filter rules which will generated during runtime depending on the input of the input
parameter-value-matrix"""

from typing import Dict, Callable
import packaging.version
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_bashi.globals import RT_HOST_COMPILER_CUDA_SUPPORT
from alpaka_bashi.versions import get_alpaka_version


class HostCompilerSupportsCuda:
    """Check if for the given host compiler is a supported CUDA SDK version."""

    def __init__(self, gcc_version: bashi.ValueVersion, clang_version: bashi.ValueVersion):
        self.max_versions: Dict[bashi.ValueName, bashi.ValueVersion] = {
            GCC: gcc_version,
            CLANG: clang_version,
        }

    def get_max_version(self, compiler_name: bashi.ValueName) -> bashi.ValueVersion:
        """Return the maximum compiler version, which is supported by a CUDA SDK.

        Args:
            compiler_name (bashi.ValueName): Name of the compiler

        Returns:
            bashi.ValueVersion: Maximum supported version
        """
        return self.max_versions[compiler_name]

    def __call__(
        self, compiler_name: bashi.ValueName, compiler_version: bashi.ValueVersion
    ) -> bool:
        return compiler_version <= self.max_versions[compiler_name]


def get_rt_func_host_compiler_supports_cuda(
    input_versions: Dict[str, List[str | int | float]] | None = None,
) -> Callable[..., bool]:
    """This runtime info object contains the information for the given GCC, Clang and Nvcc versions,
    which is the highest GCC or Clang version which can be used by a Nvcc compiler.

    Args:
        input_versions (Dict[str, List[str  |  int  |  float]] | None, optional): Dict containing
            the GCC, CLANG and NVCC versions which are used to generate the parameter-value-matrix.
            If the value is None, use return value of alpaka_bashi.versions.get_version(). Defaults
            to None.

    Returns:
        Callable[..., bool]: Return HostCompilerSupportsCuda object.
    """
    max_gcc_version = packaging.version.parse("0")
    max_clang_version = packaging.version.parse("0")
    if input_versions is None:
        input_versions = get_alpaka_version()

    max_cuda_sdk_version = packaging.version.parse(str(max(input_versions[NVCC])))

    for cuda_sdk_gcc in sorted(bashi.versions.NVCC_GCC_MAX_VERSION):
        if cuda_sdk_gcc.nvcc <= max_cuda_sdk_version:
            max_gcc_version = cuda_sdk_gcc.host

    for cuda_sdk_gcc in sorted(bashi.versions.NVCC_CLANG_MAX_VERSION):
        if cuda_sdk_gcc.nvcc <= max_cuda_sdk_version:
            max_clang_version = cuda_sdk_gcc.host

    return HostCompilerSupportsCuda(max_gcc_version, max_clang_version)


def get_runtime_infos() -> Dict[str, Callable[..., bool]]:
    """Get several runtime filter rules for the given input parameter-value-matrix

    Args:
        parameter_value_matrix (ParameterValueMatrix): parameter-value-matrix

    Returns:
        Dict[str, Callable[..., bool]]: Dict of filter functions
    """
    runtime_infos: Dict[str, Callable[..., bool]] = {}
    runtime_infos[RT_HOST_COMPILER_CUDA_SUPPORT] = get_rt_func_host_compiler_supports_cuda()

    return runtime_infos
