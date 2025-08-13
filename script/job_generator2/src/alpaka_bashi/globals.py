"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

This module contains constants used for the alpaka job generation.
"""

from typing import List, Dict, Union
import packaging.version
import bashi

# possible values of BUILD_TYPE
BUILD_TYPE: bashi.Parameter = "build_type"
CMAKE_RELEASE: int = 0
CMAKE_DEBUG: int = 1
BUILD_TYPES: List[Union[str, int, float]] = [CMAKE_RELEASE, CMAKE_DEBUG]
BUILD_TYPES_NAMES: Dict[int, str] = {CMAKE_RELEASE: "Release", CMAKE_DEBUG: "Debug"}

# possible values of TEST_TYPE
JOB_EXECUTION_TYPE: bashi.Parameter = "job_execution_type"
JOB_EXECUTION_COMPILE_ONLY: int = 0
JOB_EXECUTION_RUNTIME: int = 1
JOB_EXECUTION_TYPES: List[Union[str, int, float]] = [
    JOB_EXECUTION_COMPILE_ONLY,
    JOB_EXECUTION_RUNTIME,
]
JOB_EXECUTION_TYPES_NAMES: Dict[int, str] = {
    JOB_EXECUTION_COMPILE_ONLY: "compile_only",
    JOB_EXECUTION_RUNTIME: "runtime",
}

# enable mdspan support
MDSPAN: bashi.Parameter = "mdspan"


def get_version_aliases() -> Dict[bashi.ValueName, Dict[bashi.ValueVersion, str]]:
    """Return a list of value-version aliases which can be set for print_row_nice()

    Returns:
        Dict[bashi.ValueName, Dict[bashi.ValueVersion, str]]: _description_
    """
    version_aliases = {}
    for val_name, version_map in [
        (BUILD_TYPE, BUILD_TYPES_NAMES),
        (JOB_EXECUTION_TYPE, JOB_EXECUTION_TYPES_NAMES),
    ]:
        version_map_parsed: Dict[bashi.ValueVersion, str] = {}
        for ver, alias in version_map.items():
            version_map_parsed[packaging.version.parse(str(ver))] = alias
        version_aliases[val_name] = version_map_parsed

    return version_aliases
