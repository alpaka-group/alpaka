"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

alpaka_bashi package
"""

from alpaka_bashi.alpaka_filter import AlpakaFilter
from alpaka_bashi.combination import add_combinations_parameters
from alpaka_bashi.combination_modifier.job_filter import filter_combinations
from alpaka_bashi.globals import (
    BUILD_TYPE,
    BUILD_TYPES,
    BUILD_TYPES_NAMES,
    CMAKE_DEBUG,
    CMAKE_DEBUG_VER,
    CMAKE_RELEASE,
    CMAKE_RELEASE_VER,
    CMAKE_RELEASE_WITH_DEBUG_INFO,
    CMAKE_RELEASE_WITH_DEBUG_INFO_VER,
    HWLOC,
    get_version_aliases,
)
from alpaka_bashi.jobs import WaveSize
from alpaka_bashi.pipeline import distribute_to_pipelines
from alpaka_bashi.utils import get_filter_name, print_warn
from alpaka_bashi.verify import verify
from alpaka_bashi.versions import (
    get_alpaka_version_relation,
    get_software_versions_for_alpaka,
    get_used_backends,
)
from alpaka_bashi.writers import write_multiple_file_job_configuration, write_single_file_job_configuration

__all__ = [
    "AlpakaFilter",
    "add_combinations_parameters",
    "filter_combinations",
    "BUILD_TYPE",
    "BUILD_TYPES",
    "BUILD_TYPES_NAMES",
    "CMAKE_DEBUG",
    "CMAKE_DEBUG_VER",
    "CMAKE_RELEASE",
    "CMAKE_RELEASE_VER",
    "CMAKE_RELEASE_WITH_DEBUG_INFO",
    "CMAKE_RELEASE_WITH_DEBUG_INFO_VER",
    "HWLOC",
    "WaveSize",
    "distribute_to_pipelines",
    "get_filter_name",
    "print_warn",
    "verify",
    "get_version_aliases",
    "get_alpaka_version_relation",
    "get_used_backends",
    "get_software_versions_for_alpaka",
    "write_multiple_file_job_configuration",
    "write_single_file_job_configuration",
]
