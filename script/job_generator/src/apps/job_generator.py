"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Generates the GitLab CI jobs for alpaka.
"""

import argparse
import os
import random
import shutil
import sys

import bashi
import termcolor

import alpaka_bashi
from alpaka_bashi.globals import CI_PIPELINE_COMPILE_ONLY_VER, CI_PIPELINE_NAME_MAPPING, CI_PIPELINE_RUNTIME_CPU_VER
from alpaka_bashi.versions import get_spack_alias


def get_args() -> argparse.Namespace:
    """Define and parse the commandline arguments.

    Returns:
        argparse.Namespace: The commandline arguments.
    """
    parser = argparse.ArgumentParser(description="Calculate job matrix and create GitLab CI .yml.")

    parser.add_argument("version", type=float, help="Version number of the used CI container.")
    parser.add_argument(
        "--print-combinations",
        action="store_true",
        help="Display combination list.",
    )

    parser.add_argument(
        "--no-image-check",
        action="store_false",
        help="Disable registry check for existing Docker image.",
    )

    parser.add_argument(
        "--no-verification",
        action="store_true",
        help="Disable verification of the combination and continue generating the GitLab CI yaml code.",
    )

    parser.add_argument(
        "--no-spack",
        action="store_true",
        help="Disable job generation to build spack packages and push to the cache.",
    )

    parser.add_argument(
        "--print-spack",
        action="store_true",
        help="Display all spack packages which are missing the build cache.",
    )

    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Filter the jobs with a Python regex that checks the job names.",
    )

    parser.add_argument(
        "--split-pipeline",
        action="store_true",
        help="Write job pipelines in separate output files.",
    )

    for wave_name in CI_PIPELINE_NAME_MAPPING:
        parser.add_argument(
            f"--pipeline-out-{wave_name}",
            type=str,
            required="--split-pipeline" in sys.argv,
            # add `all` and remove `JOB_UNKNOWN` from the choices
            help=f"Output path of the job yaml for the pipeline {wave_name}",
        )

    parser.add_argument(
        "--debug-print",
        type=bashi.FilterDebugMode,
        choices=list(bashi.FilterDebugMode),
        default="off",
        help="Display Indicate which combinations passed through the filter chain and which did "
        "not.Green text indicates that the combination passed through the filter chain; red text"
        " indicates that it did not. Add the keyword `passed` if colored output is not available."
        "Option `normal` is easy human readable output. If `args` is set, the output can be "
        "directly passed to the validator",
    )

    return parser.parse_args()


def setup_row_printer() -> None:
    """Set extra configurations for the bashi.print_row_nice() function"""
    bashi.add_print_row_nice_parameter_alias(alpaka_bashi.BUILD_TYPE, "buildType")
    # bashi.add_print_row_nice_parameter_alias(alpaka_bashi.JOB_EXECUTION_TYPE, "jobType")

    for val_name, aliases in alpaka_bashi.get_version_aliases().items():
        bashi.add_print_row_nice_version_alias(val_name, aliases)


def get_missing_cached_spack_packages(combination_list: bashi.CombinationList) -> bashi.SpackBuilder:
    """
    Calculate all spack packages, which needs to be build for the given combination-list.

    Args:
        combination_list (bashi.CombinationList): The SpackBuilder reads the combination-list and extract the required
            spack packages from it.

    Returns:
        SpackBuilder: The spack builder contains all information about required spack packages. For example, which can
            be build, which are already exist in the build cache and which one are needs to be build.
    """
    if not shutil.which("spack"):
        # If the environment variable CI is not set, we are on a local system and spack is optional
        if "CI" in os.environ:
            bashi.exit_error("spack is not installed")
        else:
            alpaka_bashi.print_warn(
                "spack is not installed and run in non-CI environment. Skip generating spack package build jobs."
            )
            return bashi.SpackBuilder([], {}, bashi.get_spack_build_cache_url)

    spack_builder = bashi.SpackBuilder(
        combination_list, alpaka_bashi.get_spack_alias(), bashi.get_spack_build_cache_url
    )
    missing_packages_str = "The following spack packages does not exist:\n"
    for ubuntu, spack_packages in spack_builder.get_required_available_spack_packages().missing_packages.items():
        if spack_packages.values():
            missing_packages_str += f"=== Ubuntu {bashi.ubuntu_version_to_string(ubuntu)} ===\n"
            for pkg_name, pkg_version in sorted(bashi.pkg_iter(spack_packages)):
                missing_packages_str += f"  {spack_builder.spack_packages[pkg_name].package_name}@{pkg_version}\n"
    alpaka_bashi.print_warn(missing_packages_str)

    return spack_builder


def print_spack_info(spack_builder: bashi.SpackBuilder):
    """
    Prints all required information about the spack packages which are required for testing.

    Args:
        spack_builder (bashi.SpackBuilder): The SpackBuilder which contains the information.
    """
    for ubuntu, spack_packages in spack_builder.get_build_cache_packages().existing_packages.items():
        if spack_packages.values():
            print(
                termcolor.colored(
                    f"=== Packages in build cache for Ubuntu {bashi.ubuntu_version_to_string(ubuntu)} ===", "green"
                ),
                file=sys.stderr,
            )
            for pkg_name, pkg_version in sorted(bashi.pkg_iter(spack_packages)):
                print(termcolor.colored(f"  {get_spack_alias()[pkg_name].package_name}@{pkg_version}", "green"))

    print()

    for ubuntu, spack_packages in spack_builder.get_build_cache_packages().missing_packages.items():
        if spack_packages.values():
            print(
                termcolor.colored(
                    f"=== Missing packages in build cache for Ubuntu {bashi.ubuntu_version_to_string(ubuntu)} ===",
                    "red",
                )
            )
            for pkg_name, pkg_version in sorted(bashi.pkg_iter(spack_packages)):
                print(termcolor.colored(f"  {get_spack_alias()[pkg_name].package_name}@{pkg_version}", "red"))


def main() -> None:
    """The main entry point."""
    args = get_args()

    setup_row_printer()

    software_versions = alpaka_bashi.get_software_versions_for_alpaka()
    param_matrix: bashi.ParameterValueMatrix = bashi.get_parameter_value_matrix(
        software_versions=software_versions, backends=alpaka_bashi.get_used_backends()
    )

    version_relation = alpaka_bashi.get_alpaka_version_relation()
    alpaka_filter = alpaka_bashi.AlpakaFilter()
    runtime_infos = bashi.get_runtime_infos(param_matrix, version_relation)

    comb_list: bashi.CombinationList = bashi.generate_combination_list(
        parameter_value_matrix=param_matrix,
        runtime_infos=runtime_infos,
        custom_filter=alpaka_filter,
        version_relation=version_relation,
        debug_print=args.debug_print,
    )
    print(f"number of combinations: {len(comb_list)}", file=sys.stderr)

    comb_list = alpaka_bashi.add_combinations_parameters(comb_list)

    if not args.no_verification:
        if not alpaka_bashi.verify(comb_list, param_matrix, version_relation, runtime_infos):
            print("ERROR: Result is incorrect", file=sys.stderr)
            sys.exit(1)
        else:
            print("Result is correct", file=sys.stderr)
    else:
        alpaka_bashi.print_warn("Skip verification step")

    job_filter_name = alpaka_bashi.get_filter_name(args)
    if job_filter_name:
        comb_list = alpaka_bashi.filter_combinations(comb_list, job_filter_name)
        print(f"number of filtered combinations: {len(comb_list)}", file=sys.stderr)

    if args.print_combinations:
        for c in comb_list:
            bashi.print_row_nice(c)
        sys.exit(0)

    spack_builder = bashi.SpackBuilder([], {}, bashi.get_spack_build_cache_url)

    if not args.no_spack:
        spack_builder = get_missing_cached_spack_packages(comb_list)
    else:
        alpaka_bashi.print_warn("Skip generating spack package build jobs")

    if args.print_spack:
        print_spack_info(spack_builder)
        sys.exit(0)

    # shuffle jobs to increase the chance to run different compiler in the first wave
    random.Random(42).shuffle(comb_list)

    pipelines = alpaka_bashi.distribute_to_pipelines(comb_list)

    # TODO: remove me, if all CI dependencies are ported to spack packages.
    # We cannot port all dependencies to spack package at the same time.
    # Therefore remove all packages, which we don't want to build yet.
    for ubuntu, spack_packages in spack_builder.get_build_cache_packages().missing_packages.items():
        allowed_spack_packages: dict[bashi.ValueName, list[bashi.ValueVersion]] = {}
        for sw_name, sw_version in bashi.pkg_iter(spack_packages):
            if sw_name == bashi.ALPAKA_ACC_GPU_CUDA_ENABLE and bashi.ValueVersion(
                "12.6"
            ) <= sw_version <= bashi.ValueVersion("12.8"):
                bashi.SpackBuilder._dict_add_unique_list_entry(sw_name, sw_version, allowed_spack_packages)
        spack_builder.get_build_cache_packages().missing_packages[ubuntu] = allowed_spack_packages

    # If the pipelines are not split and therefore written to different files, write everything
    # to stdout.
    # We split up the pipelines and merge again, because in the meantime reorder operations can be
    # applied on the different pipelines.
    # By the way, it also automatically sort the jobs by pipeline.
    if not args.split_pipeline:
        alpaka_bashi.write_single_file_job_configuration(pipelines, spack_builder, args, sys.stdout)
    else:
        wave_sizes = {
            CI_PIPELINE_COMPILE_ONLY_VER: alpaka_bashi.WaveSize(30, 2),
            CI_PIPELINE_RUNTIME_CPU_VER: alpaka_bashi.WaveSize(30, 2),
        }

        alpaka_bashi.write_multiple_file_job_configuration(pipelines, spack_builder, wave_sizes, args)

    sys.exit(0)


if __name__ == "__main__":
    main()
