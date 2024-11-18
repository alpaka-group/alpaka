#!/bin/bash

#
# Copyright 2022 Simeon Ehrig
# SPDX-License-Identifier: MPL-2.0
#

# set exit on error manually instead using setup_utilities because
# otherwise the begin of the job log looks not helpful
if [ -z ${alpaka_DISABLE_EXIT_FAILURE+x} ]; then
    set -e
fi

# print all environment variables, which are required to reproduce alpaka build
#
# @param $1 (optional): set `export_env` to print all environment variables in the shape of:
#    export VAR_NAME=VAR_VALUE \
function print_env() {
    if [[ $# -ge 1 ]] && [[ "$1" == "export_env" ]]; then
        export_cmd=true
    else
        export_cmd=false
    fi

    # take all env variables, filter it and display it with a `export` prefix
    env_name_prefixes=("^ALPAKA_*" "^alpaka_*" "^CMAKE_*" "^BOOST_*" "^CUDA_*")
    for reg in ${env_name_prefixes[@]}; do
        if printenv | grep -qE ${reg}; then
            printenv | grep -E ${reg} | sort | while read -r line; do
                if $export_cmd == true; then
                    echo "export $line \\"
                else
                    echo "$line"
                fi
            done
        fi
    done
}

# on GitLab CI print all instructions to run test locally
# on GitHub Actions, simply print environment variables
if [ ! -z "${GITLAB_CI+x}" ]; then
    # display output with yellow color
    echo -e "\033[0;33mSteps to setup container locally"

    # display the correct docker run command
    first_step_prefix="1. Run docker image via:"
    if [ "${CMAKE_CXX_COMPILER:-}" == "nvc++" ] || [ "${alpaka_ACC_GPU_CUDA_ENABLE}" == "ON" ]; then
        if [ "${ALPAKA_CI_RUN_TESTS}" == "ON" ]; then
            echo "${first_step_prefix} docker run --gpus=all -it ${CI_JOB_IMAGE} bash"
        else
            echo "${first_step_prefix} docker run -it ${CI_JOB_IMAGE} bash"
        fi
    elif [ "${alpaka_ACC_GPU_HIP_ENABLE}" == "ON" ]; then
        if [ "${ALPAKA_CI_RUN_TESTS}" == "ON" ]; then
            echo "${first_step_prefix} docker run -it --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video ${CI_JOB_IMAGE} bash"
        else
            echo "${first_step_prefix} docker run -it ${CI_JOB_IMAGE} bash"
        fi
    else
        echo "${first_step_prefix} docker run -it ${CI_JOB_IMAGE} bash"
    fi

    echo -e "2. Run the following export commands in the container to setup environment\n"
    print_env export_env

    echo "export CI_RUNNER_TAGS='${CI_RUNNER_TAGS}' \\"

    # the variable is not set, but should be set if a job is debugged locally in a container
    echo 'export alpaka_DISABLE_EXIT_FAILURE=true \'
    echo 'export GITLAB_CI=true'
    echo ""

    echo "3. install git: apt update && apt install -y git"
    echo "4. clone alpaka repository: git clone https://gitlab.com/hzdr/crp/alpaka.git --depth 1 -b ${CI_COMMIT_BRANCH}"
    echo "5. Run the following script: cd alpaka && ./script/gitlab_ci_run.sh"
    # reset the color
    echo -e "\033[0m"
else
    print_env
fi
