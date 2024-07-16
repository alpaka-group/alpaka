#!/usr/bin/env bash

# SPDX-License-Identifier: MPL-2.0

# serveral helper function and tools for the CI

: "${ALPAKA_CI_OS_NAME?'ALPAKA_CI_OS_NAME must be specified'}"

# enable Terminal colors for MacOS
if [ "$ALPAKA_CI_OS_NAME" = "macOS" ]; then
    export CLICOLOR=1
    export LSCOLORS=ExFxCxDxBxegedabagacad
fi

# display a message in green
echo_green() {
    echo -e "\e[1;32m$1\e[0m"
}

# display a message in yellow
echo_yellow() {
    echo -e "\e[1;33m$1\e[0m"
}

# display a message in red
echo_red() {
    echo -e "\e[1;31m$1\e[0m"
}
