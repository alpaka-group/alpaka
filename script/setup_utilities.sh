#!/usr/bin/env bash

# SPDX-License-Identifier: MPL-2.0

# serveral helper function and tools for the CI

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
