#!/bin/bash
# Define global variables
SCRIPT_DIR=$(dirname "$(realpath "$0")")
results_dir="$SCRIPT_DIR/test-results"
mkdir -p "$results_dir"

# Function to print directory information
print_directory_info() {
    echo "Current Working Directory: $(pwd)"
    echo "Script Directory: $(dirname "$(realpath "$0")")"
}

# Function to clone or update the Alpaka repository
clone_or_update_alpaka() {
    if [ -d "alpaka" ]; then
        echo "Updating Alpaka repository..."
        cd alpaka || exit 1
        git checkout develop
        git pull origin develop
    else
        echo "Cloning Alpaka repository..."
        git clone https://github.com/alpaka-group/alpaka.git --branch develop
        cd ./alpaka
    fi
}

# Function to set up the environment for cpu-omp2b
setup_environment_cpu_omp2b() {
    echo "Setting up environment for cpu-omp2b..."
    # source /etc/profile.d/modules.sh
    # source /opt/spack/share/spack/setup-env.sh

    module load git
    module load cmake/3.26.1
    module load gcc/12.2.0 || { echo "Failed to load gcc/12.2.0"; return 1; }
    module load python/3.10.4 || { echo "Failed to load python/3.10.4"; return 1; }
    module load boost/1.82.0 || { echo "Failed to load Boost"; return 1; }  # Specific hash for Boost
   
    # Ensure OpenMP is available
    which gcc > /dev/null || { echo "Error: GCC (with OpenMP support) not found."; return 1; }

    echo "Environment setup completed successfully."
}

# Function to configure, build, and run the benchmark for cpu-omp2b
build_and_run_cpu_omp2b() {
    local preset="cpu-omp2b"
    local num_cores=$(( $(nproc) - 2 ))
    num_cores=$(( num_cores < 1 ? 1 : num_cores ))  # Ensure at least 1 core is used

    boost_path=$(spack location -i /u3oct6d)/include
    echo "Using Boost include directory: $boost_path"

    extra_flags=""

    if [ "$(basename "$(pwd)")" == "build" ]; then
        cd .. || exit 1
    fi

    if [ ! -f "CMakePresets.json" ]; then
        echo "Error: Script must be executed from the alpaka root directory."
        exit 1
    fi

    echo "Configuring for preset: $preset"
    cmake --preset "$preset" \
          -DBoost_INCLUDE_DIR="$boost_path" \
          -Dalpaka_BUILD_BENCHMARKS=ON \
          -DCMAKE_BUILD_TYPE=RELEASE \
          $extra_flags

    build_dir="build/$preset"
    cd "$build_dir" || exit 1
    echo "Building for preset: $preset"
    cmake --build . --target babelstream -j "$num_cores"

    # Define the results file path in the test-results directory
    datetime_now=$(date +"%Y-%m-%d_%H-%M")
    commit_hash=$(git rev-parse --short=8 HEAD 2>/dev/null || echo "unknown")
    results_file="$results_dir/babelstream-$preset-$datetime_now-$commit_hash.txt"

    # Run the benchmark directly
    echo "Running BabelStream benchmark locally..."
    echo "Results will be saved to: $results_file"

    babelstream_executable="./benchmarks/babelstream/babelstream"
    if [ ! -x "$babelstream_executable" ]; then
        echo "Error: BabelStream executable not found or not executable."
        exit 1
    fi

    "$babelstream_executable" --array-size=33554432 --number-runs=10 > "$results_file" 2>&1
    echo "Benchmark completed. Results saved to $results_file."
}

# Main script execution
print_directory_info
clone_or_update_alpaka
echo "Current Working Directory: $(pwd)"

if [ "$(basename "$(pwd)")" != "alpaka" ]; then
    echo "Error: Script must be executed from the alpaka root directory."
    exit 1
fi

if ! setup_environment_cpu_omp2b; then
    echo "Failed to set up the environment. Exiting."
    exit 1
fi

build_and_run_cpu_omp2b
