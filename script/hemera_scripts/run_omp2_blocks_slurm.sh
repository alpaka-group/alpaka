i#!/bin/bash

# Define global variables
SCRIPT_DIR=$(dirname "$(realpath "$0")")
results_dir="$SCRIPT_DIR/test-results"
mkdir -p "$results_dir"

# Function to print directory information
print_directory_info() {
    echo "Current Working Directory: $(pwd)"
    echo "Script Directory: $(dirname "$(realpath "$0")")"
}

# Function to initialize the module system (based on your interactive environment)
initialize_modules() {
    echo "Initializing module system..."

    # Set MODULEPATH as seen in your interactive environment
    export MODULEPATH=/trinity/shared/lmod/modulefiles/Linux:/trinity/shared/lmod/modulefiles/Core:/trinity/shared/lmod/lmod/modulefiles/Core/tools:/trinity/shared/lmod/lmod/modulefiles/Core/ansys:/trinity/shared/lmod/lmod/modulefiles/Core/analysis:/trinity/shared/lmod/lmod/modulefiles/Core/simulation:/trinity/shared/lmod/lmod/modulefiles/Core/devel:/trinity/shared/lmod/lmod/modulefiles/Core/compiler

    # Source the Lmod initialization script
    if [ -f /trinity/shared/lmod/lmod/init/bash ]; then
        source /trinity/shared/lmod/lmod/init/bash
    elif [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    else
        echo "Error: Module system initialization script not found."
        exit 1
    fi

    echo "Module system initialized successfully."
}

# Function to clone or update the Alpaka repository
clone_or_update_alpaka() {
    if [ -d "alpaka" ]; then
        echo "Updating Alpaka repository..."
        cd alpaka || exit 1

        # Explicitly set PATH for git (based on your interactive environment)
        export PATH=/trinity/shared/pkg/devel/git/2.37.1/bin:$PATH
        git checkout develop
        git pull origin develop
    else
        echo "Cloning Alpaka repository..."

        # Explicitly set PATH for git (based on your interactive environment)
        export PATH=/trinity/shared/pkg/devel/git/2.37.1/bin:$PATH
        git clone https://github.com/alpaka-group/alpaka.git --branch develop
        cd ./alpaka
    fi
}

# Function to set up the environment for cpu-omp2b
setup_environment_cpu_omp2b() {
    echo "Setting up environment for cpu-omp2b..."

    # Initialize the module system
    initialize_modules

    # Load necessary modules (based on your interactive environment)
    module load git
    module load cmake/3.26.1
    module load gcc/12.2.0 || { echo "Failed to load gcc/12.2.0"; return 1; }
    module load python/3.10.4 || { echo "Failed to load python/3.10.4"; return 1; }
    module load boost/1.82.0 || { echo "Failed to load Boost"; return 1; }

    # Ensure OpenMP is available
    which gcc > /dev/null || { echo "Error: GCC (with OpenMP support) not found."; return 1; }

    echo "Environment setup completed successfully."
}

# Function to configure and build the project for cpu-omp2b
build_cpu_omp2b() {
    local preset="cpu-omp2b"
    local num_cores=$(( $(nproc) - 2 ))
    num_cores=$(( num_cores < 1 ? 1 : num_cores ))  # Ensure at least 1 core is used

    extra_flags="-Dalpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE=ON"

    if [ "$(basename "$(pwd)")" == "build" ]; then
        cd .. || exit 1
    fi

    if [ ! -f "CMakePresets.json" ]; then
        echo "Error: Script must be executed from the alpaka root directory."
        exit 1
    fi

    echo "Configuring for preset: $preset"
    cmake --preset "$preset" \
          -Dalpaka_BUILD_BENCHMARKS=ON \
          -DCMAKE_BUILD_TYPE=RELEASE \
          $extra_flags

    build_dir="build/$preset"
    cd "$build_dir" || exit 1

    echo "Building for preset: $preset"
    cmake --build . --target babelstream -j "$num_cores"
}

# Function to submit the benchmark for cpu-omp2b using Slurm
submit_cpu_omp2b_benchmark() {
    local preset="cpu-omp2b"
    datetime_now=$(date +"%Y-%m-%d_%H-%M")
    commit_hash=$(git rev-parse --short=8 HEAD 2>/dev/null || echo "unknown")
    results_file="$results_dir/babelstream-$preset-$datetime_now-$commit_hash.txt"

    echo "Submitting BabelStream benchmark to Slurm..."
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=babelstream-cpu-omp2b
#SBATCH --output=$results_dir/babelstream-slurm-%j.out  # Store Slurm metadata here
#SBATCH --error=$results_dir/babelstream-slurm-%j.err   # Store Slurm errors here
#SBATCH --time=00:30:00
#SBATCH --partition=milan                            # Use CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Reinitialize the module system inside the Slurm job
export MODULEPATH=/trinity/shared/lmod/modulefiles/Linux:/trinity/shared/lmod/modulefiles/Core:/trinity/shared/lmod/lmod/modulefiles/Core/tools:/trinity/shared/lmod/lmod/modulefiles/Core/ansys:/trinity/shared/lmod/lmod/modulefiles/Core/analysis:/trinity/shared/lmod/lmod/modulefiles/Core/simulation:/trinity/shared/lmod/lmod/modulefiles/Core/devel:/trinity/shared/lmod/lmod/modulefiles/Core/compiler
source /trinity/shared/lmod/lmod/init/bash

# Load necessary modules inside the Slurm job
module load gcc/12.2.0
module load boost/1.82.0

# To prevent libstdc++ not found error
export LD_LIBRARY_PATH=/trinity/shared/pkg/compiler/gcc/12.2.0/lib64:/trinity/shared/pkg/compiler/gcc/12.2.0/lib:\$LD_LIBRARY_PATH

# Path to the babelstream executable
babelstream_executable="\$(pwd)/benchmarks/babelstream/babelstream"

# Define the results file for the benchmark output
benchmark_results_file="$results_file"

# Run the benchmark and redirect output to the benchmark results file
echo "Running BabelStream benchmark on \$(hostname)..."
"\$babelstream_executable" --array-size=33554432 --number-runs=10 > "\$benchmark_results_file" 2>&1
echo "Benchmark completed."
EOF

    echo "BabelStream benchmark submitted to Slurm. Check $results_file for results."
}

# Main script execution
print_directory_info

# Ensure the module system is initialized
initialize_modules

# Clone or update Alpaka repository
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

# Build the project
build_cpu_omp2b

# Submit the benchmark to Slurm
submit_cpu_omp2b_benchmark

echo "Script execution completed."
