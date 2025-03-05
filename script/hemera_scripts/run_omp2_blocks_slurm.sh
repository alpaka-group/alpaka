#!/bin/bash

# Define global variables
SCRIPT_DIR=$(dirname "$(realpath "$0")")
results_dir="$SCRIPT_DIR/test-results"
mkdir -p "$results_dir"

# Global array of required modules
declare -a required_modules=("gcc/12.2.0" "python/3.10.4" "boost/1.82.0" "cmake")

# Function to print directory information
print_directory_info() {
    echo "Current Working Directory: $(pwd)"
    echo "Script Directory: $SCRIPT_DIR"
}

# Function to initialize the module system (based on your interactive environment)
initialize_modules() {
    echo "Initializing module system..."
    # Set MODULEPATH as seen in your interactive environment
    export MODULEPATH=/trinity/shared/lmod/modulefiles/Linux:/trinity/shared/lmod/modulefiles/Core:/trinity/shared/lmod/lmod/modulefiles/Core/tools:/trinity/shared/lmod/lmod/modulefiles/Core/ansys:/trinity/shared/lmod/lmod/modulefiles/Core/analysis:/trinity/shared/lmod/lmod/modulefiles/Core/simulation:/trinity/shared/lmod/lmod/modulefiles/Core/devel:/trinity/shared/lmod/lmod/modulefiles/Core/compiler
    # Source the Lmod initialization script
    if [ -f /trinity/shared/lmod/lmod/init/bash ]; then
        source /trinity/shared/lmod/lmod/init/bash
        echo "Lmod initialization script sourced successfully."
    elif [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
        echo "Modules.sh sourced successfully."
    else
        echo "Error: Module system initialization script not found."
        exit 1
    fi
    echo "Module system initialized successfully."
}

# Function to clone or update the Alpaka repository
clone_or_update_alpaka() {
    local alpaka_dir="$SCRIPT_DIR/alpaka"
    if [ -d "$alpaka_dir" ]; then
        echo "Updating Alpaka repository..."
        cd "$alpaka_dir" || { echo "Failed to enter alpaka directory."; exit 1; }
        # Explicitly set PATH for git (based on your interactive environment)
        export PATH=/trinity/shared/pkg/devel/git/2.37.1/bin:$PATH
        echo "Git path set to: $PATH"
        git checkout develop
        if ! git pull origin develop; then
            echo "Error: Failed to pull latest changes from the Alpaka repository."
            exit 1
        fi
        echo "Alpaka repository updated successfully."
    else
        echo "Cloning Alpaka repository..."
        # Explicitly set PATH for git (based on your interactive environment)
        export PATH=/trinity/shared/pkg/devel/git/2.37.1/bin:$PATH
        echo "Git path set to: $PATH"
        if ! git clone https://github.com/alpaka-group/alpaka.git --branch develop "$alpaka_dir"; then
            echo "Error: Failed to clone the Alpaka repository."
            exit 1
        fi
        cd "$alpaka_dir" || { echo "Failed to enter alpaka directory."; exit 1; }
        echo "Alpaka repository cloned successfully."
    fi
}

# Function to load required modules and log their names
load_and_log_modules() {
    echo "Loading required modules..."
    for module_name in "${required_modules[@]}"; do
        echo "Loading module: $module_name"
        module load "$module_name" || { echo "Failed to load $module_name"; exit 1; }
    done
    echo "Modules loaded successfully."
}

# Function to submit the combined Slurm job for cpu-omp2b
submit_combined_job_cpu_omp2b() {
    local datetime_now=$(date +"%Y-%m-%d_%H-%M")
    local commit_hash=$(git rev-parse --short=8 HEAD 2>/dev/null || echo "unknown")
    local results_file="$results_dir/babelstream-cpu-omp2b-$datetime_now-$commit_hash.txt"
    echo "Submitting combined Slurm job (compilation + benchmarking) for cpu-omp2b..."
    echo "Benchmark results file: $results_file"
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=combined-babelstream-cpu-omp2b
#SBATCH --output=$results_dir/babelstr-slurminfo-cpu-omp2b-%j-$datetime_now-$commit_hash.out  # Store Slurm metadata here
#SBATCH --error=$results_dir/babelstr-slurminfo-cpu-omp2b-%j-$datetime_now-$commit_hash.err   # Store Slurm errors here
#SBATCH --time=01:00:00
#SBATCH --partition=milan                              # Use CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# Reinitialize the module system inside the Slurm job
export MODULEPATH=/trinity/shared/lmod/modulefiles/Linux:/trinity/shared/lmod/modulefiles/Core:/trinity/shared/lmod/lmod/modulefiles/Core/tools:/trinity/shared/lmod/lmod/modulefiles/Core/ansys:/trinity/shared/lmod/lmod/modulefiles/Core/analysis:/trinity/shared/lmod/lmod/modulefiles/Core/simulation:/trinity/shared/lmod/lmod/modulefiles/Core/devel:/trinity/shared/lmod/lmod/modulefiles/Core/compiler
source /trinity/shared/lmod/lmod/init/bash
# Load necessary modules inside the Slurm job
for module_name in ${required_modules[@]}; do
    echo "Loading module: \$module_name"
    module load "\$module_name" || { echo "Failed to load \$module_name"; exit 1; }
done
echo "Modules loaded successfully."
# To prevent libstdc++ not found error
export LD_LIBRARY_PATH=/trinity/shared/pkg/compiler/gcc/12.2.0/lib64:/trinity/shared/pkg/compiler/gcc/12.2.0/lib:\$LD_LIBRARY_PATH
# Clone or update Alpaka repository
cd "$SCRIPT_DIR/alpaka" || { echo "Failed to enter alpaka directory."; exit 1; }
# Log system information
{
    echo "System Information:"
    echo "-------------------"
    uname -a
    echo ""
    echo "NUMA Configuration:"
    echo "-------------------"
    numactl --hardware
    echo ""
    echo "Hardware Topology:"
    echo "------------------"
    hwloc-ls
    echo ""
    echo "Loaded Modules:"
    echo "---------------"
    for module_name in ${required_modules[@]}; do
        echo "\$module_name"
    done
    echo ""
    echo "Partition and Node:"
    echo "-------------------"
    echo "Partition: milan"
    echo "Node: \$(hostname)"
    echo ""
    echo "Alpaka Repository Hash:"
    echo "-----------------------"
    git rev-parse --short=8 HEAD
    echo ""
} > "$results_file"
# Configure and build the project
echo "Configuring and building the project for cpu-omp2b..."
preset_dir="cpu-omp2b"
num_cores=\$(( \$(nproc) - 2 ))
num_cores=\$(( num_cores < 1 ? 1 : num_cores ))  # Ensure at least 1 core is used
extra_flags="-Dalpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE=ON -DCMAKE_CXX_FLAGS=-march=native"
if [ ! -f "CMakePresets.json" ]; then
    echo "Error: CMakePresets.json not found. Exiting."
    exit 1
fi
cmake --preset "\$preset_dir" \
      -Dalpaka_BUILD_BENCHMARKS=ON \
      -DCMAKE_BUILD_TYPE=RELEASE \
      \$extra_flags || { echo "CMake configuration failed."; exit 1; }
build_dir="$SCRIPT_DIR/alpaka/build/\$preset_dir"
mkdir -p "\$build_dir"
cd "\$build_dir" || { echo "Error: Failed to enter build directory."; exit 1; }
cmake --build . --target babelstream -j "\$num_cores" || { echo "Build failed."; exit 1; }
echo "Compilation completed."
# Path to the babelstream executable
babelstream_executable="\$build_dir/benchmarks/babelstream/babelstream"
if [ ! -f "\$babelstream_executable" ]; then
    echo "Error: BabelStream executable not found at \$babelstream_executable. Exiting."
    exit 1
fi
# Add header information to the beginning of the results file
temp_file="\$(mktemp)"
{
    echo "ExecutableName: Babelstream"
    echo "Time And Date: $datetime_now"
    echo "Preset: \$preset_dir"
    cat "$results_file"
} > "\$temp_file"
mv "\$temp_file" "$results_file"
# Run the benchmark and append output to the benchmark results file
echo "Running BabelStream benchmark on \$(hostname)..."
echo "Benchmark Results:" >> "$results_file"
echo "------------------" >> "$results_file"
"\$babelstream_executable" --array-size=33554432 --number-runs=10 >> "$results_file" 2>&1
echo "Benchmark completed."
EOF
    echo "Combined Slurm job submitted for cpu-omp2b. Check $results_file for results."
}

# Main script execution
echo "Starting script execution..."
print_directory_info
# Ensure the module system is initialized
initialize_modules
# Load required modules and log their names
load_and_log_modules
# Clone or update Alpaka repository
clone_or_update_alpaka
echo "Current Working Directory: $(pwd)"
if [ "$(basename "$(pwd)")" != "alpaka" ]; then
    echo "Error: Script must be executed from the alpaka root directory."
    exit 1
fi
# Submit the combined Slurm job for cpu-omp2b
echo "Submitting combined Slurm job (compilation + benchmarking) for cpu-omp2b..."
submit_combined_job_cpu_omp2b
echo "Script execution completed."
