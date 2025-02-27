#!/bin/bash

# Define global variables
SCRIPT_DIR=$(dirname "$(realpath "$0")")
results_dir="$SCRIPT_DIR/test-results"
mkdir -p "$results_dir"

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

# Function to submit the combined Slurm job for cpu-omp2t
submit_combined_job_cpu_omp2t() {
    local datetime_now=$(date +"%Y-%m-%d_%H-%M")
    local commit_hash=$(git rev-parse --short=8 HEAD 2>/dev/null || echo "unknown")
    local results_file="$results_dir/babelstream-cpu-omp2t-$datetime_now-$commit_hash.txt"
    echo "Submitting combined Slurm job (compilation + benchmarking) for cpu-omp2t..."
    echo "Benchmark results file: $results_file"
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=combined-babelstream-cpu-omp2t
#SBATCH --output=$results_dir/combined-slurm-%j.out  # Store Slurm metadata here
#SBATCH --error=$results_dir/combined-slurm-%j.err   # Store Slurm errors here
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
module load python/3.10.4 || { echo "Failed to load python/3.10.4"; exit 1; }
module load gcc/12.2.0 || { echo "Failed to load gcc/12.2.0"; exit 1; }
module load boost/1.82.0 || { echo "Failed to load boost/1.82.0"; exit 1; }
module load cmake || { echo "Failed to load cmake"; exit 1; }
echo "Modules loaded successfully."
# To prevent libstdc++ not found error
export LD_LIBRARY_PATH=/trinity/shared/pkg/compiler/gcc/12.2.0/lib64:/trinity/shared/pkg/compiler/gcc/12.2.0/lib:\$LD_LIBRARY_PATH
# Clone or update Alpaka repository
cd "$SCRIPT_DIR/alpaka" || { echo "Failed to enter alpaka directory."; exit 1; }
# Configure and build the project
echo "Configuring and building the project for cpu-omp2t..."
preset="cpu-omp2t"
num_cores=\$(( \$(nproc) - 2 ))
num_cores=\$(( num_cores < 1 ? 1 : num_cores ))  # Ensure at least 1 core is used
extra_flags="-Dalpaka_ACC_CPU_B_SEQ_T_OMP2_ENABLE=ON -DCMAKE_CXX_FLAGS=-march=native"
if [ ! -f "CMakePresets.json" ]; then
    echo "Error: CMakePresets.json not found. Exiting."
    exit 1
fi
cmake --preset "\$preset" \
      -Dalpaka_BUILD_BENCHMARKS=ON \
      -DCMAKE_BUILD_TYPE=RELEASE \
      \$extra_flags || { echo "CMake configuration failed."; exit 1; }
build_dir="$SCRIPT_DIR/alpaka/build/\$preset"
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
# Run the benchmark and redirect output to the benchmark results file
echo "Running BabelStream benchmark on \$(hostname)..."
"\$babelstream_executable" --array-size=33554432 --number-runs=10 > "$results_file" 2>&1
echo "Benchmark completed."
EOF
    echo "Combined Slurm job submitted for cpu-omp2t. Check $results_file for results."
}

# Main script execution
echo "Starting script execution..."
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
# Submit the combined Slurm job for cpu-omp2t
echo "Submitting combined Slurm job (compilation + benchmarking) for cpu-omp2t..."
submit_combined_job_cpu_omp2t
echo "Script execution completed."
