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

# Function to initialize the module system (based on env output)
initialize_modules() {
    echo "Initializing module system..."
    # Set MODULEPATH as seen in the env output
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
        # Explicitly set PATH for git (based on env output)
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
        # Explicitly set PATH for git (based on env output)
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

# Function to configure and build the project for gpu-cuda-nvcc
build_gpu_cuda_nvcc() {
    echo "Loading necessary modules for building..."
    module load python/3.10.4 || { echo "Failed to load python/3.10.4"; exit 1; }
    module load gcc/12.2.0 || { echo "Failed to load gcc/12.2.0"; exit 1; }
    module load boost/1.82.0 || { echo "Failed to load boost/1.82.0"; exit 1; }
    module load cuda/12.1 || { echo "Failed to load CUDA 12.1"; exit 1; }
    module load cmake || { echo "Failed to load cmake"; exit 1; }

    # Verify nvcc availability
    if ! which nvcc > /dev/null; then
        echo "Error: nvcc not found. Ensure CUDA is loaded properly."
        exit 1
    fi

    local preset="gpu-cuda-nvcc"
    local num_cores=$(( $(nproc) - 2 ))
    num_cores=$(( num_cores < 1 ? 1 : num_cores ))  # Ensure at least 1 core is used
    extra_flags="-Dalpaka_ACC_GPU_CUDA_ENABLE=ON \
                 -Dalpaka_ACC_GPU_CUDA_ONLY_MODE=ON \
                 -Dalpaka_ACC_CPU_B_SEQ_T_SEQ_ENABLE=OFF \
                 -DCMAKE_CUDA_COMPILER=$(which nvcc) \
                 -DCMAKE_CUDA_ARCHITECTURES=52"
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
    mkdir -p "$build_dir"
    cd "$build_dir" || exit 1
    echo "Building for preset: $preset"
    cmake --build . --target babelstream -j "$num_cores"
}

# Function to submit the benchmark for gpu-cuda-nvcc using Slurm
submit_gpu_cuda_nvcc_benchmark() {
    local datetime_now=$(date +"%Y-%m-%d_%H-%M")
    local commit_hash=$(git rev-parse --short=8 HEAD 2>/dev/null || echo "unknown")
    local results_file="$results_dir/babelstream-gpu-cuda-nvcc-$datetime_now-$commit_hash.txt"
    local preset="gpu-cuda-nvcc"
    echo "Submitting BabelStream benchmark to Slurm..."
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=babelstream-gpu-cuda-nvcc
#SBATCH --output=$results_dir/babelstream-slurm-%j.out  # Store Slurm metadata here
#SBATCH --error=$results_dir/babelstream-slurm-%j.err   # Store Slurm errors here
#SBATCH --time=00:30:00
#SBATCH --partition=casus_a100                              # Use GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1                                 # Request 1 GPU
# Reinitialize the module system inside the Slurm job
export MODULEPATH=/trinity/shared/lmod/modulefiles/Linux:/trinity/shared/lmod/modulefiles/Core:/trinity/shared/lmod/lmod/modulefiles/Core/tools:/trinity/shared/lmod/lmod/modulefiles/Core/ansys:/trinity/shared/lmod/lmod/modulefiles/Core/analysis:/trinity/shared/lmod/lmod/modulefiles/Core/simulation:/trinity/shared/lmod/lmod/modulefiles/Core/devel:/trinity/shared/lmod/lmod/modulefiles/Core/compiler
source /trinity/shared/lmod/lmod/init/bash
# Load necessary modules inside the Slurm job
module load python/3.10.4 || { echo "Failed to load python/3.10.4"; exit 1; }
module load gcc/12.2.0 || { echo "Failed to load gcc/12.2.0"; exit 1; }
module load boost/1.82.0 || { echo "Failed to load boost/1.82.0"; exit 1; }
module load cuda/12.1 || { echo "Failed to load CUDA 12.1"; exit 1; }
# To prevent libstdc++ not found error
export LD_LIBRARY_PATH=/trinity/shared/pkg/compiler/gcc/12.2.0/lib64:/trinity/shared/pkg/compiler/gcc/12.2.0/lib:\$LD_LIBRARY_PATH
# Path to the babelstream executable
preset="gpu-cuda-nvcc"
build_dir="$SCRIPT_DIR/alpaka/build/\$preset"
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
    echo "BabelStream benchmark submitted to Slurm. Check $results_file for results."
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

# Build the project (on the head node)
echo "Building the project for gpu-cuda-nvcc..."
build_gpu_cuda_nvcc

# Submit the benchmark to Slurm
echo "Submitting BabelStream benchmark to Slurm..."
submit_gpu_cuda_nvcc_benchmark
echo "Script execution completed."
