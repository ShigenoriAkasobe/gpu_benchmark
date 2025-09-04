#!/bin/bash

# CUDA C++ Benchmark Library Setup Script
# Automated setup for Python + CUDA C++ integration

set -e  # Exit on any error

echo "CUDA C++ Benchmark Library Setup"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running from correct directory
if [[ ! -f "app.py" || ! -d "cpp" ]]; then
    print_error "Please run this script from the gpu_benchmark directory"
    exit 1
fi

# Function to check command existence
check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        print_success "$1 is available"
        return 0
    else
        print_error "$1 is not available"
        return 1
    fi
}

# Function to check CUDA installation
check_cuda() {
    print_status "Checking CUDA installation..."

    if check_command "nvcc"; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        print_success "CUDA $CUDA_VERSION detected"
    else
        print_warning "CUDA toolkit not found. GPU benchmarks will not work."
        return 1
    fi

    if check_command "nvidia-smi"; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        print_success "GPU detected: $GPU_INFO"
    else
        print_warning "nvidia-smi not found. Cannot detect GPU."
        return 1
    fi

    return 0
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking system dependencies..."

    local missing_deps=()

    # Check build tools
    if ! check_command "make"; then
        missing_deps+=("build-essential")
    fi

    if ! check_command "gcc"; then
        missing_deps+=("gcc")
    fi

    # Check OpenBLAS
    if ! ldconfig -p | grep -q "blas"; then
        missing_deps+=("libopenblas-dev")
    else
        print_success "BLAS library found"
    fi

    # Check OpenMP
    if ! echo '#include <omp.h>' | gcc -E -fopenmp - >/dev/null 2>&1; then
        missing_deps+=("libomp-dev")
    else
        print_success "OpenMP support found"
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_warning "Missing dependencies: ${missing_deps[*]}"
        print_status "Install with: sudo apt install ${missing_deps[*]}"
        read -p "Would you like to install these dependencies? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo apt update
            sudo apt install "${missing_deps[@]}"
            print_success "Dependencies installed"
        else
            print_warning "Continuing without installing dependencies. Build may fail."
        fi
    else
        print_success "All system dependencies are available"
    fi
}

# Function to setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."

    # Check if we're in a virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        print_success "Using virtual environment: $VIRTUAL_ENV"
    else
        print_warning "Not in a virtual environment. Consider using one."
    fi

    # Install Python dependencies
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt || {
        print_warning "requirements.txt not found, installing basic dependencies..."
        pip install numpy flask waitress GPUtil psutil
    }

    print_success "Python dependencies installed"
}

# Function to build CUDA library
build_cuda_library() {
    print_status "Building CUDA C++ library..."

    cd cpp

    # Show GPU detection info
    if make -f Makefile.lib show-gpu-info; then
        print_success "GPU auto-detection successful"
    else
        print_warning "GPU auto-detection failed, using default settings"
    fi

    # Build the library
    if make -f Makefile.lib clean all; then
        print_success "CUDA library built successfully"
    else
        print_error "Failed to build CUDA library"
        cd ..
        return 1
    fi

    # Test the library
    print_status "Testing the built library..."
    if make -f Makefile.lib test-lib; then
        print_success "Library test passed"
    else
        print_warning "Library test failed, but library may still work"
    fi

    cd ..
    return 0
}

# Function to run integration tests
run_tests() {
    print_status "Running integration tests..."

    if python test_cuda_lib.py; then
        print_success "All integration tests passed"
        return 0
    else
        print_warning "Some integration tests failed"
        return 1
    fi
}

# Function to check web app
test_web_app() {
    print_status "Testing web application startup..."

    # Start the web app in background
    python app.py &
    WEB_PID=$!

    # Wait a bit for startup
    sleep 3

    # Check if it's responding
    if curl -s http://localhost:5000 >/dev/null; then
        print_success "Web application is responding"
        kill $WEB_PID 2>/dev/null || true
        return 0
    else
        print_warning "Web application is not responding"
        kill $WEB_PID 2>/dev/null || true
        return 1
    fi
}

# Function to show usage instructions
show_usage() {
    print_status "Setup completed!"
    echo
    echo "Usage Instructions:"
    echo "=================="
    echo
    echo "1. Start the web application:"
    echo "   python app.py"
    echo
    echo "2. Access the web interface:"
    echo "   http://localhost:5000"
    echo
    echo "3. Use the Python library directly:"
    echo "   from cuda_benchmark import CudaBenchmarkLibrary"
    echo "   lib = CudaBenchmarkLibrary()"
    echo "   result = lib.run_all_benchmarks(1024, 1)"
    echo
    echo "4. Available benchmark types:"
    echo "   - cpu_single: CPU single-core baseline"
    echo "   - cpu_optimized: CPU optimized (OpenMP + AVX)"
    echo "   - cpu_openblas: CPU OpenBLAS"
    echo "   - cuda_naive: CUDA naive implementation"
    echo "   - cublas: cuBLAS optimized"
    echo "   - cublas_tensorcore: cuBLAS + Tensor Core"
    echo "   - wmma: WMMA Tensor Core manual implementation"
    echo "   - all: All benchmarks comparison"
    echo
    echo "5. Run tests anytime:"
    echo "   python test_cuda_lib.py"
    echo
    echo "6. Rebuild library if needed:"
    echo "   cd cpp && make -f Makefile.lib clean all"
    echo
}

# Main setup process
main() {
    print_status "Starting automated setup..."

    # Step 1: Check CUDA
    CUDA_AVAILABLE=false
    if check_cuda; then
        CUDA_AVAILABLE=true
    fi

    # Step 2: Check dependencies
    check_dependencies

    # Step 3: Setup Python environment
    setup_python_env

    # Step 4: Build CUDA library (if CUDA is available)
    if [[ "$CUDA_AVAILABLE" == "true" ]]; then
        if build_cuda_library; then
            print_success "CUDA library setup completed"
        else
            print_error "CUDA library setup failed"
            exit 1
        fi
    else
        print_warning "Skipping CUDA library build (CUDA not available)"
        print_status "Only traditional Python benchmarks will be available"
    fi

    # Step 5: Run integration tests
    if [[ "$CUDA_AVAILABLE" == "true" ]]; then
        run_tests
    fi

    # Step 6: Test web application
    test_web_app

    # Step 7: Show usage instructions
    show_usage

    print_success "Setup completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    "--cuda-only")
        print_status "Building CUDA library only..."
        if check_cuda; then
            build_cuda_library
        else
            print_error "CUDA not available"
            exit 1
        fi
        ;;
    "--test-only")
        print_status "Running tests only..."
        run_tests
        ;;
    "--deps-only")
        print_status "Installing dependencies only..."
        check_dependencies
        setup_python_env
        ;;
    "--help"|"-h")
        echo "CUDA C++ Benchmark Setup Script"
        echo "Usage: $0 [option]"
        echo
        echo "Options:"
        echo "  --cuda-only   Build CUDA library only"
        echo "  --test-only   Run tests only"
        echo "  --deps-only   Install dependencies only"
        echo "  --help        Show this help"
        echo
        echo "No option: Run full setup"
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
