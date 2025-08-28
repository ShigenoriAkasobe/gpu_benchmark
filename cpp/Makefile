# Makefile for CUDA WMMA benchmark with automatic GPU architecture detection
# 7-way performance comparison: CPU (single-core, manual optimized, OpenBLAS) vs GPU (naive CUDA, cuBLAS, cuBLAS+TensorCore, WMMA)

# Configuration
NVCC = nvcc
TARGET = matmul_wmma_benchmark.out
SOURCE = matmul_wmma_benchmark.cu
LIBS = -lcublas -lopenblas
# CPU optimization flags
CPU_FLAGS = -fopenmp -mavx2 -mfma -O3

# Auto-detect GPU architecture
GPU_NAME := $(shell nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
GPU_COMPUTE := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d '.')

# Map compute capability to architecture
ifeq ($(GPU_COMPUTE),75)
    ARCH = sm_75
    GPU_GENERATION = Turing
else ifeq ($(GPU_COMPUTE),80)
    ARCH = sm_80
    GPU_GENERATION = Ampere
else ifeq ($(GPU_COMPUTE),86)
    ARCH = sm_86
    GPU_GENERATION = Ampere
else ifeq ($(GPU_COMPUTE),89)
    ARCH = sm_89
    GPU_GENERATION = Ada Lovelace
else ifeq ($(GPU_COMPUTE),90)
    ARCH = sm_90
    GPU_GENERATION = Hopper
else
    # Default fallback
    ARCH = sm_75
    GPU_GENERATION = Unknown (defaulting to Turing)
endif

CFLAGS = -O3 -arch=$(ARCH)

# Default target
.PHONY: all
all: show-gpu-info $(TARGET)

# Show detected GPU information
.PHONY: show-gpu-info
show-gpu-info:
	@echo "Detected GPU: $(GPU_NAME)"
	@echo "Compute Capability: $(GPU_COMPUTE)"
	@echo "Architecture: $(ARCH)"
	@echo "Generation: $(GPU_GENERATION)"
	@echo ""

# Compile the CUDA program
$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) $(SOURCE) -o $(TARGET) $(LIBS) -Xcompiler "$(CPU_FLAGS)"

# Run the benchmark
.PHONY: run
run: $(TARGET)
	./$(TARGET)

# Quick benchmark tests with different sizes
.PHONY: test-small
test-small: $(TARGET)
	@echo "=== Quick test with small matrix (512x512) ==="
	./$(TARGET) 512

.PHONY: test-medium
test-medium: $(TARGET)
	@echo "=== Medium test (1024x1024) ==="
	./$(TARGET) 1024

.PHONY: test-large
test-large: $(TARGET)
	@echo "=== Large test (4096x4096, CPU will be slow) ==="
	@echo "Note: CPU benchmarks will take several minutes"
	./$(TARGET) 4096

.PHONY: test-gpu-only
test-gpu-only: $(TARGET)
	@echo "=== GPU-only test (2048x2048, skipping CPU benchmarks) ==="
	@echo "Note: CPU benchmarks skipped for performance (would take >20 minutes)"
	./$(TARGET) 2048 --gpu-only

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(TARGET)

# Display detailed GPU information
.PHONY: info
info:
	nvidia-smi

# Check CUDA installation and GPU compute capability
.PHONY: check-cuda
check-cuda:
	@echo "=== CUDA Version ==="
	nvcc --version
	@echo ""
	@echo "=== GPU Information ==="
	nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
	@echo ""
	@echo "=== Detected Configuration ==="
	@echo "GPU: $(GPU_NAME)"
	@echo "Compute Capability: $(GPU_COMPUTE)"
	@echo "Selected Architecture: $(ARCH)"

# Debug build
.PHONY: debug
debug: CFLAGS = -g -G -arch=$(ARCH)
debug: show-gpu-info
	$(NVCC) $(CFLAGS) $(SOURCE) -o $(TARGET) $(LIBS) -Xcompiler "$(CPU_FLAGS)"

# Profile with nvprof (if available)
.PHONY: profile
profile: $(TARGET)
	nvprof ./$(TARGET)

# Force build for specific architecture (override auto-detection)
.PHONY: force-turing
force-turing:
	$(NVCC) -O3 -arch=sm_75 $(SOURCE) -o $(TARGET) $(LIBS) -Xcompiler "$(CPU_FLAGS)"

.PHONY: force-ampere
force-ampere:
	$(NVCC) -O3 -arch=sm_80 $(SOURCE) -o $(TARGET) $(LIBS) -Xcompiler "$(CPU_FLAGS)"

.PHONY: force-ada
force-ada:
	$(NVCC) -O3 -arch=sm_89 $(SOURCE) -o $(TARGET) $(LIBS) -Xcompiler "$(CPU_FLAGS)"

# Help target
.PHONY: help
help:
	@echo "CUDA WMMA Benchmark Makefile with Auto-Detection & 7-Way Comparison"
	@echo "====================================================================="
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build the benchmark (default, with GPU auto-detection)"
	@echo "  run          - Run the benchmark (default 1024x1024)"
	@echo "  test-small   - Quick test with 512x512 matrix (includes CPU)"
	@echo "  test-medium  - Medium test with 1024x1024 matrix (includes CPU)"
	@echo "  test-large   - Large test with 4096x4096 matrix (CPU will be slow)"
	@echo "  test-gpu-only - GPU-only test with 2048x2048 (CPU too slow)"
	@echo "  clean        - Remove build artifacts"
	@echo "  info         - Display detailed GPU information"
	@echo "  check-cuda   - Check CUDA installation and detected GPU"
	@echo "  debug        - Build with debug information"
	@echo "  profile      - Run with nvprof profiler"
	@echo "  show-gpu-info - Show detected GPU information"
	@echo ""
	@echo "Comparison includes:"
	@echo "  1. CPU single-core (baseline)"
	@echo "  2. CPU manual optimized (OpenMP + AVX + cache blocking)"
	@echo "  3. CPU OpenBLAS (industry-standard optimized BLAS)"
	@echo "  4. CUDA-Core naive"
	@echo "  5. cuBLAS optimized"
	@echo "  6. cuBLAS + Tensor Core"
	@echo "  7. WMMA manual implementation"
	@echo ""
	@echo "Force specific architecture (override auto-detection):"
	@echo "  force-turing - Force build for Turing (sm_75)"
	@echo "  force-ampere - Force build for Ampere (sm_80)"
	@echo "  force-ada    - Force build for Ada Lovelace (sm_89)"
	@echo ""
	@echo "Manual override:"
	@echo "  make ARCH=sm_XX  # Manually specify architecture"
	@echo ""
	@echo "Examples:"
	@echo "  make test-small  # Quick 7-way comparison with small matrices"
	@echo "  make test-medium # Medium size with optimized CPU vs GPU comparison"
	@echo "  make test-large  # Large size (CPU will take several minutes)"
	@echo "  make run         # Standard benchmark (1024x1024, reasonable performance)"
	@echo "  make clean all   # Clean and rebuild with auto-detection"
	@echo "  make check-cuda  # Check CUDA and show detected configuration"
