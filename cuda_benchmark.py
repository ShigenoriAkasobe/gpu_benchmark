"""
Python wrapper for CUDA benchmark library using ctypes
Provides easy-to-use Python interface for C++/CUDA benchmark functions
"""

import ctypes
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Union


@dataclass
class BenchmarkResult:
    """Python representation of benchmark results"""

    cpu_single_core_ms: float = 0.0
    cpu_optimized_ms: float = 0.0
    cpu_openblas_ms: float = 0.0
    cuda_naive_ms: float = 0.0
    cublas_ms: float = 0.0
    cublas_tensorcore_ms: float = 0.0
    wmma_ms: float = 0.0
    matrix_size: int = 0
    iterations: int = 0
    success: bool = False
    error_message: str = ""

    def to_dict(self) -> Dict[str, Union[float, int, bool, str]]:
        """Convert to dictionary for JSON serialization"""
        return {
            "cpu_single_core_ms": self.cpu_single_core_ms,
            "cpu_optimized_ms": self.cpu_optimized_ms,
            "cpu_openblas_ms": self.cpu_openblas_ms,
            "cuda_naive_ms": self.cuda_naive_ms,
            "cublas_ms": self.cublas_ms,
            "cublas_tensorcore_ms": self.cublas_tensorcore_ms,
            "wmma_ms": self.wmma_ms,
            "matrix_size": self.matrix_size,
            "iterations": self.iterations,
            "success": self.success,
            "error_message": self.error_message,
        }

    def get_speedup_analysis(self) -> Dict[str, float]:
        """Calculate speedup ratios between different methods"""
        analysis = {}

        if self.cpu_single_core_ms > 0:
            baseline = self.cpu_single_core_ms

            if self.cpu_optimized_ms > 0:
                analysis["cpu_optimized_vs_single"] = baseline / self.cpu_optimized_ms
            if self.cpu_openblas_ms > 0:
                analysis["cpu_openblas_vs_single"] = baseline / self.cpu_openblas_ms
            if self.cuda_naive_ms > 0:
                analysis["cuda_naive_vs_cpu_single"] = baseline / self.cuda_naive_ms
            if self.cublas_ms > 0:
                analysis["cublas_vs_cpu_single"] = baseline / self.cublas_ms
            if self.cublas_tensorcore_ms > 0:
                analysis["cublas_tc_vs_cpu_single"] = baseline / self.cublas_tensorcore_ms
            if self.wmma_ms > 0:
                analysis["wmma_vs_cpu_single"] = baseline / self.wmma_ms

        # GPU-specific comparisons
        if self.cuda_naive_ms > 0:
            if self.cublas_ms > 0:
                analysis["cublas_vs_cuda_naive"] = self.cuda_naive_ms / self.cublas_ms
            if self.cublas_tensorcore_ms > 0:
                analysis["cublas_tc_vs_cuda_naive"] = self.cuda_naive_ms / self.cublas_tensorcore_ms
            if self.wmma_ms > 0:
                analysis["wmma_vs_cuda_naive"] = self.cuda_naive_ms / self.wmma_ms

        if self.cublas_ms > 0 and self.cublas_tensorcore_ms > 0:
            analysis["cublas_tc_vs_cublas"] = self.cublas_ms / self.cublas_tensorcore_ms

        return analysis


class CudaBenchmarkResultStruct(ctypes.Structure):
    """C structure representation for ctypes"""

    _fields_ = [
        ("cpu_single_core_ms", ctypes.c_double),
        ("cpu_optimized_ms", ctypes.c_double),
        ("cpu_openblas_ms", ctypes.c_double),
        ("cuda_naive_ms", ctypes.c_double),
        ("cublas_ms", ctypes.c_double),
        ("cublas_tensorcore_ms", ctypes.c_double),
        ("wmma_ms", ctypes.c_double),
        ("matrix_size", ctypes.c_int),
        ("iterations", ctypes.c_int),
        ("success", ctypes.c_int),
        ("error_message", ctypes.c_char * 256),
    ]


class CudaBenchmarkLibrary:
    """Python wrapper for CUDA benchmark library"""

    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize the library wrapper

        Args:
            lib_path: Path to the shared library. If None, tries to find it automatically.
        """
        self.lib = None
        self._load_library(lib_path)
        self._setup_function_signatures()

    def _find_library(self) -> str:
        """Find the library in common locations"""
        search_paths = [
            # Local paths
            os.path.join(os.path.dirname(__file__), "lib", "libbenchmark.so"),
            os.path.join(os.path.dirname(__file__), "cpp", "libbenchmark.so"),
            os.path.join(os.path.dirname(__file__), "libbenchmark.so"),
            # System paths
            "/usr/local/lib/libbenchmark.so",
            "/usr/lib/libbenchmark.so",
        ]

        for path in search_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Could not find libbenchmark.so in any of these locations:\n"
            + "\n".join(search_paths)
            + "\nPlease build the library first with 'make -f Makefile.lib' in the cpp directory."
        )

    def _load_library(self, lib_path: Optional[str]):
        """Load the shared library"""
        if lib_path is None:
            lib_path = self._find_library()

        try:
            self.lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise RuntimeError(f"Failed to load library from {lib_path}: {e}")

    def _setup_function_signatures(self):
        """Setup function signatures for type safety"""
        # All benchmark functions return pointer to BenchmarkResult
        result_ptr_type = ctypes.POINTER(CudaBenchmarkResultStruct)

        # Function signatures
        functions = [
            "run_cpu_single_core_benchmark",
            "run_cpu_optimized_benchmark",
            "run_cpu_openblas_benchmark",
            "run_cuda_naive_benchmark",
            "run_cublas_benchmark",
            "run_cublas_tensorcore_benchmark",
            "run_wmma_benchmark",
        ]

        for func_name in functions:
            func = getattr(self.lib, func_name)
            func.argtypes = [ctypes.c_int, ctypes.c_int]  # matrix_size, iterations
            func.restype = result_ptr_type

        # run_all_benchmarks has extra parameter
        self.lib.run_all_benchmarks.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]  # matrix_size, iterations, gpu_only
        self.lib.run_all_benchmarks.restype = result_ptr_type

        # Utility functions
        self.lib.free_benchmark_result.argtypes = [result_ptr_type]
        self.lib.free_benchmark_result.restype = None

        self.lib.check_gpu_availability.argtypes = []
        self.lib.check_gpu_availability.restype = ctypes.c_int

    def _call_benchmark_function(
        self, func_name: str, matrix_size: int, iterations: int, **kwargs
    ) -> BenchmarkResult:
        """Generic function to call benchmark functions and handle results"""
        try:
            func = getattr(self.lib, func_name)

            # Call function with appropriate arguments
            if func_name == "run_all_benchmarks":
                gpu_only = kwargs.get("gpu_only", 0)
                result_ptr = func(matrix_size, iterations, gpu_only)
            else:
                result_ptr = func(matrix_size, iterations)

            if not result_ptr:
                return BenchmarkResult(
                    matrix_size=matrix_size,
                    iterations=iterations,
                    success=False,
                    error_message="Function returned null pointer",
                )

            # Convert C struct to Python dataclass
            c_result = result_ptr.contents
            result = BenchmarkResult(
                cpu_single_core_ms=c_result.cpu_single_core_ms,
                cpu_optimized_ms=c_result.cpu_optimized_ms,
                cpu_openblas_ms=c_result.cpu_openblas_ms,
                cuda_naive_ms=c_result.cuda_naive_ms,
                cublas_ms=c_result.cublas_ms,
                cublas_tensorcore_ms=c_result.cublas_tensorcore_ms,
                wmma_ms=c_result.wmma_ms,
                matrix_size=c_result.matrix_size,
                iterations=c_result.iterations,
                success=bool(c_result.success),
                error_message=(
                    c_result.error_message.decode("utf-8") if c_result.error_message else ""
                ),
            )

            # Free C memory
            self.lib.free_benchmark_result(result_ptr)

            return result

        except Exception as e:
            return BenchmarkResult(
                matrix_size=matrix_size,
                iterations=iterations,
                success=False,
                error_message=f"Exception in {func_name}: {str(e)}",
            )

    def check_gpu_availability(self) -> bool:
        """Check if GPU is available"""
        try:
            return bool(self.lib.check_gpu_availability())
        except Exception:
            return False

    def run_cpu_single_core_benchmark(
        self, matrix_size: int, iterations: int = 1
    ) -> BenchmarkResult:
        """Run CPU single-core benchmark"""
        return self._call_benchmark_function(
            "run_cpu_single_core_benchmark", matrix_size, iterations
        )

    def run_cpu_optimized_benchmark(self, matrix_size: int, iterations: int = 1) -> BenchmarkResult:
        """Run CPU optimized benchmark (OpenMP + AVX + cache blocking)"""
        return self._call_benchmark_function("run_cpu_optimized_benchmark", matrix_size, iterations)

    def run_cpu_openblas_benchmark(self, matrix_size: int, iterations: int = 1) -> BenchmarkResult:
        """Run CPU OpenBLAS benchmark"""
        return self._call_benchmark_function("run_cpu_openblas_benchmark", matrix_size, iterations)

    def run_cuda_naive_benchmark(self, matrix_size: int, iterations: int = 1) -> BenchmarkResult:
        """Run CUDA naive benchmark"""
        return self._call_benchmark_function("run_cuda_naive_benchmark", matrix_size, iterations)

    def run_cublas_benchmark(self, matrix_size: int, iterations: int = 1) -> BenchmarkResult:
        """Run cuBLAS benchmark"""
        return self._call_benchmark_function("run_cublas_benchmark", matrix_size, iterations)

    def run_cublas_tensorcore_benchmark(
        self, matrix_size: int, iterations: int = 1
    ) -> BenchmarkResult:
        """Run cuBLAS + Tensor Core benchmark"""
        return self._call_benchmark_function(
            "run_cublas_tensorcore_benchmark", matrix_size, iterations
        )

    def run_wmma_benchmark(self, matrix_size: int, iterations: int = 1) -> BenchmarkResult:
        """Run WMMA Tensor Core benchmark"""
        return self._call_benchmark_function("run_wmma_benchmark", matrix_size, iterations)

    def run_all_benchmarks(
        self, matrix_size: int, iterations: int = 1, gpu_only: bool = False
    ) -> BenchmarkResult:
        """Run all benchmarks"""
        return self._call_benchmark_function(
            "run_all_benchmarks", matrix_size, iterations, gpu_only=int(gpu_only)
        )


# Global instance for easy access
_benchmark_lib = None


def get_benchmark_library() -> CudaBenchmarkLibrary:
    """Get global instance of benchmark library"""
    global _benchmark_lib
    if _benchmark_lib is None:
        _benchmark_lib = CudaBenchmarkLibrary()
    return _benchmark_lib


# Convenience functions for direct access
def check_gpu_availability() -> bool:
    """Check if GPU is available"""
    return get_benchmark_library().check_gpu_availability()


def run_benchmark(benchmark_type: str, matrix_size: int, iterations: int = 1) -> BenchmarkResult:
    """
    Run a specific benchmark type

    Args:
        benchmark_type: One of 'cpu_single', 'cpu_optimized', 'cpu_openblas',
                       'cuda_naive', 'cublas', 'cublas_tensorcore', 'wmma', 'all'
        matrix_size: Size of square matrices
        iterations: Number of iterations to average

    Returns:
        BenchmarkResult object
    """
    lib = get_benchmark_library()

    benchmark_map = {
        "cpu_single": lib.run_cpu_single_core_benchmark,
        "cpu_optimized": lib.run_cpu_optimized_benchmark,
        "cpu_openblas": lib.run_cpu_openblas_benchmark,
        "cuda_naive": lib.run_cuda_naive_benchmark,
        "cublas": lib.run_cublas_benchmark,
        "cublas_tensorcore": lib.run_cublas_tensorcore_benchmark,
        "wmma": lib.run_wmma_benchmark,
        "all": lib.run_all_benchmarks,
    }

    if benchmark_type not in benchmark_map:
        raise ValueError(
            f"Unknown benchmark type: {benchmark_type}. Available: {list(benchmark_map.keys())}"
        )

    return benchmark_map[benchmark_type](matrix_size, iterations)


if __name__ == "__main__":
    # Simple test
    print("Testing CUDA benchmark library...")

    try:
        lib = CudaBenchmarkLibrary()
        print(f"GPU available: {lib.check_gpu_availability()}")

        # Quick test with small matrix
        print("\nRunning quick test (512x512 matrix)...")
        result = lib.run_cpu_single_core_benchmark(512, 1)

        if result.success:
            print(f"CPU single-core: {result.cpu_single_core_ms:.2f} ms")
        else:
            print(f"Test failed: {result.error_message}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
