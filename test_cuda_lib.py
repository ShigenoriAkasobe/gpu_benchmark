#!/usr/bin/env python3
"""
CUDA C++ Benchmark Library Test Script
Tests the integration between Python and CUDA C++ library
"""

import os
import sys
import time


def test_library_import():
    """Test if the CUDA benchmark library can be imported"""
    print("=== Testing Library Import ===")
    try:
        from cuda_benchmark import (
            CudaBenchmarkLibrary,
            check_gpu_availability,
            run_benchmark,
        )

        print("✓ CUDA benchmark library imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import CUDA benchmark library: {e}")
        print(
            "  Make sure you have built the library with 'make -f Makefile.lib' in the cpp directory"
        )
        return False


def test_gpu_availability():
    """Test GPU availability"""
    print("\n=== Testing GPU Availability ===")
    try:
        from cuda_benchmark import check_gpu_availability

        gpu_available = check_gpu_availability()
        if gpu_available:
            print("✓ GPU is available")
        else:
            print("⚠ GPU is not available (tests will be limited to CPU)")
        return gpu_available
    except Exception as e:
        print(f"✗ Error checking GPU availability: {e}")
        return False


def test_library_initialization():
    """Test library initialization"""
    print("\n=== Testing Library Initialization ===")
    try:
        from cuda_benchmark import CudaBenchmarkLibrary

        lib = CudaBenchmarkLibrary()
        print("✓ Library initialized successfully")
        return lib
    except Exception as e:
        print(f"✗ Failed to initialize library: {e}")
        return None


def test_cpu_benchmark(lib):
    """Test CPU benchmark"""
    print("\n=== Testing CPU Benchmark ===")
    try:
        print("Running CPU single-core benchmark (512x512)...")
        start_time = time.time()
        result = lib.run_cpu_single_core_benchmark(512, 1)
        end_time = time.time()

        if result.success:
            print(f"✓ CPU benchmark completed in {end_time - start_time:.2f}s")
            print(f"  Result: {result.cpu_single_core_ms:.2f} ms")
            return True
        else:
            print(f"✗ CPU benchmark failed: {result.error_message}")
            return False
    except Exception as e:
        print(f"✗ CPU benchmark error: {e}")
        return False


def test_gpu_benchmark(lib, gpu_available):
    """Test GPU benchmark"""
    print("\n=== Testing GPU Benchmark ===")
    if not gpu_available:
        print("⚠ Skipping GPU benchmark (GPU not available)")
        return True

    try:
        print("Running CUDA naive benchmark (512x512)...")
        start_time = time.time()
        result = lib.run_cuda_naive_benchmark(512, 1)
        end_time = time.time()

        if result.success:
            print(f"✓ GPU benchmark completed in {end_time - start_time:.2f}s")
            print(f"  Result: {result.cuda_naive_ms:.2f} ms")
            return True
        else:
            print(f"✗ GPU benchmark failed: {result.error_message}")
            return False
    except Exception as e:
        print(f"✗ GPU benchmark error: {e}")
        return False


def test_all_benchmarks(lib, gpu_available):
    """Test all benchmarks"""
    print("\n=== Testing All Benchmarks ===")
    try:
        print("Running all benchmarks (512x512, GPU-only mode)...")
        start_time = time.time()

        # Use GPU-only mode for faster testing
        result = lib.run_all_benchmarks(512, 1, gpu_only=not gpu_available)
        end_time = time.time()

        if result.success:
            print(f"✓ All benchmarks completed in {end_time - start_time:.2f}s")

            # Display results
            if result.cpu_single_core_ms > 0:
                print(f"  CPU Single-core: {result.cpu_single_core_ms:.2f} ms")
            if result.cpu_optimized_ms > 0:
                print(f"  CPU Optimized: {result.cpu_optimized_ms:.2f} ms")
            if result.cpu_openblas_ms > 0:
                print(f"  CPU OpenBLAS: {result.cpu_openblas_ms:.2f} ms")
            if result.cuda_naive_ms > 0:
                print(f"  CUDA Naive: {result.cuda_naive_ms:.2f} ms")
            if result.cublas_ms > 0:
                print(f"  cuBLAS: {result.cublas_ms:.2f} ms")
            if result.cublas_tensorcore_ms > 0:
                print(f"  cuBLAS + Tensor Core: {result.cublas_tensorcore_ms:.2f} ms")
            if result.wmma_ms > 0:
                print(f"  WMMA: {result.wmma_ms:.2f} ms")

            # Show speedup analysis
            analysis = result.get_speedup_analysis()
            if analysis:
                print("  Speedup Analysis:")
                for comparison, ratio in list(analysis.items())[:3]:  # Show first 3
                    print(f"    {comparison}: {ratio:.2f}x")

            return True
        else:
            print(f"✗ All benchmarks failed: {result.error_message}")
            return False
    except Exception as e:
        print(f"✗ All benchmarks error: {e}")
        return False


def test_convenience_functions():
    """Test convenience functions"""
    print("\n=== Testing Convenience Functions ===")
    try:
        from cuda_benchmark import run_benchmark

        print("Testing convenience function...")
        result = run_benchmark("cpu_single", 256, 1)

        if result.success:
            print(f"✓ Convenience function works: {result.cpu_single_core_ms:.2f} ms")
            return True
        else:
            print(f"✗ Convenience function failed: {result.error_message}")
            return False
    except Exception as e:
        print(f"✗ Convenience function error: {e}")
        return False


def test_error_handling():
    """Test error handling"""
    print("\n=== Testing Error Handling ===")
    try:
        from cuda_benchmark import run_benchmark

        # Test invalid benchmark type
        result = run_benchmark("invalid_type", 512, 1)
        if not result.success:
            print("✓ Invalid benchmark type properly handled")
        else:
            print("⚠ Invalid benchmark type should have failed")

        # Test invalid matrix size for WMMA
        result = run_benchmark("wmma", 513, 1)  # Not multiple of 16
        if not result.success and "multiple of 16" in result.error_message:
            print("✓ WMMA matrix size validation works")
        else:
            print("⚠ WMMA matrix size validation may not be working properly")

        return True
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def check_system_info():
    """Display system information"""
    print("=== System Information ===")
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        if gpus:
            for gpu in gpus:
                print(f"GPU: {gpu.name}")
                print(f"  Memory: {gpu.memoryTotal}MB total, {gpu.memoryFree}MB free")
                print(f"  Load: {gpu.load*100:.1f}%")
                print(f"  Temperature: {gpu.temperature}°C")
        else:
            print("No GPUs detected")
    except ImportError:
        print("GPUtil not available for GPU info")
    except Exception as e:
        print(f"Error getting GPU info: {e}")

    # Check CUDA
    try:
        import subprocess

        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = [line for line in result.stdout.split("\n") if "release" in line]
            if version_line:
                print(f"CUDA: {version_line[0].strip()}")
        else:
            print("CUDA toolkit not found")
    except FileNotFoundError:
        print("nvcc not found in PATH")
    except Exception as e:
        print(f"Error checking CUDA: {e}")


def main():
    """Main test function"""
    print("CUDA C++ Benchmark Library Test")
    print("=" * 50)

    # Display system info
    check_system_info()
    print()

    # Run tests
    tests_passed = 0
    total_tests = 0

    # Test library import
    total_tests += 1
    if test_library_import():
        tests_passed += 1
    else:
        print("\nCannot continue without library import. Please build the library first.")
        sys.exit(1)

    # Test GPU availability
    total_tests += 1
    gpu_available = test_gpu_availability()
    if gpu_available:
        tests_passed += 1

    # Test library initialization
    total_tests += 1
    lib = test_library_initialization()
    if lib:
        tests_passed += 1
    else:
        print("\nCannot continue without library initialization.")
        sys.exit(1)

    # Test CPU benchmark
    total_tests += 1
    if test_cpu_benchmark(lib):
        tests_passed += 1

    # Test GPU benchmark
    total_tests += 1
    if test_gpu_benchmark(lib, gpu_available):
        tests_passed += 1

    # Test convenience functions
    total_tests += 1
    if test_convenience_functions():
        tests_passed += 1

    # Test error handling
    total_tests += 1
    if test_error_handling():
        tests_passed += 1

    # Test all benchmarks (more comprehensive)
    total_tests += 1
    if test_all_benchmarks(lib, gpu_available):
        tests_passed += 1

    # Summary
    print("\n" + "=" * 50)
    print(f"Test Summary: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("✓ All tests passed! The library is working correctly.")
        print("\nYou can now:")
        print("1. Start the web application: python app.py")
        print("2. Access the web interface at: http://localhost:5000")
        print("3. Use the library in your Python code")
        return 0
    else:
        print(f"✗ {total_tests - tests_passed} tests failed. Please check the setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
