#ifndef BENCHMARK_LIB_H
#define BENCHMARK_LIB_H

#ifdef __cplusplus
extern "C" {
#endif

// Result structure for benchmark results
typedef struct {
    double cpu_single_core_ms;
    double cpu_optimized_ms;
    double cpu_openblas_ms;
    double cuda_naive_ms;
    double cublas_ms;
    double cublas_tensorcore_ms;
    double wmma_ms;
    int matrix_size;
    int iterations;
    int success;
    char error_message[256];
} BenchmarkResult;

// Function prototypes for C interface
BenchmarkResult* run_cpu_single_core_benchmark(int matrix_size, int iterations);
BenchmarkResult* run_cpu_optimized_benchmark(int matrix_size, int iterations);
BenchmarkResult* run_cpu_openblas_benchmark(int matrix_size, int iterations);
BenchmarkResult* run_cuda_naive_benchmark(int matrix_size, int iterations);
BenchmarkResult* run_cublas_benchmark(int matrix_size, int iterations);
BenchmarkResult* run_cublas_tensorcore_benchmark(int matrix_size, int iterations);
BenchmarkResult* run_wmma_benchmark(int matrix_size, int iterations);
BenchmarkResult* run_all_benchmarks(int matrix_size, int iterations, int gpu_only);

// Memory management
void free_benchmark_result(BenchmarkResult* result);

// GPU availability check
int check_gpu_availability();

#ifdef __cplusplus
}
#endif

#endif // BENCHMARK_LIB_H
