#include "benchmark_lib.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cublas_v2.h>
#include <omp.h>
#include <immintrin.h>
#include <cblas.h>

using namespace nvcuda;
using namespace std::chrono;

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return nullptr; \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, stat); \
        return nullptr; \
    } \
} while(0)

// Helper function to create result structure
static BenchmarkResult* create_result(int matrix_size, int iterations) {
    BenchmarkResult* result = (BenchmarkResult*)malloc(sizeof(BenchmarkResult));
    if (!result) return nullptr;

    memset(result, 0, sizeof(BenchmarkResult));
    result->matrix_size = matrix_size;
    result->iterations = iterations;
    result->success = 1;
    return result;
}

// Helper function to set error
static void set_error(BenchmarkResult* result, const char* message) {
    if (result) {
        result->success = 0;
        strncpy(result->error_message, message, sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
    }
}

// Include the original implementations (copied from matmul_wmma_benchmark.cu)
void init_matrix_float(float* M, int N, unsigned int seed=123) {
    srand(seed);
    for (long long i = 0; i < (long long)N * N; ++i) {
        M[i] = (float)(rand() % 100) / 100.0f;
    }
}

void matmul_cpu_single_core(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_cpu_manual_optimized(const float* A, const float* B, float* C, int N) {
    const int BLOCK_SIZE = 64;

    #pragma omp parallel for
    for (int i = 0; i < N * N; ++i) {
        C[i] = 0.0f;
    }

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                int i_end = std::min(ii + BLOCK_SIZE, N);
                int j_end = std::min(jj + BLOCK_SIZE, N);
                int k_end = std::min(kk + BLOCK_SIZE, N);

                for (int i = ii; i < i_end; ++i) {
                    for (int j = jj; j < j_end; j += 8) {
                        if (j + 8 <= j_end) {
                            __m256 sum_vec = _mm256_load_ps(&C[i * N + j]);

                            for (int k = kk; k < k_end; ++k) {
                                __m256 a_vec = _mm256_broadcast_ss(&A[i * N + k]);
                                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                            }

                            _mm256_store_ps(&C[i * N + j], sum_vec);
                        } else {
                            for (int j_rem = j; j_rem < j_end; ++j_rem) {
                                for (int k = kk; k < k_end; ++k) {
                                    C[i * N + j_rem] += A[i * N + k] * B[k * N + j_rem];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void matmul_cpu_openblas(const float* A, const float* B, float* C, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
}

__global__ void matmul_cuda_core(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void float_to_half_kernel(const float* in, half* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __float2half(in[idx]);
    }
}

__global__ void matmul_wmma(const half* A, const half* B, float* C, int N) {
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < N; k += 16) {
        const half* tileA = A + (warpM * 16) * N + k;
        const half* tileB = B + k * N + (warpN * 16);

        wmma::load_matrix_sync(a_frag, tileA, N);
        wmma::load_matrix_sync(b_frag, tileB, N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    float* tileC = C + (warpM * 16) * N + (warpN * 16);
    wmma::store_matrix_sync(tileC, acc_frag, N, wmma::mem_row_major);
}

// C interface implementations
extern "C" {

BenchmarkResult* run_cpu_single_core_benchmark(int matrix_size, int iterations) {
    BenchmarkResult* result = create_result(matrix_size, iterations);
    if (!result) return nullptr;

    try {
        size_t bytes = (size_t)matrix_size * matrix_size * sizeof(float);
        float* A = (float*)aligned_alloc(32, bytes);
        float* B = (float*)aligned_alloc(32, bytes);
        float* C = (float*)aligned_alloc(32, bytes);

        if (!A || !B || !C) {
            set_error(result, "Memory allocation failed");
            free(A); free(B); free(C);
            return result;
        }

        init_matrix_float(A, matrix_size, 123);
        init_matrix_float(B, matrix_size, 456);

        // Warmup
        matmul_cpu_single_core(A, B, C, matrix_size);

        auto start = high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            matmul_cpu_single_core(A, B, C, matrix_size);
        }
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(end - start);
        result->cpu_single_core_ms = duration.count() / (double)iterations / 1000.0;

        free(A); free(B); free(C);

    } catch (const std::exception& e) {
        set_error(result, e.what());
    }

    return result;
}

BenchmarkResult* run_cpu_optimized_benchmark(int matrix_size, int iterations) {
    BenchmarkResult* result = create_result(matrix_size, iterations);
    if (!result) return nullptr;

    try {
        size_t bytes = (size_t)matrix_size * matrix_size * sizeof(float);
        float* A = (float*)aligned_alloc(32, bytes);
        float* B = (float*)aligned_alloc(32, bytes);
        float* C = (float*)aligned_alloc(32, bytes);

        if (!A || !B || !C) {
            set_error(result, "Memory allocation failed");
            free(A); free(B); free(C);
            return result;
        }

        init_matrix_float(A, matrix_size, 123);
        init_matrix_float(B, matrix_size, 456);

        // Warmup
        matmul_cpu_manual_optimized(A, B, C, matrix_size);

        auto start = high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            matmul_cpu_manual_optimized(A, B, C, matrix_size);
        }
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(end - start);
        result->cpu_optimized_ms = duration.count() / (double)iterations / 1000.0;

        free(A); free(B); free(C);

    } catch (const std::exception& e) {
        set_error(result, e.what());
    }

    return result;
}

BenchmarkResult* run_cpu_openblas_benchmark(int matrix_size, int iterations) {
    BenchmarkResult* result = create_result(matrix_size, iterations);
    if (!result) return nullptr;

    try {
        size_t bytes = (size_t)matrix_size * matrix_size * sizeof(float);
        float* A = (float*)malloc(bytes);
        float* B = (float*)malloc(bytes);
        float* C = (float*)malloc(bytes);

        if (!A || !B || !C) {
            set_error(result, "Memory allocation failed");
            free(A); free(B); free(C);
            return result;
        }

        init_matrix_float(A, matrix_size, 123);
        init_matrix_float(B, matrix_size, 456);

        // Warmup
        matmul_cpu_openblas(A, B, C, matrix_size);

        auto start = high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            matmul_cpu_openblas(A, B, C, matrix_size);
        }
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<microseconds>(end - start);
        result->cpu_openblas_ms = duration.count() / (double)iterations / 1000.0;

        free(A); free(B); free(C);

    } catch (const std::exception& e) {
        set_error(result, e.what());
    }

    return result;
}

BenchmarkResult* run_cuda_naive_benchmark(int matrix_size, int iterations) {
    BenchmarkResult* result = create_result(matrix_size, iterations);
    if (!result) return nullptr;

    try {
        size_t bytes = (size_t)matrix_size * matrix_size * sizeof(float);

        // Host memory
        float* h_A = (float*)malloc(bytes);
        float* h_B = (float*)malloc(bytes);

        if (!h_A || !h_B) {
            set_error(result, "Host memory allocation failed");
            free(h_A); free(h_B);
            return result;
        }

        init_matrix_float(h_A, matrix_size, 123);
        init_matrix_float(h_B, matrix_size, 456);

        // Device memory
        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
        CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
        CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

        CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((matrix_size + block.x - 1) / block.x, (matrix_size + block.y - 1) / block.y);

        // Warmup
        matmul_cuda_core<<<grid, block>>>(d_A, d_B, d_C, matrix_size);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            matmul_cuda_core<<<grid, block>>>(d_A, d_B, d_C, matrix_size);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms_total = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        result->cuda_naive_ms = ms_total / (double)iterations;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        free(h_A); free(h_B);

    } catch (const std::exception& e) {
        set_error(result, e.what());
    }

    return result;
}

BenchmarkResult* run_cublas_benchmark(int matrix_size, int iterations) {
    BenchmarkResult* result = create_result(matrix_size, iterations);
    if (!result) return nullptr;

    try {
        size_t bytes = (size_t)matrix_size * matrix_size * sizeof(float);

        float* h_A = (float*)malloc(bytes);
        float* h_B = (float*)malloc(bytes);

        if (!h_A || !h_B) {
            set_error(result, "Host memory allocation failed");
            free(h_A); free(h_B);
            return result;
        }

        init_matrix_float(h_A, matrix_size, 123);
        init_matrix_float(h_B, matrix_size, 456);

        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
        CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
        CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

        CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        cublasHandle_t cublasH;
        CHECK_CUBLAS(cublasCreate(&cublasH));

        const float alpha = 1.0f, beta = 0.0f;

        // Warmup
        CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                matrix_size, matrix_size, matrix_size,
                                &alpha, d_B, matrix_size, d_A, matrix_size,
                                &beta, d_C, matrix_size));
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                    matrix_size, matrix_size, matrix_size,
                                    &alpha, d_B, matrix_size, d_A, matrix_size,
                                    &beta, d_C, matrix_size));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms_total = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        result->cublas_ms = ms_total / (double)iterations;

        CHECK_CUBLAS(cublasDestroy(cublasH));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        free(h_A); free(h_B);

    } catch (const std::exception& e) {
        set_error(result, e.what());
    }

    return result;
}

BenchmarkResult* run_cublas_tensorcore_benchmark(int matrix_size, int iterations) {
    BenchmarkResult* result = create_result(matrix_size, iterations);
    if (!result) return nullptr;

    try {
        size_t bytes_f = (size_t)matrix_size * matrix_size * sizeof(float);
        size_t bytes_h = (size_t)matrix_size * matrix_size * sizeof(half);

        float* h_A = (float*)malloc(bytes_f);
        float* h_B = (float*)malloc(bytes_f);

        if (!h_A || !h_B) {
            set_error(result, "Host memory allocation failed");
            free(h_A); free(h_B);
            return result;
        }

        init_matrix_float(h_A, matrix_size, 123);
        init_matrix_float(h_B, matrix_size, 456);

        float *d_Af, *d_Bf, *d_C;
        half *d_Ah, *d_Bh;
        CHECK_CUDA(cudaMalloc((void**)&d_Af, bytes_f));
        CHECK_CUDA(cudaMalloc((void**)&d_Bf, bytes_f));
        CHECK_CUDA(cudaMalloc((void**)&d_C, bytes_f));
        CHECK_CUDA(cudaMalloc((void**)&d_Ah, bytes_h));
        CHECK_CUDA(cudaMalloc((void**)&d_Bh, bytes_h));

        CHECK_CUDA(cudaMemcpy(d_Af, h_A, bytes_f, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_Bf, h_B, bytes_f, cudaMemcpyHostToDevice));

        // Convert to half
        int convert_threads = 256;
        int convert_blocks = (matrix_size * matrix_size + convert_threads - 1) / convert_threads;
        float_to_half_kernel<<<convert_blocks, convert_threads>>>(d_Af, d_Ah, matrix_size * matrix_size);
        float_to_half_kernel<<<convert_blocks, convert_threads>>>(d_Bf, d_Bh, matrix_size * matrix_size);
        CHECK_CUDA(cudaGetLastError());

        cublasHandle_t cublasH;
        CHECK_CUBLAS(cublasCreate(&cublasH));
        CHECK_CUBLAS(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));

        const float alpha = 1.0f, beta = 0.0f;

        // Warmup
        CHECK_CUBLAS(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                 matrix_size, matrix_size, matrix_size,
                                 &alpha,
                                 d_Bh, CUDA_R_16F, matrix_size,
                                 d_Ah, CUDA_R_16F, matrix_size,
                                 &beta,
                                 d_C, CUDA_R_32F, matrix_size,
                                 CUDA_R_32F,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            CHECK_CUBLAS(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                     matrix_size, matrix_size, matrix_size,
                                     &alpha,
                                     d_Bh, CUDA_R_16F, matrix_size,
                                     d_Ah, CUDA_R_16F, matrix_size,
                                     &beta,
                                     d_C, CUDA_R_32F, matrix_size,
                                     CUDA_R_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms_total = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        result->cublas_tensorcore_ms = ms_total / (double)iterations;

        CHECK_CUBLAS(cublasDestroy(cublasH));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_Af));
        CHECK_CUDA(cudaFree(d_Bf));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUDA(cudaFree(d_Ah));
        CHECK_CUDA(cudaFree(d_Bh));
        free(h_A); free(h_B);

    } catch (const std::exception& e) {
        set_error(result, e.what());
    }

    return result;
}

BenchmarkResult* run_wmma_benchmark(int matrix_size, int iterations) {
    BenchmarkResult* result = create_result(matrix_size, iterations);
    if (!result) return nullptr;

    if (matrix_size % 16 != 0) {
        set_error(result, "Matrix size must be multiple of 16 for WMMA");
        return result;
    }

    try {
        size_t bytes_f = (size_t)matrix_size * matrix_size * sizeof(float);
        size_t bytes_h = (size_t)matrix_size * matrix_size * sizeof(half);

        float* h_A = (float*)malloc(bytes_f);
        float* h_B = (float*)malloc(bytes_f);

        if (!h_A || !h_B) {
            set_error(result, "Host memory allocation failed");
            free(h_A); free(h_B);
            return result;
        }

        init_matrix_float(h_A, matrix_size, 123);
        init_matrix_float(h_B, matrix_size, 456);

        float *d_Af, *d_Bf, *d_C;
        half *d_Ah, *d_Bh;
        CHECK_CUDA(cudaMalloc((void**)&d_Af, bytes_f));
        CHECK_CUDA(cudaMalloc((void**)&d_Bf, bytes_f));
        CHECK_CUDA(cudaMalloc((void**)&d_C, bytes_f));
        CHECK_CUDA(cudaMalloc((void**)&d_Ah, bytes_h));
        CHECK_CUDA(cudaMalloc((void**)&d_Bh, bytes_h));

        CHECK_CUDA(cudaMemcpy(d_Af, h_A, bytes_f, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_Bf, h_B, bytes_f, cudaMemcpyHostToDevice));

        // Convert to half
        int convert_threads = 256;
        int convert_blocks = (matrix_size * matrix_size + convert_threads - 1) / convert_threads;
        float_to_half_kernel<<<convert_blocks, convert_threads>>>(d_Af, d_Ah, matrix_size * matrix_size);
        float_to_half_kernel<<<convert_blocks, convert_threads>>>(d_Bf, d_Bh, matrix_size * matrix_size);
        CHECK_CUDA(cudaGetLastError());

        dim3 grid_wmma(matrix_size / 16, matrix_size / 16);
        dim3 block_wmma(32, 1, 1);

        // Warmup
        matmul_wmma<<<grid_wmma, block_wmma>>>(d_Ah, d_Bh, d_C, matrix_size);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            matmul_wmma<<<grid_wmma, block_wmma>>>(d_Ah, d_Bh, d_C, matrix_size);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms_total = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms_total, start, stop));
        result->wmma_ms = ms_total / (double)iterations;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_Af));
        CHECK_CUDA(cudaFree(d_Bf));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUDA(cudaFree(d_Ah));
        CHECK_CUDA(cudaFree(d_Bh));
        free(h_A); free(h_B);

    } catch (const std::exception& e) {
        set_error(result, e.what());
    }

    return result;
}

BenchmarkResult* run_all_benchmarks(int matrix_size, int iterations, int gpu_only) {
    BenchmarkResult* result = create_result(matrix_size, iterations);
    if (!result) return nullptr;

    try {
        if (!gpu_only) {
            // CPU benchmarks
            BenchmarkResult* cpu_single = run_cpu_single_core_benchmark(matrix_size, iterations);
            BenchmarkResult* cpu_opt = run_cpu_optimized_benchmark(matrix_size, iterations);
            BenchmarkResult* cpu_blas = run_cpu_openblas_benchmark(matrix_size, iterations);

            if (cpu_single && cpu_single->success) {
                result->cpu_single_core_ms = cpu_single->cpu_single_core_ms;
            }
            if (cpu_opt && cpu_opt->success) {
                result->cpu_optimized_ms = cpu_opt->cpu_optimized_ms;
            }
            if (cpu_blas && cpu_blas->success) {
                result->cpu_openblas_ms = cpu_blas->cpu_openblas_ms;
            }

            free_benchmark_result(cpu_single);
            free_benchmark_result(cpu_opt);
            free_benchmark_result(cpu_blas);
        }

        // GPU benchmarks
        BenchmarkResult* cuda_naive = run_cuda_naive_benchmark(matrix_size, iterations);
        BenchmarkResult* cublas = run_cublas_benchmark(matrix_size, iterations);
        BenchmarkResult* cublas_tc = run_cublas_tensorcore_benchmark(matrix_size, iterations);
        BenchmarkResult* wmma = run_wmma_benchmark(matrix_size, iterations);

        if (cuda_naive && cuda_naive->success) {
            result->cuda_naive_ms = cuda_naive->cuda_naive_ms;
        }
        if (cublas && cublas->success) {
            result->cublas_ms = cublas->cublas_ms;
        }
        if (cublas_tc && cublas_tc->success) {
            result->cublas_tensorcore_ms = cublas_tc->cublas_tensorcore_ms;
        }
        if (wmma && wmma->success) {
            result->wmma_ms = wmma->wmma_ms;
        }

        free_benchmark_result(cuda_naive);
        free_benchmark_result(cublas);
        free_benchmark_result(cublas_tc);
        free_benchmark_result(wmma);

    } catch (const std::exception& e) {
        set_error(result, e.what());
    }

    return result;
}

void free_benchmark_result(BenchmarkResult* result) {
    if (result) {
        free(result);
    }
}

int check_gpu_availability() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0) ? 1 : 0;
}

} // extern "C"
