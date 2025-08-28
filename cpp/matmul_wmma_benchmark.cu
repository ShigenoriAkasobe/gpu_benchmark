#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>        // for strcmp
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cublas_v2.h>
#include <omp.h>          // OpenMP for CPU parallelization
#include <immintrin.h>    // AVX intrinsics
#include <cblas.h>        // OpenBLAS for optimized CPU operations

using namespace nvcuda;
using namespace std::chrono;

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, stat); \
        exit(1); \
    } \
} while(0)

////////////////////////////////////////////////////////////////////////////////
// CPU Single-Core GEMM (FP32) - Baseline implementation
// Compute C = A * B (naive triple loop)
////////////////////////////////////////////////////////////////////////////////
void matmul_cpu_single_core(const float* A, const float* B, float* C, int N) {
    // Simple triple loop - no optimization
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

////////////////////////////////////////////////////////////////////////////////
// CPU Manual Optimized GEMM (FP32) - Multi-core + SIMD + Cache blocking
// Compute C = A * B with OpenMP + AVX + blocking (manual implementation)
////////////////////////////////////////////////////////////////////////////////
void matmul_cpu_manual_optimized(const float* A, const float* B, float* C, int N) {
    const int BLOCK_SIZE = 64;  // Cache-friendly block size

    // Initialize result matrix to zero
    #pragma omp parallel for
    for (int i = 0; i < N * N; ++i) {
        C[i] = 0.0f;
    }

    // Blocked matrix multiplication with OpenMP parallelization
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                // Block boundaries
                int i_end = std::min(ii + BLOCK_SIZE, N);
                int j_end = std::min(jj + BLOCK_SIZE, N);
                int k_end = std::min(kk + BLOCK_SIZE, N);

                // Inner block computation with AVX optimization
                for (int i = ii; i < i_end; ++i) {
                    for (int j = jj; j < j_end; j += 8) {  // Process 8 elements at once with AVX
                        if (j + 8 <= j_end) {
                            // AVX vectorized inner loop
                            __m256 sum_vec = _mm256_load_ps(&C[i * N + j]);

                            for (int k = kk; k < k_end; ++k) {
                                __m256 a_vec = _mm256_broadcast_ss(&A[i * N + k]);
                                __m256 b_vec = _mm256_load_ps(&B[k * N + j]);
                                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                            }

                            _mm256_store_ps(&C[i * N + j], sum_vec);
                        } else {
                            // Handle remaining elements (non-vectorizable)
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

////////////////////////////////////////////////////////////////////////////////
// CPU OpenBLAS Optimized GEMM (FP32) - Industry-standard optimized BLAS
// Compute C = A * B using cblas_sgemm (highly optimized)
////////////////////////////////////////////////////////////////////////////////
void matmul_cpu_openblas(const float* A, const float* B, float* C, int N) {
    // OpenBLAS cblas_sgemm performs: C = alpha * A * B + beta * C
    // Parameters: (order, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
    // CblasRowMajor: row-major storage
    // CblasNoTrans: no transpose
    // M=N, N=N, K=N for square matrices
    // alpha=1.0, beta=0.0 for C = A * B
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,                    // M, N, K dimensions
                1.0f,                       // alpha
                A, N,                       // A matrix, leading dimension
                B, N,                       // B matrix, leading dimension
                0.0f,                       // beta
                C, N);                      // C matrix, leading dimension
}

////////////////////////////////////////////////////////////////////////////////
// Simple CUDA-Core GEMM (FP32)
// Compute C = A * B
////////////////////////////////////////////////////////////////////////////////
__global__ void matmul_cuda_core(const float* A, const float* B, float* C, int N) {
    // blockDim: (tile, tile)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        // simple inner loop (not optimized cache blocking), but still uses many CUDA cores in parallel
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Helper kernel: convert float array -> half (__half)
////////////////////////////////////////////////////////////////////////////////
__global__ void float_to_half_kernel(const float* in, half* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __float2half(in[idx]);
    }
}

////////////////////////////////////////////////////////////////////////////////
// WMMA Tensor Core GEMM (FP16 inputs, FP32 accumulate) using 16x16x16 tiles
// Each warp computes one 16x16 tile (typical simple mapping).
////////////////////////////////////////////////////////////////////////////////
__global__ void matmul_wmma(const half* A, const half* B, float* C, int N) {
    // gridDim: (N/16, N/16)
    // blockDim.x must be 32 (one warp per block in this simple mapping)
    int warpM = blockIdx.x; // tile row index
    int warpN = blockIdx.y; // tile col index

    // fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // iterate k in steps of 16
    for (int k = 0; k < N; k += 16) {
        // pointers to tile starting positions
        const half* tileA = A + (warpM * 16) * N + k;
        const half* tileB = B + k * N + (warpN * 16);

        // load fragments (wmma handles the necessary lanes internally)
        wmma::load_matrix_sync(a_frag, tileA, N);
        wmma::load_matrix_sync(b_frag, tileB, N);
        // matrix multiply-accumulate
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // store result tile back to C
    float* tileC = C + (warpM * 16) * N + (warpN * 16);
    wmma::store_matrix_sync(tileC, acc_frag, N, wmma::mem_row_major);
}

////////////////////////////////////////////////////////////////////////////////
// Host-side utility: random init, verification
////////////////////////////////////////////////////////////////////////////////
void init_matrix_float(float* M, int N, unsigned int seed=123) {
    srand(seed);
    for (long long i = 0; i < (long long)N * N; ++i) {
        // small values to avoid overflow in naive kernel for large N
        M[i] = (float)(rand() % 100) / 100.0f;
    }
}

double max_abs_diff(const float* A, const float* B, int N) {
    double maxd = 0.0;
    for (long long i = 0; i < (long long)N * N; ++i) {
        double d = fabs((double)A[i] - (double)B[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

int main(int argc, char** argv) {
    int N = 1024; // default size (must be multiple of 16)
    bool gpu_only = false; // flag for GPU-only benchmark

    // Parse command line arguments
    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3 && strcmp(argv[2], "--gpu-only") == 0) {
        gpu_only = true;
        printf("GPU-only mode enabled (skipping CPU benchmarks)\n");
    }

    if (N % 16 != 0) {
        fprintf(stderr, "N must be multiple of 16 for WMMA tiles. Given N=%d\n", N);
        return 1;
    }

    printf("Matrix multiply benchmark: N=%d x %d\n", N, N);

    size_t bytes_f = (size_t)N * N * sizeof(float);
    size_t bytes_h = (size_t)N * N * sizeof(half);

    // host allocations (aligned for AVX)
    float* h_A = (float*)aligned_alloc(32, bytes_f);  // 32-byte aligned for AVX
    float* h_B = (float*)aligned_alloc(32, bytes_f);
    float* h_C_cpu = (float*)aligned_alloc(32, bytes_f);
    float* h_C_cpu_opt = (float*)aligned_alloc(32, bytes_f);
    float* h_C_cpu_blas = (float*)malloc(bytes_f);  // OpenBLAS result buffer
    float* h_C_cuda = (float*)malloc(bytes_f);
    float* h_C_cublas = (float*)malloc(bytes_f);
    float* h_C_cublas_tc = (float*)malloc(bytes_f);
    float* h_C_wmma = (float*)malloc(bytes_f);

    init_matrix_float(h_A, N, 123);
    init_matrix_float(h_B, N, 456);

    // device allocations
    float *d_Af = nullptr, *d_Bf = nullptr, *d_C_cuda = nullptr, *d_C_cublas = nullptr, *d_C_cublas_tc = nullptr, *d_C_wmma = nullptr;
    half *d_Ah = nullptr, *d_Bh = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_Af, bytes_f));
    CHECK_CUDA(cudaMalloc((void**)&d_Bf, bytes_f));
    CHECK_CUDA(cudaMalloc((void**)&d_C_cuda, bytes_f));
    CHECK_CUDA(cudaMalloc((void**)&d_C_cublas, bytes_f));
    CHECK_CUDA(cudaMalloc((void**)&d_C_cublas_tc, bytes_f));
    CHECK_CUDA(cudaMalloc((void**)&d_C_wmma, bytes_f)); // output as float

    CHECK_CUDA(cudaMalloc((void**)&d_Ah, bytes_h));
    CHECK_CUDA(cudaMalloc((void**)&d_Bh, bytes_h));

    // copy host floats to device floats
    CHECK_CUDA(cudaMemcpy(d_Af, h_A, bytes_f, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Bf, h_B, bytes_f, cudaMemcpyHostToDevice));

    // convert device float -> half (device kernel)
    int convert_threads = 256;
    int convert_blocks = (N * N + convert_threads - 1) / convert_threads;
    float_to_half_kernel<<<convert_blocks, convert_threads>>>(d_Af, d_Ah, N * N);
    float_to_half_kernel<<<convert_blocks, convert_threads>>>(d_Bf, d_Bh, N * N);
    CHECK_CUDA(cudaGetLastError());

    // Warm up
    CHECK_CUDA(cudaMemset(d_C_cuda, 0, bytes_f));
    CHECK_CUDA(cudaMemset(d_C_cublas, 0, bytes_f));
    CHECK_CUDA(cudaMemset(d_C_cublas_tc, 0, bytes_f));
    CHECK_CUDA(cudaMemset(d_C_wmma, 0, bytes_f));

    // Initialize cuBLAS
    cublasHandle_t cublasH;
    CHECK_CUBLAS(cublasCreate(&cublasH));

    // Enable Tensor Core usage for cuBLAS
    CHECK_CUBLAS(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));

    // timing with events/chrono
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int repeat = 3;

    // Increase repeat count for small matrices to improve measurement accuracy
    if (N <= 512) {
        repeat = 10;
        printf("Using %d repetitions for accurate timing of small matrices\n", repeat);
    }

    // Variables for CPU benchmark results (will be set only if not gpu_only)
    double avg_ms_cpu = 0.0, avg_ms_cpu_opt = 0.0, avg_ms_cpu_blas = 0.0;

    if (!gpu_only) {
        // --- CPU Single-Core GEMM (baseline) ---
        printf("Running CPU single-core benchmark...\n");

        // warmup
        matmul_cpu_single_core(h_A, h_B, h_C_cpu, N);

        auto cpu_start = high_resolution_clock::now();
        for (int i = 0; i < repeat; ++i) {
            matmul_cpu_single_core(h_A, h_B, h_C_cpu, N);
        }
        auto cpu_end = high_resolution_clock::now();
        auto cpu_duration = duration_cast<microseconds>(cpu_end - cpu_start);
        avg_ms_cpu = cpu_duration.count() / (double)repeat / 1000.0; // Convert to ms

        printf("CPU single-core GEMM avg time: %f ms (avg over %d runs)\n", avg_ms_cpu, repeat);

        // --- CPU Optimized GEMM (Multi-core + SIMD + Cache blocking) ---
        printf("Running CPU optimized benchmark (OpenMP + AVX)...\n");
        printf("Using %d CPU threads\n", omp_get_max_threads());

        // warmup
        matmul_cpu_manual_optimized(h_A, h_B, h_C_cpu_opt, N);

        auto cpu_opt_start = high_resolution_clock::now();
        for (int i = 0; i < repeat; ++i) {
            matmul_cpu_manual_optimized(h_A, h_B, h_C_cpu_opt, N);
        }
        auto cpu_opt_end = high_resolution_clock::now();
        auto cpu_opt_duration = duration_cast<microseconds>(cpu_opt_end - cpu_opt_start);
        avg_ms_cpu_opt = cpu_opt_duration.count() / (double)repeat / 1000.0; // Convert to ms

        printf("CPU optimized GEMM avg time: %f ms (avg over %d runs)\n", avg_ms_cpu_opt, repeat);

        // --- CPU OpenBLAS GEMM (Industry-standard optimized BLAS) ---
        printf("Running CPU OpenBLAS benchmark (industry-standard BLAS)...\n");

        // warmup
        matmul_cpu_openblas(h_A, h_B, h_C_cpu_blas, N);

        auto cpu_blas_start = high_resolution_clock::now();
        for (int i = 0; i < repeat; ++i) {
            matmul_cpu_openblas(h_A, h_B, h_C_cpu_blas, N);
        }
        auto cpu_blas_end = high_resolution_clock::now();
        auto cpu_blas_duration = duration_cast<microseconds>(cpu_blas_end - cpu_blas_start);
        avg_ms_cpu_blas = cpu_blas_duration.count() / (double)repeat / 1000.0; // Convert to ms

        printf("CPU OpenBLAS GEMM avg time: %f ms (avg over %d runs)\n", avg_ms_cpu_blas, repeat);

    } else {
        printf("Skipping CPU benchmarks (GPU-only mode)\n");
    }

    // --- CUDA Core GEMM (naive) ---
    printf("Running CUDA naive benchmark...\n");
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // warmup
    matmul_cuda_core<<<grid, block>>>(d_Af, d_Bf, d_C_cuda, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        matmul_cuda_core<<<grid, block>>>(d_Af, d_Bf, d_C_cuda, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_cuda = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_cuda, start, stop));
    double avg_ms_cuda = ms_cuda / (double)repeat;

    printf("CUDA-Core naive GEMM avg time: %f ms (avg over %d runs)\n", avg_ms_cuda, repeat);

    // --- cuBLAS optimized GEMM (FP32) ---
    printf("Running cuBLAS optimized benchmark...\n");
    // cuBLAS uses column-major format, so we compute C = B^T * A^T = (A * B)^T
    // Then we interpret the result as row-major C = A * B
    const float alpha = 1.0f, beta = 0.0f;

    // warmup
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, N, N,
                            &alpha,
                            d_Bf, N,  // B matrix
                            d_Af, N,  // A matrix
                            &beta,
                            d_C_cublas, N));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                N, N, N,
                                &alpha,
                                d_Bf, N,  // B matrix
                                d_Af, N,  // A matrix
                                &beta,
                                d_C_cublas, N));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_cublas = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_cublas, start, stop));
    double avg_ms_cublas = ms_cublas / (double)repeat;

    printf("cuBLAS optimized GEMM avg time: %f ms (avg over %d runs)\n", avg_ms_cublas, repeat);

    // --- cuBLAS + Tensor Core GEMM (FP16 with automatic mixed precision) ---
    printf("Running cuBLAS + Tensor Core benchmark...\n");
    const float alpha_f = 1.0f, beta_f = 0.0f;

    // warmup
    CHECK_CUBLAS(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha_f,
                             d_Bh, CUDA_R_16F, N,  // B matrix (FP16)
                             d_Ah, CUDA_R_16F, N,  // A matrix (FP16)
                             &beta_f,
                             d_C_cublas_tc, CUDA_R_32F, N,  // C matrix (FP32)
                             CUDA_R_32F,  // Computation type
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        CHECK_CUBLAS(cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, N,
                                 &alpha_f,
                                 d_Bh, CUDA_R_16F, N,  // B matrix (FP16)
                                 d_Ah, CUDA_R_16F, N,  // A matrix (FP16)
                                 &beta_f,
                                 d_C_cublas_tc, CUDA_R_32F, N,  // C matrix (FP32)
                                 CUDA_R_32F,  // Computation type
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_cublas_tc = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_cublas_tc, start, stop));
    double avg_ms_cublas_tc = ms_cublas_tc / (double)repeat;

    printf("cuBLAS + Tensor Core GEMM avg time: %f ms (avg over %d runs)\n", avg_ms_cublas_tc, repeat);

    // --- WMMA Tensor Core GEMM ---
    printf("Running WMMA manual implementation benchmark...\n");
    // grid: N/16 x N/16 (one warp per tile approach)
    dim3 grid_wmma(N / 16, N / 16);
    dim3 block_wmma(32, 1, 1); // one warp per block (simple mapping)

    // warmup
    matmul_wmma<<<grid_wmma, block_wmma>>>(d_Ah, d_Bh, d_C_wmma, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        matmul_wmma<<<grid_wmma, block_wmma>>>(d_Ah, d_Bh, d_C_wmma, N);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_wmma = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_wmma, start, stop));
    double avg_ms_wmma = ms_wmma / (double)repeat;

    printf("WMMA Tensor-Core GEMM avg time: %f ms (avg over %d runs)\n", avg_ms_wmma, repeat);

    // copy back results
    CHECK_CUDA(cudaMemcpy(h_C_cuda, d_C_cuda, bytes_f, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_cublas, d_C_cublas, bytes_f, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_cublas_tc, d_C_cublas_tc, bytes_f, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_wmma, d_C_wmma, bytes_f, cudaMemcpyDeviceToHost));

    // verify differences (using CPU single-core as reference - only if CPU benchmarks were run)
    double maxd_cpu_cpu_opt = 0.0, maxd_cpu_cpu_blas = 0.0, maxd_cpu_cuda = 0.0, maxd_cpu_cublas = 0.0, maxd_cpu_cublas_tc = 0.0, maxd_cpu_wmma = 0.0;

    if (!gpu_only) {
        maxd_cpu_cpu_opt = max_abs_diff(h_C_cpu, h_C_cpu_opt, N);
        maxd_cpu_cpu_blas = max_abs_diff(h_C_cpu, h_C_cpu_blas, N);
        maxd_cpu_cuda = max_abs_diff(h_C_cpu, h_C_cuda, N);
        maxd_cpu_cublas = max_abs_diff(h_C_cpu, h_C_cublas, N);
        maxd_cpu_cublas_tc = max_abs_diff(h_C_cpu, h_C_cublas_tc, N);
        maxd_cpu_wmma = max_abs_diff(h_C_cpu, h_C_wmma, N);
    }

    printf("\n=== Performance Summary ===\n");
    if (!gpu_only) {
        printf("1. CPU single-core     : %8.3f ms (baseline)\n", avg_ms_cpu);
        printf("2. CPU manual optimized: %8.3f ms (%.2fx faster than single-core)\n",
               avg_ms_cpu_opt, avg_ms_cpu / avg_ms_cpu_opt);
        printf("3. CPU OpenBLAS        : %8.3f ms (%.2fx faster than single-core, %.2fx vs manual)\n",
               avg_ms_cpu_blas, avg_ms_cpu / avg_ms_cpu_blas, avg_ms_cpu_opt / avg_ms_cpu_blas);
        printf("4. CUDA-Core naive     : %8.3f ms (%.2fx faster than CPU single-core)\n",
               avg_ms_cuda, avg_ms_cpu / avg_ms_cuda);
        printf("5. cuBLAS optimized    : %8.3f ms (%.2fx faster than CPU single-core, %.2fx faster than CUDA naive)\n",
               avg_ms_cublas, avg_ms_cpu / avg_ms_cublas, avg_ms_cuda / avg_ms_cublas);
        printf("6. cuBLAS + TensorCore : %8.3f ms (%.2fx faster than CPU single-core, %.2fx faster than cuBLAS)\n",
               avg_ms_cublas_tc, avg_ms_cpu / avg_ms_cublas_tc, avg_ms_cublas / avg_ms_cublas_tc);
        printf("7. WMMA manual impl    : %8.3f ms (%.2fx faster than CPU single-core, %.2fx vs cuBLAS)\n",
               avg_ms_wmma, avg_ms_cpu / avg_ms_wmma, avg_ms_cublas / avg_ms_wmma);

        printf("\n=== CPU vs GPU Comparison ===\n");
        printf("Best CPU (OpenBLAS) vs Best GPU (cuBLAS+TC): %.2fx GPU advantage\n",
               avg_ms_cpu_blas / avg_ms_cublas_tc);
    } else {
        printf("GPU-only benchmark results:\n");
        printf("1. CUDA-Core naive    : %8.3f ms\n", avg_ms_cuda);
        printf("2. cuBLAS optimized   : %8.3f ms (%.2fx faster than CUDA naive)\n",
               avg_ms_cublas, avg_ms_cuda / avg_ms_cublas);
        printf("3. cuBLAS + TensorCore: %8.3f ms (%.2fx faster than cuBLAS)\n",
               avg_ms_cublas_tc, avg_ms_cublas / avg_ms_cublas_tc);
        printf("4. WMMA manual impl   : %8.3f ms (%.2fx vs cuBLAS)\n",
               avg_ms_wmma, avg_ms_cublas / avg_ms_wmma);

        printf("\n=== GPU Performance Comparison ===\n");
        printf("Best GPU (cuBLAS+TC) vs CUDA naive: %.2fx improvement\n",
               avg_ms_cuda / avg_ms_cublas_tc);
    }

    if (!gpu_only) {
        printf("\n=== Accuracy Verification (vs CPU single-core baseline) ===\n");
        printf("Max diff (CPU single vs CPU manual opt): %e\n", maxd_cpu_cpu_opt);
        printf("Max diff (CPU single vs CPU OpenBLAS)  : %e\n", maxd_cpu_cpu_blas);
        printf("Max diff (CPU single vs CUDA naive)    : %e\n", maxd_cpu_cuda);
        printf("Max diff (CPU single vs cuBLAS)        : %e\n", maxd_cpu_cublas);
        printf("Max diff (CPU single vs cuBLAS+TC)     : %e\n", maxd_cpu_cublas_tc);
        printf("Max diff (CPU single vs WMMA)          : %e\n", maxd_cpu_wmma);
    } else {
        printf("\n=== Accuracy Verification (GPU-only mode) ===\n");
        printf("GPU accuracy verification skipped (no CPU baseline in GPU-only mode)\n");
        printf("Use cuBLAS as reference for GPU implementations\n");
    }

    // Clean up cuBLAS
    CHECK_CUBLAS(cublasDestroy(cublasH));

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_Af)); CHECK_CUDA(cudaFree(d_Bf));
    CHECK_CUDA(cudaFree(d_C_cuda)); CHECK_CUDA(cudaFree(d_C_cublas)); CHECK_CUDA(cudaFree(d_C_cublas_tc)); CHECK_CUDA(cudaFree(d_C_wmma));
    CHECK_CUDA(cudaFree(d_Ah)); CHECK_CUDA(cudaFree(d_Bh));

    // Free host memory
    free(h_A); free(h_B); free(h_C_cuda); free(h_C_cublas); free(h_C_cublas_tc); free(h_C_wmma);
    free(h_C_cpu); free(h_C_cpu_opt); free(h_C_cpu_blas);

    return 0;
}
