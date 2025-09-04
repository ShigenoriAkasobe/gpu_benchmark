import time

import GPUtil
import numpy as np

# CUDAが利用可能かチェック（オプション）
try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

# PyTorchが利用可能かチェック
try:
    import torch

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# CUDA C++ benchmark library
try:
    from cuda_benchmark import (
        CudaBenchmarkLibrary,
        check_gpu_availability,
        run_benchmark,
    )

    CUDA_CPP_AVAILABLE = True
except ImportError as e:
    print(f"CUDA C++ benchmark library not available: {e}")
    CUDA_CPP_AVAILABLE = False


class GPUBenchmark:
    def __init__(self):
        self.results = {}
        # Initialize CUDA C++ library if available
        self.cuda_lib = None
        if CUDA_CPP_AVAILABLE:
            try:
                self.cuda_lib = CudaBenchmarkLibrary()
            except Exception as e:
                print(f"Failed to initialize CUDA C++ library: {e}")
                self.cuda_lib = None

    def get_gpu_info(self):
        """GPU情報を取得"""
        gpu_info = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                info = {
                    "id": gpu.id,
                    "name": gpu.name,
                    "driver": gpu.driver,
                    "memory_total": gpu.memoryTotal,
                    "memory_free": gpu.memoryFree,
                    "memory_used": gpu.memoryUsed,
                    "temperature": gpu.temperature,
                    "load": gpu.load,
                }
                gpu_info.append(info)
        except Exception as e:
            print(f"GPU情報の取得に失敗: {e}")

        return gpu_info

    def cpu_matrix_multiply(self, size=2000, iterations=1):
        """CPUでの行列乗算ベンチマーク"""
        total_time = 0
        total_ops = 0

        for i in range(iterations):
            start_time = time.time()
            a = np.random.random((size, size)).astype(np.float32)
            b = np.random.random((size, size)).astype(np.float32)
            c = np.dot(a, b)
            end_time = time.time()

            total_time += end_time - start_time
            total_ops += 2.0 * size * size * size

        avg_time = total_time / iterations
        avg_gflops = total_ops / (total_time * 1e9)

        return {
            "time": avg_time,
            "total_time": total_time,
            "gflops": avg_gflops,
            "matrix_size": size,
            "iterations": iterations,
        }

    def gpu_matrix_multiply_cupy(self, size=2000, iterations=1):
        """CuPyを使用したGPU行列乗算ベンチマーク"""
        if not CUDA_AVAILABLE:
            return None

        try:
            # GPU初期化とウォームアップ
            cp.cuda.Device().synchronize()

            total_time = 0
            total_ops = 0

            for i in range(iterations):
                start_time = time.time()
                a = cp.random.random((size, size), dtype=cp.float32)
                b = cp.random.random((size, size), dtype=cp.float32)
                c = cp.dot(a, b)
                cp.cuda.Device().synchronize()
                end_time = time.time()

                total_time += end_time - start_time
                total_ops += 2.0 * size * size * size

                # メモリ解放
                del a, b, c
                cp.cuda.Device().synchronize()

            # CuPyのメモリプールを完全にクリア
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

            avg_time = total_time / iterations
            avg_gflops = total_ops / (total_time * 1e9)

            return {
                "time": avg_time,
                "total_time": total_time,
                "gflops": avg_gflops,
                "matrix_size": size,
                "iterations": iterations,
            }
        except Exception as e:
            # エラー時もメモリをクリア
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except:
                pass
            print(f"CuPy GPU テストエラー: {e}")
            return None

    def gpu_matrix_multiply_torch(self, size=2000, iterations=1):
        """PyTorchを使用したGPU行列乗算ベンチマーク"""
        if not PYTORCH_AVAILABLE:
            return None

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cpu":
                return None

            # GPU初期化とウォームアップ
            torch.cuda.synchronize()

            total_time = 0
            total_ops = 0

            for i in range(iterations):
                start_time = time.time()
                a = torch.randn(size, size, device=device, dtype=torch.float32)
                b = torch.randn(size, size, device=device, dtype=torch.float32)
                c = torch.mm(a, b)
                torch.cuda.synchronize()
                end_time = time.time()

                total_time += end_time - start_time
                total_ops += 2.0 * size * size * size

                # メモリ解放
                del a, b, c
                torch.cuda.synchronize()

            # PyTorchのキャッシュをクリア
            torch.cuda.empty_cache()

            avg_time = total_time / iterations
            avg_gflops = total_ops / (total_time * 1e9)

            return {
                "time": avg_time,
                "total_time": total_time,
                "gflops": avg_gflops,
                "matrix_size": size,
                "iterations": iterations,
            }
        except Exception as e:
            # エラー時もメモリをクリア
            try:
                torch.cuda.empty_cache()
            except:
                pass
            print(f"PyTorch GPU テストエラー: {e}")
            return None

    def memory_bandwidth_test(self, size_mb=100):
        """メモリ帯域幅テスト"""
        if not CUDA_AVAILABLE:
            return None

        try:
            size = int(size_mb * 1024 * 1024 / 4)  # float32要素数

            # GPU -> CPU
            start_time = time.time()
            gpu_array = cp.random.random(size, dtype=cp.float32)
            cp.cuda.Device().synchronize()
            cpu_array = cp.asnumpy(gpu_array)
            end_time = time.time()
            gpu_to_cpu_bw = (size * 4) / (end_time - start_time) / 1e9  # GB/s

            # CPU -> GPU
            start_time = time.time()
            gpu_array2 = cp.asarray(cpu_array)
            cp.cuda.Device().synchronize()
            end_time = time.time()
            cpu_to_gpu_bw = (size * 4) / (end_time - start_time) / 1e9  # GB/s

            # メモリ解放
            del gpu_array, gpu_array2, cpu_array
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

            return {
                "gpu_to_cpu_bandwidth": gpu_to_cpu_bw,
                "cpu_to_gpu_bandwidth": cpu_to_gpu_bw,
                "data_size_mb": size_mb,
            }
        except Exception as e:
            # エラー時もメモリをクリア
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except:
                pass
            print(f"メモリ帯域幅テストエラー: {e}")
            return None

    def run_all_benchmarks(
        self, matrix_size=2000, iterations=1, memory_size=100, progress_callback=None
    ):
        """全ベンチマークを実行"""
        results = {}

        tests = [
            ("GPU情報取得", self.get_gpu_info),
            ("CPU行列乗算", lambda: self.cpu_matrix_multiply(matrix_size, iterations)),
            ("GPU行列乗算 (CuPy)", lambda: self.gpu_matrix_multiply_cupy(matrix_size, iterations)),
            (
                "GPU行列乗算 (PyTorch)",
                lambda: self.gpu_matrix_multiply_torch(matrix_size, iterations),
            ),
            ("メモリ帯域幅", lambda: self.memory_bandwidth_test(memory_size)),
        ]

        total_tests = len(tests)

        for i, (test_name, test_func) in enumerate(tests):
            if progress_callback:
                progress_callback(test_name, int((i / total_tests) * 100))

            try:
                result = test_func()
                results[test_name] = result

                # 各テスト後にメモリクリーンアップ
                if "GPU" in test_name:
                    self.cleanup_gpu_memory()

                time.sleep(0.5)  # 視覚的な進捗のため
            except Exception as e:
                results[test_name] = f"エラー: {str(e)}"
                # エラー時もメモリクリーンアップ
                self.cleanup_gpu_memory()

        # 最終メモリクリーンアップ
        self.cleanup_gpu_memory()

        if progress_callback:
            progress_callback("完了", 100)

        return results

    def run_cuda_cpp_benchmark(
        self, benchmark_type="all", matrix_size=2000, iterations=1, progress_callback=None
    ):
        """
        Run CUDA C++ benchmarks

        Args:
            benchmark_type: Type of benchmark ('cpu_single', 'cpu_optimized', 'cpu_openblas',
                           'cuda_naive', 'cublas', 'cublas_tensorcore', 'wmma', 'all')
            matrix_size: Size of square matrices
            iterations: Number of iterations to average
            progress_callback: Optional callback function for progress updates

        Returns:
            Dict containing benchmark results
        """
        if not self.cuda_lib:
            return {"error": "CUDA C++ library not available"}

        try:
            if progress_callback:
                progress_callback(f"CUDA C++ {benchmark_type}", 0)

            # For large matrices and CPU benchmarks, use gpu_only mode
            gpu_only = matrix_size > 1024 and benchmark_type == "all"

            if benchmark_type == "all":
                result = self.cuda_lib.run_all_benchmarks(matrix_size, iterations, gpu_only)
            else:
                result = run_benchmark(benchmark_type, matrix_size, iterations)

            if progress_callback:
                progress_callback(f"CUDA C++ {benchmark_type}", 100)

            if not result.success:
                return {"error": result.error_message}

            # Convert to dict and add analysis
            result_dict = result.to_dict()
            result_dict["speedup_analysis"] = result.get_speedup_analysis()

            return result_dict

        except Exception as e:
            return {"error": f"CUDA C++ benchmark failed: {str(e)}"}

    def run_individual_cuda_cpp_benchmark(
        self, benchmark_type, matrix_size, iterations=1, progress_callback=None
    ):
        """
        Run individual CUDA C++ benchmark for finer control

        Args:
            benchmark_type: Specific benchmark type
            matrix_size: Matrix size
            iterations: Number of iterations
            progress_callback: Progress callback function

        Returns:
            Dict with benchmark result
        """
        return self.run_cuda_cpp_benchmark(
            benchmark_type, matrix_size, iterations, progress_callback
        )

    def cleanup_gpu_memory(self):
        """GPUメモリのクリーンアップ"""
        try:
            # CuPyメモリプールのクリア
            if CUDA_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                cp.cuda.Device().synchronize()
        except Exception as e:
            print(f"CuPyメモリクリーンアップエラー: {e}")

        try:
            # PyTorchキャッシュのクリア
            if PYTORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            print(f"PyTorchメモリクリーンアップエラー: {e}")

        # 少し待機してメモリ解放を確実にする
        time.sleep(0.1)


# 利用可能性をエクスポート
def get_availability():
    """利用可能なベンチマーク機能を確認"""
    availability = {
        "cuda_available": CUDA_AVAILABLE,
        "pytorch_available": PYTORCH_AVAILABLE,
        "cuda_cpp_available": CUDA_CPP_AVAILABLE,
        "gpu_devices": [],
    }

    # GPU情報を取得
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            availability["gpu_devices"].append(
                {"id": gpu.id, "name": gpu.name, "memory_total": gpu.memoryTotal}
            )
    except Exception as e:
        print(f"GPU情報取得エラー: {e}")

    # CUDA C++ library GPU check
    if CUDA_CPP_AVAILABLE:
        try:
            availability["cuda_cpp_gpu_available"] = check_gpu_availability()
        except Exception:
            availability["cuda_cpp_gpu_available"] = False
    else:
        availability["cuda_cpp_gpu_available"] = False

    return availability
