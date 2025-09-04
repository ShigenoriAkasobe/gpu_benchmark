# CUDA C++ Benchmark Library セットアップガイド

## 概要

このガイドでは、既存のPython GPU benchmarkアプリケーションにCUDA C++ベンチマークライブラリを統合する方法を説明します。これにより、Pythonから高性能なCUDA C++ベンチマークを呼び出すことができるようになります。

## 前提条件

1. **CUDA Toolkit** (11.0以上推奨)
2. **NVIDIA GPU** (Compute Capability 7.5以上推奨)
3. **OpenBLAS** または **Intel MKL**
4. **OpenMP対応コンパイラ** (gcc, clang)
5. **Python 3.7以上**

## 依存関係のインストール

### Ubuntu/Debian

```bash
# CUDA開発パッケージ
sudo apt update
sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev

# OpenBLASとOpenMP
sudo apt install libopenblas-dev libomp-dev

# ビルドツール
sudo apt install build-essential

# Python依存関係
pip install numpy flask waitress GPUtil psutil
```

### CentOS/RHEL

```bash
# CUDA開発パッケージ（NVIDIA公式リポジトリから）
sudo yum install cuda-toolkit cuda-devel

# OpenBLAS
sudo yum install openblas-devel

# ビルドツール
sudo yum groupinstall "Development Tools"

# Python依存関係
pip install numpy flask waitress GPUtil psutil
```

## ビルド手順

### 1. 共有ライブラリのビルド

```bash
cd cpp
make -f Makefile.lib
```

これにより以下が作成されます：
- `libbenchmark.so` - 共有ライブラリ
- `../lib/libbenchmark.so` - インストール済みライブラリ
- `../lib/benchmark_lib.h` - Pythonで使用するヘッダファイル

### 2. GPU自動検出の確認

```bash
# GPU情報とビルド設定を確認
make -f Makefile.lib check-cuda

# 簡単なテスト実行
make -f Makefile.lib test-lib
```

### 3. システム全体へのインストール（オプション）

```bash
# システム全体にライブラリをインストール（sudo権限必要）
sudo make -f Makefile.lib install-system
```

## 使用方法

### 1. Pythonからの基本的な使用

```python
from cuda_benchmark import CudaBenchmarkLibrary, run_benchmark

# ライブラリの初期化
lib = CudaBenchmarkLibrary()

# GPU利用可能性の確認
if lib.check_gpu_availability():
    print("GPU利用可能")
else:
    print("GPU利用不可")

# 個別ベンチマークの実行
result = lib.run_cpu_single_core_benchmark(1024, 3)
if result.success:
    print(f"CPU single-core: {result.cpu_single_core_ms:.2f} ms")

# 全ベンチマークの実行
result = lib.run_all_benchmarks(2048, 1, gpu_only=True)
if result.success:
    print("全ベンチマーク完了")
    analysis = result.get_speedup_analysis()
    for comparison, ratio in analysis.items():
        print(f"{comparison}: {ratio:.2f}x")
```

### 2. Webアプリケーションからの使用

1. **Flask アプリケーションの起動**:
   ```bash
   python app.py
   ```

2. **ブラウザアクセス**: `http://localhost:5000`

3. **ベンチマーク設定**:
   - 行列サイズ: 100-20000 (WMMAは16の倍数)
   - 反復回数: 1-10
   - ベンチマークタイプ選択

4. **実行**:
   - "CUDA C++ ベンチマーク" ボタンをクリック
   - または従来の "ベンチマーク開始" を選択

## ベンチマーク種類

| タイプ | 説明 |
|--------|------|
| `cpu_single` | CPU シングルコア（基準） |
| `cpu_optimized` | CPU 最適化（OpenMP + AVX + cache blocking） |
| `cpu_openblas` | CPU OpenBLAS（業界標準BLAS） |
| `cuda_naive` | CUDA ナイーブ実装 |
| `cublas` | cuBLAS最適化 |
| `cublas_tensorcore` | cuBLAS + Tensor Core |
| `wmma` | WMMA Tensor Core手動実装 |
| `all` | 全ベンチマーク比較 |

## パフォーマンス最適化

### 行列サイズの推奨事項

- **小規模テスト** (512x512): 全CPUベンチマーク含む
- **中規模テスト** (1024-2048): バランスの取れた比較
- **大規模テスト** (4096+): GPU専用モード推奨
- **WMMA使用時**: 16の倍数必須

### GPU専用モードの使用

大きな行列サイズではCPUベンチマークが非常に遅くなるため、GPU専用モードを使用：

```python
# 大きな行列ではGPU専用モードを自動使用
result = lib.run_all_benchmarks(4096, 1, gpu_only=True)
```

## トラブルシューティング

### 1. ライブラリが見つからない

```bash
# ライブラリの場所を確認
find . -name "libbenchmark.so"

# 環境変数の設定
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/lib"
```

### 2. CUDA エラー

```bash
# GPU情報の確認
nvidia-smi

# CUDA インストールの確認
nvcc --version

# コンパイル時のGPU設定確認
make -f Makefile.lib show-gpu-info
```

### 3. OpenBLAS エラー

```bash
# OpenBLASのインストール確認
ldconfig -p | grep blas

# 代替インストール方法
conda install openblas
# または
sudo apt install libatlas-base-dev
```

### 4. メモリエラー

- 行列サイズを削減
- GPU専用モードの使用
- 反復回数の削減

### 5. 性能が期待値より低い

- CPU周波数の確認（CPUテスト）
- GPU使用率の確認（`nvidia-smi`）
- 他のプロセスによるリソース使用の確認

## 高度な使用方法

### カスタムライブラリパスの指定

```python
from cuda_benchmark import CudaBenchmarkLibrary

# カスタムパスでライブラリを初期化
lib = CudaBenchmarkLibrary("/path/to/custom/libbenchmark.so")
```

### 結果の詳細分析

```python
result = lib.run_all_benchmarks(2048, 3)
if result.success:
    # 辞書形式で取得
    data = result.to_dict()
    
    # 詳細な性能分析
    analysis = result.get_speedup_analysis()
    
    # 最速の手法を特定
    times = {k: v for k, v in data.items() if k.endswith('_ms') and v > 0}
    fastest = min(times, key=times.get)
    print(f"最速: {fastest} ({times[fastest]:.2f} ms)")
```

### プログレス監視

```python
def progress_callback(test_name, progress):
    print(f"{test_name}: {progress}%")

# コールバック付きでベンチマーク実行
# 注意: 現在のC++ライブラリではコールバックは直接サポートされていません
# Webアプリケーション側で実装
```

## アーキテクチャ図

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │  Flask Backend  │    │  CUDA C++ Lib  │
│   (JavaScript)  │◄──►│    (Python)     │◄──►│  (ctypes/API)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Traditional     │    │ High Performance│
                       │ Benchmarks      │    │ CUDA Kernels    │
                       │ (CuPy/PyTorch)  │    │ (cuBLAS/WMMA)   │
                       └─────────────────┘    └─────────────────┘
```

## ライセンスと謝辞

このプロジェクトはMITライセンスの下で提供されています。

使用ライブラリ：
- NVIDIA CUDA Toolkit
- OpenBLAS
- Flask
- その他Python依存関係

## 参考資料

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [WMMA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [OpenBLAS Documentation](https://github.com/xianyi/OpenBLAS)
