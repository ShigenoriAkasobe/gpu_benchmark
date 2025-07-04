import os
import time
import json
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request
import numpy as np
import psutil
import GPUtil

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

app = Flask(__name__)

# グローバル変数
benchmark_results = {}
is_running = False
current_test = ""
progress = 0

class GPUBenchmark:
    def __init__(self):
        self.results = {}
        
    def get_gpu_info(self):
        """GPU情報を取得"""
        gpu_info = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                info = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'driver': gpu.driver,
                    'memory_total': gpu.memoryTotal,
                    'memory_free': gpu.memoryFree,
                    'memory_used': gpu.memoryUsed,
                    'temperature': gpu.temperature,
                    'load': gpu.load
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
            
            total_time += (end_time - start_time)
            total_ops += 2.0 * size * size * size
        
        avg_time = total_time / iterations
        avg_gflops = total_ops / (total_time * 1e9)
        
        return {
            'time': avg_time,
            'total_time': total_time,
            'gflops': avg_gflops,
            'matrix_size': size,
            'iterations': iterations
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
                
                total_time += (end_time - start_time)
                total_ops += 2.0 * size * size * size
            
            avg_time = total_time / iterations
            avg_gflops = total_ops / (total_time * 1e9)
            
            return {
                'time': avg_time,
                'total_time': total_time,
                'gflops': avg_gflops,
                'matrix_size': size,
                'iterations': iterations
            }
        except Exception as e:
            print(f"CuPy GPU テストエラー: {e}")
            return None
    
    def gpu_matrix_multiply_torch(self, size=2000, iterations=1):
        """PyTorchを使用したGPU行列乗算ベンチマーク"""
        if not PYTORCH_AVAILABLE:
            return None
            
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cpu':
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
                
                total_time += (end_time - start_time)
                total_ops += 2.0 * size * size * size
            
            avg_time = total_time / iterations
            avg_gflops = total_ops / (total_time * 1e9)
            
            return {
                'time': avg_time,
                'total_time': total_time,
                'gflops': avg_gflops,
                'matrix_size': size,
                'iterations': iterations
            }
        except Exception as e:
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
            gpu_array = cp.asarray(cpu_array)
            cp.cuda.Device().synchronize()
            end_time = time.time()
            cpu_to_gpu_bw = (size * 4) / (end_time - start_time) / 1e9  # GB/s
            
            return {
                'gpu_to_cpu_bandwidth': gpu_to_cpu_bw,
                'cpu_to_gpu_bandwidth': cpu_to_gpu_bw,
                'data_size_mb': size_mb
            }
        except Exception as e:
            print(f"メモリ帯域幅テストエラー: {e}")
            return None
    
    def run_all_benchmarks(self, matrix_size=2000, iterations=1, memory_size=100):
        """全ベンチマークを実行"""
        global is_running, current_test, progress, benchmark_results
        
        is_running = True
        progress = 0
        benchmark_results = {}
        
        tests = [
            ("GPU情報取得", self.get_gpu_info),
            ("CPU行列乗算", lambda: self.cpu_matrix_multiply(matrix_size, iterations)),
            ("GPU行列乗算 (CuPy)", lambda: self.gpu_matrix_multiply_cupy(matrix_size, iterations)),
            ("GPU行列乗算 (PyTorch)", lambda: self.gpu_matrix_multiply_torch(matrix_size, iterations)),
            ("メモリ帯域幅", lambda: self.memory_bandwidth_test(memory_size)),
        ]
        
        total_tests = len(tests)
        
        for i, (test_name, test_func) in enumerate(tests):
            current_test = test_name
            progress = int((i / total_tests) * 100)
            
            try:
                result = test_func()
                benchmark_results[test_name] = result
                time.sleep(0.5)  # 視覚的な進捗のため
            except Exception as e:
                benchmark_results[test_name] = f"エラー: {str(e)}"
        
        progress = 100
        is_running = False
        current_test = "完了"

# ベンチマークインスタンス
gpu_benchmark = GPUBenchmark()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_benchmark', methods=['POST'])
def start_benchmark():
    global is_running
    
    if is_running:
        return jsonify({'error': 'ベンチマークは既に実行中です'})
    
    # パラメータを取得
    data = request.get_json() or {}
    matrix_size = data.get('matrix_size', 2000)
    iterations = data.get('iterations', 1)
    memory_size = data.get('memory_size', 100)
    
    # パラメータ検証
    try:
        matrix_size = int(matrix_size)
        iterations = int(iterations)
        memory_size = int(memory_size)
        
        if matrix_size < 100 or matrix_size > 20000:
            return jsonify({'error': '行列サイズは100-20000の範囲で設定してください'})
        if iterations < 1 or iterations > 10:
            return jsonify({'error': '実行回数は1-10の範囲で設定してください'})
        if memory_size < 10 or memory_size > 2000:
            return jsonify({'error': 'メモリサイズは10-2000MBの範囲で設定してください'})
            
    except ValueError:
        return jsonify({'error': '無効なパラメータです'})
    
    # 別スレッドでベンチマークを実行
    thread = threading.Thread(target=gpu_benchmark.run_all_benchmarks, args=(matrix_size, iterations, memory_size))
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': f'ベンチマークを開始しました (行列:{matrix_size}x{matrix_size}, 回数:{iterations}, メモリ:{memory_size}MB)'})

@app.route('/benchmark_status')
def benchmark_status():
    return jsonify({
        'is_running': is_running,
        'current_test': current_test,
        'progress': progress,
        'results': benchmark_results
    })

@app.route('/gpu_info')
def gpu_info():
    return jsonify(gpu_benchmark.get_gpu_info())

@app.route('/system_info')
def system_info():
    return jsonify({
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'cuda_available': CUDA_AVAILABLE,
        'pytorch_available': PYTORCH_AVAILABLE,
        'torch_cuda_available': torch.cuda.is_available() if PYTORCH_AVAILABLE else False
    })

# HTMLテンプレート
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Performance Benchmark</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #4a5568;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .info-section {
            background: #f7fafc;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #4299e1;
        }
        .benchmark-section {
            background: #fff5f5;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #f56565;
        }
        button {
            background: linear-gradient(45deg, #4299e1, #667eea);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            margin: 10px 5px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        button:disabled {
            background: #a0aec0;
            cursor: not-allowed;
            transform: none;
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e2e8f0;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(45deg, #48bb78, #38a169);
            transition: width 0.3s;
            border-radius: 15px;
        }
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .result-card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .result-card h3 {
            color: #2d3748;
            margin-top: 0;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #f1f5f9;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-value {
            font-weight: bold;
            color: #4299e1;
        }
        .status {
            text-align: center;
            font-size: 18px;
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            background: #f0fff4;
            border: 1px solid #9ae6b4;
        }
        .error {
            background: #fed7d7;
            border: 1px solid #feb2b2;
            color: #c53030;
        }
        .settings-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .setting-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        .setting-item label {
            display: block;
            font-weight: bold;
            margin-bottom: 5px;
            color: #495057;
        }
        .setting-item input {
            width: 100%;
            padding: 8px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
        }
        .setting-item small {
            display: block;
            margin-top: 5px;
            color: #6c757d;
            font-size: 12px;
        }
        .preset-buttons {
            margin-bottom: 20px;
        }
        .preset-buttons button {
            background: linear-gradient(45deg, #28a745, #20c997);
            margin-right: 10px;
        }
        .preset-buttons button:hover {
            background: linear-gradient(45deg, #218838, #1ea085);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 GPU Performance Benchmark</h1>
        
        <div class="info-section">
            <h2>システム情報</h2>
            <div id="system-info">
                <p>システム情報を読み込み中...</p>
            </div>
        </div>
        
        <div class="info-section">
            <h2>GPU情報</h2>
            <div id="gpu-info">
                <p>GPU情報を読み込み中...</p>
            </div>
        </div>
        
        <div class="benchmark-section">
            <h2>ベンチマーク設定</h2>
            
            <div class="settings-grid">
                <div class="setting-item">
                    <label for="matrix-size">行列サイズ (N×N):</label>
                    <input type="number" id="matrix-size" value="4000" min="100" max="20000" step="100">
                    <small>大きいほど負荷が高い (推奨: CPU 2000-4000, GPU 4000-8000)</small>
                </div>
                
                <div class="setting-item">
                    <label for="iterations">実行回数:</label>
                    <input type="number" id="iterations" value="3" min="1" max="10">
                    <small>複数回実行して平均値を取る</small>
                </div>
                
                <div class="setting-item">
                    <label for="memory-size">メモリテストサイズ (MB):</label>
                    <input type="number" id="memory-size" value="500" min="10" max="2000" step="10">
                    <small>メモリ帯域幅テストのデータサイズ</small>
                </div>
            </div>
            
            <div class="preset-buttons">
                <button onclick="setPreset('light')">軽量テスト</button>
                <button onclick="setPreset('medium')">標準テスト</button>
                <button onclick="setPreset('heavy')">重負荷テスト</button>
            </div>
            
            <button id="start-btn" onclick="startBenchmark()">ベンチマーク開始</button>
            <button onclick="loadSystemInfo()">システム情報更新</button>
            
            <div id="progress-section" style="display: none;">
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill"></div>
                </div>
                <div id="current-test" class="status">準備中...</div>
            </div>
            
            <div id="results-section">
                <div class="results-grid" id="results-grid"></div>
            </div>
        </div>
    </div>

    <script>
        let benchmarkInterval;
        
        function loadSystemInfo() {
            fetch('/system_info')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('system-info').innerHTML = `
                        <div class="metric">
                            <span>CPU コア数:</span>
                            <span class="metric-value">${data.cpu_count}</span>
                        </div>
                        <div class="metric">
                            <span>総メモリ:</span>
                            <span class="metric-value">${(data.memory_total / 1024 / 1024 / 1024).toFixed(2)} GB</span>
                        </div>
                        <div class="metric">
                            <span>利用可能メモリ:</span>
                            <span class="metric-value">${(data.memory_available / 1024 / 1024 / 1024).toFixed(2)} GB</span>
                        </div>
                        <div class="metric">
                            <span>CUDA利用可能:</span>
                            <span class="metric-value">${data.cuda_available ? 'はい' : 'いいえ'}</span>
                        </div>
                        <div class="metric">
                            <span>PyTorch利用可能:</span>
                            <span class="metric-value">${data.pytorch_available ? 'はい' : 'いいえ'}</span>
                        </div>
                        <div class="metric">
                            <span>PyTorch CUDA:</span>
                            <span class="metric-value">${data.torch_cuda_available ? 'はい' : 'いいえ'}</span>
                        </div>
                    `;
                })
                .catch(error => {
                    document.getElementById('system-info').innerHTML = '<p class="error">システム情報の取得に失敗しました</p>';
                });
        }
        
        function loadGPUInfo() {
            fetch('/gpu_info')
                .then(response => response.json())
                .then(data => {
                    if (data.length === 0) {
                        document.getElementById('gpu-info').innerHTML = '<p>GPUが検出されませんでした</p>';
                        return;
                    }
                    
                    let html = '';
                    data.forEach(gpu => {
                        html += `
                            <div class="result-card">
                                <h3>GPU ${gpu.id}: ${gpu.name}</h3>
                                <div class="metric">
                                    <span>ドライバー:</span>
                                    <span class="metric-value">${gpu.driver}</span>
                                </div>
                                <div class="metric">
                                    <span>総メモリ:</span>
                                    <span class="metric-value">${gpu.memory_total} MB</span>
                                </div>
                                <div class="metric">
                                    <span>使用メモリ:</span>
                                    <span class="metric-value">${gpu.memory_used} MB</span>
                                </div>
                                <div class="metric">
                                    <span>空きメモリ:</span>
                                    <span class="metric-value">${gpu.memory_free} MB</span>
                                </div>
                                <div class="metric">
                                    <span>温度:</span>
                                    <span class="metric-value">${gpu.temperature}°C</span>
                                </div>
                                <div class="metric">
                                    <span>使用率:</span>
                                    <span class="metric-value">${(gpu.load * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        `;
                    });
                    document.getElementById('gpu-info').innerHTML = html;
                })
                .catch(error => {
                    document.getElementById('gpu-info').innerHTML = '<p class="error">GPU情報の取得に失敗しました</p>';
                });
        }
        
        function startBenchmark() {
            const matrixSize = parseInt(document.getElementById('matrix-size').value);
            const iterations = parseInt(document.getElementById('iterations').value);
            const memorySize = parseInt(document.getElementById('memory-size').value);
            
            const requestData = {
                matrix_size: matrixSize,
                iterations: iterations,
                memory_size: memorySize
            };
            
            fetch('/start_benchmark', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert(data.error);
                        return;
                    }
                    
                    document.getElementById('start-btn').disabled = true;
                    document.getElementById('progress-section').style.display = 'block';
                    document.getElementById('results-grid').innerHTML = '';
                    
                    benchmarkInterval = setInterval(updateBenchmarkStatus, 1000);
                });
        }
        
        function setPreset(preset) {
            const matrixSizeInput = document.getElementById('matrix-size');
            const iterationsInput = document.getElementById('iterations');
            const memorySizeInput = document.getElementById('memory-size');
            
            switch(preset) {
                case 'light':
                    matrixSizeInput.value = '1000';
                    iterationsInput.value = '1';
                    memorySizeInput.value = '100';
                    break;
                case 'medium':
                    matrixSizeInput.value = '4000';
                    iterationsInput.value = '3';
                    memorySizeInput.value = '500';
                    break;
                case 'heavy':
                    matrixSizeInput.value = '8000';
                    iterationsInput.value = '5';
                    memorySizeInput.value = '1000';
                    break;
            }
        }
        
        function updateBenchmarkStatus() {
            fetch('/benchmark_status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('progress-fill').style.width = data.progress + '%';
                    document.getElementById('current-test').textContent = data.current_test;
                    
                    if (!data.is_running) {
                        clearInterval(benchmarkInterval);
                        document.getElementById('start-btn').disabled = false;
                        document.getElementById('progress-section').style.display = 'none';
                        displayResults(data.results);
                    }
                });
        }
        
        function displayResults(results) {
            let html = '';
            
            for (const [testName, result] of Object.entries(results)) {
                if (typeof result === 'string') {
                    html += `
                        <div class="result-card">
                            <h3>${testName}</h3>
                            <p class="error">${result}</p>
                        </div>
                    `;
                } else if (Array.isArray(result)) {
                    // GPU情報の場合
                    continue; // 既に表示済み
                } else if (result && typeof result === 'object') {
                    html += `
                        <div class="result-card">
                            <h3>${testName}</h3>
                    `;
                    
                    for (const [key, value] of Object.entries(result)) {
                        let displayValue = value;
                        if (typeof value === 'number') {
                            displayValue = value.toFixed(2);
                        }
                        html += `
                            <div class="metric">
                                <span>${key}:</span>
                                <span class="metric-value">${displayValue}</span>
                            </div>
                        `;
                    }
                    html += '</div>';
                }
            }
            
            document.getElementById('results-grid').innerHTML = html;
        }
        
        // 初期化
        loadSystemInfo();
        loadGPUInfo();
    </script>
</body>
</html>
'''

# HTMLテンプレートをファイルに保存
def create_template_file():
    template_dir = 'templates'
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    with open(os.path.join(template_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)

if __name__ == '__main__':
    create_template_file()
    print("GPU Performance Benchmark WebApp")
    print("=" * 50)
    print("必要なライブラリ:")
    print("- Flask")
    print("- numpy")
    print("- psutil")
    print("- GPUtil")
    print("- cupy (CUDA GPU用、オプション)")
    print("- torch (PyTorch GPU用、オプション)")
    print("=" * 50)
    print("アプリケーションを起動中...")
    print("ブラウザで http://localhost:5000 にアクセスしてください")
    app.run(debug=True, host='0.0.0.0', port=5000)
