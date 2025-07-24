import argparse
import socket
import threading

import psutil
from flask import Flask, jsonify, render_template, request
from waitress import serve

from benchmark import GPUBenchmark, get_availability

app = Flask(__name__)

# グローバル変数
benchmark_results = {}
is_running = False
current_test = ""
progress = 0

# ベンチマークインスタンス
gpu_benchmark = GPUBenchmark()


def progress_callback(test_name, progress_value):
    """ベンチマーク進捗のコールバック"""
    global current_test, progress
    current_test = test_name
    progress = progress_value


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_benchmark", methods=["POST"])
def start_benchmark():
    global is_running, benchmark_results

    if is_running:
        return jsonify({"error": "ベンチマークは既に実行中です"})

    # パラメータを取得
    data = request.get_json() or {}
    matrix_size = data.get("matrix_size", 2000)
    iterations = data.get("iterations", 1)
    memory_size = data.get("memory_size", 100)

    # パラメータ検証
    try:
        matrix_size = int(matrix_size)
        iterations = int(iterations)
        memory_size = int(memory_size)

        if matrix_size < 100 or matrix_size > 20000:
            return jsonify({"error": "行列サイズは100-20000の範囲で設定してください"})
        if iterations < 1 or iterations > 10:
            return jsonify({"error": "実行回数は1-10の範囲で設定してください"})
        if memory_size < 10 or memory_size > 2000:
            return jsonify({"error": "メモリサイズは10-2000MBの範囲で設定してください"})

    except ValueError:
        return jsonify({"error": "無効なパラメータです"})

    def run_benchmark():
        global is_running, benchmark_results
        is_running = True
        benchmark_results = gpu_benchmark.run_all_benchmarks(
            matrix_size, iterations, memory_size, progress_callback
        )
        is_running = False

    # 別スレッドでベンチマークを実行
    thread = threading.Thread(target=run_benchmark)
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "message": f"ベンチマークを開始しました (行列:{matrix_size}x{matrix_size}, 回数:{iterations}, メモリ:{memory_size}MB)"
        }
    )


@app.route("/benchmark_status")
def benchmark_status():
    return jsonify(
        {
            "is_running": is_running,
            "current_test": current_test,
            "progress": progress,
            "results": benchmark_results,
        }
    )


@app.route("/gpu_info")
def gpu_info():
    return jsonify(gpu_benchmark.get_gpu_info())


@app.route("/cleanup_gpu_memory", methods=["POST"])
def cleanup_gpu_memory():
    """手動でGPUメモリをクリーンアップ"""
    try:
        gpu_benchmark.cleanup_gpu_memory()
        return jsonify({"message": "GPUメモリをクリーンアップしました"})
    except Exception as e:
        return jsonify({"error": f"メモリクリーンアップエラー: {str(e)}"})


@app.route("/system_info")
def system_info():
    availability = get_availability()
    return jsonify(
        {
            "host_name": socket.gethostname(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            **availability,
        }
    )


def get_args():
    parser = argparse.ArgumentParser(description="GPU Performance Benchmark WebApp")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the app in debug mode (default: False)",
    )
    return parser.parse_args()


def show_dev_info():
    """開発者向け情報を表示"""
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


if __name__ == "__main__":
    host_num = "0.0.0.0"
    port_num = 5000

    show_dev_info()

    print("アプリケーションを起動中...")
    print(f"ブラウザで http://localhost:{port_num} にアクセスしてください")

    args = get_args()
    if args.debug is True:
        app.run(debug=True, host=host_num, port=port_num)
    else:
        serve(app, host=host_num, port=port_num)
