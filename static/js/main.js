let benchmarkInterval;

function loadSystemInfo() {
    fetch('/system_info')
        .then(response => response.json())
        .then(data => {
            document.getElementById('system-info').innerHTML = `
                <div class="metric">
                    <span>ホスト名:</span>
                    <span class="metric-value">${data.host_name}</span>
                </div>
                <div class="metric">
                    <span>CPUコア数:</span>
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

function cleanupGPUMemory() {
    fetch('/cleanup_gpu_memory', {method: 'POST'})
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                alert(data.message);
                loadGPUInfo(); // GPU情報を更新
            }
        })
        .catch(error => {
            alert('メモリクリーンアップに失敗しました');
        });
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

// 初期化処理
document.addEventListener('DOMContentLoaded', function() {
    loadSystemInfo();
    loadGPUInfo();

    // フォームのデフォルト送信を防ぐ
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
        });
    }

    // エンターキーでのベンチマーク開始
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            const startBtn = document.getElementById('start-btn');
            if (startBtn && !startBtn.disabled) {
                startBenchmark();
            }
        }
    });
});
