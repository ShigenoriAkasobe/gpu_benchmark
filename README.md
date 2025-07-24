# GPU Benchmark App

## Setup
```sh
pip install -r requirements.txt
```

## Run
- For live
```sh
python app.py
```
- For debug
```sh
python app.py --debug
```

## Register app to systemd
- Create a systemd service file for the Flask app.
```sh
sudo vi /etc/systemd/system/flaskapp-gpu_benchmark.service
```
- Then, write the following.
```ini
[Unit]
Description=Flask Web Application for GPU Benchmark
After=network.target

[Service]
User=your-user-name
WorkingDirectory=/home/your-user-name/flask-app
ExecStart=/home/your-user-name/start_flask.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
- Register app to systemd by script.
```sh
cd script
./register_app_to_systemd.sh
```
- Restart service.
```sh
sudo systemctl restart flaskapp-gpu_benchmark.service
```
- Check status.
```sh
sudo systemctl status flaskapp-gpu_benchmark.service
```
```sh
journalctl -u flaskapp-gpu_benchmark.service -f
```
