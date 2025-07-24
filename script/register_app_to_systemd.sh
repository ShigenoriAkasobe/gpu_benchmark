#!/bin/bash
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable flaskapp-gpu_benchmark.service
sudo systemctl start flaskapp-gpu_benchmark.service
