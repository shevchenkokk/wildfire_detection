docker build --tag=wildfire-detection-app .
docker run -p 8501:8501 --gpus=all --network==host wildfire-detection-app
