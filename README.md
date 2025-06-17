# Wildfire Satellite Image Detection

This repository contains a code for wildfire detection task on satellite images.

## Project structure 

- `notebooks` - folder contains jupyter notebooks for this project
- `scripts` - folder contains scripts for this project such as download dataset and etc.
- `src` - folder contains main code for this project

## Training models 

1. First step create virtual enviroment and download requirements for this project. Run the following commands: 

```bash
python3 -m venv venv # or conda create --name wildfire-detection
source venv/bin/activate # or conda activate wildfire-detection
pip install -r requirements.txt
```

2. Donwload dataset for this project

To download data you need to log in to [roboflow]( https://app.roboflow.com/). Next go to account settings -> workspace -> API keys, copy your API key, create a .env file in the root of the project and enter

```bash
ROBOFLOW_API_KEY=<YOUR_API_KEY>
```

Next, run the script to download the data in coco format

```bash
python3 scripts/download.py
```

Optionally if you need data in yolo format. Run

```bash
python3 scripts/convert_coco_to_yolo.py
```

3. Run script for training model


## Application

Go to the following path `src/app/`

Run script 

```bash
./run_app.sh
```

This script create web app for this project available on localhost:8501.