import streamlit as st
import numpy as np
import ultralytics
import torch
import torch.nn as nn
import cv2
import argparse

from PIL import Image


torch.classes.__path__ = [] # add this line to manually set it to empty. 

# parser = argparse.ArgumentParser()
# parser.add_argument("--path-to-weights", type=str, help="Path to model weights")
# args = parser.parse_args()
path_to_weights = "weights/yolo/best.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ultralytics.YOLO(path_to_weights).to(device)


def draw_bboxes(model: nn.Module, img: np.ndarray):
    with torch.no_grad():
        results = model(img)
    for result in results:
        for box in result.boxes:
            bbox_coords = box.xyxy.cpu().numpy()[0]
            confidence = float(box.conf.cpu())
            label = model.names[int(box.cls[0])]

            left, top, right, bottom = bbox_coords
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)

            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, f"{label}: {confidence:.3f}", (left, bottom+20), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)


st.title("Поиск лесных пожаров по фотографии со спутника")

image_input = st.file_uploader(
    "Загрузить фотографию со спутникового снимка",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)


if st.button("Получить детекции лесных пожаров"):
    if image_input:
        img = np.array(Image.open(image_input))
        draw_bboxes(model, img)
        st.image(img, caption="Загруженное изображение", use_container_width=True)
