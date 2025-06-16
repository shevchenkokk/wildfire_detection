from super_image import EdsrModel, ImageLoader
from PIL import Image
import requests
import os
import torch.nn as nn
import shutil
from torchvision.transforms.functional import to_pil_image

SCALE_FACTOR = 3
TARGET_IMAGE_SIZE = (768, 768)



def resize_image(upscale_model: nn.Module, path_to_image: str):
    image = Image.open(path_to_image)
    if image.size == (256, 256):
        inputs = ImageLoader.load_image(image)
        preds = upscale_model(inputs)
        image = to_pil_image(preds.squeeze()).convert("RGB")
    image.save(path_to_image.replace("merged_wildfire_dataset", "upscaled_merged_wildfire_dataset"))

    

def main():
    path_to_dataset = "data/merged_wildfire_dataset"
    upscale_model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=SCALE_FACTOR)
    for folder in ("train", "valid", "test"):
        path_to_images = os.path.join(path_to_dataset, folder)
        print(path_to_images)
        os.makedirs(path_to_images.replace("merged_wildfire_dataset", "upscaled_merged_wildfire_dataset"), exist_ok=True)
        for image_or_label in os.listdir(path_to_images):
            if not image_or_label.endswith(".jpg"):
                continue
            resize_image(upscale_model, os.path.join(path_to_images, image_or_label))
            path_to_annotations = os.path.join(path_to_images, image_or_label.replace(".jpg", ".txt"))
            if os.path.exists(path_to_annotations):
                shutil.copy(
                    path_to_annotations,
                    path_to_annotations.replace("merged_wildfire_dataset", "upscaled_merged_wildfire_dataset"),
                )


if __name__ == "__main__":
    exit(main())
