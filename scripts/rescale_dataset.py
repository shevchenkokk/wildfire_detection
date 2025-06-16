import os
import json
from PIL import Image
from tqdm import tqdm
from super_image import EdsrModel, ImageLoader
from torchvision.transforms.functional import to_pil_image

# huggingface-hub==0.25.2 transformers==4.44.2

SCALE_FACTOR = 3

INPUT_DATASET_DIR = "data/merged_wildfire_dataset"
OUTPUT_DATASET_DIR = "data/upscaled_merged_wildfire_dataset"


def process_split(split_name: str, upscale_model):
    input_split_dir = os.path.join(INPUT_DATASET_DIR, split_name)
    output_split_dir = os.path.join(OUTPUT_DATASET_DIR, split_name)
    input_ann_file = os.path.join(input_split_dir, "_annotations.coco.json")
    output_ann_file = os.path.join(output_split_dir, "_annotations.coco.json")

    if not os.path.exists(input_ann_file):
        print(f"annotation file not found for split {split_name}")
        return

    os.makedirs(output_split_dir, exist_ok=True)

    with open(input_ann_file, 'r') as f:
        coco_data = json.load(f)

    new_coco_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data["categories"],
        "images": [],
        "annotations": []
    }

    for image_info in tqdm(coco_data['images']):
        image_path = os.path.join(input_split_dir, image_info['file_name'])
        output_image_path = os.path.join(output_split_dir, image_info['file_name'])

        if not os.path.exists(image_path):
            print(f"image file not found {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")

        inputs = ImageLoader.load_image(image)
        preds = upscale_model(inputs)
        upscaled_image = to_pil_image(preds.squeeze()).convert("RGB")

        upscaled_image.save(output_image_path)

        new_image_info = image_info.copy()
        new_image_info['width'] = upscaled_image.width
        new_image_info['height'] = upscaled_image.height
        new_coco_data['images'].append(new_image_info)

    if 'annotations' in coco_data:
        for ann in tqdm(coco_data['annotations']):
            new_ann = ann.copy()

            bbox = new_ann['bbox']
            new_ann['bbox'] = [
                bbox[0] * SCALE_FACTOR,
                bbox[1] * SCALE_FACTOR,
                bbox[2] * SCALE_FACTOR,
                bbox[3] * SCALE_FACTOR,
            ]

            if 'area' in new_ann:
                new_ann['area'] = new_ann['area'] * (SCALE_FACTOR ** 2)

            new_coco_data['annotations'].append(new_ann)

    with open(output_ann_file, 'w') as f:
        json.dump(new_coco_data, f, indent=2)


def main():
    upscale_model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=SCALE_FACTOR)

    for split in ("train", "valid", "test"):
        if os.path.exists(os.path.join(INPUT_DATASET_DIR, split)):
            process_split(split, upscale_model)


if __name__ == "__main__":
    main()