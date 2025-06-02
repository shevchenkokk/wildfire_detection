import json
import os
from tqdm import tqdm


def convert_coco_to_yolo(coco_json_path: str, images_dir_path):
    """
    Converts COCO JSON annotations to YOLO format (.txt files).

    Args:
        coco_json_path (str): Path to the COCO JSON annotation file.
        images_dir_path (str): Path to the directory containing images for this split.
                               The .txt label files will be created in this same directory.
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    category_id_to_yolo_class_id = {
        1: 0, # COCO 'Wildfire' (id 1) -> YOLO class 0
        2: 1  # COCO 'No Wildfire' (id 2) -> YOLO class 1
    }

    image_info = {img['id']: {'file_name': img['file_name'], 'width': img['width'], 'height': img['height']}
                  for img in coco_data['images']}

    print(f"Converting annotations from: {coco_json_path}")
    print(f"Saving YOLO labels to: {images_dir_path}")

    # Process annotations
    for ann in tqdm(coco_data['annotations'], desc="Converting annotations"):
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox_coco = ann['bbox'] # [x_min, y_min, width, height]

        if image_id not in image_info:
            print(f"Warning: Image ID {image_id} not found in image_info. Skipping annotation.")
            continue
        
        if category_id not in category_id_to_yolo_class_id:
            print(f"Warning: COCO category ID {category_id} not mapped to a YOLO class ID. Skipping annotation.")
            continue

        img_width = image_info[image_id]['width']
        img_height = image_info[image_id]['height']

        x_min, y_min, bbox_w, bbox_h = bbox_coco
        x_center = (x_min + bbox_w / 2) / img_width
        y_center = (y_min + bbox_h / 2) / img_height
        norm_bbox_w = bbox_w / img_width
        norm_bbox_h = bbox_h / img_height

        yolo_class_id = category_id_to_yolo_class_id[category_id]

        yolo_line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_bbox_w:.6f} {norm_bbox_h:.6f}\n"

        image_file_name = image_info[image_id]['file_name']
        label_file_name = os.path.splitext(image_file_name)[0] + '.txt'
        label_file_path = os.path.join(images_dir_path, label_file_name)

        with open(label_file_path, 'a') as f:
            f.write(yolo_line)

    print("Conversion complete!")


def create_yolo_yaml(
        output_path: str,
        dataset_base_path: str,
        classes: list,
        train_sub_dir: str = 'train',
        val_sub_dir: str = 'valid',
        test_sub_dir: str = 'test'):
    """
    Creates the YOLOv8 dataset configuration YAML file.

    Args:
        output_path (str): The full path where the wildfire_data.yaml file will be created.
        dataset_base_path (str): The relative path to the root of your dataset (e.g., "data/merged_wildfire_dataset").
        classes (list): A list of class names (e.g., ['Wildfire']).
        train_sub_dir (str): Subdirectory name for training images/labels relative to dataset_base_path.
        val_sub_dir (str): Subdirectory name for validation images/labels relative to dataset_base_path.
        test_sub_dir (str): Subdirectory name for test images/labels relative to dataset_base_path.
    """
    yaml_content = f"""
    # wildfire_data.yaml
    # Путь к корневой папке датасета
    path: {dataset_base_path}

    # Относительные пути к папкам изображений внутри 'path'
    train: {train_sub_dir}
    val: {val_sub_dir}
    test: {test_sub_dir}

    # Количество классов и их имена
    nc: {len(classes)}
    names: {classes}
    """
    yaml_content = "\n".join([line.strip() for line in yaml_content.split("\n") if line.strip()])

    with open(output_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nCreated YOLO dataset config file: {output_path}")


if __name__ == '__main__':
    base_data_dir = "data/merged_wildfire_dataset"

    train_dir = os.path.join(base_data_dir, "train")
    train_ann_file = os.path.join(train_dir, "_annotations.coco.json")

    val_dir = os.path.join(base_data_dir, "valid")
    val_ann_file = os.path.join(val_dir, "_annotations.coco.json")

    test_dir = os.path.join(base_data_dir, "test")
    test_ann_file = os.path.join(test_dir, "_annotations.coco.json")

    print("\n--- Converting Training Data ---")
    convert_coco_to_yolo(train_ann_file, train_dir)

    print("\n--- Converting Validation Data ---")
    convert_coco_to_yolo(val_ann_file, val_dir)

    print("\n--- Converting Test Data ---")
    convert_coco_to_yolo(test_ann_file, test_dir)

    create_yolo_yaml(
        output_path="data/wildfire_data.yaml",
        dataset_base_path="data/merged_wildfire_dataset",
        classes=['Wildfire'],
        train_sub_dir="train",
        val_sub_dir="valid",
        test_sub_dir="test"
    )

    print("\nAll conversions finished. You can now try training YOLO.")