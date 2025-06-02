import torch
from torch.utils.data import Dataset
import os
import json
from PIL import Image
from collections import defaultdict


class WildfireDetectionDataset(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.img_folder = img_folder
        self.transforms = transforms

        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        self.img_id_to_info = {img['id']: img for img in self.coco_data['images']}

        temp_img_id_to_anns = defaultdict(list)
        for ann in self.coco_data.get('annotations', []):
            xmin, ymin, width, height = ann['bbox']
            if width > 0 and height > 0:
                temp_img_id_to_anns[ann['image_id']].append(ann)

        self.ids = []
        self.img_id_to_anns = {}

        for img_id in self.img_id_to_info.keys():
            if temp_img_id_to_anns[img_id]:
                self.ids.append(img_id)
                self.img_id_to_anns[img_id] = temp_img_id_to_anns[img_id]

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # Find image info
        img_info = next((item for item in self.coco_data['images'] if item["id"] == img_id), None)
        if img_info is None:
            raise ValueError(f"Image with id {img_id} not found in image_infos.")

        img_path = os.path.join(self.img_folder, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # Get annotations for this image
        anns = self.img_id_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            # COCO bbox format is [x, y, width, height]
            # PyTorch expects [xmin, ymin, xmax, ymax]
            xmin = ann['bbox'][0]
            ymin = ann['bbox'][1]
            xmax = xmin + ann['bbox'][2]
            ymax = ymin + ann['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([img_id])
        target["area"] = torch.as_tensor(areas, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    # Example usage
    print("--- Attempting to initialize WildfireDetectionDataset... ---")

    dummy_train_img_dir = 'data/merged_wildfire_dataset/train'
    dummy_train_ann_file = os.path.join(dummy_train_img_dir, '_annotations.coco.json')

    try:
        train_dataset = WildfireDetectionDataset(
            img_folder=dummy_train_img_dir,
            ann_file=dummy_train_ann_file
        )
        print(f"Successfully initialized dataset. Number of samples: {len(train_dataset)}")
        if len(train_dataset) > 0:
            img, target = train_dataset[0]
            print("Sample image loaded:", type(img))
            print("Sample target:", target)
    except Exception as e:
        print(f"Error initializing or using WildfireDetectionDataset: {e}")
        print("This might be due to incorrect paths or issues with the dummy data setup") 