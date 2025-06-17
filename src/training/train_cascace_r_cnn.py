import os
import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

PATH_TO_TRAIN_DATASET = "data/merged_wildfire_dataset/train/"
PATH_TO_VAL_DATASET = "data/merged_wildfire_dataset/valid/"
PATH_TO_TEST_DATASET = "data/merged_wildfire_dataset/test/"
CLASS_NAMES = ["wildfire"]
MODEL_CONFIG = "https://github.com/facebookresearch/detectron2/blob/main/configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
MODEL_WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"



class TrainerWithMetrics(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def get_yolo_dicts(img_dir: str, label_dir: str):
    """
    Загружает датасет в формате YOLO и конвертирует в формат Detectron2.
    """
    dataset_dicts = []
    for idx, img_file in enumerate(os.listdir(img_dir)):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        record = {}
        image_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))

        im = cv2.imread(image_path)
        height, width = im.shape[:2]

        record["file_name"] = image_path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id, xc, yc, w, h = map(float, parts)
                    xc *= width
                    yc *= height
                    w *= width
                    h *= height
                    x1 = xc - w / 2
                    y1 = yc - h / 2
                    x2 = xc + w / 2
                    y2 = yc + h / 2

                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask[int(y1):int(y2), int(x1):int(x2)] = 1

                    polygon = [
                        [x1, y1, x2, y1, x2, y2, x1, y2]
                    ]

                    objs.append({
                        "bbox": [x1, y1, x2, y2],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": int(class_id),
                        "segmentation": polygon, 
                        "gt_masks": mask,
                    })

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def register_yolo_dataset(name, img_dir, label_dir, class_names):
    DatasetCatalog.register(name, lambda: get_yolo_dicts(img_dir, label_dir))
    MetadataCatalog.get(name).set(thing_classes=class_names)


def main():
    register_yolo_dataset(
        "train_dataset", 
        PATH_TO_TRAIN_DATASET, 
        PATH_TO_TRAIN_DATASET,
        CLASS_NAMES,
    )

    register_yolo_dataset(
        "val_dataset", 
        PATH_TO_VAL_DATASET,
        PATH_TO_VAL_DATASET,
        CLASS_NAMES,
    )

    register_yolo_dataset(
        "test_dataset", 
        PATH_TO_TEST_DATASET,
        PATH_TO_TEST_DATASET, 
        CLASS_NAMES,
    )

  
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.VAL = ("val_dataset",)
    cfg.DATASETS.TEST = ("test_dataset",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 64
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 20
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES) # Число классов
    
    predictor = DefaultPredictor(cfg)

    trainer = TrainerWithMetrics(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    trainer.test(cfg, trainer.model)

    model = trainer.model
    torch.save(model.state_dict(), "model_final.pth")



if __name__ == "__main__":
    main()

