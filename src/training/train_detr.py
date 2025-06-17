import albumentations as A
import argparse
import json
import numpy as np
import os
import sys
import torch
from PIL import Image
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value, Sequence
from functools import partial
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import (
    TrainingArguments,
    Trainer,
)
from transformers.image_transforms import center_to_corners_format
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput
from transformers.trainer_utils import EvalPrediction, set_seed
import src.models.detr as models


def create_dataset_generator(ann_file, img_dir):
    """
    Generator that reads COCO annotations and yields samples.
    """
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    img_id_to_info = {img['id']: img for img in coco_data['images']}
    img_id_to_annotations = {img_id: [] for img_id in img_id_to_info}
    if 'annotations' in coco_data:
        for ann in coco_data['annotations']:
            img_id_to_annotations[ann['image_id']].append(ann)

    for img_id, img_info in img_id_to_info.items():
        img_path = os.path.join(img_dir, img_info['file_name'])

        assert os.path.exists(img_path), f"Image file not found {img_path}"

        annotations = img_id_to_annotations[img_id]

        objects = {
            "image_id": [ann['image_id'] for ann in annotations],
            "area": [ann['area'] for ann in annotations],
            "bbox": [ann['bbox'] for ann in annotations],
            "category": [ann['category_id'] - 1 for ann in annotations],  # COCO IDs to 0-based
        }

        yield {"image": img_path, "objects": objects}


def format_annotations_for_processor(image_id, categories, areas, bboxes):
    """
    Formats annotations for a single image into the format expected by DetrImageProcessor.
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        annotations.append({
            "image_id": image_id,
            "category_id": category,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0,
        })
    return {"image_id": image_id, "annotations": annotations}


def transform_aug_ann(examples, image_processor, transform):
    """
    Opens images, applies augmentations and image_processor transformations.
    """
    images = []
    annotations = []

    for image_path, objects in zip(examples["image"], examples["objects"]):
        image = Image.open(image_path).convert("RGB")

        transformed = transform(
            image=np.array(image),
            bboxes=objects["bbox"],
            category=objects["category"],
        )

        images.append(transformed["image"])
        image_id = objects["image_id"][0] if objects["image_id"] else -1

        formatted_ann = format_annotations_for_processor(
            image_id=image_id,
            categories=transformed["category"],
            areas=objects["area"],
            bboxes=transformed["bboxes"],
        )
        annotations.append(formatted_ann)

    return image_processor(images=images, annotations=annotations, return_tensors="pt")


def collate_fn(batch):
    """
    Custom data collator for batching samples.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Converts bounding boxes from YOLO format to Pascal VOC format.
    """
    boxes = center_to_corners_format(boxes)
    height, width = image_size
    boxes = boxes * torch.tensor([width, height, width, height], device=boxes.device)
    return boxes


def compute_metrics(eval_pred: EvalPrediction, id2label: dict, image_processor):
    """
    Computes COCO-style metrics for object detection.
    """
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    model_outputs, all_labels = eval_pred.predictions, eval_pred.label_ids

    for i, batch_labels in enumerate(all_labels):
        logits = torch.from_numpy(model_outputs[i][1])
        pred_boxes = torch.from_numpy(model_outputs[i][2])
        orig_sizes = torch.stack([torch.tensor(lbl["orig_size"]) for lbl in batch_labels])

        post_processed_preds = image_processor.post_process_object_detection(
            outputs=DetrObjectDetectionOutput(logits=logits, pred_boxes=pred_boxes),
            threshold=0.2,
            target_sizes=orig_sizes
        )

        targets = []
        for label in batch_labels:
            boxes = convert_bbox_yolo_to_pascal(
                torch.from_numpy(label["boxes"]), label["orig_size"]
            )
            targets.append({
                "boxes": boxes,
                "labels": torch.from_numpy(label["class_labels"])
            })

        metric.update(post_processed_preds, targets)

    metrics = metric.compute()

    if "classes" in metrics:
        classes_tensor = metrics.pop("classes")
        map_per_class_tensor = metrics.pop("map_per_class")
        mar_100_per_class_tensor = metrics.pop("mar_100_per_class")

        if classes_tensor.numel() == 1:
            class_id = classes_tensor.item()
            class_name = id2label[class_id]
            metrics[f"map_{class_name}"] = round(map_per_class_tensor.item(), 4)
            metrics[f"mar_100_{class_name}"] = round(mar_100_per_class_tensor.item(), 4)
        else:
            classes = classes_tensor.tolist()
            map_per_class = map_per_class_tensor.tolist()
            mar_100_per_class = mar_100_per_class_tensor.tolist()
            for i, class_id in enumerate(classes):
                class_name = id2label[class_id]
                metrics[f"map_{class_name}"] = round(map_per_class[i], 4)
                metrics[f"mar_100_{class_name}"] = round(mar_100_per_class[i], 4)

    metrics = {k: round(v.item(), 4) for k, v in metrics.items() if isinstance(v, torch.Tensor)}
    return metrics


def main(args):
    set_seed(args.seed)

    train_dir = os.path.join(args.data_path, "train")
    train_ann_file = os.path.join(train_dir, "_annotations.coco.json")
    val_dir = os.path.join(args.data_path, "valid")
    val_ann_file = os.path.join(val_dir, "_annotations.coco.json")
    test_dir = os.path.join(args.data_path, "test")
    test_ann_file = os.path.join(test_dir, "_annotations.coco.json")

    with open(train_ann_file, 'r') as f:
        category_names = [cat['name'] for cat in json.load(f)['categories']]

    features = Features({
        'image': Value(dtype='string'),
        'objects': Sequence({
            'image_id': Value(dtype='int64'),
            'area': Value(dtype='float32'),
            'bbox': Sequence(Value(dtype='float32'), length=4),
            'category': ClassLabel(names=category_names),
        })
    })

    raw_datasets = DatasetDict({
        "train": Dataset.from_generator(create_dataset_generator,
                                        gen_kwargs={"ann_file": train_ann_file, "img_dir": train_dir},
                                        features=features),
        "validation": Dataset.from_generator(create_dataset_generator,
                                             gen_kwargs={"ann_file": val_ann_file, "img_dir": val_dir},
                                             features=features),
        "test": Dataset.from_generator(create_dataset_generator,
                                       gen_kwargs={"ann_file": test_ann_file, "img_dir": test_dir}, features=features)
    })

    print(f"Train dataset size: {len(raw_datasets['train'])}")
    print(f"Validation dataset size: {len(raw_datasets['validation'])}")
    print(f"Validation dataset size: {len(raw_datasets['test'])}")

    image_processor = models.get_detr_image_processor(args.model_path)

    transform_train = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.ColorJitter(p=0.3),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], min_visibility=0.3, clip=True),
    )

    transform_val_test = A.Compose(
        [],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True)
    )

    # raw_datasets["train"] = raw_datasets["train"].select(range(100))
    # raw_datasets["validation"] = raw_datasets["validation"].select(range(50))

    train_dataset_transformed = raw_datasets["train"].with_transform(
        partial(transform_aug_ann, image_processor=image_processor, transform=transform_train)
    )
    val_dataset_transformed = raw_datasets["validation"].with_transform(
        partial(transform_aug_ann, image_processor=image_processor, transform=transform_val_test)
    )
    test_dataset_transformed = raw_datasets["test"].with_transform(
        partial(transform_aug_ann, image_processor=image_processor, transform=transform_val_test)
    )

    id2label = {i: name for i, name in enumerate(category_names)}
    label2id = {name: i for i, name in enumerate(category_names)}

    if args.mode == 'train':
        model = models.get_detr_model(
            model_path=args.model_path,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            logging_dir=f"{args.output_dir}/logs",
            num_train_epochs=args.epochs,
            logging_strategy="steps",
            logging_steps=250,
            eval_strategy="steps",
            eval_steps=1000,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="map",
            learning_rate=3e-5,
            weight_decay=1e-4,
            remove_unused_columns=False,
            eval_do_concat_batches=False,
            report_to="tensorboard",
            seed=args.seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            train_dataset=train_dataset_transformed,
            eval_dataset=val_dataset_transformed,
            tokenizer=image_processor,  # Correct parameter name
            compute_metrics=partial(compute_metrics, id2label=id2label, image_processor=image_processor),
        )

        print("\n--- Starting training ---")
        trainer.train()
        trainer.evaluate()
        trainer.save_model()
        print("--- Training finished. ---")

    elif args.mode == 'eval':
        model = models.get_detr_model(
            model_path=args.model_path,
            id2label=id2label,
            label2id=label2id
        )

        eval_args = TrainingArguments(
            output_dir=f"{args.output_dir}/eval_only",
            per_device_eval_batch_size=args.batch_size,
            remove_unused_columns=False,
            eval_do_concat_batches=False,
            fp16=True,
            seed=args.seed,
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            data_collator=collate_fn,
            tokenizer=image_processor,
            compute_metrics=partial(compute_metrics, id2label=id2label, image_processor=image_processor),
        )

        print("\n--- Evaluating on Val Set ---")
        metrics = trainer.evaluate(eval_dataset=val_dataset_transformed, metric_key_prefix="val")
        print("\n--- Evaluation Metrics ---")
        print(metrics)

        print("\n--- Evaluating on Test Set ---")
        metrics = trainer.evaluate(eval_dataset=test_dataset_transformed, metric_key_prefix="test")
        print("\n--- Evaluation Metrics ---")
        print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate a DETR model for object detection.")
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'eval'],
                        help="Mode to run the script in: 'train' or 'eval'.")
    parser.add_argument("--data_path", type=str, default="data/merged_wildfire_dataset",
                        help="Path to the root directory of the dataset.")
    parser.add_argument("--model_path", type=str, default="facebook/detr-resnet-50",
                        help="Path to the model checkpoint. For training, this is the base model. For eval, this is the trained model to evaluate.")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints/detr_wildfire",
                        help="Directory to save training checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    main(args)
