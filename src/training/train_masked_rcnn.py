import os
import torch
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dotenv import load_dotenv
import src.models.masked_rcnn as models

class FireDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        json_path = os.path.join(root, "_annotations.coco.json")
        with open(json_path) as f:
            self.coco_data = json.load(f)
        
        self.image_info = {img['id']: img for img in self.coco_data['images']}
        self.annotations = {}
        
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        self.ids = list(sorted(self.image_info.keys()))
        self.classes = {1: "fire"}

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.image_info[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        
        img = Image.open(img_path).convert("RGB")
        
        anns = self.annotations.get(img_id, [])
        
        boxes = []
        labels = []
        masks = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            if ann['segmentation']:
                pass  # Здесь можно добавить обработку полигонов
            else:
                mask[int(y):int(y+h), int(x):int(x+w)] = 1
            masks.append(mask)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img_info['height'], img_info['width']), dtype=torch.uint8)
        else:
            # Конвертируем все маски в один numpy array перед созданием тензора
            masks = np.stack(masks, axis=0)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id])
        }
        
        if self.transforms is not None:
            img = self.transforms(img)  # Только изображение
            # Трансформации для target нужно обрабатывать отдельно
        
        return img, target

    def __len__(self):
        return len(self.ids)
    
train_losses = []
val_metrics = {'map_25': [], 'map_50': [], 'map_75': []}

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    epoch_losses = []
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_losses.append(losses.item())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if print_freq and (i % print_freq == 0):
            print(f"Epoch: {epoch}, Iter: {i}, Loss: {losses.item()}")
    
    # Сохраняем средний лосс за эпоху
    avg_loss = np.mean(epoch_losses)
    train_losses.append(avg_loss)
    print(f"Epoch: {epoch}, Average Loss: {avg_loss:.4f}")

def evaluate(model, data_loader, device):
    model.eval()
    
    metric_25 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.25])
    metric_50 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])
    metric_75 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.75])
    
    for metric in [metric_25, metric_50, metric_75]:
        metric.to(device)
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)
            
            filtered_preds = []
            for pred in predictions:
                keep = pred['scores'] > 0.2
                filtered = {k: v[keep] for k, v in pred.items()}
                filtered_preds.append(filtered)
            
            metric_25.update(filtered_preds, targets)
            metric_50.update(filtered_preds, targets)
            metric_75.update(filtered_preds, targets)
    
    results_25 = metric_25.compute()
    results_50 = metric_50.compute()
    results_75 = metric_75.compute()
    
    print(f"mAP@25: {results_25['map'].item():.4f}")
    print(f"mAP@50: {results_50['map'].item():.4f}")
    print(f"mAP@75: {results_75['map'].item():.4f}")
    
    # Сохраняем метрики для визуализации
    val_metrics['map_25'].append(results_25['map'].item())
    val_metrics['map_50'].append(results_50['map'].item())
    val_metrics['map_75'].append(results_75['map'].item())
    
    return {
        'map_25': results_25['map'],
        'map_50': results_50['map'],
        'map_75': results_75['map']
    }

def collate_fn(batch):
    return tuple(zip(*batch))

def main():

    model_path = '/home/student/temp/mask_rcnn_fire_detection.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_classes = 2

    print("Loading datasets...")
    dataset_train = FireDataset('./data/merged_wildfire_dataset/train', get_transform(train=True))
    dataset_val = FireDataset('./data/merged_wildfire_dataset/valid', get_transform(train=False))
    dataset_test = FireDataset('./data/merged_wildfire_dataset/test', get_transform(train=False))

    data_loader_train = DataLoader(
        dataset_train, batch_size=16, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    data_loader_val = DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    print("Creating model...")
    model = models(num_classes, True)
    print(f"Loading model weights from {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params, lr=0.001, weight_decay=0.0001)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=4, gamma=0.1)

    num_epochs = 10

    for param in model.backbone.parameters():
        param.requires_grad = True
        
    print("Starting training...")
    for epoch in range(num_epochs):
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=100)
        lr_scheduler.step()
        evaluate(model, data_loader_val, device=device)

    print("Testing on test set...")
    evaluate(model, data_loader_test, device=device)

    torch.save(model.state_dict(), model_path)
    print("Model saved to mask_rcnn_fire_detection.pth")

if __name__ == '__main__':
    main()