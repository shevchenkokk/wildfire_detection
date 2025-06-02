import torch
import os
import time
import json
import torchvision
from torch.utils.data import DataLoader
from PIL import Image
import random
import src.data_handling.dataset as dataset
from src.data_handling.transforms import get_transforms_pipeline_functional
import src.models.faster_rcnn as models
from src.utils.logger import TensorBoardLogger
from src.utils.metrics import compute_detection_metrics

# Helper function to draw boxes on image for visualization
def draw_boxes_on_image(image_tensor, gt_boxes=None, pred_boxes=None, 
                        gt_labels=None, pred_labels=None, scores=None, 
                        class_names=None, score_threshold=0.5):
    """
    Draws bounding boxes and labels on an image tensor.
    Assumes image_tensor is C, H, W and values are in [0,1] or [0,255].
    Converts to uint8 for drawing.
    """
    img_to_draw = image_tensor.cpu().clone()

    if img_to_draw.is_floating_point():
        if img_to_draw.max() <= 1.0:
            img_to_draw = (img_to_draw * 255).to(torch.uint8)
        else:
            img_to_draw = img_to_draw.to(torch.uint8)

    if img_to_draw.ndim == 2:
        img_to_draw = img_to_draw.unsqueeze(0)
    if img_to_draw.shape[0] == 1:
        img_to_draw = img_to_draw.repeat(3, 1, 1)
    
    drawn_boxes_img = img_to_draw
    
    # Draw Ground Truth boxes (green)
    if gt_boxes is not None and len(gt_boxes) > 0:
        gt_box_labels = []
        if gt_labels is not None and class_names is not None:
            for label in gt_labels:
                if int(label) < len(class_names):
                    gt_box_labels.append(class_names[int(label)])
                else:
                    gt_box_labels.append(f"GT_ERR_{int(label)}")
        drawn_boxes_img = torchvision.utils.draw_bounding_boxes(drawn_boxes_img, gt_boxes, labels=gt_box_labels, colors="green", width=2)

    # Draw Predicted boxes (red)
    if pred_boxes is not None and len(pred_boxes) > 0 and scores is not None:
        # Filter predictions by score threshold
        selected_preds_mask = scores >= score_threshold
        if selected_preds_mask.any(): # Only draw if there are predictions above threshold
            pred_boxes_to_draw = pred_boxes[selected_preds_mask]
            pred_labels_to_draw = pred_labels[selected_preds_mask]
            pred_scores_to_draw = scores[selected_preds_mask]
            
            pred_box_labels = []
            if class_names is not None:
                for label, score in zip(pred_labels_to_draw, pred_scores_to_draw):
                    if int(label) < len(class_names): # Safety check
                        pred_box_labels.append(f"{class_names[int(label)]}: {score:.2f}")
                    else:
                        pred_box_labels.append(f"CLS_ERR_{int(label)}: {score:.2f}")

            drawn_boxes_img = torchvision.utils.draw_bounding_boxes(drawn_boxes_img, pred_boxes_to_draw, labels=pred_box_labels, colors="red", width=2)
            
    return drawn_boxes_img


# Collate function for DataLoader (handles varying number of objects per image)
def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate(model, data_loader, device, class_names, iou_threshold=0.5, epoch_num=None, logger=None, log_prefix="Val"):
    """
    Evaluates the model on the given data_loader.
    Calculates mAP, precision, and recall using the custom metrics_calculator.
    Logs metrics and a batch of example images with predictions to TensorBoard.
    """
    model.eval()
    all_predictions = []
    all_ground_truths = []
    
    print(f"\n--- Running Evaluation ({log_prefix}) ---")
    start_time = time.time()
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            images_device = [img.to(device) for img in images]
            
            outputs = model(images_device)

            for j in range(len(outputs)):
                cpu_output = {k: v.cpu() for k, v in outputs[j].items()}
                all_predictions.append(cpu_output)

                cpu_target = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in targets[j].items()}
                all_ground_truths.append(cpu_target)

            if i == 0 and logger and epoch_num is not None:
                try:
                    drawn_images_viz = []
                    for k in range(min(len(images), 4)): 
                        img_tensor = images[k]

                        gt_boxes_viz = targets[k]['boxes'].cpu()
                        gt_labels_viz = targets[k]['labels'].cpu()
                        
                        pred_boxes_viz = outputs[k]['boxes'].cpu()
                        pred_labels_viz = outputs[k]['labels'].cpu()
                        pred_scores_viz = outputs[k]['scores'].cpu()
                        
                        drawn_img = draw_boxes_on_image(
                            img_tensor, 
                            gt_boxes_viz, pred_boxes_viz, 
                            gt_labels_viz, pred_labels_viz, pred_scores_viz,
                            class_names=class_names, 
                            score_threshold=0.3
                        )
                        drawn_images_viz.append(drawn_img)
                    
                    if drawn_images_viz:
                        img_grid = torchvision.utils.make_grid(drawn_images_viz)
                        logger.log_image_grid(f'{log_prefix}_Detections/epoch_{epoch_num}', img_grid, global_step=epoch_num)
                except Exception as e:
                    print(f"Error during {log_prefix} image logging: {e}")

    if not all_predictions and not all_ground_truths:
        print(f"No predictions or ground truths collected during {log_prefix} evaluation. Skipping metrics.")
        return {f"mAP@{iou_threshold}": 0.0, "per_class_AP": {}, "overall_precision": 0.0, "overall_recall": 0.0}
    
    if not all_predictions:
         print(f"No predictions collected during {log_prefix} evaluation. mAP will be 0.0.")
    if not all_ground_truths:
        print(f"No ground truths collected during {log_prefix} evaluation. Recall will be 0.0.")

    # Calculate metrics using the custom calculator
    metrics = compute_detection_metrics(all_predictions, all_ground_truths,
                                        model_class_names=class_names, 
                                        iou_threshold=iou_threshold,
                                        verbose=True)
    
    # Log metrics to TensorBoard
    if logger and epoch_num is not None:
        logger.log_scalar(f'Metrics/{log_prefix}_mAP@{iou_threshold}', metrics.get(f"mAP@{iou_threshold}", 0.0), epoch_num)
        logger.log_scalar(f'Metrics/{log_prefix}_Precision_Overall', metrics.get("overall_precision", 0.0), epoch_num)
        logger.log_scalar(f'Metrics/{log_prefix}_Recall_Overall', metrics.get("overall_recall", 0.0), epoch_num)
        for cls_name, ap_val in metrics["per_class_AP"].items():
            logger.log_scalar(f'AP/{log_prefix}_{cls_name}', ap_val, epoch_num)
            
    eval_duration = time.time() - start_time
    print(f"--- {log_prefix} Evaluation Complete for Epoch {epoch_num if epoch_num is not None else 'Final'} (Duration: {eval_duration:.2f}s) ---")
    print(f"mAP@{iou_threshold}: {metrics.get(f'mAP@{iou_threshold}', 0):.4f}, Precision: {metrics.get('overall_precision',0):.4f}, Recall: {metrics.get('overall_recall',0):.4f}")
    return metrics


def main():
    print("--- Starting training and evaluation for Faster R-CNN... ---")
    
    # --- Configuration ---
    experiment_name = "wildfire_detection_frcnn" # Unique name for TensorBoard logs
    base_data_dir = "data/merged_wildfire_dataset" # Base directory for all data splits
    
    # Paths for training data
    train_dir = os.path.join(base_data_dir, "train")
    train_ann_file = os.path.join(train_dir, "_annotations.coco.json")
    
    # Paths for validation data
    val_dir = os.path.join(base_data_dir, "valid")
    val_ann_file = os.path.join(val_dir, "_annotations.coco.json")
    
    # Paths for test data
    test_dir = os.path.join(base_data_dir, "test")
    test_ann_file = os.path.join(test_dir, "_annotations.coco.json")

    output_dir = "models/checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    class_names = ["background", "Wildfire"] 
    num_classes = len(class_names) # Total number of classes including background
    
    img_target_size = (640, 640) # All images will be resized to (640, 640)
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.005
    momentum = 0.9
    weight_decay = 0.0005 # L2 regularization
    log_images_train_interval = 5 # Log training images with predictions every N epochs

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # --- TensorBoard Logger Initialization ---
    hparams_to_log = {
        "learning_rate": learning_rate, "batch_size": batch_size, "num_epochs": num_epochs,
        "optimizer": "SGD", "momentum": momentum, "weight_decay": weight_decay,
        "num_classes": num_classes, "model_architecture": "FasterRCNN_ResNet50_FPN",
        "image_size": f"{img_target_size[0]}x{img_target_size[1]}"
    }
    logger = TensorBoardLogger(log_dir_base="runs", experiment_name=experiment_name, hparams=hparams_to_log)

    # --- Data Loading & Transforms ---
    print("Setting up data transforms...")
    transforms_train = get_transforms_pipeline_functional(is_train=True, target_size_hw=img_target_size)
    transforms_val_test = get_transforms_pipeline_functional(is_train=False, target_size_hw=img_target_size)

    print("Loading datasets...")
    try:
        dataset_train = dataset.WildfireDetectionDataset(
            img_folder=train_dir, ann_file=train_ann_file, transforms=transforms_train
        )
        dataset_val = dataset.WildfireDetectionDataset(
            img_folder=val_dir, ann_file=val_ann_file, transforms=transforms_val_test
        )
        dataset_test = dataset.WildfireDetectionDataset(
            img_folder=test_dir, ann_file=test_ann_file, transforms=transforms_val_test
        )

        data_loader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn 
        )
        data_loader_val = DataLoader(
            dataset_val, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
        )
        data_loader_test = DataLoader(
            dataset_test, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
        )
        print(f"Training dataset: {len(dataset_train)} samples. Validation: {len(dataset_val)}. Test: {len(dataset_test)}")
    except Exception as e:
        print(f"Error loading datasets. Please ensure dummy data is generated or real data exists: {e}")
        logger.close()
        return

    # --- Model Initialization ---
    print(f"Initializing Faster R-CNN model for {num_classes} classes.")
    model = models.get_faster_rcnn_model(num_classes=num_classes, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 

    # --- Training Loop ---
    print("\n--- Starting training loop ---")
    global_step = 0 # Counter for TensorBoard logging
    best_val_map = 0.0 # To keep track of the best validation mAP for saving the best model
    save_path_best = os.path.join(output_dir, f"{experiment_name}_best_model.pth") # Path for best model checkpoint

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time_epoch = time.time()

        for i, (images, targets) in enumerate(data_loader_train):
            images = list(image.to(device) for image in images)
            targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets_device)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            batch_loss = losses.item()
            epoch_loss += batch_loss
            
            logger.log_scalar('Loss/train_batch', batch_loss, global_step) # Log batch loss
            global_step += 1
            
            if (i + 1) % 5 == 0: # Log progress every few batches
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(data_loader_train)}], Loss: {batch_loss:.4f}")

        avg_epoch_loss = epoch_loss / len(data_loader_train)
        logger.log_scalar('Loss/train_epoch', avg_epoch_loss, epoch) # Log average epoch loss
        logger.log_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch) # Log current learning rate
        epoch_duration = time.time() - start_time_epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Avg Loss: {avg_epoch_loss:.4f}. Duration: {epoch_duration:.2f}s")
        
        # Step the learning rate scheduler
        if lr_scheduler:
            lr_scheduler.step()

        # --- Log Training Images with Detections (Periodically) ---
        if (epoch + 1) % log_images_train_interval == 0 and len(dataset_train) > 0:
            model.eval()
            with torch.no_grad():
                try:
                    sample_images_viz, sample_targets_viz = next(iter(data_loader_train)) 
                    sample_images_viz_dev = list(img.to(device) for img in sample_images_viz)
                    predictions_train_viz = model(sample_images_viz_dev)

                    drawn_images_train = []
                    # Visualize up to 4 images
                    for j in range(min(len(sample_images_viz), 4)): 
                        img_tensor = sample_images_viz[j] # Original CPU tensor
                        gt_boxes = sample_targets_viz[j]['boxes'].cpu()
                        gt_labels = sample_targets_viz[j]['labels'].cpu()
                        pred_boxes = predictions_train_viz[j]['boxes'].cpu()
                        pred_labels = predictions_train_viz[j]['labels'].cpu()
                        pred_scores = predictions_train_viz[j]['scores'].cpu()
                        
                        drawn_img = draw_boxes_on_image(
                            img_tensor, gt_boxes, pred_boxes, gt_labels, pred_labels, pred_scores,
                            class_names=class_names, score_threshold=0.5
                        )
                        drawn_images_train.append(drawn_img)
                    
                    if drawn_images_train:
                        img_grid_train = torchvision.utils.make_grid(drawn_images_train)
                        logger.log_image_grid(f'Train_Detections/epoch_{epoch+1}', img_grid_train, global_step=epoch)
                except Exception as e:
                    print(f"Error during training image logging for epoch {epoch+1}: {e}")
            model.train()

        # --- Validation Loop ---
        # Evaluate model performance on the validation set after each epoch
        if len(dataset_val) > 0:
            val_metrics = evaluate(model, data_loader_val, device, class_names, 
                                   iou_threshold=0.5, epoch_num=epoch, logger=logger, log_prefix="Val")
            current_val_map = val_metrics.get(f"mAP@0.5", 0.0) # Get mAP@0.5 from metrics
            
            # Save best model based on validation mAP
            if current_val_map > best_val_map:
                best_val_map = current_val_map
                torch.save(model.state_dict(), save_path_best)
                print(f"Epoch {epoch+1}: New best validation mAP: {best_val_map:.4f}. Model saved to {save_path_best}")
        else:
            print("Validation dataset is empty or not configured. Skipping validation.")

    print("--- Training finished. ---")
    
    # --- Save Final Model ---
    final_model_path = os.path.join(output_dir, f"{experiment_name}_final_model_epoch{num_epochs}.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # --- Final Test Evaluation ---
    if len(dataset_test) > 0:
        print("\n--- Running Final Test Evaluation ---")
        
        # Load the best model's weights for final testing
        if os.path.exists(save_path_best):
            print(f"Loading best model from {save_path_best} for final testing.")
            model.load_state_dict(torch.load(save_path_best))
            model.to(device)
        else:
            print("Best model checkpoint not found. Testing with the model state at the end of training.")

        test_metrics = evaluate(model, data_loader_test, device, class_names, 
                                iou_threshold=0.5, epoch_num=num_epochs, logger=logger, log_prefix="Test") # Use num_epochs for step
        print("\n--- Test Metrics Summary ---")
        for k, v in test_metrics.items():
            if isinstance(v, dict):
                for cls_name, ap_val in v.items(): print(f"  {k}/{cls_name}: {ap_val:.4f}")
            else: print(f"  {k}: {v:.4f}")
    else:
        print("Test dataset is empty or not configured. Skipping final testing.")
    
    logger.close()
    print(f"\nTo view logs, run in your terminal: tensorboard --logdir={os.path.abspath(os.path.dirname(logger.log_dir))}")


if __name__ == '__main__':
    main()