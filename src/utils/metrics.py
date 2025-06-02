import torch
import numpy as np


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    Boxes are in [xmin, ymin, xmax, ymax] format.
    box1, box2 are Tensors.
    """
    x_left = torch.max(box1[0], box2[0])
    y_top = torch.max(box1[1], box2[1])
    x_right = torch.min(box1[2], box2[2])
    y_bottom = torch.min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / (box1_area + box2_area - intersection_area + 1e-6) # Add epsilon for stability
    return iou.item()


def calculate_ap_for_class(gt_boxes_all_images, gt_labels_all_images,
                           pred_boxes_all_images, pred_labels_all_images, pred_scores_all_images,
                           target_class_id, iou_threshold=0.5):
    """
    Calculates Average Precision (AP) for a single class.
    Assumes inputs are lists of tensors (one tensor per image).
    """
    class_preds_info = [] # Stores {'score': float, 'is_tp': bool, 'image_idx': int}
    num_total_gt_for_class = 0

    for img_idx in range(len(gt_boxes_all_images)):
        gt_boxes_img = gt_boxes_all_images[img_idx]
        gt_labels_img = gt_labels_all_images[img_idx]
        
        pred_boxes_img = pred_boxes_all_images[img_idx]
        pred_labels_img = pred_labels_all_images[img_idx]
        pred_scores_img = pred_scores_all_images[img_idx]

        # Filter GTs for the current class on this image
        gt_class_mask = (gt_labels_img == target_class_id)
        gt_boxes_class_img = gt_boxes_img[gt_class_mask]
        num_total_gt_for_class += gt_boxes_class_img.shape[0]
        
        gt_matched_on_img = [False] * gt_boxes_class_img.shape[0]

        pred_class_mask = (pred_labels_img == target_class_id)
        if not pred_class_mask.any():
            continue
            
        pred_boxes_class_img_on_img = pred_boxes_img[pred_class_mask]
        pred_scores_class_img_on_img = pred_scores_img[pred_class_mask]

        sorted_indices = torch.argsort(pred_scores_class_img_on_img, descending=True)
        
        for pred_local_idx in sorted_indices:
            pred_box = pred_boxes_class_img_on_img[pred_local_idx]
            pred_score = pred_scores_class_img_on_img[pred_local_idx].item()
            
            is_tp = False
            if gt_boxes_class_img.shape[0] > 0: # Only match if there are GTs for this class in the image
                best_iou = 0.0
                best_gt_match_local_idx = -1

                for gt_local_idx in range(gt_boxes_class_img.shape[0]):
                    if gt_matched_on_img[gt_local_idx]:
                        continue 
                    iou = calculate_iou(pred_box, gt_boxes_class_img[gt_local_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_match_local_idx = gt_local_idx
                
                if best_iou >= iou_threshold and best_gt_match_local_idx != -1:
                    if not gt_matched_on_img[best_gt_match_local_idx]:
                        is_tp = True
                        gt_matched_on_img[best_gt_match_local_idx] = True
            
            class_preds_info.append({'score': pred_score, 'is_tp': is_tp})

    if not class_preds_info:
        return 0.0, 0, 0, num_total_gt_for_class # AP, TPs, FPs, Num_GTs

    class_preds_info.sort(key=lambda x: x['score'], reverse=True)
    
    tp_cumsum = np.zeros(len(class_preds_info))
    fp_cumsum = np.zeros(len(class_preds_info))
    
    current_tps = 0
    current_fps = 0
    for i, pred_info in enumerate(class_preds_info):
        if pred_info['is_tp']:
            current_tps += 1
        else:
            current_fps += 1
        tp_cumsum[i] = current_tps
        fp_cumsum[i] = current_fps

    recall_curve = tp_cumsum / (num_total_gt_for_class + 1e-9) if num_total_gt_for_class > 0 else np.zeros_like(tp_cumsum)
    precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)

    ap = 0.0
    if num_total_gt_for_class > 0 and len(precision_curve) > 0:
        mrec = np.concatenate(([0.], recall_curve, [recall_curve[-1]])) 
        mpre = np.concatenate(([1.], precision_curve, [0.]))     

        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])

        indices = np.where(mrec[1:] != mrec[:-1])[0] + 1
        ap = np.sum((mrec[indices] - mrec[indices-1]) * mpre[indices])
        
    return ap, current_tps, current_fps, num_total_gt_for_class


def compute_detection_metrics(all_predictions, all_ground_truths, 
                              model_class_names, # Includes background, e.g., ["background", "class1", "class2"]
                              iou_threshold=0.5, verbose=False):
    """
    Calculates mAP@iou_threshold and overall precision/recall for foreground classes.

    Args:
        all_predictions (list of dicts): Model outputs per image.
            Each dict: {'boxes': Tensor, 'labels': Tensor, 'scores': Tensor}
        all_ground_truths (list of dicts): Ground truths per image.
            Each dict: {'boxes': Tensor, 'labels': Tensor}
        model_class_names (list): Names of classes model predicts, including background at index 0.
        iou_threshold (float): IoU threshold.
        verbose (bool): Print per-class APs.
    Returns:
        dict: Metrics: 'mAP', 'per_class_AP' (dict), 'overall_precision', 'overall_recall'.
    """
    
    # Ensure data is on CPU for numpy operations
    gt_boxes_all = [gt['boxes'].cpu() for gt in all_ground_truths]
    gt_labels_all = [gt['labels'].cpu() for gt in all_ground_truths]
    
    pred_boxes_all = [pred['boxes'].cpu() for pred in all_predictions]
    pred_labels_all = [pred['labels'].cpu() for pred in all_predictions]
    pred_scores_all = [pred['scores'].cpu() for pred in all_predictions]

    per_class_ap = {}
    aggregated_tps = 0
    aggregated_fps = 0 # FPs are specific to a class prediction being wrong for that class
    aggregated_gts = 0
    
    # Iterate over foreground classes (skip background at index 0)
    for class_idx in range(1, len(model_class_names)):
        class_name = model_class_names[class_idx]
        
        ap, tps_class, fps_class, gts_class = calculate_ap_for_class(
            gt_boxes_all, gt_labels_all,
            pred_boxes_all, pred_labels_all, pred_scores_all,
            target_class_id=class_idx, # Model predicts this label for this class
            iou_threshold=iou_threshold
        )
        per_class_ap[class_name] = ap
        aggregated_tps += tps_class
        aggregated_fps += fps_class # Sum of FPs specific to each class
        aggregated_gts += gts_class
        
        if verbose:
            class_prec = tps_class / (tps_class + fps_class) if (tps_class + fps_class) > 0 else 0
            class_rec = tps_class / gts_class if gts_class > 0 else 0
            print(f"Class: {class_name} (ID: {class_idx}) | AP@{iou_threshold}: {ap:.4f} | P: {class_prec:.4f} | R: {class_rec:.4f} | GTs: {gts_class}")

    mean_ap = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
    
    # Overall micro-averaged precision and recall for foreground classes
    # Precision = Sum of TPs for all fg classes / Sum of (TPs+FPs) for all fg classes
    # Recall = Sum of TPs for all fg classes / Sum of GTs for all fg classes
    overall_precision = aggregated_tps / (aggregated_tps + aggregated_fps) if (aggregated_tps + aggregated_fps) > 0 else 0.0
    overall_recall = aggregated_tps / aggregated_gts if aggregated_gts > 0 else 0.0

    if verbose:
        print(f"\n--- Summary ---")
        print(f"mAP@{iou_threshold}: {mean_ap:.4f}")
        print(f"Overall Precision (micro-avg for fg classes): {overall_precision:.4f}")
        print(f"Overall Recall (micro-avg for fg classes): {overall_recall:.4f}")
        
    return {
        f"mAP@{iou_threshold}": mean_ap,
        "per_class_AP": per_class_ap,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall
    }