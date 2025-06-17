from transformers import AutoModelForObjectDetection, AutoImageProcessor

def get_detr_model(model_path, id2label=None, label2id=None, ignore_mismatched_sizes=False):
    """
    Loads a DETR model for object detection.
    """
    return AutoModelForObjectDetection.from_pretrained(
        model_path,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=ignore_mismatched_sizes
    )


def get_detr_image_processor(model_path):
    """
    Loads the corresponding image processor for the DETR model.
    """
    return AutoImageProcessor.from_pretrained(model_path)
