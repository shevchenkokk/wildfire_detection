import torch
import torchvision.transforms.functional as F
from torchvision import transforms as T
import random
from PIL import Image


def resize_image_and_target(image: Image.Image, target: dict, size: tuple) -> tuple[Image.Image, dict]:
    """
    Resizes an image and corresponding bounding boxes.

    Args:
        image (PIL.Image): Input image to resize
        target (dict): Dictionary containing bounding box annotations with keys:
            - boxes (torch.Tensor): Bounding boxes in [xmin, ymin, xmax, ymax] format
            - labels (torch.Tensor): Class labels for each box
            - area (torch.Tensor, optional): Area of each box
            - iscrowd (torch.Tensor, optional): Crowd instance indicators
        size (tuple): Target size as (height, width)

    Returns:
        tuple: Tuple containing:
            - PIL.Image: Resized image
            - dict: Updated target dictionary with resized boxes
    """
    w_orig, h_orig = image.size # PIL Image size is (width, height)
    image = F.resize(image, size) 
    
    if target is not None and "boxes" in target and target["boxes"].numel() > 0:
        boxes = target["boxes"].clone() # [xmin, ymin, xmax, ymax]
        
        h_new, w_new = size # New size (height, width)
        
        boxes[:, 0] = boxes[:, 0] * (w_new / w_orig) # xmin
        boxes[:, 1] = boxes[:, 1] * (h_new / h_orig) # ymin
        boxes[:, 2] = boxes[:, 2] * (w_new / w_orig) # xmax
        boxes[:, 3] = boxes[:, 3] * (h_new / h_orig) # ymax
        
        boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], 0, w_new) # Clamp x coords
        boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], 0, h_new) # Clamp y coords
        
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        target["boxes"] = boxes[valid_boxes]
        target["labels"] = target["labels"][valid_boxes]
        
        if "area" in target:
             target["area"] = (target["boxes"][:, 2] - target["boxes"][:, 0]) * \
                              (target["boxes"][:, 3] - target["boxes"][:, 1])
        
        if "iscrowd" in target:
            target["iscrowd"] = target["iscrowd"][valid_boxes]
            
    return image, target


def random_horizontal_flip_image_and_target(image: Image.Image, target: dict, p: float = 0.5) -> tuple[Image.Image, dict]:
    """
    Randomly flips image and bounding boxes horizontally.

    Args:
        image (PIL.Image): Input image to flip
        target (dict): Dictionary containing bounding box annotations
        p (float, optional): Probability of flipping. Defaults to 0.5

    Returns:
        tuple: Tuple containing:
            - PIL.Image: Flipped or original image
            - dict: Updated target dictionary with flipped boxes if applicable
    """
    if random.random() < p:
        w_orig, _ = image.size
        image = F.hflip(image)
        if target is not None and "boxes" in target and target["boxes"].numel() > 0:
            boxes = target["boxes"].clone() # [xmin, ymin, xmax, ymax]
            new_xmin = w_orig - boxes[:, 2]
            new_xmax = w_orig - boxes[:, 0]
            boxes[:, 0] = new_xmin
            boxes[:, 2] = new_xmax
            target["boxes"] = boxes
    return image, target


def color_jitter_image(image: Image.Image, target: dict, brightness: float = 0.2, contrast: float = 0.2, 
                      saturation: float = 0.2, hue: float = 0.1) -> tuple[Image.Image, dict]:
    """
    Applies random color jittering to the image.

    Args:
        image (PIL.Image): Input image to transform
        target (dict): Target dictionary (unchanged)
        brightness (float, optional): Brightness jitter range. Defaults to 0.2
        contrast (float, optional): Contrast jitter range. Defaults to 0.2
        saturation (float, optional): Saturation jitter range. Defaults to 0.2
        hue (float, optional): Hue jitter range. Defaults to 0.1

    Returns:
        tuple: Tuple containing:
            - PIL.Image: Color jittered image
            - dict: Unchanged target dictionary
    """
    transform = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    image = transform(image)
    return image, target


def to_tensor_image_and_target(image: Image.Image, target: dict) -> tuple[torch.Tensor, dict]:
    """
    Converts PIL image to PyTorch tensor.

    Args:
        image (PIL.Image): Input PIL image
        target (dict): Target dictionary containing tensor annotations

    Returns:
        tuple: Tuple containing:
            - torch.Tensor: Converted image tensor
            - dict: Unchanged target dictionary
    """
    image = F.to_tensor(image)
    return image, target


def normalize_image_and_target(image: torch.Tensor, target: dict) -> tuple[torch.Tensor, dict]:
    """
    Normalizes a PyTorch tensor image using ImageNet mean and std.
    """
    # ImageNet statistics for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize_transform = T.Normalize(mean=mean, std=std)
    image = normalize_transform(image)
    return image, target


def compose_transforms(transforms_list: list) -> callable:
    """
    Creates a single function that sequentially applies a list of transform functions.

    Args:
        transforms_list (list): List of transform functions to compose

    Returns:
        callable: Composed transform function that takes image and target as arguments
    """
    def composed_transform(image, target):
        for t_func in transforms_list:
            image, target = t_func(image, target)
        return image, target
    return composed_transform


def get_transforms_pipeline_functional(is_train: bool, target_size_hw: tuple = (640, 640)) -> callable:
    """
    Returns a composed transform function for the training/validation pipeline.

    Args:
        is_train (bool): Whether to include training-specific transforms
        target_size_hw (tuple, optional): Target size as (height, width). Defaults to (640, 640)

    Returns:
        callable: Composed transform function that applies all transforms in sequence
    """
    transform_list = []

    transform_list.append(lambda img, tgt: resize_image_and_target(img, tgt, size=target_size_hw))
    
    if is_train:
        # RandomHorizontalFlip (with default probability 0.5)
        transform_list.append(lambda img, tgt: random_horizontal_flip_image_and_target(img, tgt, p=0.5))
        # ColorJitter (with default parameters)
        transform_list.append(lambda img, tgt: color_jitter_image(img, tgt, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))

    # ToTensor
    transform_list.append(to_tensor_image_and_target)

    # Normalize
    transform_list.append(normalize_image_and_target)
    
    return compose_transforms(transform_list)