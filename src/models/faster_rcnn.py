import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_faster_rcnn_model(num_classes, pretrained=True):
    """
    Constructs a Faster R-CNN model with a ResNet-50 FPN backbone.

    Args:
        num_classes (int): The number of classes for the model to predict (including background).
                           For example, if you have "Wildfire" and "No Wildfire", num_classes should be 3.
        pretrained (bool): If True, loads weights pre-trained on COCO.

    Returns:
        torchvision.models.detection.FasterRCNN: The Faster R-CNN model.
    """
    if pretrained:
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    else:
        weights = None

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

if __name__ == '__main__':
    # Example of how to get the model
    # num_classes should be number of your object categories + 1 (for background)
    # e.g., Wildfire (1), No Wildfire (2) -> num_classes = 3
    model = get_faster_rcnn_model(num_classes=3, pretrained=True)
    print("Faster R-CNN model created successfully.")
    # print(model) # Uncomment to see model structure 