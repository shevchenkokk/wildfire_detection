import torch
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn


def get_model_masked_rcnn(num_classes, pretrained, model_path = None):
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)
    
    # Модифицируем головы
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = MaskRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    if model_path is not None:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)

    return model

if __name__ == '__main__':

    model = get_model_masked_rcnn(num_classes=2, pretrained=True)
    print("Masked R-CNN model created successfully.")
 