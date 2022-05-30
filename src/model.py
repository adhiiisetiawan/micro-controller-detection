from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes):
    # load model
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define new head with custom num_classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model