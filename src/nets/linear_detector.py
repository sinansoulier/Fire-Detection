import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.utils.hardware import device

class FireDetectionNetLinear(nn.Module):
    """
    Fire Detection Network with Linear layers for bounding box prediction.

    •	Encoder: ResNet-50 (pretrained, frozen).
	•	Linear Layers: Two-layer fully connected network for bounding box prediction.
	•	Dropout: 0.4 rate for regularization.
	•	Bounding Box Head: Linear layer predicting 4 coordinates per box.
	•	Correctness Head: Two-layer fully connected network with softmax output for box correctness scores.
    """
    def __init__(self, num_supported_boxes: int = 5):
        super(FireDetectionNetLinear, self).__init__()

        self.num_supported_boxes = num_supported_boxes
        self.encoder = models.resnet50(weights='ResNet50_Weights.DEFAULT').to(device())
        self.l_bbox = nn.Linear(self.encoder.fc.out_features, 512)
        self.l_bbox2 = nn.Linear(512, 4 * num_supported_boxes)

        self.l_correctness = nn.Linear(self.encoder.fc.out_features, 256)
        self.l_correctness2 = nn.Linear(256, num_supported_boxes)

    def forward(self, x):
        x = self.encoder(x)
        bbox = F.relu(self.l_bbox(x))
        correctness = F.relu(self.l_correctness(x))

        # Use linear layers to predict bounding boxes
        bboxes = self.l_bbox2(bbox)
        correctness = F.softmax(self.l_correctness2(correctness), dim=1)
        return bboxes.reshape(-1, self.num_supported_boxes, 4), correctness