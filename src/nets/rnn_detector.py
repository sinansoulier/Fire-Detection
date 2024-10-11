import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.utils.hardware import device

class FireDetectionNetRNN(nn.Module):
    """
    Fire Detection Network with RNN for bounding box prediction.

    •	Encoder: ResNet-50 (pretrained, frozen).
	•	RNN: 2-layer LSTM for bounding box prediction.
	•	Dropout: 0.4 rate for regularization.
	•	Bounding Box Head: Linear layer predicting 4 coordinates per box.
	•	Correctness Head: Two-layer fully connected network with softmax output for box correctness scores.
    """
    def __init__(self, num_supported_boxes: int = 5):
        super(FireDetectionNetRNN, self).__init__()
        self.num_supported_boxes = num_supported_boxes
        self.encoder = models.resnet50(weights='ResNet50_Weights.DEFAULT').to(device())
        self.encoder.requires_grad_(False) # Freeze the encoder (ResNet) weights
        # RNN for bounding box regression
        self.rnn = nn.LSTM(self.encoder.fc.out_features, 256, 2, batch_first=True)
        self.l = nn.Linear(256, 4 * num_supported_boxes)
        self.dropout = nn.Dropout(0.4)

        self.l_correctness = nn.Linear(self.encoder.fc.out_features, 256)
        self.l_correctness2 = nn.Linear(256, num_supported_boxes)

    def forward(self, x):
        x = self.encoder(x)
        # Use RNN to predict bounding boxes
        bbox, _ = self.rnn(x)
        bbox = self.dropout(bbox)
        correctness = F.relu(self.l_correctness(x))

        bbox = self.l(bbox)
        correctness = F.softmax(self.l_correctness2(correctness), dim=1)

        return bbox.reshape(-1, self.num_supported_boxes, 4), correctness