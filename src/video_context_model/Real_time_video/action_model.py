import os
import cv2
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from datetime import datetime
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

try:
    from ucf101_config import UCF101_CLASSES
    UCF101_AVAILABLE = True
except ImportError:
    UCF101_AVAILABLE = False
    print("[WARNING] UCF101 config not found, using default configuration")

class CNN_GRU(nn.Module):
    def __init__(self, cnn_model='mobilenetv2', hidden_size=128, num_layers=1,
                 num_classes=5, dropout=0.5, FREEZE_BACKBONE=True):
        super(CNN_GRU, self).__init__()

        if cnn_model == 'mobilenetv2':
            cnn = models.mobilenet_v2(pretrained=True)
            self.cnn_out_features = cnn.last_channel
            self.cnn = cnn.features
        elif cnn_model == 'efficientnet_b0':
            import timm
            cnn = timm.create_model('efficientnet_b0', pretrained=True)
            self.cnn_out_features = cnn.classifier.in_features
            cnn.classifier = nn.Identity()
            self.cnn = cnn
        else:
            raise ValueError("Invalid CNN model")

        if FREEZE_BACKBONE:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.gru = nn.GRU(self.cnn_out_features,
                          hidden_size,
                          num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        feats = self.cnn(x)
        feats = F.adaptive_avg_pool2d(feats, 1).view(b, t, -1)
        out, _ = self.gru(feats)
        out = self.dropout(out[:, -1])
        return self.fc(out)


def get_transform(resize=(112, 112), augment=False):
    transforms_list = [
        transforms.ToPILImage(),
        transforms.Resize(resize),
    ]

    if augment:
        transforms_list.extend([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(transforms_list)


def preprocess_frames(frames, seq_len=16, resize=(112, 112), augment=False):
    transform = get_transform(resize=resize, augment=augment)
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    total_frames = len(rgb_frames)

    if total_frames >= seq_len:
        indices = np.linspace(0, total_frames - 1, seq_len, dtype=int)
    else:
        indices = np.pad(np.arange(total_frames), (0, seq_len - total_frames), mode='wrap')

    sampled_frames = [rgb_frames[i] for i in indices]
    transformed_frames = [transform(frame) for frame in sampled_frames]
    frames_tensor = torch.stack(transformed_frames)  # [T, C, H, W]
    return frames_tensor


def load_action_model(model_path="best_model.pt", device='cpu',
                      num_classes=5, hidden_size=128):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None
    model = CNN_GRU(num_classes=num_classes, hidden_size=hidden_size)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model from {model_path} on {device}")
    return model


def predict_action(model, frames_tensor, label_map_path="label_map.json", device="cpu", top_k=3):
    if model is None:
        return {"action": "Model not loaded", "confidence": 0.0, "top_predictions": []}

    idx_to_class = {}
    if os.path.exists(label_map_path):
        try:
            with open(label_map_path, 'r') as f:
                label_map = json.load(f)
            idx_to_class = {v: k for k, v in label_map.items()}
            print(f"[INFO] Loaded {len(idx_to_class)} classes from {label_map_path}")
        except Exception as e:
            print(f"[WARNING] Could not load label map: {e}")

    if not idx_to_class and UCF101_AVAILABLE:
        idx_to_class = {i: class_name for i, class_name in enumerate(UCF101_CLASSES)}
        print("[INFO] Using default UCF101 class mapping")
    elif not idx_to_class:
        idx_to_class = {0: 'CricketShot', 1: 'PlayingCello', 2: 'Punch',
                        3: 'ShavingBeard', 4: 'TennisSwing'}
        print("[WARNING] Using minimal default labels.")

    try:
        with torch.no_grad():
            frames_tensor = frames_tensor.unsqueeze(0).to(device)  # [1, T, C, H, W]
            output = model(frames_tensor)
            probabilities = torch.softmax(output, dim=1)
            top_k_probs, top_k_indices = torch.topk(probabilities, min(top_k, probabilities.size(1)))

            predicted_idx = top_k_indices[0][0].item()
            predicted_class = idx_to_class.get(predicted_idx, f"Class_{predicted_idx}")
            confidence = top_k_probs[0][0].item()

            top_predictions = [
                {"class": idx_to_class.get(idx.item(), f"Class_{idx.item()}"),
                 "confidence": prob.item()}
                for prob, idx in zip(top_k_probs[0], top_k_indices[0])
            ]

            return {
                "action": predicted_class,
                "confidence": confidence,
                "top_predictions": top_predictions
            }
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return {"action": "Error", "confidence": 0.0, "top_predictions": []}


def log_action_prediction(action_label, confidence, log_file="logs/action_log.txt"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] ACTION: {action_label} (confidence: {confidence:.2f})\n")