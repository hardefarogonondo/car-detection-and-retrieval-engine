import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import batched_nms
import torchvision.transforms as T
import cv2
import numpy as np
import requests
import json
import os
import time
import math
from types import SimpleNamespace
from PIL import Image

class SSD300(nn.Module):
    def __init__(self, num_classes):
        super(SSD300, self).__init__()
        self.num_classes = num_classes
        vgg_features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        vgg_features[16] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        vgg_features[23] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.vgg_part1 = nn.ModuleList(vgg_features[:23])
        self.vgg_part2 = nn.ModuleList(vgg_features[23:30])
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.aux_convs = nn.ModuleList([
            nn.Sequential( # conv8
                nn.Conv2d(1024, 256, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True)
            ),
            nn.Sequential( # conv9
                nn.Conv2d(512, 128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True)
            ),
            nn.Sequential( # conv10
                nn.Conv2d(256, 128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=0), nn.ReLU(inplace=True)
            ),
            nn.Sequential( # conv11
                nn.Conv2d(256, 128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=0), nn.ReLU(inplace=True)
            )
        ])
        self.boxes_per_loc = [4, 6, 6, 6, 4, 4]
        self.loc_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        in_channels = [512, 1024, 512, 256, 256, 256]
        for i in range(len(in_channels)):
            self.loc_convs.append(nn.Conv2d(in_channels[i], self.boxes_per_loc[i] * 4, kernel_size=3, padding=1))
            self.cls_convs.append(nn.Conv2d(in_channels[i], self.boxes_per_loc[i] * self.num_classes, kernel_size=3, padding=1))
        self.init_weights()

    def init_weights(self):
        layers_to_init = [self.conv6, self.conv7, self.aux_convs, self.loc_convs, self.cls_convs]
        for layer_group in layers_to_init:
            for m in layer_group.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        features = []
        for layer in self.vgg_part1:
            x = layer(x)
        features.append(x)
        for layer in self.vgg_part2:
            x = layer(x)
        x = self.pool5(x)
        x = F.relu(self.conv6(x), inplace=True)
        x = F.relu(self.conv7(x), inplace=True)
        features.append(x)
        for conv_block in self.aux_convs:
            x = conv_block(x)
            features.append(x)
        locs, confs = [], []
        for i, feature in enumerate(features):
            loc = self.loc_convs[i](feature).permute(0, 2, 3, 1).contiguous()
            locs.append(loc.view(loc.size(0), -1))
            conf = self.cls_convs[i](feature).permute(0, 2, 3, 1).contiguous()
            confs.append(conf.view(conf.size(0), -1))
        locs = torch.cat(locs, 1).view(locs[0].size(0), -1, 4)
        confs = torch.cat(confs, 1).view(confs[0].size(0), -1, self.num_classes)
        return locs, confs

def generate_default_boxes():
    """Generate default boxes for SSD300."""
    fmap_dims = {"conv4_3": 38, "conv7": 19, "conv8_2": 10, "conv9_2": 5, "conv10_2": 3, "conv11_2": 1}
    obj_scales = {"conv4_3": 0.1, "conv7": 0.2, "conv8_2": 0.375, "conv9_2": 0.55, "conv10_2": 0.725, "conv11_2": 0.9}
    aspect_ratios = {
        "conv4_3": [1., 2., 0.5], "conv7": [1., 2., 3., 0.5, 0.333], "conv8_2": [1., 2., 3., 0.5, 0.333],
        "conv9_2": [1., 2., 3., 0.5, 0.333], "conv10_2": [1., 2., 0.5], "conv11_2": [1., 2., 0.5]
    }
    default_boxes = []
    fmaps = list(fmap_dims.keys())
    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx, cy = (j + 0.5) / fmap_dims[fmap], (i + 0.5) / fmap_dims[fmap]
                for ratio in aspect_ratios[fmap]:
                    default_boxes.append([cx, cy, obj_scales[fmap] * math.sqrt(ratio), obj_scales[fmap] / math.sqrt(ratio)])
                    if ratio == 1.:
                        try:
                            additional_scale = math.sqrt(obj_scales[fmap] * obj_scales[fmaps[k+1]])
                        except IndexError:
                            additional_scale = 1.
                        default_boxes.append([cx, cy, additional_scale, additional_scale])
    default_boxes = torch.tensor(default_boxes, dtype=torch.float32)
    default_boxes.clamp_(0, 1)
    return default_boxes

def post_process_predictions(predicted_locs, predicted_scores, default_boxes, num_classes, img_size, conf_thresh=0.5, nms_thresh=0.45):
    """Post-process raw SSD output to get final detections."""
    batch_size = predicted_locs.size(0)
    predicted_scores = F.softmax(predicted_scores, dim=2)
    db_cx, db_cy, db_w, db_h = default_boxes[:, 0], default_boxes[:, 1], default_boxes[:, 2], default_boxes[:, 3]
    pred_cx = db_cx.unsqueeze(0) + predicted_locs[:, :, 0] * 0.1 * db_w.unsqueeze(0)
    pred_cy = db_cy.unsqueeze(0) + predicted_locs[:, :, 1] * 0.1 * db_h.unsqueeze(0)
    pred_w = db_w.unsqueeze(0) * torch.exp(predicted_locs[:, :, 2] * 0.2)
    pred_h = db_h.unsqueeze(0) * torch.exp(predicted_locs[:, :, 3] * 0.2)
    pred_boxes_xmin = (pred_cx - pred_w / 2) * img_size
    pred_boxes_ymin = (pred_cy - pred_h / 2) * img_size
    pred_boxes_xmax = (pred_cx + pred_w / 2) * img_size
    pred_boxes_ymax = (pred_cy + pred_h / 2) * img_size
    pred_boxes = torch.stack([pred_boxes_xmin, pred_boxes_ymin, pred_boxes_xmax, pred_boxes_ymax], dim=2)
    all_batch_preds = []
    for i in range(batch_size):
        scores = predicted_scores[i][:, 1:]
        boxes = pred_boxes[i]
        conf_scores, conf_labels = torch.max(scores, dim=1)
        mask = conf_scores > conf_thresh
        if mask.sum() == 0:
            all_batch_preds.append({"boxes": torch.empty(0, 4), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.int64)})
            continue
        final_boxes = boxes[mask]
        final_scores = conf_scores[mask]
        final_labels = conf_labels[mask] + 1 # Add 1 to correct class index
        indices_to_keep = batched_nms(final_boxes, final_scores, final_labels.float(), nms_thresh)
        all_batch_preds.append({
            "boxes": final_boxes[indices_to_keep],
            "scores": final_scores[indices_to_keep],
            "labels": final_labels[indices_to_keep],
        })
    return all_batch_preds

def load_models_and_labels(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading SSD300 detection model...")
    ID_TO_CLASS = {0: "background", 1: "bus", 2: "car", 3: "microbus", 4: "motorbike", 5: "pickup-van", 6: "truck"}
    NUM_CLASSES = len(ID_TO_CLASS)
    detection_model = SSD300(num_classes=NUM_CLASSES)
    try:
        detection_model.load_state_dict(torch.load(config.model_path, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model weights not found at '{config.model_path}'. Please check the path.")
        return None, None, None, None, None, None
    detection_model.to(device)
    detection_model.eval()
    print("Custom SSD300 model loaded successfully.")
    default_boxes = generate_default_boxes().to(device)
    print("Loading ResNet50 classification model...")
    classification_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    classification_model.to(device)
    classification_model.eval()
    print("Loading ImageNet class labels...")
    try:
        response = requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json")
        response.raise_for_status()
        imagenet_class_index = json.loads(response.text)
        IMAGENET_CLASSES = {int(k): v[1] for k, v in imagenet_class_index.items()}
        print("Successfully loaded ImageNet class labels.")
    except Exception as e:
        print(f"Error loading ImageNet labels: {e}. Classification will show 'Unknown'.")
        IMAGENET_CLASSES = {}
    return device, detection_model, default_boxes, ID_TO_CLASS, classification_model, IMAGENET_CLASSES

def process_video(config, device, detection_model, default_boxes, ID_TO_CLASS, classification_model, IMAGENET_CLASSES):
    detection_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((config.img_size, config.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    classification_transform = T.Compose([
        T.ToPILImage(), T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if not os.path.exists(config.input):
        print(f"Error: Input video not found at {config.input}"); return
    cap = cv2.VideoCapture(config.input)
    if not cap.isOpened():
        print(f"Error: Could not open video {config.input}"); return
    output_dir = os.path.dirname(config.output); os.makedirs(output_dir, exist_ok=True)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.output, fourcc, fps, (frame_width, frame_height))
    frame_count = 0
    start_time = time.time()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("\nStarting video processing...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}/{total_frames}...")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = detection_transform(frame_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_locs, predicted_scores = detection_model(img_tensor)
            processed_preds = post_process_predictions(
                predicted_locs, predicted_scores, default_boxes,
                len(ID_TO_CLASS), config.img_size,
                conf_thresh=config.confidence
            )
        detections = processed_preds[0]
        for i in range(len(detections['boxes'])):
            box = detections['boxes'][i].cpu().numpy()
            label_id = detections['labels'][i].item()
            score = detections['scores'][i].item()
            class_name = ID_TO_CLASS.get(label_id, 'Unknown')
            if class_name != 'car':
                continue
            xmin = int(box[0] * (frame_width / config.img_size))
            ymin = int(box[1] * (frame_height / config.img_size))
            xmax = int(box[2] * (frame_width / config.img_size))
            ymax = int(box[3] * (frame_height / config.img_size))
            crop_ymin, crop_xmin = max(0, ymin), max(0, xmin)
            crop_ymax, crop_xmax = min(frame_height, ymax), min(frame_width, xmax)
            if crop_ymin >= crop_ymax or crop_xmin >= crop_xmax: continue
            car_crop = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            input_batch = classification_transform(car_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                output = classification_model(input_batch)
                _, predicted_idx = torch.max(output, 1)
                car_type_name = IMAGENET_CLASSES.get(predicted_idx.item(), "Unknown type")
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label_text = f"Car: {car_type_name} ({score:.2f})"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_ymin = max(ymin, text_height + 10)
            cv2.rectangle(frame, (xmin, label_ymin - text_height - 10), (xmin + text_width, label_ymin), (0, 255, 0), -1)
            cv2.putText(frame, label_text, (xmin, label_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        out.write(frame)
        if config.display:
            cv2.imshow('Video Processing', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    end_time = time.time()
    cap.release(); out.release(); cv2.destroyAllWindows()
    total_time = end_time - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print("\n" + "="*50)
    print("Processing Complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average processing FPS: {avg_fps:.2f}")
    print(f"Output video saved to: {config.output}")
    print("="*50)

if __name__ == "__main__":
    config_dict = {
        "input": "../references/traffic_test.mp4",
        "output": "../reports/tested_model.mp4",
        "model_path": "../models/logs/best_ssd_model.pth",
        "confidence": 0.5, # Confidence threshold for your SSD model
        "display": True,
        "img_size": 300, # Input size for your SSD model
    }
    config = SimpleNamespace(**config_dict)
    models_data = load_models_and_labels(config)
    if all(m is not None for m in models_data):
        device, detection_model, default_boxes, ID_TO_CLASS, classification_model, IMAGENET_CLASSES = models_data
        process_video(config, device, detection_model, default_boxes, ID_TO_CLASS, classification_model, IMAGENET_CLASSES)

