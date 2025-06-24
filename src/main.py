from models import SSD300, VGG16
from types import SimpleNamespace
from utils import generate_default_boxes, post_process_predictions
import cv2
import os
import time
import torch
import torchvision.transforms as T

def load_models_and_labels(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading SSD300 detection model...")
    DETECTION_CLASS_MAP = {0: "background", 1: "bus", 2: "car", 3: "microbus", 4: "motorbike", 5: "pickup-van", 6: "truck"}
    detection_model = SSD300(num_classes=len(DETECTION_CLASS_MAP))
    try:
        detection_model.load_state_dict(torch.load(config.ssd_model_path, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: SSD model weights not found at '{config.ssd_model_path}'."); return (None,)*6
    detection_model.to(device); detection_model.eval()
    print("SSD300 model loaded successfully.")
    default_boxes = generate_default_boxes().to(device)
    print("Loading VGG16 classification model...")
    CLASSIFICATION_CLASS_MAP = {
        0: "Toyota Inova", 1: "Toyota Fortuner", 2: "Toyota Alphard", 3: "Suzuki Ertiga",
        4: "Mitsubishi Xpander", 5: "Honda Freed", 6: "Honda Brio", 7: "Daihatsu Ayla",
        8: "Mitsubishi Pajero", 9: "Honda Jazz", 10: "Toyota Camry",
        11: "New Toyota Inova", 12: "Toyota Avanza", 13: "Unknown Vehicle"
    }
    classification_model = VGG16(num_classes=len(CLASSIFICATION_CLASS_MAP))
    try:
        classification_model.load_state_dict(torch.load(config.vgg_model_path, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: VGG model weights not found at '{config.vgg_model_path}'."); return (None,)*6
    classification_model.to(device); classification_model.eval()
    print("VGG16 model loaded successfully.")
    return device, detection_model, default_boxes, DETECTION_CLASS_MAP, classification_model, CLASSIFICATION_CLASS_MAP

def process_video(config, device, detection_model, default_boxes, DETECTION_CLASS_MAP, classification_model, CLASSIFICATION_CLASS_MAP):
    detection_transform = T.Compose([
        T.ToPILImage(), T.Resize((config.ssd_img_size, config.ssd_img_size)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    classification_transform = T.Compose([
        T.ToPILImage(), T.Resize((config.vgg_img_size, config.vgg_img_size)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if not os.path.exists(config.input):
        print(f"Error: Input video not found at {config.input}"); return
    cap = cv2.VideoCapture(config.input)
    if not cap.isOpened():
        print(f"Error: Could not open video {config.input}"); return
    output_dir = os.path.dirname(config.output)
    os.makedirs(output_dir, exist_ok=True)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(config.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    frame_count, start_time = 0, time.time()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("\nStarting video processing...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if frame_count % 30 == 0: print(f"Processing frame {frame_count}/{total_frames}...")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # --- Stage 1: Object Detection ---
        img_tensor_ssd = detection_transform(frame_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_locs, predicted_scores = detection_model(img_tensor_ssd)
            processed_preds = post_process_predictions(
                predicted_locs, predicted_scores, default_boxes, len(DETECTION_CLASS_MAP),
                config.ssd_img_size, conf_thresh=config.confidence
            )
        detections = processed_preds[0]
        for i in range(len(detections["boxes"])):
            box, label_id, score = detections["boxes"][i].cpu().numpy(), detections["labels"][i].item(), detections["scores"][i].item()
            class_name = DETECTION_CLASS_MAP.get(label_id, "Unknown")
            if class_name != "car": continue
            xmin = int(box[0] * (frame_width / config.ssd_img_size))
            ymin = int(box[1] * (frame_height / config.ssd_img_size))
            xmax = int(box[2] * (frame_width / config.ssd_img_size))
            ymax = int(box[3] * (frame_height / config.ssd_img_size))
            # --- Stage 2: Object Classification ---
            crop_ymin, crop_xmin = max(0, ymin), max(0, xmin)
            crop_ymax, crop_xmax = min(frame_height, ymax), min(frame_width, xmax)
            if crop_ymin >= crop_ymax or crop_xmin >= crop_xmax: continue
            car_crop = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            img_tensor_vgg = classification_transform(car_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                output = classification_model(img_tensor_vgg)
                _, predicted_idx = torch.max(output, 1)
                car_type_name = CLASSIFICATION_CLASS_MAP.get(predicted_idx.item(), "Unknown")
            # Draw results on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label_text = f"Car: {car_type_name} ({score:.2f})"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_ymin = max(ymin, text_height + 10)
            cv2.rectangle(frame, (xmin, label_ymin - text_height - 10), (xmin + text_width, label_ymin), (0, 255, 0), -1)
            cv2.putText(frame, label_text, (xmin, label_ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        out.write(frame)
        if config.display:
            cv2.imshow("Video Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    end_time = time.time(); cap.release(); out.release(); cv2.destroyAllWindows()
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
        "output": "../reports/model_demo.mp4",
        "ssd_model_path": "../backup/models/best_ssd_model.pth",
        "vgg_model_path": "../backup/models/best_vgg_model.pth",
        "confidence": 0.5,
        "display": True,
        "ssd_img_size": 300,
        "vgg_img_size": 224,
    }
    config = SimpleNamespace(**config_dict)
    models_data = load_models_and_labels(config)
    if all(m is not None for m in models_data):
        process_video(config, *models_data)