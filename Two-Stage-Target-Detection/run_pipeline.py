#!/usr/bin/env python3
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from ultralytics import YOLO

# -----------------------------
# Paths (edit these as needed)
# -----------------------------
CLASSIFIER_WEIGHTS = "weights/fake_gate_mnv3.pth"

YOLO_WEIGHTS = "weights/best.pt"

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load classifier (MobileNetV3-Small)
# -----------------------------
classifier = models.mobilenet_v3_small(weights=None)

# Safer way to replace the final layer (last layer is at index -1)
in_features = classifier.classifier[-1].in_features
classifier.classifier[-1] = nn.Linear(in_features, 2)  # 2 classes

# Load weights
state = torch.load(CLASSIFIER_WEIGHTS, map_location=device)
classifier.load_state_dict(state)
classifier = classifier.to(device).eval()

# -----------------------------
# Preprocessing (match training)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# IMPORTANT: Order must match how you trained (ImageFolder alphabetical).
# If your training folders were `fake/` then `not_fake/` this likely maps to [0,1].
# Adjust these labels if they appear inverted in practice.
class_names = ["Fake", "Real"]

# -----------------------------
# Load YOLO
# -----------------------------
yolo_weights_path = str(Path(YOLO_WEIGHTS))
yolo = YOLO(yolo_weights_path)

# -----------------------------
# Helpers
# -----------------------------
def classify_frame(frame_bgr):
    """Return (pred_idx, prob_vector) for the whole frame."""
    # OpenCV BGR -> RGB -> PIL Image
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Transform -> (1, C, H, W)
    x = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(x)
        probs = torch.softmax(logits, dim=1)  # shape (1, 2)

    pred_idx = int(torch.argmax(probs, dim=1).item())
    prob_vec = probs.squeeze(0).cpu().tolist()  # [p_fake, p_real] if class_names == ["Fake","Real"]
    return pred_idx, prob_vec

def put_text(img, text, org=(30, 30), color=(255, 255, 0), scale=1.0, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# -----------------------------
# Webcam loop
# -----------------------------
cap = cv2.VideoCapture(0)  # change to a video path if needed
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0).")

last_t = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # 1) Classify the whole frame
    pred_idx, prob_vec = classify_frame(frame)
    pred_label = class_names[pred_idx]
    conf = prob_vec[pred_idx]  # confidence of the predicted class

    # Show classifier decision & confidence
    put_text(frame, f"Classifier: {pred_label} ({conf*100:.1f}%)", (30, 30), (255, 255, 0), 1.0, 2)

    # 2) If NOT Fake -> run YOLO to localize real targets
    if pred_label == "Fake":
        put_text(frame, "FAKE TARGET DETECTED", (30, 70), (0, 0, 255), 1.0, 2)
    else:
        try:
            # You can set conf threshold here if desired, e.g. yolo.predict(..., conf=0.4)
            results = yolo.predict(source=frame, verbose=False)
            frame = results[0].plot()  # draw YOLO boxes on the frame
        except Exception as e:
            # If YOLO fails for any reason, show the error once on-screen and continue
            put_text(frame, f"YOLO error: {str(e)[:60]}", (30, 70), (0, 0, 255), 0.7, 2)

    # 3) FPS overlay
    now = time.time()
    dt = now - last_t
    last_t = now
    fps = 1.0 / dt if dt > 0 else 0.0
    put_text(frame, f"FPS: {fps:.1f}", (30, 105), (0, 255, 0), 0.8, 2)

    cv2.imshow("Classifier → YOLO Pipeline", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
