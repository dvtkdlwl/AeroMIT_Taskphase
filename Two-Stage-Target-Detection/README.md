# Real vs Fake (Two-Stage) Target Detection

Detects bullseye targets from a live camera feed and rejects fake ones (same bullseye shape, different colours). Built to recognise correct landing zones, intended to run on a compute-constrained edge device (such as a flight computer).

<p align="center">
  <img src="https://github.com/user-attachments/assets/5a110f35-6192-46e1-9623-6d7b675036b0" width="49%" alt="Fake target rejected" />
  <img src="https://github.com/user-attachments/assets/91eeba21-6c98-45fd-b7dc-4eb0b3e50d4f" width="49%" alt="Real target detected" />
</p>

## Pipeline

Each frame goes through two models:

1. A **MobileNetV3-Small classifier** checks whether the frame has a fake target. If it does, "FAKE TARGET DETECTED" is shown and the second stage is skipped.
2. Otherwise, a **YOLOv8s detector** looks for a real target and draws a box around it. If there's no target in the frame, nothing is drawn.

The classifier's "not fake" class includes both real targets and plain background, so it only has to catch fakes and YOLO handles the rest.

## Training

- **Classifier:** MobileNetV3-Small (ImageNet weights) fine-tuned on two classes, `fake` and `not_fake`, for 10 epochs.
- **Detector:** YOLOv8s fine-tuned on one class, `target`. Images without targets get empty label files (`make_empty_labels.py`) so the model learns what isn't a target.

<!-- ![Training results](assets/results.png) -->

## Files

- `run_pipeline.py`: live classifier → YOLO pipeline
- `train_classifier.py`: trains the classifier
- `make_empty_labels.py`: creates empty labels for background images
- `data.yaml`: YOLO dataset config
- `Target_Identification_YOLO.ipynb`: first version, YOLO only. It couldn't tell a fake target from an empty frame, which is why I added the classifier.
- `weights/`: trained classifier and YOLO weights
