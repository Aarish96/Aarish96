TrafficViolationDetection/
│
├── data/                   # Folder for storing video datasets
├── scripts/                # Folder for scripts
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── violation_detection.py
│   └── evaluation.py
├── models/                 # Folder for saving trained models
├── outputs/                # Folder for saving outputs like results and logs
├── README.md               # Documentation file
└── requirements.txt        # Required libraries
numpy
opencv-python
matplotlib
torch
torchvision
pandas
scikit-learn
import cv2
import os
import random
import numpy as np

def extract_frames(video_path, output_dir, frame_rate=5):
    """Extract frames from video at the specified frame rate."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    success = True

    while success:
        success, frame = cap.read()
        if count % frame_rate == 0 and success:
            frame_path = os.path.join(output_dir, f'frame_{count}.jpg')
            cv2.imwrite(frame_path, frame)
        count += 1

    cap.release()
    print(f'Frames extracted to {output_dir}')

def augment_image(image, brightness=0.5):
    """Apply brightness augmentation to the image."""
    augmented = cv2.convertScaleAbs(image, alpha=1, beta=random.uniform(-brightness, brightness))
    return augmented

# Example Usage
extract_frames('data/video1.mp4', 'data/frames/')
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

def train_model(train_loader, model, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
        
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

# Initialize model, optimizer, and dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Example Training Call
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# train_model(train_loader, model, optimizer)
import cv2
import torch

def detect_violations(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    writer = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Object detection
        inputs = [transforms.ToTensor()(frame).to(device)]
        outputs = model(inputs)

        # Draw bounding boxes for detected vehicles
        for box in outputs[0]['boxes']:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Write frame to output video
        if writer is None:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (frame.shape[1], frame.shape[0]))
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Violation detection video saved at {output_path}")

# Example Call
# detect_violations(model, 'data/video1.mp4', 'outputs/violation_output.avi')
from sklearn.metrics import precision_score, recall_score

def evaluate_model(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

# Example Evaluation Call
# true_labels = [...]
# predicted_labels = [...]
# evaluate_model(true_labels, predicted_labels)
