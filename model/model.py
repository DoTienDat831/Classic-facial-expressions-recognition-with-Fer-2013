"""
Real-time Facial Emotion Recognition (FER) webcam demo.

Loads the model + class mapping from `fer_checkpoint.pth` (saved after
training) and runs live predictions on faces detected in your webcam feed.

Run LOCALLY (not in Colab/Kaggle) -- this needs a real webcam + display:
    pip install opencv-python torch torchvision pillow
    python webcam_fer_test.py

Make sure `fer_checkpoint.pth` is in the same folder as this script.
Press 'q' to quit.
"""

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Rebuild the exact architecture used during training
# ---------------------------------------------------------------------------
def build_resnet18_fer(num_classes=7):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# 2. Load the trained checkpoint
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# weights_only=False: this checkpoint also stores class_to_idx + optimizer/
# scheduler state, not just tensors -- fine since it's a file you saved yourself
checkpoint = torch.load("model/fer_checkpoint.pth", map_location=device, weights_only=False)

model = build_resnet18_fer(num_classes=7)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
print(f"Loaded model (best val acc: {checkpoint['best_val_acc']:.3f})")
print("Classes:", idx_to_class)


# ---------------------------------------------------------------------------
# 3. Preprocessing -- must match training: grayscale, 224x224, mean/std=0.5
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ---------------------------------------------------------------------------
# 4. Face detection + webcam loop
# ---------------------------------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (check camera index / permissions)")

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face_crop = gray[y:y + h, x:x + w]
        pil_face = Image.fromarray(face_crop)

        input_tensor = transform(pil_face).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item() * 100

        label = f"{idx_to_class[pred_idx]} {confidence:.0f}%"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("FER Webcam Test - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()