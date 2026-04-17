import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =========================
# 1. LOAD MODEL
# =========================
checkpoint = torch.load("skin_model.pth", map_location=torch.device('cpu'))

classes = checkpoint['classes']

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Model loaded ✅")

# =========================
# 2. TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# 3. LOAD IMAGE
# =========================
image_path = "test.jpg"  # بدليها باسم الصورة تاعك

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

# =========================
# 4. PREDICTION
# =========================
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

disease_names = {
    0: "Actinic keratoses",
    1: "Basal cell carcinoma",
    2: "Benign keratosis-like lesions",
    3: "Chickenpox",
    4: "Cowpox",
    5: "Dermatofibroma",
    6: "Healthy",
    7: "HFMD",
    8: "Measles",
    9: "Melanocytic nevi",
    10: "Melanoma",
    11: "Monkeypox",
    12: "Squamous cell carcinoma",
    13: "Vascular lesions"
}

pred_idx = predicted.item()
print("Prediction index:", pred_idx)
print("Prediction disease:", disease_names[pred_idx])
model.eval()

import torch.nn.functional as F

outputs = model(image)
probs = F.softmax(outputs, dim=1)

# top 3 predictions
top3_prob, top3_idx = torch.topk(probs, 3)

print("\nTop 3 predictions:")
for i in range(3):
    idx = top3_idx[0][i].item()
    prob = top3_prob[0][i].item()
    print(f"{disease_names[idx]}: {prob:.4f}")