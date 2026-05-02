from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

app = Flask(__name__)

# ❗ نفس الكلاسات بدون Healthy (كما في training)
classes_list = [
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis',
    'Chickenpox',
    'Cowpox',
    'Dermatofibroma',
    'HFMD',
    'Measles',
    'Melanocytic nevi',
    'Melanoma',
    'Monkeypox',
    'Squamous cell carcinoma',
    'Vascular lesions'
]

NUM_CLASSES = len(classes_list)

# ✅ نفس الموديل EXACTLY كما في training
device = torch.device("cpu")

model = models.efficientnet_v2_s(weights=None)

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, NUM_CLASSES)
)

# تحميل weights
model.load_state_dict(torch.load("best_modell.pth", map_location=device))
model.eval()

# نفس transform تاع validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api/predict/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files['file']
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

    top3_prob, top3_idx = torch.topk(probs, 3)

    results = []
    for i in range(3):
        idx = top3_idx[0][i].item()
        prob = top3_prob[0][i].item()

        results.append({
            "label": classes_list[idx],
            "confidence": round(prob * 100, 2)
        })

    return {"top3": results}

if __name__ == "__main__":
    app.run(debug=True)
