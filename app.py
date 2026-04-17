from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

app = Flask(__name__)

# 1. قائمة الأصناف مرتبة أبجدياً (نفس ترتيب مجلد train)
classes_list = [
    'Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
    'Chickenpox', 'Cowpox', 'Dermatofibroma', 'HFMD', 'Healthy',
    'Measles', 'Melanocytic nevi', 'Melanoma', 'Monkeypox',
    'Squamous cell carcinoma', 'Vascular lesions'
]

# 2. إعداد الموديل (لازم يكون MobileNetV2 كما في التدريب)
device = torch.device('cpu')
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 14) # 14 صنف

# تحميل الأوزان (استعمال weights_only=True لتفادي التحذير)
checkpoint = torch.load("best_model.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

# 3. التحويلات (Transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

    # الحصول على أفضل 3 نتائج
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