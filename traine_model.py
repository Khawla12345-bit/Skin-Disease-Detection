import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
from PIL import Image

# =========================
# 1. CONFIG
# =========================
BATCH_SIZE = 32
EPOCHS = 25
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================
# 3. LOAD DATA (HF)
# =========================
hf_dataset = load_dataset("ahmed-ai/skin-lesions-classification-dataset")

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"]

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        label = item["label"]

        if self.transform:
            img = self.transform(img)

        return img, label

train_dataset = HFDataset(hf_dataset["train"], train_transform)
val_dataset = HFDataset(hf_dataset["validation"], val_transform)

# =========================
# 4. SAMPLER (imbalance)
# =========================
targets = [hf_dataset["train"][i]["label"] for i in range(len(hf_dataset["train"]))]

counts = Counter(targets)
weights = 1. / torch.tensor([counts[i] for i in range(len(counts))], dtype=torch.float)
samples_weights = weights[targets]

sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =========================
# 5. MODEL
# =========================
model = models.mobilenet_v2(weights="IMAGENET1K_V1")

# Freeze layers
for param in model.features.parameters():
    param.requires_grad = False

# New classifier
model.classifier[1] = nn.Sequential(
    nn.Linear(model.last_channel, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 14)
)

model.to(device)

# =========================
# 6. OPTIMIZER
# =========================
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)

criterion = nn.CrossEntropyLoss()

# =========================
# 7. TRAINING
# =========================
best_acc = 0

for epoch in range(EPOCHS):
    print(f"\n================ Epoch {epoch+1}/{EPOCHS} ================")

    # ===== TRAIN =====
    model.train()
    correct, total = 0, 0

    loop = tqdm(train_loader)

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        loop.set_postfix(train_acc=f"{100*correct/total:.2f}%")

    train_acc = 100 * correct / total
    print(f"🎯 Train Accuracy: {train_acc:.2f}%")

    # ===== VALIDATION =====
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"📊 Validation Accuracy: {val_acc:.2f}%")

    scheduler.step(val_acc)

    # SAVE BEST MODEL
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"🔥 NEW BEST MODEL: {best_acc:.2f}%")

# =========================
# FINAL RESULT
# =========================
print("\n🏆 FINAL RESULTS")
print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Validation Accuracy: {val_acc:.2f}%")

