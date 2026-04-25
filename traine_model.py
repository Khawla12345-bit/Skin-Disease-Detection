# =========================


# =========================
# IMPORTS
# =========================
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
# CONFIG
# =========================
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.3,0.3,0.3,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================
# LOAD DATA
# =========================
dataset = load_dataset("ahmed-ai/skin-lesions-classification-dataset")

# ⚠️ healthy = 6 (كما قلت)
healthy_label = 6

# =========================
# REMOVE HEALTHY
# =========================
train_data = dataset["train"].filter(lambda x: x["label"] != healthy_label)
val_data = dataset["validation"].filter(lambda x: x["label"] != healthy_label)

# =========================
# REMAP LABELS (مهم)
# =========================
def remap(example):
    if example["label"] > healthy_label:
        example["label"] -= 1
    return example

train_data = train_data.map(remap)
val_data = val_data.map(remap)

# =========================
# DATASET CLASS
# =========================
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item["image"]
        label = item["label"]

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label

train_ds = MyDataset(train_data, train_transform)
val_ds = MyDataset(val_data, val_transform)

# =========================
# SAMPLER
# =========================
targets = [train_data[i]["label"] for i in range(len(train_data))]
counts = Counter(targets)

weights = 1. / torch.tensor([counts[i] for i in range(len(counts))], dtype=torch.float)
samples_weights = weights[targets]

sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# =========================
# MODEL
# =========================
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Freeze شوية layers
for param in model.features[:5].parameters():
    param.requires_grad = False

model.classifier[1] = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 13)  # 🔥 13 classes
)

model.to(device)

# =========================
# OPTIMIZER
# =========================
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

criterion = nn.CrossEntropyLoss()

# =========================
# TRAINING
# =========================
best_acc = 0

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    # Unfreeze بعد 5 epochs
    if epoch == 5:
        for param in model.features.parameters():
            param.requires_grad = True
        print("🔥 Unfreezing all layers")

    # TRAIN
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

        _, preds = torch.max(outputs,1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        loop.set_postfix(train_acc=100*correct/total)

    train_acc = 100 * correct / total
    print(f"🎯 Train Acc: {train_acc:.2f}%")

    # VALIDATION
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            _, preds = torch.max(outputs,1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"📊 Val Acc: {val_acc:.2f}%")

    scheduler.step(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"🔥 BEST MODEL SAVED: {best_acc:.2f}%")

print(f"\n🏆 FINAL BEST: {best_acc:.2f}%")
