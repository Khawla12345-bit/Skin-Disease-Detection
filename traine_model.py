import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# =========================
# DATASET
# =========================
dataset = load_dataset("ahmed-ai/skin-lesions-classification-dataset")

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


train_ds = MyDataset(dataset["train"], train_transform)
val_ds = MyDataset(dataset["validation"], val_transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=True)

# =========================
# MODEL (MobileNetV2)
# =========================
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.last_channel, 14)
model = model.to(device)

# Freeze backbone (first stage training)
for param in model.features.parameters():
    param.requires_grad = False

# =========================
# LOSS + OPTIMIZER
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# =========================
# TRAINING
# =========================
num_epochs = 25

for epoch in range(num_epochs):

    # ===== TRAIN =====
    model.train()
    train_loss = 0
    correct, total = 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

        loop.set_postfix(loss=loss.item(), acc=100*correct/total)

    train_acc = 100 * correct / total
    train_loss = train_loss / len(train_loader)

    # ===== VALIDATION =====
    model.eval()
    val_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = 100 * correct / total
    val_loss = val_loss / len(val_loader)

    scheduler.step()

    print(f"""
========================
Epoch {epoch+1}/{num_epochs}
Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%
Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%
========================
""")

# =========================
# UNFREEZE (fine-tuning)
# =========================
print("Unfreezing backbone for fine-tuning...")

for param in model.features.parameters():
    param.requires_grad = True

# continue training if needed
