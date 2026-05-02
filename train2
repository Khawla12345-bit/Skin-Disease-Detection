import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
from PIL import Image
import os

# =========================
# 1. الإعدادات العامة (CONFIG)
# =========================
torch.manual_seed(42)
BATCH_SIZE = 32
EPOCHS = 15  # زدنا عدد الدورات قليلاً مع وجود Early Stopping
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOP_PATIENCE = 7
HEALTHY_LABEL = 6  # الصنف المراد حذفه

# =========================
# 2. تحسين معالجة الصور (Enhanced Augmentation)
# =========================
# أضفنا تحسينات لمحاكاة الصور الخارجية (إضاءة مختلفة، زوايا مختلفة)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================
# 3. تحميل وتصفية البيانات
# =========================
print("⏳ Loading dataset...")
dataset = load_dataset("ahmed-ai/skin-lesions-classification-dataset")

def remap_labels(example):
    if example["label"] > HEALTHY_LABEL:
        example["label"] -= 1
    return example

# حذف صنف الـ Healthy وإعادة ترتيب الباقي
train_data = dataset["train"].filter(lambda x: x["label"] != HEALTHY_LABEL).map(remap_labels)
val_data = dataset["validation"].filter(lambda x: x["label"] != HEALTHY_LABEL).map(remap_labels)

class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img, label = item["image"], item["label"]
        if not isinstance(img, Image.Image): img = Image.fromarray(img)
        img = img.convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

train_ds = SkinDataset(train_data, train_transform)
val_ds = SkinDataset(val_data, val_transform)

# معالجة عدم توازن البيانات (Imbalance)
targets = train_data["label"]
counts = Counter(targets)
NUM_CLASSES = len(counts)
class_weights = 1. / torch.tensor([counts[i] for i in range(NUM_CLASSES)], dtype=torch.float)
samples_weights = class_weights[targets]
sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

# =========================
# 4. بناء الموديل (EfficientNet-V2-S)
# =========================
model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")

# تجميد الطبقات الأولى فقط لضمان تعلم الخصائص العميقة
for param in model.features[:3].parameters():
    param.requires_grad = False

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, NUM_CLASSES)
)
model.to(DEVICE)

# =========================
# 5. الخسارة والمحسن (Loss & Optimizer)
# =========================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # تقليل الثقة الزائدة للموديل
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

# =========================
# 6. حلقة التدريب (Training Loop)
# =========================
best_acc = 0
early_stop_counter = 0

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    
    # تدريب الموديل بالكامل بعد فترة معينة (Fine-tuning)
    if epoch == 10:
        for param in model.parameters():
            param.requires_grad = True
        for g in optimizer.param_groups:
            g['lr'] = LR * 0.1
        print("🔓 Unfrozen all layers for fine-tuning...")

    # --- TRAIN ---
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    loop = tqdm(train_loader, desc="Training")
    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()
        loop.set_postfix(acc=f"{(100 * train_correct / train_total):.2f}%")

    # --- VAL ---
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    avg_val_acc = 100 * val_correct / val_total
    print(f"📊 Val Acc: {avg_val_acc:.2f}% | Train Acc: {(100 * train_correct / train_total):.2f}%")

    scheduler.step(avg_val_acc)

    # حفظ أفضل موديل
    if avg_val_acc > best_acc:
        best_acc = avg_val_acc
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"⭐ New Best Model Saved! ({best_acc:.2f}%)")
    else:
        early_stop_counter += 1
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"🛑 Early Stopping at epoch {epoch+1}")
            break

print(f"\n🏆 Training Finished. Best Val Accuracy: {best_acc:.2f}%")
