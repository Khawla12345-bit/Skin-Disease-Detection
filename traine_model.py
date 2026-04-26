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
# 1. CONFIGURATION
# =========================
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. DATA AUGMENTATION (Avoid Overfitting)
# =========================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), # مهم جداً لصور الجلد
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================
# 3. LOAD & CLEAN DATA
# =========================
dataset = load_dataset("ahmed-ai/skin-lesions-classification-dataset")
healthy_label = 6

def remap(example):
    if example["label"] > healthy_label:
        example["label"] -= 1
    return example

train_data = dataset["train"].filter(lambda x: x["label"] != healthy_label).map(remap)
val_data = dataset["validation"].filter(lambda x: x["label"] != healthy_label).map(remap)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img, label = item["image"], item["label"]
        if not isinstance(img, Image.Image): img = Image.fromarray(img)
        if self.transform: img = self.transform(img)
        return img, label

train_ds = MyDataset(train_data, train_transform)
val_ds = MyDataset(val_data, val_transform)

# Sampler لتحقيق التوازن بين الأصناف
targets = [train_data[i]["label"] for i in range(len(train_data))]
counts = Counter(targets)
weights = 1. / torch.tensor([counts[i] for i in range(len(counts))], dtype=torch.float)
samples_weights = weights[targets]
sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# =========================
# 4. MODEL (EfficientNet-V2-S)
# =========================
model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")

# Freeze early layers initially
for param in model.features[:4].parameters():
    param.requires_grad = False

num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 13) 
)
model.to(DEVICE)

# =========================
# 5. LOSS & OPTIMIZER
# =========================
# Label Smoothing تساعد الموديل باش ما يحفظش الصور حرفياً
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

# =========================
# 6. TRAINING LOOP
# =========================
best_acc = 0

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    
    # Unfreeze all layers after epoch 5 for fine-tuning
    if epoch == 5:
        for param in model.parameters():
            param.requires_grad = True
        print("🔓 Unfrozen all layers for fine-tuning...")

    # --- TRAIN PHASE ---
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    
    loop = tqdm(train_loader, desc="Training")
    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100*train_correct/train_total)

    avg_train_loss = train_loss / train_total
    avg_train_acc = 100 * train_correct / train_total

    # --- VAL PHASE ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    avg_val_loss = val_loss / val_total
    avg_val_acc = 100 * val_correct / val_total

    # --- PRINT RESULTS ---
    print(f"\n📊 Results Epoch {epoch+1}:")
    print(f"   [Train] Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.2f}%")
    print(f"   [Val]   Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.2f}%")

    scheduler.step(avg_val_acc)

    if avg_val_acc > best_acc:
        best_acc = avg_val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"⭐ New Best Model Saved! ({best_acc:.2f}%)")

print(f"\n🏆 Training Finished! Final Best Accuracy: {best_acc:.2f}%")
