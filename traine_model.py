import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRANSFORMS
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# DATASET (HuggingFace)
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

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# MODEL
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.last_channel, 14)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# TRAIN
for epoch in range(10):
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

        loop.set_postfix(acc=100*correct/total)

    print(f"Epoch {epoch+1} Train Acc: {100*correct/total:.2f}%")

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

    print(f"Validation Acc: {100*correct/total:.2f}%\n")
