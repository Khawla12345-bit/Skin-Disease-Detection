import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from tqdm import tqdm
import os

# 1. Configuration des dossiers
model_save_path = "checkpoints"
checkpoint_file = "last_training_state.pth" 
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# 2. Transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Chargement des données
train_dataset = datasets.ImageFolder("C:/Users/Dell/Desktop/DL/data/train", transform=train_transform)
val_dataset = datasets.ImageFolder("C:/Users/Dell/Desktop/DL/data/validation", transform=train_transform)

targets = torch.tensor(train_dataset.targets)
counts = Counter(train_dataset.targets)
weights = 1. / torch.tensor([counts[i] for i in range(len(counts))], dtype=torch.float)
samples_weights = weights[targets]
sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=16)

# 4. Modèle (MobileNetV2 pour CPU)
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.last_channel, 14) 

device = torch.device("cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ==========================================
# 5. Logique de Reprise (Resume Training)
# ==========================================
start_epoch = 0
best_acc = 0.0

if os.path.exists(checkpoint_file):
    print(f"--- Checkpoint trouvé ({checkpoint_file}), chargement en cours... ---")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    print(f"--- Reprise à partir de l'Epoch {start_epoch + 1} ---")
else:
    print("--- Aucun checkpoint trouvé, début du nouvel entraînement ---")

# 6. Boucle d'entraînement
num_epochs = 20

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    
    # tqdm en français
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for imgs, lbls in loop:
        imgs, lbls = imgs.to(device), lbls.to(device)
        
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(out, 1)
        total += lbls.size(0)
        correct += (predicted == lbls).sum().item()
        
        loop.set_postfix(acc=f"{100*correct/total:.2f}%")

    epoch_acc = 100 * correct / total

    # A) Sauvegarde de l'état complet pour la reprise (Resume)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, checkpoint_file)

    # B) Sauvegarde du point de contrôle spécifique à l'Epoch
    save_name = f"{model_save_path}/model_epoch_{epoch+1}_acc_{epoch_acc:.2f}.pth"
    torch.save(model.state_dict(), save_name)
    
    # C) Sauvegarde du meilleur modèle pour app.py
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"--- Nouveau record ! Meilleur modèle sauvegardé: {best_acc:.2f}% ---")