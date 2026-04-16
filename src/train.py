import os
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# -------------------------
# Configuration
# -------------------------
DATA_DIR    = "data/raw/images"
MODEL_DIR   = "models"
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
BEST_MODEL   = os.path.join(MODEL_DIR, "best_model.pth")

NUM_EPOCHS  = 15
BATCH_SIZE  = 16
LR          = 1e-4
NUM_WORKERS = 0          # 0 = safe on Windows
SEED        = 42

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (le reste)

os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(SEED)

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# -------------------------
# Transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# -------------------------
# Dataset & splits
# -------------------------
full_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=train_transform,
    is_valid_file=lambda p: p.lower().endswith((".jpg", ".jpeg", ".png"))
)

classes = full_dataset.classes
num_classes = len(classes)
print(f"Classes ({num_classes}) : {classes}")
print(f"Total images : {len(full_dataset)}")

n_total = len(full_dataset)
n_train = int(n_total * TRAIN_RATIO)
n_val   = int(n_total * VAL_RATIO)
n_test  = n_total - n_train - n_val

train_set, val_set, test_set = random_split(
    full_dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED)
)

# Appliquer le bon transform sur val et test
val_set.dataset  = copy.deepcopy(full_dataset)
test_set.dataset = copy.deepcopy(full_dataset)
val_set.dataset.transform  = val_test_transform
test_set.dataset.transform = val_test_transform

print(f"Split → train: {n_train} | val: {n_val} | test: {n_test}")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# -------------------------
# Modèle ResNet-18
# -------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Geler toutes les couches sauf la dernière
for param in model.parameters():
    param.requires_grad = False

# Remplacer la tête de classification
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# -------------------------
# Boucle d'entraînement
# -------------------------
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
best_weights = None

print("\n=== Début de l'entraînement ===")

for epoch in range(NUM_EPOCHS):

    # --- TRAIN ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)

    train_loss = running_loss / total
    train_acc  = correct / total

    # --- VAL ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss    += loss.item() * images.size(0)
            _, preds     = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_loss = val_loss / val_total
    val_acc  = val_correct / val_total

    scheduler.step()

    history["train_loss"].append(round(train_loss, 4))
    history["train_acc"].append(round(train_acc, 4))
    history["val_loss"].append(round(val_loss, 4))
    history["val_acc"].append(round(val_acc, 4))

    print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
          f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
          f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

    # Sauvegarder le meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(best_weights, BEST_MODEL)
        print(f"  ✔ Meilleur modèle sauvegardé (val_acc={val_acc:.4f})")

# -------------------------
# Évaluation sur le test set
# -------------------------
model.load_state_dict(best_weights)
model.eval()

test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        test_total   += labels.size(0)

test_acc = test_correct / test_total
print(f"\n=== Test accuracy : {test_acc:.4f} ===")

# -------------------------
# Sauvegarde des métriques
# -------------------------
metrics = {
    "classes": classes,
    "num_classes": num_classes,
    "epochs": NUM_EPOCHS,
    "best_val_acc": round(best_val_acc, 4),
    "test_acc": round(test_acc, 4),
    "history": history
}

with open(METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Métriques sauvegardées dans {METRICS_FILE}")
print(f"Modèle sauvegardé dans {BEST_MODEL}")