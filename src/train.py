import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image

DATA_DIR = "data/raw"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MODEL_DIR = "data/models"
os.makedirs(MODEL_DIR, exist_ok=True)
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
IMG_SIZE = 224
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")

class FoodDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, label

def load_samples():
    categories = sorted(os.listdir(IMAGES_DIR))
    label_map = {cat: i for i, cat in enumerate(categories)}
    print(f"Categories : {label_map}")
    samples = []
    for cat in categories:
        folder = os.path.join(IMAGES_DIR, cat)
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(folder, fname), label_map[cat]))
    print(f"Total images : {len(samples)}")
    return samples, label_map

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def train():
    samples, label_map = load_samples()
    num_classes = len(label_map)
    total = len(samples)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    n_test = total - n_train - n_val
    full_dataset = FoodDataset(samples, transform=train_transform)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    val_set.dataset = FoodDataset(samples, transform=val_transform)
    test_set.dataset = FoodDataset(samples, transform=val_transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        model.eval()
        correct, total_val = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total_val += labels.size(0)
        val_acc = correct / total_val
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_model.pth")
            print(f"  Meilleur modele sauvegarde (val_acc={val_acc:.4f})")

    model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model.pth"))
    model.eval()
    correct, total_test = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_test += labels.size(0)
    test_acc = correct / total_test
    print(f"\nTest Accuracy : {test_acc:.4f}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Val Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/training_curves.png")
    print(f"Courbes sauvegardees dans {MODEL_DIR}/training_curves.png")

if __name__ == "__main__":
    train()