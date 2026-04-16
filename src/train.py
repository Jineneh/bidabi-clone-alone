import os
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib
matplotlib.use("Agg")          # pas besoin d'écran (serveur / CI)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
#  Configuration
# ============================================================
DATA_DIR     = "data/raw/images"
MODEL_DIR    = "models"
REPORTS_DIR  = os.path.join(MODEL_DIR, "reports")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
BEST_MODEL   = os.path.join(MODEL_DIR, "best_model.pth")

NUM_EPOCHS  = 15
BATCH_SIZE  = 16
LR          = 1e-4
NUM_WORKERS = 0
SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
torch.manual_seed(SEED)

# Palette cohérente
COLORS = {
    "train"      : "#4F8EF7",
    "val"        : "#F7874F",
    "test"       : "#4FC78E",
    "background" : "#0F1117",
    "surface"    : "#1A1D2E",
    "text"       : "#E8EAF6",
    "accent"     : "#7C83FD",
    "grid"       : "#2A2D3E",
}

plt.rcParams.update({
    "figure.facecolor"  : COLORS["background"],
    "axes.facecolor"    : COLORS["surface"],
    "axes.edgecolor"    : COLORS["grid"],
    "axes.labelcolor"   : COLORS["text"],
    "xtick.color"       : COLORS["text"],
    "ytick.color"       : COLORS["text"],
    "text.color"        : COLORS["text"],
    "grid.color"        : COLORS["grid"],
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.5,
    "font.family"       : "monospace",
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
})

# ============================================================
#  Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*55}")
print(f"    Device : {device}")
print(f"{'='*55}")

# ============================================================
#  Transforms
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============================================================
#  Dataset & splits
# ============================================================
full_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=train_transform,
    is_valid_file=lambda p: p.lower().endswith((".jpg", ".jpeg", ".png"))
)

classes     = full_dataset.classes
num_classes = len(classes)
n_total     = len(full_dataset)
n_train     = int(n_total * TRAIN_RATIO)
n_val       = int(n_total * VAL_RATIO)
n_test      = n_total - n_train - n_val

print(f"\n   Classes ({num_classes}) : {classes}")
print(f"    Total images : {n_total}")
print(f"    Split → train: {n_train} | val: {n_val} | test: {n_test}\n")

train_set, val_set, test_set = random_split(
    full_dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED)
)

val_set.dataset        = copy.deepcopy(full_dataset)
test_set.dataset       = copy.deepcopy(full_dataset)
val_set.dataset.transform  = val_test_transform
test_set.dataset.transform = val_test_transform

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ============================================================
#  Visualisation 1 — Distribution du dataset
# ============================================================
def plot_dataset_distribution(classes, n_train, n_val, n_test, n_total):
    class_counts = [len([i for i in full_dataset.targets
                         if full_dataset.classes[i] == c]) for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(" Distribution du Dataset", fontsize=16,
                 color=COLORS["accent"], fontweight="bold", y=1.02)

    # Barres par classe
    ax = axes[0]
    bars = ax.bar(classes, class_counts,
                  color=[COLORS["train"], COLORS["val"], COLORS["test"]],
                  edgecolor=COLORS["accent"], linewidth=1.5, width=0.5)
    for bar, count in zip(bars, class_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(count), ha="center", va="bottom",
                color=COLORS["text"], fontsize=12, fontweight="bold")
    ax.set_title("Images par classe", pad=10)
    ax.set_ylabel("Nombre d'images")
    ax.set_ylim(0, max(class_counts) * 1.2)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    # Donut splits
    ax2 = axes[1]
    splits = [n_train, n_val, n_test]
    labels = [f"Train\n{n_train} imgs\n({n_train/n_total*100:.0f}%)",
              f"Val\n{n_val} imgs\n({n_val/n_total*100:.0f}%)",
              f"Test\n{n_test} imgs\n({n_test/n_total*100:.0f}%)"]
    wedge_colors = [COLORS["train"], COLORS["val"], COLORS["test"]]
    wedges, texts = ax2.pie(
        splits, labels=labels, colors=wedge_colors,
        startangle=90, wedgeprops=dict(width=0.5, edgecolor=COLORS["background"], linewidth=2),
        textprops=dict(color=COLORS["text"], fontsize=10)
    )
    ax2.set_title("Répartition Train / Val / Test", pad=10)
    ax2.text(0, 0, f"{n_total}\nimages", ha="center", va="center",
             fontsize=13, fontweight="bold", color=COLORS["accent"])

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "01_dataset_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=COLORS["background"])
    plt.close()
    print(f"   Graphe sauvegardé : {path}")

plot_dataset_distribution(classes, n_train, n_val, n_test, n_total)

# ============================================================
#  Modèle ResNet-18
# ============================================================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ============================================================
#  Boucle d'entraînement
# ============================================================
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
best_weights = None

print("="*55)
print("   Début de l'entraînement")
print("="*55)

for epoch in range(NUM_EPOCHS):

    # TRAIN
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

    # VAL
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

    flag = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(best_weights, BEST_MODEL)
        flag = "  ✔ meilleur modèle"

    print(f"  Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
          f"loss: {train_loss:.4f} acc: {train_acc:.4f} | "
          f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}{flag}")

# ============================================================
#  Visualisation 2 — Courbes d'entraînement
# ============================================================
def plot_training_curves(history, NUM_EPOCHS):
    epochs = range(1, NUM_EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(" Courbes d'Entraînement", fontsize=16,
                 color=COLORS["accent"], fontweight="bold")

    for ax, metric, title, ylabel in [
        (axes[0], "loss", "Loss (Train vs Val)", "Loss"),
        (axes[1], "acc",  "Accuracy (Train vs Val)", "Accuracy"),
    ]:
        ax.plot(epochs, history[f"train_{metric}"],
                color=COLORS["train"], linewidth=2.5,
                marker="o", markersize=5, label="Train")
        ax.plot(epochs, history[f"val_{metric}"],
                color=COLORS["val"], linewidth=2.5,
                marker="s", markersize=5, label="Val", linestyle="--")

        # Zone entre les deux courbes
        ax.fill_between(epochs,
                        history[f"train_{metric}"],
                        history[f"val_{metric}"],
                        alpha=0.1, color=COLORS["accent"])

        # Meilleure val_acc
        if metric == "acc":
            best_epoch = history["val_acc"].index(max(history["val_acc"])) + 1
            best_value = max(history["val_acc"])
            ax.axvline(x=best_epoch, color=COLORS["test"],
                       linestyle=":", linewidth=1.5, alpha=0.8)
            ax.annotate(f"Best val\n{best_value:.4f}",
                        xy=(best_epoch, best_value),
                        xytext=(best_epoch + 1, best_value - 0.03),
                        color=COLORS["test"], fontsize=9,
                        arrowprops=dict(arrowstyle="->",
                                        color=COLORS["test"]))

        ax.set_title(title, pad=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(facecolor=COLORS["surface"],
                  edgecolor=COLORS["grid"],
                  labelcolor=COLORS["text"])
        ax.grid(True)
        ax.set_xlim(1, NUM_EPOCHS)

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "02_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=COLORS["background"])
    plt.close()
    print(f"   Graphe sauvegardé : {path}")

plot_training_curves(history, NUM_EPOCHS)

# ============================================================
#  Évaluation finale sur le test set
# ============================================================
model.load_state_dict(best_weights)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

print(f"\n{'='*55}")
print(f"   Test accuracy : {test_acc:.4f}")
print(f"{'='*55}\n")

# ============================================================
#  Visualisation 3 — Matrice de confusion
# ============================================================
def plot_confusion_matrix(all_labels, all_preds, classes):
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("🔢 Matrice de Confusion", fontsize=16,
                 color=COLORS["accent"], fontweight="bold")

    for ax, data, title, fmt in [
        (axes[0], cm,      "Valeurs absolues", "d"),
        (axes[1], cm_norm, "Normalisée (%)",   ".2f"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, ax=ax,
            xticklabels=classes, yticklabels=classes,
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.5, linecolor=COLORS["background"],
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 13, "weight": "bold",
                       "color": COLORS["background"]},
        )
        ax.set_title(title, pad=10)
        ax.set_xlabel("Prédiction", labelpad=8)
        ax.set_ylabel("Vraie classe", labelpad=8)
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "03_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=COLORS["background"])
    plt.close()
    print(f"   Graphe sauvegardé : {path}")

plot_confusion_matrix(all_labels, all_preds, classes)

# ============================================================
#  Visualisation 4 — Rapport de classification (tableau)
# ============================================================
def plot_classification_report(all_labels, all_preds, classes, test_acc):
    report = classification_report(
        all_labels, all_preds,
        target_names=classes, output_dict=True
    )

    rows = []
    for cls in classes:
        r = report[cls]
        rows.append([
            cls,
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1-score']:.3f}",
            str(int(r['support'])),
        ])
    rows.append(["─" * 10] * 5)
    rows.append([
        "Accuracy", "", "", f"{test_acc:.3f}",
        str(len(all_labels))
    ])
    rows.append([
        "Macro avg",
        f"{report['macro avg']['precision']:.3f}",
        f"{report['macro avg']['recall']:.3f}",
        f"{report['macro avg']['f1-score']:.3f}",
        str(int(report['macro avg']['support'])),
    ])
    rows.append([
        "Weighted avg",
        f"{report['weighted avg']['precision']:.3f}",
        f"{report['weighted avg']['recall']:.3f}",
        f"{report['weighted avg']['f1-score']:.3f}",
        str(int(report['weighted avg']['support'])),
    ])

    col_labels = ["Classe", "Precision", "Recall", "F1-Score", "Support"]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    ax.axis("off")
    ax.set_title(" Rapport de Classification", fontsize=16,
                 color=COLORS["accent"], fontweight="bold", pad=20)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 2.2)

    header_color  = COLORS["accent"]
    row_colors    = [COLORS["surface"], "#1F2235"]
    special_color = "#2A3550"

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["grid"])
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color=COLORS["background"],
                                fontweight="bold")
        elif rows[row - 1][0] in ["Accuracy", "Macro avg", "Weighted avg"]:
            cell.set_facecolor(special_color)
            cell.set_text_props(color=COLORS["test"], fontweight="bold")
        elif rows[row - 1][0].startswith("─"):
            cell.set_facecolor(COLORS["background"])
            cell.set_text_props(color=COLORS["grid"])
        else:
            cell.set_facecolor(row_colors[(row - 1) % 2])
            cell.set_text_props(color=COLORS["text"])

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "04_classification_report.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=COLORS["background"])
    plt.close()
    print(f"   Graphe sauvegardé : {path}")

plot_classification_report(all_labels, all_preds, classes, test_acc)

# ============================================================
#  Visualisation 5 — Dashboard résumé
# ============================================================
def plot_summary_dashboard(history, test_acc, best_val_acc, classes,
                           n_train, n_val, n_test, NUM_EPOCHS):
    fig = plt.figure(figsize=(18, 10), facecolor=COLORS["background"])
    fig.suptitle(" Dashboard — Résumé de l'Entraînement v3.0",
                 fontsize=20, color=COLORS["accent"],
                 fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.35)

    epochs = range(1, NUM_EPOCHS + 1)

    # --- Loss ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history["train_loss"],
             color=COLORS["train"], linewidth=2, label="Train")
    ax1.plot(epochs, history["val_loss"],
             color=COLORS["val"], linewidth=2, linestyle="--", label="Val")
    ax1.set_title("Loss")
    ax1.legend(facecolor=COLORS["surface"],
               labelcolor=COLORS["text"],
               edgecolor=COLORS["grid"], fontsize=9)
    ax1.grid(True)

    # --- Accuracy ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history["train_acc"],
             color=COLORS["train"], linewidth=2, label="Train")
    ax2.plot(epochs, history["val_acc"],
             color=COLORS["val"], linewidth=2, linestyle="--", label="Val")
    ax2.axhline(test_acc, color=COLORS["test"],
                linestyle=":", linewidth=2, label=f"Test {test_acc:.3f}")
    ax2.set_title("Accuracy")
    ax2.legend(facecolor=COLORS["surface"],
               labelcolor=COLORS["text"],
               edgecolor=COLORS["grid"], fontsize=9)
    ax2.grid(True)

    # --- KPI cards ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    kpis = [
        (" Test Acc",     f"{test_acc:.2%}",     COLORS["test"]),
        (" Best Val Acc", f"{best_val_acc:.2%}", COLORS["val"]),
        (" Epochs",       str(NUM_EPOCHS),        COLORS["train"]),
        ("  Total imgs",  str(n_train+n_val+n_test), COLORS["accent"]),
    ]
    for i, (label, value, color) in enumerate(kpis):
        y = 0.85 - i * 0.22
        rect = FancyBboxPatch((0.05, y - 0.08), 0.9, 0.17,
                              boxstyle="round,pad=0.02",
                              facecolor=COLORS["surface"],
                              edgecolor=color, linewidth=2,
                              transform=ax3.transAxes)
        ax3.add_patch(rect)
        ax3.text(0.12, y + 0.01, label, transform=ax3.transAxes,
                 fontsize=10, color=COLORS["text"], va="center")
        ax3.text(0.88, y + 0.01, value, transform=ax3.transAxes,
                 fontsize=13, color=color, va="center",
                 ha="right", fontweight="bold")

    # --- Confusion matrix ---
    ax4 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax4,
                xticklabels=classes, yticklabels=classes,
                cmap="Blues", linewidths=0.5,
                linecolor=COLORS["background"],
                annot_kws={"size": 12, "weight": "bold",
                            "color": COLORS["background"]},
                cbar=False)
    ax4.set_title("Confusion Matrix")
    ax4.set_xlabel("Prédit")
    ax4.set_ylabel("Réel")

    # --- Per-class F1 ---
    ax5 = fig.add_subplot(gs[1, 1])
    report = classification_report(
        all_labels, all_preds,
        target_names=classes, output_dict=True
    )
    f1_scores = [report[c]["f1-score"] for c in classes]
    bar_colors = [COLORS["train"], COLORS["val"], COLORS["test"]]
    bars = ax5.barh(classes, f1_scores, color=bar_colors,
                    edgecolor=COLORS["accent"], height=0.5)
    for bar, score in zip(bars, f1_scores):
        ax5.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{score:.3f}", va="center", fontsize=11,
                 color=COLORS["text"], fontweight="bold")
    ax5.set_xlim(0, 1.15)
    ax5.set_title("F1-Score par classe")
    ax5.set_xlabel("F1-Score")
    ax5.grid(axis="x")
    ax5.set_axisbelow(True)

    # --- Tableau récapitulatif ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    table_data = [
        ["Split",  "Images", "Ratio"],
        ["Train",  n_train,  f"{n_train/(n_train+n_val+n_test):.0%}"],
        ["Val",    n_val,    f"{n_val/(n_train+n_val+n_test):.0%}"],
        ["Test",   n_test,   f"{n_test/(n_train+n_val+n_test):.0%}"],
        ["─"*8,   "─"*4,    "─"*5],
        ["Modèle", "ResNet-18", ""],
        ["LR",     LR,          ""],
        ["Batch",  BATCH_SIZE,  ""],
    ]
    tbl = ax6.table(cellText=table_data[1:],
                    colLabels=table_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(COLORS["grid"])
        if r == 0:
            cell.set_facecolor(COLORS["accent"])
            cell.set_text_props(color=COLORS["background"],
                                fontweight="bold")
        else:
            cell.set_facecolor(COLORS["surface"])
            cell.set_text_props(color=COLORS["text"])
    ax6.set_title("Hyperparamètres & Splits", pad=15)

    path = os.path.join(REPORTS_DIR, "05_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=COLORS["background"])
    plt.close()
    print(f"   Graphe sauvegardé : {path}")

plot_summary_dashboard(history, test_acc, best_val_acc, classes,
                       n_train, n_val, n_test, NUM_EPOCHS)

# ============================================================
#  Sauvegarde des métriques JSON
# ============================================================
report_dict = classification_report(
    all_labels, all_preds,
    target_names=classes, output_dict=True
)

metrics = {
    "classes"      : classes,
    "num_classes"  : num_classes,
    "epochs"       : NUM_EPOCHS,
    "best_val_acc" : round(best_val_acc, 4),
    "test_acc"     : round(test_acc, 4),
    "history"      : history,
    "per_class"    : {
        cls: {
            "precision": round(report_dict[cls]["precision"], 4),
            "recall"   : round(report_dict[cls]["recall"],    4),
            "f1-score" : round(report_dict[cls]["f1-score"],  4),
            "support"  : int(report_dict[cls]["support"]),
        } for cls in classes
    },
    "macro_avg": {
        k: round(v, 4)
        for k, v in report_dict["macro avg"].items()
        if k != "support"
    },
}

with open(METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=2)

# ============================================================
#  Résumé terminal
# ============================================================
print(f"\n{'='*55}")
print(f"   Entraînement terminé !")
print(f"{'='*55}")
print(f"  {'Métrique':<25} {'Valeur':>10}")
print(f"  {'─'*35}")
print(f"  {'Best Val Accuracy':<25} {best_val_acc:>10.4f}")
print(f"  {'Test Accuracy':<25} {test_acc:>10.4f}")
print(f"  {'─'*35}")
for cls in classes:
    f1 = report_dict[cls]["f1-score"]
    print(f"  {'F1 ' + cls:<25} {f1:>10.4f}")
print(f"  {'─'*35}")
print(f"  {'Macro F1':<25} {report_dict['macro avg']['f1-score']:>10.4f}")
print(f"{'='*55}")
print(f"\n   Modèle    → {BEST_MODEL}")
print(f"   Métriques → {METRICS_FILE}")
print(f"    Rapports  → {REPORTS_DIR}/")
print(f"      ├── 01_dataset_distribution.png")
print(f"      ├── 02_training_curves.png")
print(f"      ├── 03_confusion_matrix.png")
print(f"      ├── 04_classification_report.png")
print(f"      └── 05_dashboard.png")
print(f"\n{'='*55}\n")