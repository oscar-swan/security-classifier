import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset import train_dataset, val_dataset
from model import build_model
import wandb
from pathlib import Path

#Adds route to store model
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"

#W&B tracking
wandb.init(
    project="security-classifier",
    config={
        "epochs": 50,
        "patience": 7,
        "batch_size": 32,
        "lr_head": 1e-3,
        "lr_backbone": 1e-4,
        "optimizer": "AdamW",
    }
)

run_id = wandb.run.id
with open("wandb_run_id.txt", "w") as f:
    f.write(run_id)

#Selects GPU to use but falls back on CPU if no GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#Builds model and moves it to selected device
model = build_model(num_classes=4)
model = model.to(device)

#Loss function
criterion = nn.CrossEntropyLoss()

#Assigns learning rate to each layer
optimizer = AdamW([
    {"params": model.fc.parameters(), "lr": 1e-3},
    {"params": model.layer3.parameters(), "lr": 1e-4},
    {"params": model.layer4.parameters(), "lr": 1e-4},
])

best_val_loss = float("inf")
patience = 7
epochs_no_improve = 0

for epoch in range(50):

    #Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    #Validation phase
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    })

    #Early stop mechanism
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), CHECKPOINT_PATH)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

wandb.finish()