import torch
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from config import BATCH_SIZE, NUM_CLASSES
from dataset import test_dataset
from model import build_model

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
RUN_ID_FILE = Path(__file__).resolve().parent / "wandb_run_id.txt"


def evaluate():
    #Continues models training W&B record for evaluation
    run_id = RUN_ID_FILE.read_text().strip()
    checkpoint_path = CHECKPOINT_DIR / f"best_model_{run_id}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    wandb.init(project="security-classifier", id=run_id, resume="must")

    #Uses GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Builds model again and loads the trained weights on to it for evaluation
    model = build_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    #Wraps the test dataset in a data loader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    class_names = test_dataset.classes
    print(f"Class order: {test_dataset.class_to_idx}")

    #Relies on shuffle=False to keep images in order so misclassified images can be viewed on W&B
    sample_paths = [p for p, _ in test_dataset.samples]

    all_preds, all_labels, misclassified = [], [], []
    idx = 0

    #Run through dataset and collect predictions
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append((sample_paths[idx + i], labels[i].item(), preds[i].item()))
            idx += len(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(NUM_CLASSES)
    )

    #Print results and send them to wandb
    print(f"Test accuracy: {accuracy:.4f}")
    for i, name in enumerate(class_names):
        print(f"{name}: precision={precision[i]:.3f} recall={recall[i]:.3f} f1={f1[i]:.3f} support={support[i]}")

    wandb.summary["test_accuracy"] = accuracy
    for i, name in enumerate(class_names):
        wandb.summary[f"test_precision_{name}"] = precision[i]
        wandb.summary[f"test_recall_{name}"] = recall[i]
        wandb.summary[f"test_f1_{name}"] = f1[i]

    wandb.log({
        "test_confusion_matrix": wandb.plot.confusion_matrix(
            preds=all_preds.tolist(), y_true=all_labels.tolist(), class_names=class_names
        )
    })

    misclass_table = wandb.Table(columns=["image", "true_label", "predicted_label"])
    for path, true_idx, pred_idx in misclassified:
        misclass_table.add_data(wandb.Image(str(path)), class_names[true_idx], class_names[pred_idx])
    wandb.log({"misclassified_examples": misclass_table})

    print(f"{len(misclassified)} misclassified out of {len(all_labels)} test images")

    wandb.finish()


if __name__ == "__main__":
    evaluate()