import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from inference import load_model, preprocess_image, CLASS_NAMES
from data_prep import center_crop_to_square
from config import IMAGE_SIZE, CURRENT_MODEL

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR.parent / "checkpoints" / CURRENT_MODEL
INPUT_DIR = BASE_DIR.parent / "gradcam_inputs"
OUTPUT_DIR = BASE_DIR.parent / "gradcam_outputs"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def class_heatmaps(model, tensor):
    """Returns class probabilities and one 224x224 heatmap per class."""
    feats = {}
    hook = model.layer4.register_forward_hook(lambda m, i, o: feats.update(maps=o))
    with torch.no_grad():
        logits = model(tensor)
    hook.remove()

    maps = feats["maps"][0]          #512 x 7 x 7 feature maps
    weights = model.fc.weight.detach()

    #Weight each map by how much it matters to each class, then add them up
    cams = (weights[:, :, None, None] * maps).sum(dim=1)   # 4 x 7 x 7
    #Keep only regions that push the score UP
    cams = torch.relu(cams)
    #Stretch 7x7 up to image size
    cams = F.interpolate(cams[None], size=(IMAGE_SIZE, IMAGE_SIZE),
                         mode="bilinear", align_corners=False)[0]
    cams = cams / (cams.max() + 1e-8)

    probs = torch.softmax(logits, dim=1)[0]
    return probs, cams


def show(model, device, image_path):
    pil = Image.open(image_path)
    probs, cams = class_heatmaps(model, preprocess_image(pil).to(device))

    shown = center_crop_to_square(pil.convert("RGB")).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    top = probs.argmax().item()

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
    axes[0].imshow(shown)
    axes[0].set_title("Input")
    for i, name in enumerate(CLASS_NAMES):
        ax = axes[i + 1]
        ax.imshow(shown)
        ax.imshow(cams[i].cpu(), cmap="jet", alpha=0.45, vmin=0, vmax=1)
        ax.set_title(f"{name} {probs[i] * 100:.1f}%",
                     fontweight="bold" if i == top else "normal")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(image_path.name)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{image_path.stem}_gradcam.png", dpi=120)
    plt.close(fig)
    print(f"Saved {image_path.stem}_gradcam.png")


if __name__ == "__main__":
    device = torch.device("cpu")
    model = load_model(CHECKPOINT_PATH, device)

    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        INPUT_DIR.mkdir(exist_ok=True)
        paths = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_TYPES)
        if not paths:
            print(f"Put some images in {INPUT_DIR} and run again.")

    for path in paths:
        show(model, device, path)