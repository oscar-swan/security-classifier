import shutil
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

CLASSES = ["Animal", "Human", "Nothing", "Vehicle"]
SPLITS = ["Train", "Val", "Test"]

TARGET_SIZE = 224
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def center_crop_to_square(img: Image.Image) -> Image.Image:
    """
    Crops image to a square by trimming width or height either side equally
    """
    w, h = img.size
    if w == h:
        return img

    if w > h:
        diff = w - h
        left = diff // 2
        right = w - (diff - left)
        box = (left, 0, right, h)
    else:
        diff = h - w
        top = diff // 2
        bottom = h - (diff - top)
        box = (0, top, w, bottom)

    return img.crop(box)


def process_image(src_path: Path, dst_path: Path) -> None:
    """
    Standardises image as RGB representation, then calls center_crop_to_square
    and resizes image to fit the target size and saves to processed folder.
    """
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = center_crop_to_square(img)
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path)


def process_dataset() -> None:
    """
    Wipes processed folder then checks every raw image in every folder to make
    sure it is a compatible image type then tries to process each image and
    logs any failed attempts and skips them.
    """
    #Wipes processed folder
    if PROCESSED_ROOT.exists():
        shutil.rmtree(PROCESSED_ROOT)
    PROCESSED_ROOT.mkdir(parents=True)

    total_ok = 0
    total_skipped = 0
    skipped_files = []

    for class_name in CLASSES:
        for split_name in SPLITS:
            src_dir = RAW_ROOT / class_name / split_name

            for src_path in sorted(src_dir.iterdir()):
                if src_path.suffix.lower() not in VALID_EXTENSIONS:
                    print(f"Not an image type: {src_path}")
                    total_skipped += 1
                    skipped_files.append(str(src_path))
                    continue

                dst_path = PROCESSED_ROOT / split_name / class_name / src_path.name

                try:
                    process_image(src_path, dst_path)
                    total_ok += 1
                except Exception as e:
                    print(f"Failed to process {src_path}: {e}")
                    total_skipped += 1
                    skipped_files.append(str(src_path))

    print("\n--- data_prep.py summary ---")
    print(f"Processed successfully: {total_ok}")
    print(f"Skipped: {total_skipped}")
    if skipped_files:
        print("Skipped files:")
        for f in skipped_files:
            print(f"  {f}")


if __name__ == "__main__":
    process_dataset()