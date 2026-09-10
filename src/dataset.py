import random
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
from config import ROTATION_DEGREES, ZOOM_OUT_PROB, ZOOM_OUT_SCALE_RANGE, ZOOM_OUT_FILL


#Assigns directory route correctly
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


class RandomZoomOut:
    def __init__(self, scale_range=(0.6, 1.0), probability=0.4, fill=(128, 128, 128)):
        self.scale_range = scale_range
        self.probability = probability
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return img

        #Gets original images width
        original_size = img.size[0]

        #Pick scale factor within range
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        #Shrinks image to new scale
        new_size = int(original_size * scale)
        shrunk = img.resize((new_size, new_size), Image.LANCZOS)

        #Creates canvas the size of original image of the selected fill colour
        canvas = Image.new("RGB", (original_size, original_size), self.fill)

        #Places the image on the canvas
        max_offset = original_size - new_size
        x_offset = random.randint(0, max_offset)
        y_offset = random.randint(0, max_offset)
        canvas.paste(shrunk, (x_offset, y_offset))

        return canvas


#Training data pipeline
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(ROTATION_DEGREES),
    RandomZoomOut(scale_range=ZOOM_OUT_SCALE_RANGE, probability=ZOOM_OUT_PROB, fill=ZOOM_OUT_FILL),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#Test and Val data pipeline
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#Assigns dataset folder and transform type required to a variable
train_dataset = ImageFolder(root=PROCESSED_ROOT / "Train", transform=train_transform)
val_dataset   = ImageFolder(root=PROCESSED_ROOT / "Val", transform=eval_transform)
test_dataset  = ImageFolder(root=PROCESSED_ROOT / "Test", transform=eval_transform)

if __name__ == "__main__":
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    images, labels = next(iter(loader))
    print(images.shape)
    print(labels)
