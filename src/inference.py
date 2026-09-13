import time
import torch
from PIL import Image
from torchvision import transforms
from model import build_model
from data_prep import center_crop_to_square
from config import IMAGE_SIZE, NUM_CLASSES


CLASS_NAMES = ["Animal", "Human", "Nothing", "Vehicle"]

infer_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(checkpoint_path, device):
    #Builds the model and adds the trained weights on to it
    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    #Replicates image processing used on the dataset
    img = pil_image.convert("RGB")
    img = center_crop_to_square(img)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    tensor = infer_transform(img)
    return tensor.unsqueeze(0)


def predict(model, device, pil_image: Image.Image) -> dict:
    #Predicts what uploaded image is with confidence ratings for each class
    t0 = time.time()
    tensor = preprocess_image(pil_image).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        print(f"Inference took {time.time() - t0:.2f}s", flush=True)
    return {name: round(prob * 100, 1) for name, prob in zip(CLASS_NAMES, probs)}