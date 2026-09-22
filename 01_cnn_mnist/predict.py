"""Predict a single MNIST-style image."""

import argparse

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from model import SmallCNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to a digit image")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    image = Image.open(args.image).convert("L")
    if image.getextrema()[0] > 127:
        image = ImageOps.invert(image)
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    input_tensor = transform(image).unsqueeze(0)

    model = SmallCNN()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        probabilities = model(input_tensor).softmax(dim=1)[0]
    prediction = int(probabilities.argmax())
    print(f"prediction={prediction} confidence={probabilities[prediction].item():.4f}")


if __name__ == "__main__":
    main()
