"""Train a small CNN on MNIST with CPU or CUDA."""

import argparse
import random
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_fn(logits, labels)
        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index, e.g. 0 for the RTX 3090")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
            raise ValueError(f"--gpu must be between 0 and {torch.cuda.device_count() - 1}")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    checkpoint_dir = project_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    model = SmallCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_accuracy = 0.0

    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_fn, device, optimizer
        )
        test_loss, test_accuracy = run_epoch(model, test_loader, loss_fn, device)
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "accuracy": best_accuracy,
                    "epoch": epoch,
                },
                checkpoint_dir / "best.pt",
            )

    print(f"saved={checkpoint_dir / 'best.pt'} best_test_acc={best_accuracy:.4f}")


if __name__ == "__main__":
    main()
