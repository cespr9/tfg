"""Entry point de entrenamiento. Incluye DummyDataset para pruebas sin datos."""
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from models.build import build_model
from training.losses import build_loss
from utils.metrics import compute_metrics


class DummyDataset(Dataset):
    """Dataset falso para validar el pipeline sin datos reales."""

    def __init__(self, n_samples=16, num_classes=5, image_size=224):
        self.n_samples = n_samples
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        image = torch.randn(3, self.image_size, self.image_size)
        label = torch.randint(0, 2, (self.num_classes,)).float()
        return image, label


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    import numpy as np
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return compute_metrics(y_true, y_prob)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="densenet121", choices=["resnet50", "densenet121"])
    parser.add_argument("--strategy", default="full", choices=["feature_extraction", "partial", "full"])
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(args.model, args.num_classes, pretrained=True, freeze_strategy=args.strategy).to(device)
    criterion = build_loss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    train_loader = DataLoader(DummyDataset(num_classes=args.num_classes), batch_size=args.batch_size)
    val_loader = DataLoader(DummyDataset(num_classes=args.num_classes), batch_size=args.batch_size)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: loss={loss:.4f} metrics={metrics}")


if __name__ == "__main__":
    main()
