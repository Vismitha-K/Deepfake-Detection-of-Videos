import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from moe_model import MoEModel
from src.data.dataset import get_dataloaders


def freeze_experts(model):
    """Freeze ResNet and MobileNet parameters so only the gate trains."""
    for param in model.resnet.parameters():
        param.requires_grad = False
    for param in model.mobilenet.parameters():
        param.requires_grad = False
    for param in model.gate.parameters():
        param.requires_grad = True
    print("[Info] Experts frozen. Only gating network is trainable.")


def train_gate(model, train_loader, val_loader, device, epochs=3, lr=1e-4):
    """Train only the gating network with frozen experts."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    best_acc = 0.0
    save_dir = "../checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        total_gate_weights = torch.zeros(2, device=device)

        # -------------------- Training --------------------
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)

            out = model(imgs)
            loss = criterion(out["logits"], labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = out["logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # accumulate gate weights for analysis
            total_gate_weights += out["weights"].mean(dim=0)

        train_acc = correct / total
        train_loss = total_loss / total
        avg_gate_weights = (total_gate_weights / len(train_loader)).detach().cpu().numpy()

        # -------------------- Validation --------------------
        model.eval()
        with torch.no_grad():
            total_loss, correct, total = 0.0, 0, 0
            total_gate_weights = torch.zeros(2, device=device)

            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out["logits"], labels)
                total_loss += loss.item() * imgs.size(0)
                preds = out["logits"].argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                total_gate_weights += out["weights"].mean(dim=0)

            val_acc = correct / total
            val_loss = total_loss / total
            avg_val_gate_weights = (total_gate_weights / len(val_loader)).detach().cpu().numpy()

        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"  Val   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        print(f"  Avg Gate Weights (Train): ResNet={avg_gate_weights[0]:.3f}, MobileNet={avg_gate_weights[1]:.3f}")
        print(f"  Avg Gate Weights (Val):   ResNet={avg_val_gate_weights[0]:.3f}, MobileNet={avg_val_gate_weights[1]:.3f}")

        # -------------------- Checkpoint Saving --------------------
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "moe_gate_frozen.pth"),
            )
            print(f"[Checkpoint] Saved best model (Val Acc: {val_acc:.4f})")

    print(f"[Training Complete] Best Validation Accuracy: {best_acc:.4f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MoEModel(
        resnet_ckpt="../checkpoints/resnet50/resnet50_best.pth",
        mobilenet_ckpt="../checkpoints/mobilenet_v3_large/mobilenet_v3_large_best.pth",
        device=device,
    ).to(device)

    freeze_experts(model)

    train_loader, val_loader = get_dataloaders(
        data_root="../celebdf_frames",
        batch_size=32,
        val_split=0.2,
        num_workers=4,
    )

    train_gate(model, train_loader, val_loader, device)


if __name__ == "__main__":
    main()