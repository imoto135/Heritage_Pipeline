#!/usr/bin/env python3
"""
EfficientNet-B0 くずし字認識器の学習スクリプト
論文設定: input=224x224 letterbox, SGD lr=0.02, cosine annealing,
          weighted random sampling, early stopping patience=10
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="full_dataset/以下の書籍ディレクトリ群")
    p.add_argument("--book-list", required=True, help="学習に使う書籍IDのテキストファイル（1行1書籍）")
    p.add_argument("--out-dir", default="models/efficientnet", help="出力ディレクトリ")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=8)
    return p.parse_args()


class LetterboxResize:
    """アスペクト比保持リサイズ + 白パディング"""
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        out = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        out.paste(img, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return out


class CharDataset(torch.utils.data.Dataset):
    def __init__(self, samples, class_to_idx, transform):
        self.samples = samples  # [(path, class_idx), ...]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        return self.transform(img), label


def build_samples(data_dir: Path, book_list: list):
    """書籍ディレクトリからサンプルリストとクラスマップを構築"""
    class_counts = {}
    all_samples = []

    for book_id in book_list:
        chars_dir = data_dir / book_id / "characters"
        if not chars_dir.exists():
            print(f"[警告] {chars_dir} なし、スキップ")
            continue
        for class_dir in chars_dir.iterdir():
            if not class_dir.is_dir():
                continue
            cls = class_dir.name  # e.g. "U+3044"
            files = list(class_dir.glob("*.jpg"))
            class_counts[cls] = class_counts.get(cls, 0) + len(files)
            all_samples.extend((f, cls) for f in files)

    # クラスインデックス作成
    classes = sorted(class_counts.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = [(str(p), class_to_idx[c]) for p, c in all_samples]
    return samples, class_to_idx, class_counts


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 書籍リスト読み込み
    with open(args.book_list) as f:
        book_list = [l.strip() for l in f if l.strip()]

    print(f"学習書籍数: {len(book_list)}")
    data_dir = Path(args.data_dir)

    # サンプル構築
    print("サンプル収集中...")
    samples, class_to_idx, class_counts = build_samples(data_dir, book_list)
    num_classes = len(class_to_idx)
    print(f"クラス数: {num_classes}, サンプル数: {len(samples)}")

    # クラスマップ保存
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(out_dir / "classes.txt", "w") as f:
        for i in range(num_classes):
            f.write(idx_to_class[i] + "\n")
    with open(out_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, ensure_ascii=False)

    # Transform
    train_tf = transforms.Compose([
        LetterboxResize(args.img_size),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        LetterboxResize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # train/val split (90/10)
    np.random.seed(42)
    idx = np.random.permutation(len(samples))
    val_n = max(1, int(len(samples) * 0.1))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    # Weighted sampler（クラス不均衡対策）
    class_freq = np.array([class_counts.get(idx_to_class[i], 1) for i in range(num_classes)])
    sample_weights = np.array([1.0 / class_freq[s[1]] for s in train_samples])
    sampler = WeightedRandomSampler(sample_weights, len(train_samples), replacement=True)

    train_ds = CharDataset(train_samples, class_to_idx, train_tf)
    val_ds = CharDataset(val_samples, class_to_idx, val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # モデル
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=args.weight_decay,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(imgs)
        train_acc = correct / total
        scheduler.step()

        # Val
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += len(imgs)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch:3d} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | lr={scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_acc + 0.0005:
            best_acc = val_acc
            patience_count = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_acc": val_acc, "class_to_idx": class_to_idx},
                       out_dir / "best.pth")
            print(f"  → best model saved (val_acc={val_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\n完了: best_val_acc={best_acc:.4f}")
    print(f"  モデル: {out_dir}/best.pth")
    print(f"  クラス: {out_dir}/classes.txt")


if __name__ == "__main__":
    main()
