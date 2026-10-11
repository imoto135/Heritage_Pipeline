"""
単一画像から裏写り(bleed-through)マスクを予測する、小さいU-Netの学習。

[[project_bleedthrough_verification]]で確認済みの物理モデルで生成した合成データ
(`generate_synthetic_bleedthrough.py`の出力: clean/degraded/ghost_gt)で学習し、
本物の裏写り確認済み実例に対して汎化するかを見る「安価な早期ゲート」として使う。
segmentation_models_pytorch等の外部ライブラリはどのconda envにも入っていなかった
ため、依存を増やさないよう素のPyTorchで小さいU-Netを自前実装している。

使い方:
    conda run -n diffusion python scripts/benchmark/train_bleedthrough_unet.py \
        --data_dir tmp_work/synthetic_bleedthrough/200010454 \
        --out tmp_work/bleedthrough_model/unet_v1.pth --epochs 30
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────
# 小さいU-Net
# ─────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    """4段ダウンサンプルの軽量U-Net。入力3ch(BGR) -> 出力1ch(sigmoidロジット)。"""

    def __init__(self, base=32):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.enc1 = ConvBlock(3, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, c5)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec4 = ConvBlock(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)

        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def dice_loss(logits, target, eps=1.0):
    prob = torch.sigmoid(logits)
    prob = prob.flatten(1)
    target = target.flatten(1)
    inter = (prob * target).sum(1)
    union = prob.sum(1) + target.sum(1)
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


# ─────────────────────────────────────────────────────────
# データセット
# ─────────────────────────────────────────────────────────

class BleedthroughPatchDataset(Dataset):
    """degraded画像から256x256パッチを切り出す。ghost画素を含むパッチを7割の
    確率で選ぶオバーサンプリングで、ghost比率7〜8%程度のクラス不均衡に対応する。"""

    def __init__(self, data_dir, patch_size=256, samples_per_page=8, ghost_bias=0.7):
        self.deg_paths = sorted(glob.glob(os.path.join(data_dir, 'degraded', '*.png')))
        self.gt_dir = os.path.join(data_dir, 'ghost_gt')
        self.patch_size = patch_size
        self.samples_per_page = samples_per_page
        self.ghost_bias = ghost_bias
        assert self.deg_paths, f'degraded画像が見つかりません: {data_dir}'

    def __len__(self):
        return len(self.deg_paths) * self.samples_per_page

    def __getitem__(self, idx):
        deg_path = self.deg_paths[idx // self.samples_per_page]
        stem = os.path.splitext(os.path.basename(deg_path))[0]
        gt_path = os.path.join(self.gt_dir, stem + '.png')

        img = cv2.imread(deg_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        H, W = gt.shape
        ps = self.patch_size

        ys, xs = np.where(gt > 127)
        use_ghost = len(ys) > 0 and random.random() < self.ghost_bias
        if use_ghost:
            i = random.randrange(len(ys))
            cy, cx = int(ys[i]), int(xs[i])
            y0 = np.clip(cy - ps // 2 + random.randint(-30, 30), 0, max(0, H - ps))
            x0 = np.clip(cx - ps // 2 + random.randint(-30, 30), 0, max(0, W - ps))
        else:
            y0 = random.randint(0, max(0, H - ps))
            x0 = random.randint(0, max(0, W - ps))

        img_p = img[y0:y0 + ps, x0:x0 + ps]
        gt_p = gt[y0:y0 + ps, x0:x0 + ps]
        if img_p.shape[0] != ps or img_p.shape[1] != ps:
            img_p = cv2.copyMakeBorder(img_p, 0, ps - img_p.shape[0], 0, ps - img_p.shape[1],
                                       cv2.BORDER_REFLECT)
            gt_p = cv2.copyMakeBorder(gt_p, 0, ps - gt_p.shape[0], 0, ps - gt_p.shape[1],
                                      cv2.BORDER_REFLECT)

        if random.random() < 0.5:
            img_p = np.fliplr(img_p).copy(); gt_p = np.fliplr(gt_p).copy()
        if random.random() < 0.5:
            img_p = np.flipud(img_p).copy(); gt_p = np.flipud(gt_p).copy()

        img_t = torch.from_numpy(img_p.astype(np.float32) / 255.0).permute(2, 0, 1)
        gt_t = torch.from_numpy((gt_p > 127).astype(np.float32)).unsqueeze(0)
        return img_t, gt_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val_pages', type=int, default=8, help='末尾何ページを検証用に分けるか')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    all_deg = sorted(glob.glob(os.path.join(args.data_dir, 'degraded', '*.png')))
    val_stems = set(os.path.splitext(os.path.basename(p))[0] for p in all_deg[-args.val_pages:])

    full_ds = BleedthroughPatchDataset(args.data_dir)
    train_idx = [i for i in range(len(full_ds))
                if os.path.splitext(os.path.basename(full_ds.deg_paths[i // full_ds.samples_per_page]))[0]
                not in val_stems]
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, drop_last=True)

    model = SmallUNet(base=24).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for img_t, gt_t in train_loader:
            img_t = img_t.to(args.device); gt_t = gt_t.to(args.device)
            logits = model(img_t)
            loss = F.binary_cross_entropy_with_logits(logits, gt_t) + dice_loss(logits, gt_t)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * img_t.size(0)
            n += img_t.size(0)
        print(f'epoch {epoch+1}/{args.epochs}  loss={total_loss/n:.4f}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f'保存: {args.out}')
    print(f'検証用に分けたページ({args.val_pages}枚): {sorted(val_stems)}')


if __name__ == '__main__':
    main()
