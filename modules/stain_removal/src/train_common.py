"""
学習スクリプト共通ロジック。
train_patch.py / train_page.py から呼び出される。
"""

import os
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from natsort import natsorted
import glob

from src.models.unet.unet import UNetModel

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# データセット
# ============================================================
class CleanImageDataset(Dataset):
    """クリーン文献画像をロードするデータセット。"""
    def __init__(self, image_dir: str, image_size: int):
        self.files = natsorted(
            glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True) +
            glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True) +
            glob.glob(os.path.join(image_dir, '**', '*.jpeg'), recursive=True)
        )
        if len(self.files) == 0:
            raise FileNotFoundError(f'画像が見つかりません: {image_dir}')
        logger.info(f'学習画像数: {len(self.files)}')

        self.transform = transforms.Compose([
            transforms.Resize(
                (int(image_size * 1.12), int(image_size * 1.12)),
                transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1] に正規化
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img)


# ============================================================
# 拡散プロセス（前向き過程）
# ============================================================
class DiffusionProcess:
    """DDPMの前向き過程（ノイズ付加）と損失計算。"""
    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float, device: str):
        betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
        self.T = num_timesteps
        self.device = device

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        self.sqrt_alphas_cumprod      = torch.from_numpy(np.sqrt(alphas_cumprod)).float().to(device)
        self.sqrt_one_minus_alphas_cp = torch.from_numpy(np.sqrt(1 - alphas_cumprod)).float().to(device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_a  = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1a = self.sqrt_one_minus_alphas_cp[t].view(-1, 1, 1, 1)
        return sqrt_a * x0 + sqrt_1a * noise

    def loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = model(x_t, t)
        return nn.functional.mse_loss(pred_noise, noise)


# ============================================================
# 学習ループ（共通）
# ============================================================
def run_training(cfg):
    """
    cfg に必要なフィールド:
        image_dir, image_size, channels,
        num_diffusion_timesteps, beta_start, beta_end,
        batch_size, lr, n_epochs, save_freq, grad_clip,
        model_channels, num_res_blocks, attention_resolutions,
        save_dir, save_name, resume,
        use_wandb, wandb_project, wandb_run_name,
        device, seed, num_workers
    """
    os.makedirs(cfg.save_dir, exist_ok=True)

    # シード固定
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else 'cpu'
    logger.info(f'使用デバイス: {device}')

    # wandb 初期化
    use_wandb = cfg.use_wandb and _WANDB_AVAILABLE
    if cfg.use_wandb and not _WANDB_AVAILABLE:
        logger.warning('wandb がインストールされていません。pip install wandb でインストールしてください。')
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config={
                'image_dir':               cfg.image_dir,
                'image_size':              cfg.image_size,
                'channels':                cfg.channels,
                'num_diffusion_timesteps': cfg.num_diffusion_timesteps,
                'beta_start':              cfg.beta_start,
                'beta_end':                cfg.beta_end,
                'batch_size':              cfg.batch_size,
                'lr':                      cfg.lr,
                'n_epochs':                cfg.n_epochs,
                'save_freq':               cfg.save_freq,
                'grad_clip':               cfg.grad_clip,
                'model_channels':          cfg.model_channels,
                'num_res_blocks':          cfg.num_res_blocks,
                'attention_resolutions':   cfg.attention_resolutions,
                'save_name':               cfg.save_name,
                'device':                  device,
                'seed':                    cfg.seed,
            },
        )
        logger.info(f'wandb run: {wandb.run.name}  url: {wandb.run.url}')

    # データロード
    dataset = CleanImageDataset(cfg.image_dir, cfg.image_size)
    loader  = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # モデル
    model = UNetModel(
        in_channels=cfg.channels,
        model_channels=cfg.model_channels,
        out_channels=cfg.channels,
        num_res_blocks=cfg.num_res_blocks,
        attention_resolutions=cfg.attention_resolutions,
    ).to(device)

    if use_wandb:
        wandb.watch(model, log=None, log_freq=200)  # gradient追跡は無効（メモリ節約）

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

    # Warmup (10epoch) + CosineAnnealing で発散を防ぐ
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, cfg.n_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    diffusion  = DiffusionProcess(
        cfg.num_diffusion_timesteps, cfg.beta_start, cfg.beta_end, device
    )

    start_epoch = 1
    resume_step = 0
    if cfg.resume and os.path.exists(cfg.resume):
        ckpt = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        resume_step = ckpt.get('global_step', 0)
        logger.info(f'チェックポイントから再開: epoch {start_epoch}')

    logger.info(
        f'学習開始: {cfg.n_epochs} エポック / '
        f'バッチサイズ {cfg.batch_size} / 画像サイズ {cfg.image_size}px'
    )

    global_step = resume_step
    for epoch in range(start_epoch, cfg.n_epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f'Epoch {epoch}/{cfg.n_epochs}')
        for x0 in pbar:
            x0 = x0.to(device)
            loss = diffusion.loss(model, x0)

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
            else:
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None
                ) ** 0.5
            optimizer.step()

            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

            if use_wandb:
                wandb.log({
                    'loss/batch': loss.item(),
                    'grad_norm':  grad_norm,
                    'step':       global_step,
                }, step=global_step)

        scheduler.step()
        avg_loss = total_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f'Epoch {epoch:4d} | avg_loss: {avg_loss:.4f} | lr: {current_lr:.2e}')

        if use_wandb:
            wandb.log({
                'loss/epoch': avg_loss,
                'lr':         current_lr,
                'epoch':      epoch,
            }, step=global_step)

        if epoch % cfg.save_freq == 0:
            # save_name のステム（例: model_char）をサブフォルダ名に使う
            ckpt_subdir = os.path.join(cfg.save_dir, os.path.splitext(cfg.save_name)[0])
            os.makedirs(ckpt_subdir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_subdir, f'ckpt_epoch{epoch:04d}.pth')
            torch.save({
                'epoch':       epoch,
                'global_step': global_step,
                'model':       model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'scheduler':   scheduler.state_dict(),
            }, ckpt_path)
            logger.info(f'チェックポイント保存: {ckpt_path}')
            if use_wandb:
                wandb.save(ckpt_path)

    # 最終モデル保存（推論スクリプトが load_state_dict で直接読む形式）
    final_path = os.path.join(cfg.save_dir, cfg.save_name)
    torch.save(model.state_dict(), final_path)
    logger.info(f'学習完了。最終モデルを保存: {final_path}')

    if use_wandb:
        wandb.save(final_path)
        wandb.finish()
