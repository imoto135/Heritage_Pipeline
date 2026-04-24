import os
import shutil
from PIL import Image
import numpy as np
import cv2
import torch
from codes.Inpainting import Diffusion
from codes.random_crop import resize_image

base_dir = '/home/yoshizu/document/bunken_diffusion_folder'
src_root = os.path.join(base_dir, '徒然草', 'turezuregusa_ace')
result_root = os.path.join(base_dir, 'heritage_result', 'page_base', 'TUREZUREGUSA_RESULT_TUREZUREGUSA_GMM_ace')

class Args():
    def __init__(self):
        self.seed = 999
        self.timesteps = 20
        self.deg = 'maskshape'
        self.sigma_0 = 0.05
        self.eta = 0.5
        self.etaB = 1
        self.subset_start = 0
        self.subset_end = -1
        self.model_type = 'page'
        self.cls_cond = None
        self.num_classes=10
        self.dataset = 'orig'
        self.pad = 0
        self.mask_size = 32
        self.device = "cuda:0"
        self.cut_path_output = './result/char/0_-1.png'
        self.back_ground = './background/image.png'
        self.masking = True
        self.mask_timesteps = 50
        self.image_folder = None  # 後で設定
        self.cut_image ="cut_image"
        self.y_dir = "y"
        self.x_dir = "x"
        self.result ="result"
        self.noise_mask = "noisemask"
        self.back_mask = "backmask"
        self.mask_dir = 'mask_dir'
        self.cat_dir = 'cat_dir'

class Config():
    class data():
        dataset = "LSUN"
        category = "church_outdoor"
        image_size =1024
        channels = 3
        logit_transform = False
        uniform_dequantization = False
        gaussian_dequantization= False
        random_flip = False
        rescaled = True
        num_workers= 0
        out_of_dist= True
    class model:
        type = "orig"
        dim = 64
        channels = 3
        dim_mults=(1, 2, 4,8 )
        resnet_block_groups=4
        attn_resolutions = [16, ]
        dropout = 0.0
        var_type = 'fixedsmall'
        ema_rate = 0.999
        ema = True
        resamp_with_conv = True
    class diffusion:
        beta_schedule = 'linear'
        beta_start = 0.0001
        beta_end = 0.02
        num_diffusion_timesteps = 1000
    class training:
        batch_size = 64
        n_epochs = 1000
        n_iters = 10
        snapshot_freq = 5000
        validation_freq = 2000
    class sampling:
        batch_size =64
        last_only = True
    class optim:
        weight_deca = 0.000
        optimizer = "Adam"
        lr = 0.00002
        beta1 = 0.9
        amsgrad = False
        eps = 0.00000001

folders = [f for f in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, f))]

for target_id in folders:
    src_dir = os.path.join(src_root, target_id)
    input_path = os.path.join(src_dir, '00_input.png')
    mask_path = os.path.join(src_dir, '02_blacktext_opened.png')
    redtext_path = os.path.join(src_dir, '03_redtext_opened.png')
    result_dir = os.path.join(result_root, target_id, 'result')
    os.makedirs(result_dir, exist_ok=True)

    # 推論用画像生成
    if not os.path.exists(input_path) or not os.path.exists(mask_path):
        print(f'スキップ: {input_path} または {mask_path} が存在しません')
        continue

    image = Image.open(input_path).convert("RGB")
    width, height = image.size

    # 00_input.pngをresultにコピー
    shutil.copy(input_path, os.path.join(result_dir, '00_input.png'))
    shutil.copy(mask_path, os.path.join(result_dir, '02_blacktext_opened.png'))

    # 00_input.pngをリサイズして保存
    config = Config()
    resized_input = image.resize((config.data.image_size, config.data.image_size), Image.NEAREST)
    resized_input_path = os.path.join(result_dir, "00_input_resized.png")
    resized_input.save(resized_input_path)

    # マスクリサイズ＆保存
    mask_img = Image.open(mask_path).convert("L")
    mask_resized = mask_img.resize((config.data.image_size, config.data.image_size), Image.NEAREST)
    mask_np = np.array(mask_resized)
    mask_bin = (mask_np > 128).astype(np.uint8)
    mask_save_name = "used_mask_raw.png"
    Image.fromarray((mask_bin * 255).astype(np.uint8)).save(os.path.join(result_dir, mask_save_name))

    # 推論実行
    args = Args()
    args.image_name = src_dir
    args.image_folder = os.path.join(result_root, target_id)
    runner = Diffusion(args, config)
    runner.sample(args.image_folder)

    # 生成画像リサイズ
    png_path = os.path.join(result_dir, f"0_t{args.timesteps}.png")
    if not os.path.exists(png_path):
        print(f'スキップ: {png_path} が存在しません')
        continue
    png_img = cv2.imread(png_path)
    resized_png = cv2.resize(png_img, (width, height), interpolation=cv2.INTER_AREA)
    resize_path = os.path.join(result_dir, "0_-1_resize.png")
    cv2.imwrite(resize_path, resized_png)

    # 0_-1_resize.pngをリネーム
    renamed_path = os.path.join(result_dir, f'{target_id}.png')
    os.rename(resize_path, renamed_path)

    # 赤文字合成
    if not os.path.exists(redtext_path):
        print(f'スキップ: {redtext_path} が存在しません')
        continue
    input_img = np.array(Image.open(input_path).convert('RGB'))
    red_mask_img = np.array(Image.open(redtext_path).convert('L'))
    renamed_img = np.array(Image.open(renamed_path).convert('RGB'))
    mask = red_mask_img > 128
    renamed_img[mask] = input_img[mask]
    fin_path = os.path.join(result_dir, f'{target_id}_fin.png')
    Image.fromarray(renamed_img).save(fin_path)
    print(f'完成画像を保存: {fin_path}')
