#args.backgroundを元画像の最頻値で作成する


import traceback
import logging
import sys
import os
import torch
from src.inpainting import Diffusion
from src.random_crop import Page_Inpainting,resize_image,binary
import glob
import time
import re
import shutil
import cv2
import torchvision.utils as tvu
from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np

        
torch.set_printoptions(sci_mode=False)

class Args():
    def __init__(self):
        self.seed = 999
        #逆過程において何回推定するか
        self.timesteps = 20
        self.deg = 'maskshape'
        #char, kmnist, k49のとき'fixmask'
        #pageのときmaskshape
        self.sigma_0 = 0.05
        self.eta = 0.5
        # self.eta = 0.5
        self.etaB = 1
        self.subset_start = 0 
        self.subset_end = -1 # num of data
        #2段階修復のとき、'char'
        self.model_type = 'page' # 'char' or 'page' or 'kmnist' or 'k49'
        self.cls_cond = None #only kmnist or k49 
        self.num_classes=10 #only kmnist(10) or k49(49) cls_cond=True
        self.dataset = 'orig' # my_dataset or kmnist or k49
        ###########################
        # Parameters for fixed-mask
        ###########################
        self.pad = 0
        self.mask_size = 32
        self.device = "cuda:1"
        # self.image_name = './枕草子/2_gmm_dataset_raw'
        self.image_name = './徒然草/turezuregusa_ace_ver2'
        self.cut_path_output = './result/char/0_-1.png'
        self.back_ground = './background/image.png'
        # self.back_ground = './background/image.png'
        # self.back_ground = 'journal/output/restored_image_resized.png'
        ######################
        #noise_maskingtime
        ######################
        self.masking = True
        self.mask_timesteps = 50
        ######################
        # Save Folder
        #####################
        # self.image_folder = './heritage_result/RESULT_TUREZUREGUSA'
        self.image_folder = './heritage_result/page_base/TUREZUREGUSA_RESULT_TUREZUREGUSA_GMM_ace_ver2'
        self.cut_image ="cut_image"
        self.y_dir = "y"
        self.x_dir = "x"
        self.result ="result"
        self.noise_mask = "noisemask"
        self.back_mask = "backmask"
        self.mask_dir = 'mask_dir'
        self.cat_dir = 'cat_dir'
        # self.image_res = 'image_res'
        
        ######################
        

class Config():
    class data():
            dataset = "LSUN" #LSUN
            category = "church_outdoor"
            ##################
            # Setting imsize
            ##################e
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
        type = "orig"#"simple
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
        

def list_images_in_folder(folder_path, extensions=['jpg', 'jpeg', 'png']):
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))
    return images

def extract_sort_key(name):
    match = re.match(r'^(\d+)_(\d+)_(\d+)$', name)
    if match:
        # 固定IDは無視して、連番・サブIDをintで取得
        return (int(match.group(2)), int(match.group(3)))
    else:
        return (float('inf'), float('inf'))  # マッチしない場合は後ろに


def main(args=None):
    if args is None:
        args = Args()
    # args = Args()
    config = Config()
    # torch.cuda.set_device(args.device)
    # print("CUDA available:", torch.cuda.is_available())
    # print("CUDA device count:", torch.cuda.device_count())
    # print("Current CUDA device:", torch.cuda.current_device())
    # print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    try:
        # images = list_images_in_folder(args.image_name)#./turezuregusa/200009896_00131_2.jpg....
        images = sorted(
            [name for name in os.listdir(args.image_name) if os.path.isdir(os.path.join(args.image_name, name))],
            key=extract_sort_key
            )

        for image_path in images:
            image_name_path = os.path.join(args.image_folder, image_path)
            image_name = os.path.join(args.image_name, image_path, '00_input.png')
            #二値化画像の場合
            # mask_path1 = os.path.join(args.image_name, image_path, 'binary_otsu_local_inv.png')
            #GMMの場合
            mask_path1 = os.path.join(args.image_name, image_path, '02_text_black.png')
            # mask_path2 = os.path.join(args.image_name, image_path, '04_bleed.png')  # 追加
            
            image = Image.open(image_name).convert("RGB")
            width, height = image.size

            if not os.path.exists(image_name_path):
                os.makedirs(image_name_path)

            shutil.copy(image_name, image_name_path)
            shutil.copy(mask_path1, image_name_path)
            # if os.path.exists(mask_path2):
            #     shutil.copy(mask_path2, image_name_path)
                
            # 00_input.pngをリサイズして保存
            resized_input = image.resize((config.data.image_size, config.data.image_size), Image.NEAREST)
            resized_input_path = os.path.join(image_name_path, "00_input_resized.png")
            resized_input.save(resized_input_path)

            # --- ここで分岐 ---
            # use_combined_mask = os.path.exists(mask_path2)  # 2つ目のマスクがあれば合体
            use_combined_mask = False  # 強制的に合体マスクを使用する場合

            if use_combined_mask:
                # 2つのマスクを合体
                mask_img1 = Image.open(mask_path1).convert("L").resize((config.data.image_size, config.data.image_size), Image.NEAREST)
                # mask_img2 = Image.open(mask_path2).convert("L").resize((config.data.image_size, config.data.image_size), Image.NEAREST)
                mask_np1 = (np.array(mask_img1) > 128).astype(np.uint8)
                mask_np2 = (np.array(mask_img2) > 128).astype(np.uint8)
                mask_bin = np.clip(mask_np1 + mask_np2, 0, 1)
                mask_save_name = "used_mask_combined_raw.png"
            else:
                # 単体マスク
                mask_img = Image.open(mask_path1).convert("L")
                mask_resized = mask_img.resize((config.data.image_size, config.data.image_size), Image.NEAREST)
                mask_np = np.array(mask_resized)
                mask_bin = (mask_np > 128).astype(np.uint8)
                mask_save_name = "used_mask_raw.png"

            Image.fromarray((mask_bin * 255).astype(np.uint8)).save(os.path.join(image_name_path, mask_save_name))

            # 入力画像（リサイズ済み）を back_ground として設定
            args.back_ground = resized_input_path
            #./result/200009896_00131_2/char
            folders_page = [args.y_dir, args.x_dir, args.result, args.noise_mask, args.back_mask]
            for folder in folders_page:
                folder_path = os.path.join(image_name_path, folder)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            runner = Diffusion(args, config)
            runner.sample(image_name_path)
            png_path = os.path.join(image_name_path, "result", f"0_t{args.timesteps - 1}.png")
            png_img = cv2.imread(png_path)
            resized_png = cv2.resize(png_img, (width, height), interpolation=cv2.INTER_AREA)
            
            # 画像フォルダ名をベースにファイル名を生成
            folder_name = os.path.basename(image_path)  # 例: 200009896_00005_2
            resized_png_path = os.path.join(
                os.path.dirname(png_path),
                f"{folder_name}.png"
            )

            cv2.imwrite(resized_png_path, resized_png)

            # --- 赤テキスト合成処理 ---
            try:
                # パスの準備
                redtext_path = os.path.join(args.image_name, image_path, '03_text_red.png')
                input_path = os.path.join(args.image_name, image_path, '00_input.png')
                fin_path = os.path.join(os.path.dirname(png_path), f"{folder_name}_fin.png")

                # ファイル存在チェック
                if not os.path.exists(redtext_path):
                    print(f"Warning: Red text mask not found, skipping red text composite: {redtext_path}")
                    continue
                if not os.path.exists(input_path):
                    print(f"Warning: Input image not found, skipping red text composite: {input_path}")
                    continue

                # 画像読み込み
                input_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
                red_mask = cv2.imread(redtext_path, cv2.IMREAD_GRAYSCALE)
                base_img = cv2.imread(resized_png_path, cv2.IMREAD_COLOR)
                
                if input_img is None or red_mask is None or base_img is None:
                    print(f"Warning: Failed to load images, skipping red text composite for {image_path}")
                    continue

                # 入力画像をリサイズしてリサイズ画像と同じ大きさに
                input_img = cv2.resize(input_img, (base_img.shape[1], base_img.shape[0]), interpolation=cv2.INTER_AREA)
                red_mask = cv2.resize(red_mask, (base_img.shape[1], base_img.shape[0]), interpolation=cv2.INTER_NEAREST)

                # 赤テキスト領域が実際に存在するかチェック
                red_pixel_count = np.sum(red_mask > 128)
                if red_pixel_count == 0:
                    print(f"Warning: No red text pixels found, skipping red text composite for {image_path}")
                    continue

                # 赤テキスト部分だけ抽出（マスクが255の部分をinput_imgから取り出す）
                red_mask_bin = (red_mask > 128).astype(np.uint8)
                red_mask_bin_3ch = np.stack([red_mask_bin]*3, axis=-1)
                red_text_only = cv2.bitwise_and(input_img, input_img, mask=red_mask)

                # リサイズ画像に赤テキスト部分を貼り付け
                base_img_with_red = base_img.copy()
                base_img_with_red[red_mask_bin_3ch == 1] = red_text_only[red_mask_bin_3ch == 1]

                # 保存
                cv2.imwrite(fin_path, base_img_with_red)
                print(f"Successfully applied red text composite to {image_path} ({red_pixel_count} red pixels)")
            except Exception as e:
                logging.error(f"Red text composite failed: {e}")


    except Exception:
        logging.error(traceback.format_exc())
        
# def main_with_detailed_experiments():
#     image_sizes = [704, 512, 1024]  # 試したい画像サイズ
#     mask_timesteps_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]  # 細かいmask_timestepsの値
#     num_timesteps_list = [1000, 1200, 1500, 1800, 2000]  # 試したいnum_diffusion_timestepsの値
#     eta_values = [0.3, 0.5, 0.7]  # 試したいetaの値
#     back_ground_list = ['./background/image.png', './background/isemonogatari_2.png']  # 試したい背景画像

#     for image_size in image_sizes:
#         print(f"🚀 Running with image_size={image_size}")
#         for num_timesteps in num_timesteps_list:
#             print(f"  🔄 Running with num_diffusion_timesteps={num_timesteps}")
#             for eta in eta_values:
#                 print(f"    🌟 Running with eta={eta}")
#                 for back_ground in back_ground_list:
#                     print(f"      🖼️ Running with back_ground={back_ground}")
#                     config = Config()
#                     config.data.image_size = image_size  # 画像サイズを設定
#                     config.diffusion.num_diffusion_timesteps = num_timesteps  # 拡散プロセスのステップ数を設定

#                     for timestep in mask_timesteps_list:
#                         args = Args()
#                         args.mask_timesteps = timestep
#                         args.eta = eta  # etaの値を設定
#                         args.back_ground = back_ground  # 背景画像を設定
#                         args.image_folder = f'./RESULT_MAKURANOSOUSHI/image_size{image_size}_num_timesteps{num_timesteps}_mask_timesteps{timestep}_eta{eta}_background_{os.path.basename(back_ground)}'
#                         print(f"        ➡️ Running with mask_timesteps={timestep}")
#                         main(args)

def main_with_detailed_experiments():
    base_input_root = "./徒然草/turezuregusa_ace_ver2"
    base_output_root = "./heritage_result/page_base/TUREZUREGUSA"

    # 対象フォルダを取得（density_***）
    input_folders = sorted([
        d for d in os.listdir(base_input_root)
        if os.path.isdir(os.path.join(base_input_root, d)) and d.startswith("density_")
    ])



    for folder in input_folders:
        input_path = os.path.join(base_input_root, folder)
        output_path = os.path.join(base_output_root, f"RESULT_MAKURANOSOUSHI_{folder}")
        
        print(f"🚀 Processing folder: {folder}")
        
        config = Config()

        args = Args()
        args.image_name = input_path  # ← 入力元を上書き
        args.image_folder = output_path  # ← 出力先を上書き


        main(args)

if __name__ == "__main__":
    run_all = False  # ← Trueにすれば複数設定を一括実行、Falseなら通常通り1回だけ
    if run_all:
        main_with_detailed_experiments()
    else:
        sys.exit(main())