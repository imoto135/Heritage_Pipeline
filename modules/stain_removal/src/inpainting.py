#連続で処理するように改良
#入力画像の一枚文献画像を正方形にリサイズ、64*64で切り出し、その画像に対して
#拡散モデルを適用している。
#orinal_dataに元データ格納

import os
import sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)
# import matrix
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
import random
import torchvision.utils as tvu
from src.denoising import efficient_generalized_steps
# from functions.ckpt_util import get_ckpt_path, download
import torch.utils.data as data
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import tqdm
import numpy as np
import glob
import time
import logging

from src.models.unet.unet import UNetModel as unet
from src.diffusion import Model
from src.orig_dataset import SingleDataset
from PIL import Image
from torchvision.transforms.functional import erase

import torch
import random
import numpy as np

def torch_fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):#beta_schedule_type = linear
    betas = np.linspace(
        beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
    )
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class Diffusion(object):
    def __init__(self, args, config):
        
        self.args = args
        self.config = config
        self.device = args.device
        
        torch_fix_seed(args.seed)
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)

        
        self.num_timesteps = betas.shape[0]
        
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)#ルートをとる
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def _load_model(self):
        """モデルを初回のみロードしてキャッシュする。"""
        if hasattr(self, '_model_cache'):
            return self._model_cache
        args, config = self.args, self.config
        model = unet(in_channels=3, model_channels=64, out_channels=3,
                     num_res_blocks=2, attention_resolutions=[16, 8])
        ckpt_map = {'page': './model/model_page.pth', 'char': './model/model_char.pth'}
        ckpt = ckpt_map.get(getattr(args, 'model_type', 'char'), './model/model_char.pth')
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        model.eval()
        self._model_cache = model
        return model

    def sample(self, image_folder):
        cls_fn = None
        model = self._load_model()
        self.sample_sequence(model, cls_fn, image_folder)
        

    def sample_sequence(self, model, cls_fn=None,image_folder=None):
        args, config = self.args, self.config


        path = args.back_ground
        test_dataset = SingleDataset(
            path=path, imsize=config.data.image_size, train=False)
            
        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(
                test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        
        ###################################################
        #Get degradation matrix
        #################################################
        
        deg = args.deg
        H_funcs = None
        from src.svd_replacement import Inpainting as Inpainting
        from src.svd_replacement import GeneralH
        

        # まず合体マスクがあればそちらを優先
        combined_mask_path = os.path.join(image_folder, "used_mask_combined_raw.png")
        if os.path.exists(combined_mask_path):
            combine_img_path_binary = combined_mask_path
        else:
            combine_img_path_binary = os.path.join(image_folder, "used_mask_raw.png")

        loaded = Image.open(combine_img_path_binary).convert("L").resize(
            (self.config.data.image_size, self.config.data.image_size), Image.NEAREST
        )
        np.set_printoptions(threshold=np.inf)
        transform = transforms.Compose([transforms.ToTensor()])
        mask = torch.squeeze(transform(loaded))
        mask = 1-mask
        
        mask_seq = mask.to(self.device).reshape(-1)
        
        missing_r = torch.nonzero(mask_seq == 0).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r,missing_g, missing_b], dim=0)
        H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device) 
        sigma_0 = args.sigma_0
    
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        if torch.is_tensor(mask) == False:
            mask = torch.from_numpy(mask)
        if mask.dim() == 3:
            mask = torch.unsqueeze(mask, dim=0)
        elif mask.dim() == 2:
            mask = mask.reshape((1,1,config.data.image_size,config.data.image_size))
        mask = torch.cat((mask,mask,mask),dim=1).to(torch.float32).to(self.device)
        
        ##反転するかそのままか
        tvu.save_image(
            mask, 
            os.path.join(image_folder, "used_mask.png"),
            normalize=False
        )   
        
        '''[-1,1] -> mask (-1,1)'''
        norm = transforms.Lambda(lambda x: (x+1)/2)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            # x_orig = data_transform(self.config, x_orig)
            x_orig_copy = x_orig.clone()
            # return
            y_0 = H_funcs.H(x_orig)
            y_0 = y_0 + sigma_0 * torch.randn_like(y_0)

            pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels,
                                                self.config.data.image_size, self.config.data.image_size)

            pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))
                                        ).reshape(*pinv_y_0.shape) - 1

            # y_0 inf ?
            for i in range(len(pinv_y_0)):
                #x_orig:背景画像
                tvu.save_image(
                    x_orig_copy[i], os.path.join(image_folder, f"{args.x_dir}/x_orig{idx_so_far + i}.png")
                )
                tvu.save_image(
                    x_orig[i], os.path.join(image_folder, f"{args.x_dir}/x_mask{idx_so_far + i}.png")
                )
                tvu.save_image(
                    (pinv_y_0[i]), os.path.join(image_folder, f"{args.y_dir}/y{idx_so_far + i}.png")
                )
                tvu.save_image((mask), os.path.join(image_folder, f"{args.noise_mask}/mask{idx_so_far + i}.png")
                                )
                tvu.save_image((mask-1)*(-1), os.path.join(image_folder, f"{args.back_mask}/H{idx_so_far + i}.png")
                                )
                
            # Begin DDIM
            x = torch.randn(
                y_0.shape[0],
                
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
            with torch.no_grad():
                # x, _ ,cls_list= self.sample_image(
                #     x, model, H_funcs, y_0, sigma_0, self.device, last=False, cls_fn=cls_fn, classes=classes,mask=mask,x_orig=x_orig)
                
                x_all_steps, _, cls_list = self.sample_image(
                    x, model, H_funcs, y_0, sigma_0, self.device,
                    last=False, cls_fn=cls_fn, classes=classes,
                    mask=mask, x_orig=x_orig
                )
                

            # x = [inverse_data_transform(config, y) for y in x]
            # 復元後の全ステップを [-1,1] で正規化
            # x_all_steps = [inverse_data_transform(config, y) for y in x_all_steps]       
            # x_all_steps = x_all_steps.clamp(0, 1)
            # 各ステップを保存（x_t）
            
            
            
            for t, xt in enumerate(x_all_steps):
                for j in range(xt.size(0)):
                    tvu.save_image(
                        xt[j],
                        os.path.join(image_folder, f"{args.result}/{idx_so_far + j}_t{t}.png"),
                        normalize=True, value_range=(-1, 1)
                    )
            
            
            # for t, xt in enumerate(x_all_steps):
            #     for j in range(xt.size(0)):
            #         # 背景画像も [0,1] に変換
            #         x_orig_01 = inverse_data_transform(config, x_orig[j])
            #         # 合成
            #         composite = xt[j] * mask[0] + x_orig_01 * (1 - mask[0])
            #         # 保存
            #         tvu.save_image(
            #             composite,
            #             os.path.join(image_folder, f"{args.result}/{idx_so_far + j}_t{t}_composite.png"),
            #             normalize=False
            #         )  
            
            idx_so_far += y_0.shape[0]
        

        

    #前のコード
    # def sample_image(self, x, model, H_funcs, y_0, sigma_0, device, last=True,cls_fn=None, classes=None, mask=None, x_orig=None):
    #     skip = self.num_timesteps // self.args.timesteps
    #     seq = range(0, self.num_timesteps, skip)
    #     x = efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, sigma_0, device,
    #                                     etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta,\
    #                                         cls_fn=cls_fn, classes=classes,mask=mask,x_orig=x_orig,args=self.args)
    #     if last:
    #         x = x[0][-1]
    #     return x
    
    def sample_image(self, x, model, H_funcs, y_0, sigma_0, device, last=True,cls_fn=None, classes=None, mask=None, x_orig=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        x = efficient_generalized_steps(x, seq, model, self.betas, H_funcs, y_0, sigma_0, device,
                                        etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta,\
                                            cls_fn=cls_fn, classes=classes,mask=mask,x_orig=x_orig,args=self.args)
        if last:
            return x[0][-1], x[1], x[2]  # 最終ステップのみ返す（今まで通り）
        else:
            return x[0], x[1], x[2]      # 全ステップを返す

    
    

