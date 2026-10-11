#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:34:47 2023

@author: ihpc
"""

from PIL import Image
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

        


#resizeしてカット→保存する処理
class Page_Inpainting():
    def __init__(self,input_image_path,output_folder,threshold=50):
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # 画像を開く（いつもどおりの前処理）
        self.img = Image.open(input_image_path)
        self.width, self.height = self.img.size
        # 画像をリサイズする
        img = self.img.resize((704, 704), Image.LANCZOS)
        
        img.save(os.path.join(os.path.dirname(output_folder),'text_mask.png'))
        width, height = img.size
        #カットして保存(step_x,step_yがカットする大きさになるように定数を変更する)
        step_x = width // 11
        step_y = height // 11
        # 二値化処理を適用
        # 前
        img = img.point(lambda p: 0 if p < threshold else 255)       # ← 'L' モードを維持
        img = img.convert("L")      
        #後
        # img = img.convert("L")
        # img = img.point(lambda p: 255 if p < threshold else 0)  # 明るい→黒、暗い→白（つまり反転）
        img.save(os.path.join(os.path.dirname(output_folder),'text_mask_bit.png'))
        print("PILモード:", img.mode)
        print("PILサイズ:", img.size)
        print("PILユニーク値:", np.unique(np.array(img)))
        
        for i in range(0, width, step_x):
            for j in range(0, height, step_y):
                box = (i, j, i + step_x, j + step_y)  # カットする領域
                region = img.crop(box)
                region.save(os.path.join(output_folder, f"{i}_{j}.png"))  
    
    def get_image_size(self):
        return self.width, self.height
                
#ただリサイズするだけ
def resize_image(input_image_path, output_image_path, width, height):
    # 画像を開く
    img = Image.open(input_image_path)
    
    # 画像をリサイズする
    resized_img = img.resize((width, height), Image.LANCZOS)
    
    # リサイズした画像を保存
    resized_img.save(output_image_path)
    

def binary(input_image_path, output_image_path, threshold=125):
    # 画像を開く
    img = Image.open(input_image_path)
    
    # ２値化
    img = img.convert("L")  # グレースケールに変換
    img = img.point(lambda p: 0 if p < threshold else 255, '1')  # ２値化
    
    
    # ２値画像を保存
    img.save(output_image_path)

# resize_image('/media/ihpc/USB DISK/soturon/siroa.png','/media/ihpc/USB DISK/soturon/siroa_re.png', width = 704, height=704)

def convert_to_grayscale(input_image_path, output_image_path):
    # 画像を開く
    img = Image.open(input_image_path)
    
    # グレースケールに変換
    gray_img = img.convert('L')
    
    # グレースケールに変換した画像を保存
    gray_img.save(output_image_path)

# 入力画像と出力先ファイルパスを指定して関数を呼び出し
# convert_to_grayscale('/home/ihpc/ancient_ddrm_ver_icamechs/result_ysd/result/120.png', 'output_image_grayscale.jpg')
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
