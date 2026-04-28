#!/usr/bin/env python3
"""YOLOのbbox結果をもとにARC認識を実行し、結果画像を保存。"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageStat, ImageDraw, ImageFont
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import timm

CONV_NAME   = "convnextv2_large"
CONV_SIZE   = 384
NUM_CLASSES = 3962
EMBED_DIM   = 512
ARCFACE_S   = 16.0
FONT_PATH   = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


class SmartPadResize:
    def __init__(self, size): self.size = size
    def __call__(self, img):
        try: fill = tuple(int(v) for v in ImageStat.Stat(img).median)
        except: fill = (255, 255, 255)
        w, h = img.size
        r = self.size / max(w, h)
        nw, nh = int(w * r), int(h * r)
        img = img.resize((nw, nh), Image.BICUBIC)
        out = Image.new("RGB", (self.size, self.size), fill)
        out.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return out


class ArcFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, s=16.0, m=0.3):
        super().__init__()
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        features = features.float()
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.float()))
        if labels is None:
            return cosine * self.s
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return (cosine * (1 - one_hot) + torch.cos(theta + 0.3) * one_hot) * self.s


class KuzushijiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(CONV_NAME, pretrained=False,
                                         num_classes=0, global_pool="avg")
        self.embed = nn.Sequential(
            nn.Linear(self.encoder.num_features, EMBED_DIM),
            nn.BatchNorm1d(EMBED_DIM), nn.GELU(), nn.Dropout(0.1))
        self.arcface = ArcFaceHead(EMBED_DIM, NUM_CLASSES, s=ARCFACE_S)

    def forward(self, x, labels=None):
        return self.arcface(self.embed(self.encoder(x)), labels)


def unicode_to_char(u):
    if u and u.startswith("U+"):
        try: return chr(int(u[2:], 16))
        except: pass
    return u


def load_model(ckpt_path, device):
    model = KuzushijiModel().to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    return model


def predict(model, transform, img_pil, class_map, device):
    with torch.no_grad():
        logits = model(transform(img_pil).unsqueeze(0).to(device))
        probs = torch.softmax(logits[0], dim=0)
    top5_probs, top5_idx = torch.topk(probs, 5)
    top1_char = unicode_to_char(class_map.get(top5_idx[0].item(), "?"))
    top1_prob = top5_probs[0].item()
    top5 = [(unicode_to_char(class_map.get(i.item(), "?")), p.item())
            for i, p in zip(top5_idx, top5_probs)]
    return top1_char, top1_prob, top5


def draw_results(img_pil, predictions, font):
    draw = ImageDraw.Draw(img_pil)
    for pred in predictions:
        x1, y1, x2, y2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]
        char, prob = pred["char"], pred["prob"]
        color = "blue" if prob >= 0.5 else "orange"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{char} {prob*100:.0f}%"
        bbox = draw.textbbox((0, 0), label, font=font)
        th = bbox[3] - bbox[1]
        ty = max(0, y1 - th - 2)
        draw.text((x1, ty), label, fill=color, font=font)
    return img_pil


def main():
    parser = argparse.ArgumentParser("ARC batch inference")
    parser.add_argument("--yolo_dir", default="output/yolo_detection")
    parser.add_argument("--input_dir", default="data/inference_input")
    parser.add_argument("--output_dir", default="output/arc_recognition")
    parser.add_argument("--ckpt", default="best_arc.pth")
    parser.add_argument("--class_json", default="modules/detection/trained_class_map.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Loading ARC model...")
    model = load_model(args.ckpt, device)
    logger.info("Model loaded.")

    with open(args.class_json) as f:
        class_map = {int(k): v for k, v in json.load(f).items()}

    transform = transforms.Compose([
        SmartPadResize(CONV_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except:
        font = ImageFont.load_default()

    folders = sorted([
        d for d in os.listdir(args.yolo_dir)
        if os.path.isdir(os.path.join(args.yolo_dir, d))
    ])
    total = len(folders)
    os.makedirs(args.output_dir, exist_ok=True)

    done = skipped = 0
    for i, folder_name in enumerate(folders):
        out_folder = os.path.join(args.output_dir, folder_name)
        result_path = os.path.join(out_folder, "arc_result.jpg")
        if os.path.exists(result_path):
            skipped += 1
            if skipped % 100 == 1:
                logger.info("[{}/{}] Skip: {}", i + 1, total, folder_name)
            continue

        bbox_json = os.path.join(args.yolo_dir, folder_name, "bboxes.json")
        input_img = os.path.join(args.input_dir, folder_name, "00_input.png")
        if not os.path.exists(bbox_json) or not os.path.exists(input_img):
            logger.warning("[{}/{}] Missing files, skipping: {}", i + 1, total, folder_name)
            continue

        with open(bbox_json) as f:
            bboxes = json.load(f)

        img_pil = Image.open(input_img).convert("RGB")
        predictions = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_pil.width, x2), min(img_pil.height, y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            crop = img_pil.crop((x1, y1, x2, y2))
            char, prob, top5 = predict(model, transform, crop, class_map, device)
            predictions.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                 "char": char, "prob": prob, "top5": top5})

        os.makedirs(out_folder, exist_ok=True)
        result_img = draw_results(img_pil.copy(), predictions, font)
        result_img.save(result_path, quality=90)

        # 認識結果をJSONにも保存
        with open(os.path.join(out_folder, "arc_result.json"), "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        logger.info("[{}/{}] Done: {} ({} chars)", i + 1, total, folder_name, len(predictions))
        done += 1

    logger.info("Done. Processed: {}, Skipped: {}", done, skipped)


if __name__ == "__main__":
    main()
