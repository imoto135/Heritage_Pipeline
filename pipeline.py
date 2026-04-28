#!/usr/bin/env python3
"""
Connect_Heritage 統合パイプライン

入力: 古文書ページ画像
処理: YOLO文字検出 → クロップ → (ARC認識) → UNet++マスク生成 → NAFNet修復
出力: 修復済み文字画像 + metadata.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return (base / rel).resolve()


# ---------------------------------------------------------------------------
# クロップ
# ---------------------------------------------------------------------------

def clamp_bbox(x1, y1, x2, y2, W, H):
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(1, min(W, round(x2))))
    y2 = int(max(1, min(H, round(y2))))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


def crop_characters(image_path: Path, bboxes: list, output_dir: Path) -> list:
    """
    bboxes: [[x1,y1,x2,y2,score,...], ...]
    returns: list of saved filenames in order
    """
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"画像を読み込めません: {image_path}")
    H, W = img.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, W, H)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        crop = img[y1:y2, x1:x2]
        fname = f"char_{i:04d}.png"
        cv2.imwrite(str(output_dir / fname), crop)
        saved.append({"char_id": fname, "x": x1, "y": y1,
                      "width": x2 - x1, "height": y2 - y1,
                      "score": float(box[4]) if len(box) > 4 else None})
    return saved


# ---------------------------------------------------------------------------
# 検出 (YOLOX)
# ---------------------------------------------------------------------------

class Detector:
    """
    YOLOXを用いた文字検出ラッパー。
    conv_demo.py の YOLOXWrapper を直接importして使う。
    """

    def __init__(self, cfg: dict, base: Path, device: str = "cpu"):
        self.ckpt = str(resolve_path(base, cfg["yolox"]["weights"]))
        self.conf = cfg["yolox"]["conf_thresh"]
        self.nms = cfg["yolox"]["nms_thresh"]
        self.tsize = cfg["yolox"]["tile_size"]
        self.stride = cfg["yolox"]["stride"]
        self.device = device
        self.detection_dir = str(resolve_path(base, "modules/detection"))
        self._wrapper = None

    def _load(self):
        if self._wrapper is not None:
            return
        if not Path(self.ckpt).exists():
            return

        sys.path.insert(0, self.detection_dir)
        import torch
        from yolox.exp import get_exp
        from yolox_detector import Predictor, split_image, merge_outputs
        from yolox.data.datasets import COCO_CLASSES

        exp_file = str(Path(self.detection_dir) / "nano.py")
        exp = get_exp(exp_file, None)
        exp.test_conf = self.conf
        exp.nmsthre = self.nms
        exp.test_size = (self.tsize, self.tsize)

        model = exp.get_model()
        use_gpu = self.device == "cuda" and torch.cuda.is_available()
        if use_gpu:
            model.cuda()
        model.eval()

        ckpt = torch.load(self.ckpt, map_location="cuda" if use_gpu else "cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])

        device_str = "gpu" if use_gpu else "cpu"
        predictor = Predictor(model, exp, COCO_CLASSES, None, None, device_str, False, False)

        self._wrapper = (predictor, exp, split_image, merge_outputs)

    def detect(self, image_path: Path) -> list:
        if not Path(self.ckpt).exists():
            print(f"[警告] YOLOXの重みが見つかりません: {self.ckpt}")
            print("  → --bbox-json で外部BBoxを指定してください。")
            return []

        self._load()
        import cv2, torch
        predictor, exp, split_image, merge_outputs = self._wrapper

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"画像を読み込めません: {image_path}")

        tiles, coords = split_image(img, tile_size=self.tsize, stride=self.stride)
        outputs_list = []
        for tile in tiles:
            outputs, _ = predictor.inference(tile)
            outputs_list.append(outputs[0] if outputs is not None else None)

        merged = merge_outputs(outputs_list, coords, nms_thresh=exp.nmsthre,
                               original_img_size=img.shape[:2])
        if merged is None:
            return []

        bboxes = []
        for det in merged.cpu().numpy():
            bboxes.append([float(det[0]), float(det[1]), float(det[2]), float(det[3]),
                           float(det[4] * det[5])])
        return bboxes


# ---------------------------------------------------------------------------
# 認識 (ARC / ConvNeXtV2 + ArcFace)
# ---------------------------------------------------------------------------

class ArcRecognizer:
    """
    ArcFace付きConvNeXtV2による文字認識。
    torch / timm が利用可能な環境で直接importする。
    モデル重みが存在しない場合はスキップ。
    """

    def __init__(self, cfg: dict, base: Path, device: str = "cpu"):
        self.weights = str(resolve_path(base, cfg["arc"]["weights"]))
        self.class_map_path = str(resolve_path(base, cfg["arc"]["class_map"]))
        self.input_size = cfg["arc"]["input_size"]
        self.device = device
        self.detection_dir = str(resolve_path(base, "modules/detection"))
        self._model = None
        self._class_map = None

    def _load(self):
        if self._model is not None:
            return
        if not Path(self.weights).exists():
            return

        sys.path.insert(0, self.detection_dir)
        from conv_demo import KuzushijiModel, ArcFaceHead, SmartPadResize  # noqa: F401
        import torch
        import json
        from torchvision import transforms

        with open(self.class_map_path) as f:
            raw = json.load(f)
        self._class_map = {int(k): v for k, v in raw.items()}

        num_classes = len(self._class_map)
        model = KuzushijiModel("convnextv2_large", num_classes, embed_dim=512)
        ckpt = torch.load(self.weights, map_location=self.device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        model.load_state_dict(sd)
        model.to(self.device).eval()
        self._model = model

        self._transform = transforms.Compose([
            SmartPadResize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def recognize_crops(self, crops_dir: Path, metadata: list) -> list:
        self._load()
        if self._model is None:
            print("[警告] ARCモデルの重みが見つかりません。認識をスキップします。")
            return metadata

        import torch
        from PIL import Image

        device = torch.device(self.device)
        results = []
        for item in metadata:
            img_path = crops_dir / item["char_id"]
            if not img_path.exists():
                results.append({**item, "unicode": None, "confidence": None})
                continue
            img = Image.open(img_path).convert("RGB")
            with torch.no_grad():
                logits = self._model(
                    self._transform(img).unsqueeze(0).to(device),
                    labels=None
                )
                probs = torch.softmax(logits[0], dim=0)
            top_prob, top_idx = probs.max(dim=0)
            unicode_str = self._class_map.get(top_idx.item(), "?")
            results.append({
                **item,
                "unicode": unicode_str,
                "confidence": float(top_prob.item()),
            })
        return results


# ---------------------------------------------------------------------------
# 修復 (UNet++ → NAFNet) — subprocess呼び出し
# ---------------------------------------------------------------------------

class Restorer:
    def __init__(self, cfg: dict, base: Path, device: str = "cpu"):
        unet = cfg["unetpp"]
        naf = cfg["nafnet"]
        self.unetpp_script = str(resolve_path(base, unet["script"]))
        self.unetpp_weights = str(resolve_path(base, unet["weights"]))
        self.unetpp_img_size = str(unet["image_size"])
        self.unetpp_batch = str(unet["batch_size"])
        self.nafnet_script = str(resolve_path(base, naf["script"]))
        self.nafnet_config = str(resolve_path(base, naf["config"]))
        self.nafnet_ckpt = str(resolve_path(base, naf["checkpoint"]))
        self.nafnet_cwd = str(resolve_path(base, naf["script_dir"]))
        self.device = device

    def run_unetpp(self, input_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, self.unetpp_script,
            "--weights", self.unetpp_weights,
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--image-size", self.unetpp_img_size,
            "--batch-size", self.unetpp_batch,
        ]
        print(f"[UNet++] ダメージマスク生成: {input_dir} → {output_dir}")
        subprocess.run(cmd, check=True)

    def run_nafnet(self, input_dir: Path, mask_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, self.nafnet_script,
            "--config", self.nafnet_config,
            "--checkpoint", self.nafnet_ckpt,
            "--input-dir", str(input_dir),
            "--mask-dir", str(mask_dir),
            "--output-dir", str(output_dir),
            "--device", self.device,
        ]
        print(f"[NAFNet] 文字修復: {input_dir} → {output_dir}")
        # cwd を nafnet/ に設定: sys.path.insert(0, dirname(__file__)) の解決のため
        subprocess.run(cmd, check=True, cwd=self.nafnet_cwd)

    def restore(self, crops_dir: Path, masks_dir: Path, restored_dir: Path):
        self.run_unetpp(crops_dir, masks_dir)
        self.run_nafnet(crops_dir, masks_dir, restored_dir)


# ---------------------------------------------------------------------------
# パイプライン本体
# ---------------------------------------------------------------------------

class CharPipeline:
    def __init__(self, config_path: Path, device: str = "cpu"):
        self.base = Path(__file__).parent.resolve()
        self._cfg = load_config(config_path)
        self._device = device
        self.detector = Detector(self._cfg["detection"], self.base, device)
        self.recognizer = ArcRecognizer(self._cfg["detection"], self.base, device)
        self._restorer = None

    def _get_restorer(self) -> "Restorer":
        if self._restorer is None:
            self._restorer = Restorer(self._cfg["restoration"], self.base, self._device)
        return self._restorer

    def run(
        self,
        image_path: Path,
        output_dir: Path,
        bbox_json: Path = None,
        skip_arc: bool = False,
        with_restoration: bool = False,
        keep_temp: bool = False,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        source_name = image_path.name

        with tempfile.TemporaryDirectory() as _tmp:
            tmp = Path(_tmp)
            crops_dir = tmp / "crops"
            masks_dir = tmp / "masks"
            restored_dir = tmp / "restored"

            # --- 1. 検出 ---
            if bbox_json is not None:
                print(f"[検出] BBox JSONを読み込み: {bbox_json}")
                with open(bbox_json) as f:
                    bboxes = json.load(f)
            else:
                print(f"[検出] YOLOX実行: {image_path}")
                bboxes = self.detector.detect(image_path)
            print(f"  → {len(bboxes)} 文字検出")

            # --- 2. クロップ ---
            print(f"[クロップ] 文字領域を切り出し")
            metadata = crop_characters(image_path, bboxes, crops_dir)
            print(f"  → {len(metadata)} クロップ保存")

            if len(metadata) == 0:
                print("[警告] クロップが0件です。終了します。")
                return

            # --- 3. ARC認識 ---
            if not skip_arc:
                print(f"[ARC認識] 文字認識実行")
                metadata = self.recognizer.recognize_crops(crops_dir, metadata)

            # --- 4. 修復 ---
            if with_restoration:
                self._get_restorer().restore(crops_dir, masks_dir, restored_dir)

                # 修復済み画像を出力ディレクトリにコピー
                out_restored = output_dir / "restored"
                out_restored.mkdir(exist_ok=True)
                for f in restored_dir.iterdir():
                    import shutil
                    shutil.copy2(f, out_restored / f.name)
            else:
                # スキップ時はクロップをそのまま出力
                out_restored = output_dir / "crops"
                out_restored.mkdir(exist_ok=True)
                import shutil
                for f in crops_dir.iterdir():
                    shutil.copy2(f, out_restored / f.name)

            # --- 5. メタデータ保存 ---
            for item in metadata:
                item["source_page"] = source_name
            meta_path = output_dir / "metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            if keep_temp:
                import shutil
                shutil.copytree(str(tmp), str(output_dir / "_tmp"), dirs_exist_ok=True)

        print(f"\n完了: {output_dir}")
        print(f"  修復済み画像: {out_restored}/")
        print(f"  メタデータ  : {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Connect_Heritage 統合パイプライン: 文字検出 → 認識 → 修復"
    )
    parser.add_argument("--image", required=True, help="入力ページ画像のパス")
    parser.add_argument("--output-dir", required=True, help="出力ディレクトリ")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "pipeline.yaml"),
        help="pipeline.yaml のパス",
    )
    parser.add_argument(
        "--bbox-json",
        default=None,
        help="既存のBBox JSONファイル (YOLOXをスキップして使用)",
    )
    parser.add_argument(
        "--skip-arc", action="store_true", help="ARC文字認識をスキップ"
    )
    parser.add_argument(
        "--with-restoration", action="store_true",
        help="文字修復を有効化（UNet++ → NAFNet）。デフォルトはスキップ"
    )
    parser.add_argument(
        "--keep-temp", action="store_true", help="中間ファイルを出力ディレクトリに残す"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cuda", "cpu"], help="使用デバイス"
    )
    args = parser.parse_args()

    pipeline = CharPipeline(Path(args.config), device=args.device)
    pipeline.run(
        image_path=Path(args.image),
        output_dir=Path(args.output_dir),
        bbox_json=Path(args.bbox_json) if args.bbox_json else None,
        skip_arc=args.skip_arc,
        with_restoration=args.with_restoration,
        keep_temp=args.keep_temp,
    )


if __name__ == "__main__":
    main()
