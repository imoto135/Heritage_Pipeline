import os
import numpy as np
import scipy.ndimage as ndi
from skimage import io

# グレースケール画像から文字マスクを作成（暗い部分を文字とみなす）
def get_text_mask(img, threshold=150):
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    mask = (gray < threshold).astype(np.uint8)[:, :, None]  # 文字領域（暗）→ 1
    return mask

# ランダムなブロブ状の汚れマスク生成（RGB画像）
def generate_stain_mask(H, W, color=None, density=3e-4, size=30, roughness=2.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    numblobs = int(density * H * W)
    mask = np.zeros((H, W), 'i')
    for _ in range(numblobs):
        mask[np.random.randint(0, H), np.random.randint(0, W)] = 1

    dt = ndi.distance_transform_edt(1 - mask)
    mask = (dt < size).astype(np.float32)
    mask = ndi.gaussian_filter(mask, size / (2 * roughness))
    mask -= mask.min()
    mask /= mask.max()

    noise = np.random.rand(H, W)
    noise = ndi.gaussian_filter(noise, size / (2 * roughness))
    noise -= noise.min()
    noise /= noise.max()

    blob = (mask * noise > 0.5).astype(np.float32)

    if color is None:
        r = np.random.randint(170, 200)
        g = np.random.randint(110, 150)
        b = np.random.randint(90, 120)
        color = np.array([r, g, b], dtype=np.uint8).reshape(1, 1, 3)

    stain_mask = (blob[:, :, None] * color).astype(np.uint8)
    return stain_mask, color

# 汚れ適用処理
def apply_stain(img, stain_mask, text_mask=None, avoid_text=True, intensity=1.0):
    img = img.astype(np.float32)
    stain = stain_mask.astype(np.float32)

    if avoid_text and text_mask is not None:
        # 文字を避けたブレンド
        bg_mask = (1 - text_mask).astype(np.float32)
        stain_area = np.mean(stain, axis=2, keepdims=True) > 10  # 非黒エリア
        blend_mask = bg_mask * stain_area  # 文字以外かつ汚れ領域
        result = img * (1 - blend_mask * intensity) + stain * (blend_mask * intensity)
        result = np.clip(result, 0, 255).astype(np.uint8)

    else:
        # 文字含めて全体に汚れをブレンド
        stain_area = np.mean(stain, axis=2, keepdims=True) > 10
        result = img * (1 - stain_area * intensity) + stain * (stain_area * intensity)
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def process_images(input_dir, output_dir, density=3e-4, intensity=1.0, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    with_text_path = os.path.join(output_dir, "with_text")
    no_text_path = os.path.join(output_dir, "no_text")
    os.makedirs(with_text_path, exist_ok=True)
    os.makedirs(no_text_path, exist_ok=True)

    # ✅ 固定色（ここで一括設定）
    fixed_color = np.array([180, 130, 100], dtype=np.uint8).reshape(1, 1, 3)
    
    

    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for i, fname in enumerate(sorted(files)):
        img_path = os.path.join(input_dir, fname)
        img = io.imread(img_path)
        H, W, _ = img.shape

        # ✅ 色を固定で使用
        stain_mask, _ = generate_stain_mask(H, W, color=fixed_color, density=density, seed=seed + i)
        text_mask = get_text_mask(img)

        img_with_text = apply_stain(img, stain_mask, avoid_text=False, intensity=intensity)
        img_no_text = apply_stain(img, stain_mask, text_mask, avoid_text=True, intensity=intensity)

        io.imsave(os.path.join(with_text_path, fname), img_with_text)
        io.imsave(os.path.join(no_text_path, fname), img_no_text)

    print(f"{len(files)}枚処理完了: {with_text_path}, {no_text_path}")


# 実行例
process_images(
    input_dir="/home/yoshizu/document/bunken_diffusion_folder/枕草子/dataset/dataset",
    output_dir="/home/yoshizu/document/bunken_diffusion_folder/枕草子/dataset/dataset_stain",
    density=3e-4,
    intensity=1.0,
    seed=1000000
)
