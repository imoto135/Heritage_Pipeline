"""Human visual review aids; labels are provisional and never removal masks."""
from pathlib import Path
import hashlib
import json
from collections import Counter
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUT = BASE / 'bleed_local_review'
OLD = BASE / 'bleed_pair_registration'
PAGE = '200021071_00033_2'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def board(rows, arrays, titles, name, panel=(320, 340)):
    pw, ph = panel
    sheet = Image.new('RGB', (len(arrays)*(pw+10)+10, len(rows)*(ph+90)+10), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT, 18)
    for i, row in enumerate(rows):
        y0 = i*(ph+90)+10
        draw.text((10, y0), row['caption'], fill='black', font=font)
        for j, (arr, title) in enumerate(zip(arrays, titles)):
            x0 = 10+j*(pw+10)
            draw.text((x0, y0+28), title, fill='black', font=font)
            box = row['context_xyxy']
            cr = arr.crop(tuple(box))
            cr.thumbnail((pw, ph), Image.Resampling.LANCZOS)
            px, py = x0+(pw-cr.width)//2, y0+65
            sheet.paste(cr, (px, py))
            if 'xywh' in row:
                x, y, w, h = row['xywh']
                sx, sy = cr.width/(box[2]-box[0]), cr.height/(box[3]-box[1])
                draw.rectangle((px+(x-box[0])*sx, py+(y-box[1])*sy,
                                px+(x+w-box[0])*sx, py+(y+h-box[1])*sy), outline='#e02020', width=2)
    sheet.save(OUT/name)

def main():
    OUT.mkdir(exist_ok=True)
    protocol = json.loads((OLD/'registration_and_candidate.json').read_text())
    # Check established dependencies rather than mix earlier probe support rates.
    for path, expected in protocol['dependencies'].items():
        assert sha(ROOT/path) == expected, path
    source_path = ROOT/'data/test/200021071'/f'{PAGE}.jpg'
    paths = [source_path, OLD/'source_affine_registered.png'] + [
        BASE/'six_challenge_fixed_review/outputs'/m/PAGE/'result'/f'{PAGE}.png'
        for m in ('csc', 'sauvola', 'howe')]
    arrays = [Image.open(p).convert('RGB') for p in paths]
    assert all(im.size == arrays[0].size for im in arrays)
    categories = {8: '点・傷状の像', 10: '点・傷状の像', 13: '不明（短線）',
                  44: '不明（文字状）', 50: '不明（文字状）', 51: '不明（文字状）',
                  53: '不明（文字状）', 127: '不明（本文近傍）'}
    reasons = {
        8: '本文のない上余白の縦点。参照の濃い像との近接はあるが形状・由来は確定できない。',
        10: '上余白の小点。対応する参照の濃い本文はない。虫害・汚れ等の原因は未確定。',
        13: '上部の淡い曲線に接する短線。参照にも淡い像があるが濃い本文との対応はない。',
        44: '淡い文字状のまとまりの上端。表の淡い筆画・裏移り・他の痕跡を区別できない。',
        50: '同じ淡い文字状のまとまりの一部。参照の濃い本文との対応はない。',
        51: '同じまとまりに近い小成分。小さいことを除去根拠にしない。',
        53: '淡い文字状のまとまりの下部曲線。CSCでは消えるが文字欠損の危険を否定できない。',
        127: '濃い表文字の間にある淡い短線。表文字の薄い続きの可能性が残る。',
    }
    rows = []
    for item in protocol['pale_component_diagnostics']:
        r = dict(item)
        x, y, w, h = r['xywh']
        cx, cy = x+w//2, y+h//2
        r.update(category=categories[r['label']], cause_confirmed=False,
                 decision='保持', removal_approved=False, reason=reasons[r['label']],
                 context_xyxy=[cx-180, max(0,cy-190), cx+180, cy+190])
        r['caption'] = f"成分{r['label']}：{r['category']}／保持　直前ページ濃画素支持 {r['source_support_fraction']:.3f}"
        rows.append(r)
    board(rows, arrays, ['原画像', '直前鏡像・位置合わせ', 'CSC → DDRM',
                         'Sauvola → DDRM', 'Howe → DDRM'], '07_残存8か所_分類と比較.png')
    neighbors = ['200021071_00032_2', '200021071_00034_1']
    neighbor_paths = [ROOT/'data/test/200021071'/f'{p}.jpg' for p in neighbors]
    mirrored = [Image.open(p).convert('RGB').transpose(Image.Transpose.FLIP_LEFT_RIGHT).resize(arrays[0].size)
                for p in neighbor_paths]
    # A manually selected diagnostic glyph ROI, not a fitted global removal rule.
    windows = [(1430,610,155,205), (1440,620,145,95), (1430,715,155,100)]
    def feature(im):
        gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY).astype(np.float32)
        return np.maximum(cv2.GaussianBlur(gray, (0,0), 12)-gray, 0)
    tf = feature(arrays[0])
    fits = []
    for neighbor, im in zip(neighbors, mirrored):
        rf = feature(im)
        probes = []
        for x,y,w,h in windows:
            scores = cv2.matchTemplate(rf[y-65:y+h+65,x-65:x+w+65],
                                       tf[y:y+h,x:x+w],cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(scores)
            probes.append(dict(xywh=[x,y,w,h], ncc=float(score),
                               source_minus_target_translation_xy=[loc[0]-65,loc[1]-65]))
        fits.append(dict(page=neighbor, probes=probes))
    nextfit = fits[1]
    assert [p['source_minus_target_translation_xy'] for p in nextfit['probes']] == [[14,-3]]*3
    dx,dy = nextfit['probes'][0]['source_minus_target_translation_xy']
    next_aligned = Image.fromarray(cv2.warpAffine(np.asarray(mirrored[1]),
                   np.float32([[1,0,-dx],[0,1,-dy]]), arrays[0].size, borderValue=(255,255,255)))
    next_aligned.save(OUT/'next_page_local_translation.png')
    # Translation valid only for this glyph neighborhood; not whole-page registration.
    for row in rows:
        if row['label'] in (44,50,51,53):
            row.update(category='近隣ページ対応の裏移り・転写候補',
                       corresponding_page=neighbors[1], local_shape_match=True,
                       reason='00034_1鏡像の濃い文字状のまとまりと局所形状が対応。上下の照合で同じ移動量。画像だけで裏面透過・転写等の物理原因は確定しない。')
            row['caption'] = f"成分{row['label']}：次ページ対応の裏移り・転写候補／今回は保持"
    board(rows, [arrays[0], arrays[1], next_aligned]+arrays[2:],
          ['原画像','直前鏡像・位置合わせ','次ページ鏡像・局所移動',
           'CSC → DDRM','Sauvola → DDRM','Howe → DDRM'], '07_残存8か所_分類と比較.png')
    board([dict(caption='上部の残存像：次ページの濃い形状と対応／局所移動のみ',
          context_xyxy=[1350,480,1770,920])], [arrays[0],arrays[1],next_aligned]+arrays[2:],
          ['原画像','直前鏡像・位置合わせ','次ページ鏡像・局所移動',
           'CSC → DDRM','Sauvola → DDRM','Howe → DDRM'], '07_次ページ対応_拡大比較.png',(350,450))
    board(rows, [arrays[0], arrays[1]]+mirrored,
          ['原画像', '00033_1：位置合わせ済', '00032_2：鏡像・未位置合わせ',
           '00034_1：鏡像・未位置合わせ'], '07_残存8か所_近隣ページ確認.png')
    controls = [
        dict(id='C1', caption='C1：濃い表文字と淡い文字状像の混在／上下文脈を確認', context_xyxy=[1350,480,1770,920]),
        dict(id='C2', caption='C2：本文中の表文字を保持する対照／不明短線も含む', context_xyxy=[620,960,1100,1420]),
        dict(id='C3', caption='C3：右欄の裏移り候補と表文字／対応形状と除去結果を確認', context_xyxy=[1550,1250,1950,1800]),
    ]
    board(controls, arrays, ['原画像', '直前鏡像・位置合わせ', 'CSC → DDRM',
                            'Sauvola → DDRM', 'Howe → DDRM'], '07_表文字と裏移り候補_対照比較.png', (340,450))
    annotation = dict(status='pending_visual_review', page=PAGE, development_only=True,
                      pixel_ground_truth=False, annotation_type='provisional_visual_review',
                      counts=dict(Counter(r['category'] for r in rows)), components=rows, controls=controls,
                      mask_modified=False, inference_run=False, removal_pixels=0,
                      registration_limit=protocol['registration_limit'],
                      neighbors_locally_registered={'00032_2':False, '00034_1':'upper glyph local translation only'},
                      next_page_local_translation=dict(method='local positive dark-contrast NCC',
                         manually_selected_diagnostic_windows=True, gaussian_sigma_px=12,
                         search_radius_px=65, fits=fits,
                         applied_source_to_target_xy=[-dx,-dy],
                         validity='only upper glyph neighborhood; not whole-page alignment',
                         physically_confirmed_bleed=False),
                      dependencies={str(p.relative_to(ROOT)):sha(p) for p in paths+neighbor_paths},
                      script_sha256=sha(Path(__file__)))
    (OUT/'局所分類.json').write_text(json.dumps(annotation, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(components=len(rows), counts=annotation['counts'], mask_modified=False),ensure_ascii=False))

if __name__ == '__main__':
    main()
