from common import *
for c in cases():
 id=c['id'];x,y,w,h=c['detail_xywh']
 paths=[SOURCE/'native'/id/'original_context.png',SOURCE/'native'/id/'npj_context.png',
  HERE/'simple'/id/'sauvola_w61_k0.2.png',HERE/'external/docres'/id/'native.png',
  HERE/'external/stain'/id/'native.png',HERE/'external/stain'/id/'resized256.png']
 arrays=[read(p)[y:y+h,x:x+w] for p in paths]
 card(HERE/'external_comparisons'/f'{id}.png',arrays,
  ['原画像','NPJ系基準','Sauvola w=61','DocRes 二値化','StainRestorer 原寸','StainRestorer 256縮小'],id+' '+c['page'],
  '全例共通条件。原画像からの推定。モデル・重みは変更なし。\n原寸は512pxタイル。256縮小は条件対照。論文ベンチマークの完全再現ではない。',size=400)
print('external comparison complete',len(cases()))
