from common import *
for c in cases():
 id=c['id'];x,y,w,h=c['detail_xywh'];paths=[SOURCE/'native'/id/'original_context.png',SOURCE/'native'/id/'npj_context.png',
   HERE/'learned_real/residual'/id/'output.png',HERE/'learned_real/gain'/id/'output.png',HERE/'external/stain'/id/'native.png']
 card(HERE/'learned_comparisons'/f'{id}.png',[read(p)[y:y+h,x:x+w] for p in paths],
  ['原画像','NPJ系基準','合成学習 RGB差分','合成学習 明るさ制約','StainRestorer'],id+' '+c['page'],
  '学習は別の原画像へ加えた人工汚れ。実例での学習・チェックポイント選択なし。\n合成正解への誤差と、実物の汚れ除去・筆画保持は別の評価。',size=420)
print('learned comparisons complete')
