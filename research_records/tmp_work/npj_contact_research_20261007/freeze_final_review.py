from common import *
import time
notes='''主本文は比較的正常／軽い紙色
挿絵
表紙・題箋
本文と薄い裏写りらしい像
本文と強い灰色の裏写りらしい像
挿絵
密な本文、明瞭な局所シミは未確認
表紙
上余白の穴・シミ、本文接触なし
上余白の穴・シミ、本文接触なし
彩色挿絵
彩色挿絵
薄い題字、灰色像の表裏不明
表紙
細い注記と軽い紙色
綴じ端、本文なし
表紙・題箋
余白の軽い紙色、本文正常
表紙
綴じ端
綴じ端
表紙・挿絵
余白の水染み、灰色像の材質・表裏不明
下余白の紙色と細い本文
本文下の褐色紙肌、細かい黒マスク雑音
挿絵
一様な黄ばみと細字
表紙
細字に重なる亀裂・穴、物理損傷
挿絵
表紙・題箋・印
挿絵
密な本文と細い注記
下端の薄い本文と亀裂・摩耗
罫内の小文字と灰色像
装飾題字、余白
本文と軽い余白紙色
表紙
小さな褐色シミが草書本文に接触
彩色挿絵
細い縦文字、ページ内に朱印、灰色像は不明
無地表紙
水染みと灰色像、明瞭な表面本文なし
彩色挿絵
彩色挿絵
綴じ端近傍の薄い本文
小シミが罫線交点に接触、本文への影響は小さい
複数の薄茶シミが細い本文に重なる
裏表紙・題箋
薄い本文と綴じ端
表紙・題箋・透けた印刷
上余白の点状汚れ、本文接触なし
上端の薄いシミが罫線・最上本文近傍に接触
表紙
裏表紙
褐色シミが本文外の罫線に接触
下端の摩耗・亀裂と細い本文近傍
罫内の細字・注記、局所シミ未確認
小シミが文字のすぐ横、明瞭な接触は未確認
小褐色シミが細い縦筆画に接触、灰色像も混在
表紙・図書館ラベル
複数シミのうち一つが草書本文に重なる
余白の穴・摩耗、本文なし
挿絵
細い本文、明瞭な局所シミ未確認
上余白の穴・摩耗、本文接触なし
細い振り仮名と本文、間の茶色線は材質不明
細い振り仮名と灰色像、点状マスク雑音'''.splitlines()
assert len(notes)==68
rows=json.loads((HERE/'other_book_screen/final/selection.json').read_text())['selected']
contact={39,48,60,62};near={47,59};selected={35,39,41,47,48,59,60,62,65,67}
excluded={2,3,6,8,11,12,14,16,17,19,20,21,22,26,28,30,31,32,36,38,40,42,44,45,49,51,54,55,61,64}
review=[];cohort=[]
for n,(r,note) in enumerate(zip(rows,notes),1):
 role='contact_stain_candidate' if n in contact else 'near_stain_control' if n in near else 'excluded_cover_illustration_or_no_text' if n in excluded else 'other_material_or_text_control'
 review.append(dict(id=r['review_id'],page=r['page'],observation=note,role=role,material_GT=False,reading_GT=False,original_and_existing_masks_only=True))
 if n in selected:
  x,y,w,h=r['crop_xywh'];cohort.append(dict(id=f'F{n:02d}',page=r['page'],source=r['source'],detail_page_xywh=[x,y,w,h],role=role,original_observation=note))
dump(HERE/'other_book_screen/final/visual_review.json',dict(created=time.time(),review=review,all_68_reviewed=True,results_not_used_for_selection=True))
dump(HERE/'other_book_screen/final/cohort_C.json',dict(created=time.time(),selected=cohort,rationale='four contact candidates, two near-stain controls, four fine-writing/annotation/color controls; fixed before cohort C restoration',cases=10))
print('frozen',len(cohort),'pages',len({r['source']['book'] for r in cohort}),'source groups')
