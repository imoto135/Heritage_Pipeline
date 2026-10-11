from common import *
import csv,collections,argparse,time
def main(version,n=48):
 out=HERE/'other_book_screen'/version;out.mkdir(parents=True,exist_ok=True)
 inventory={r['page']:r for r in csv.DictReader((ROOT/'tmp_work/npj_failure_collection_20261001/page_inventory.csv').open())}
 already={r['page'] for r in json.loads((SOURCE/'screening.json').read_text())['screened_candidates']}
 if version=='final':
  already.update(r['page'] for r in json.loads((HERE/'other_book_screen/interim/selection.json').read_text())['selected'])
 rows=[json.loads(s) for s in (HERE/'other_book_screen/rows.jsonl').read_text().splitlines()]
 rows.sort(key=lambda r:-r['score']);selected=[];counts=collections.Counter()
 for r in rows:
  if r['page'] in already or r['score']<=0 or counts[r['book']]>=2:continue
  c=r['candidates'][0];x,y,w,h=c['page_xywh'];W,H=r['original_size'];cx=x+w/2;cy=y+h/2
  xx=min(max(0,round(cx-384)),W-768);yy=min(max(0,round(cy-384)),H-768)
  r.update(review_id=f'{version}_{len(selected)+1:02}',crop_xywh=[xx,yy,768,768],source=inventory[r['page']],review_status='not yet reviewed')
  selected.append(r);counts[r['book']]+=1
  if len(selected)>=n:break
 dump(out/'selection.json',dict(created=time.time(),pages_scored=len(rows),selected=selected,source_sha256=sha(__file__),before_new_method_outputs=True))
 f=ImageFont.truetype(FONT,22);sf=ImageFont.truetype(FONT,18)
 for start in range(0,len(selected),3):
  canvas=Image.new('RGB',(1500,3*592),'#faf9f5');d=ImageDraw.Draw(canvas)
  for j,r in enumerate(selected[start:start+3]):
   a=read(ROOT/r['original']);xx,yy,ww,hh=r['crop_xywh'];crop=a[yy:yy+hh,xx:xx+ww]
   write(out/'native'/r['review_id']/'original.png',crop)
   mask=read(ROOT/r['source']['npj_black'])[yy:yy+hh,xx:xx+ww];write(out/'native'/r['review_id']/'black_mask.png',mask)
   y=j*592;d.text((12,y+4),r['review_id']+' '+r['page']+f' score={r["score"]:.1f}',font=f,fill='black')
   for aa,x,label in [(crop,12,'原画像'),(mask,518,'保存黒文字マスク（修復結果ではない）')]:
    im=Image.fromarray(aa);im.thumbnail((490,490),Image.Resampling.NEAREST);d.text((x,y+40),label,font=sf,fill='black');canvas.paste(im,(x,y+76))
   thumb=Image.fromarray(a);thumb.thumbnail((440,490),Image.Resampling.LANCZOS);td=ImageDraw.Draw(thumb)
   sx,sy=thumb.width/a.shape[1],thumb.height/a.shape[0];td.rectangle([xx*sx,yy*sy,(xx+ww)*sx,(yy+hh)*sy],outline='red',width=3)
   canvas.paste(thumb,(1040,y+76))
  canvas.save(out/f'sheet_{start//3+1:02}.png')
 print('new original review prepared',version,len(selected),'from',len(rows),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--version',required=True);p.add_argument('--n',type=int,default=48);a=p.parse_args();main(a.version,a.n)
