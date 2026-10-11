from pathlib import Path
import json,hashlib,re,datetime,zipfile,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont,JpegImagePlugin

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT));from render_audit import BOXES,FONT

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return np.asarray(Image.open(p).convert('RGB'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))

def main():
    old=[]
    for name in ['ddrm_out','ddrm_out_v2fix','ddrm_out_v3fix']:
        p=ROOT/'tmp_work/speed_test_single'/name/'200010454_00002_2/result/200010454_00002_2.png'
        log=p.parents[3]/('ddrm_run'+('_v2fix' if name.endswith('v2fix') else '_v3fix' if name.endswith('v3fix') else '')+'.log')
        old.append(dict(path=str(p.relative_to(ROOT)),mtime_jst=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone(datetime.timedelta(hours=9))).isoformat(),sha256=sha(p),size=Image.open(p).size,log=str(log.relative_to(ROOT)),log_sha256=sha(log),authenticated_paper_output=False))
    save(OUT/'historical_outputs.json',old)
    save(OUT/'paper/matching_review.json',dict(searched_pages=6072,visually_reviewed='Top two warped candidates in each of three rows',confirmed_matches=0,notes='Warped top candidates lack corresponding text and have degenerate geometric maps. This does not establish absence of source pages.'))
    checks={};protocol=json.loads((OUT/'protocol.json').read_text());complete=json.loads((OUT/'complete.json').read_text());cases={c['id']:c for c in protocol['cases']}
    checks['all_source_code_and_inputs_weights_background_unchanged']=all(sha(ROOT/p)==h for p,h in {**protocol['source_code'],**protocol['dependencies']}.items())
    for row in complete['results']:
        c=cases[row['id']];p=ROOT/row['output'];a=read(p);source=ROOT/c['source']['original'];md=(ROOT/c['source']['npj_black']).parent
        input_source=md/'00_input.png' if row['variant']=='saved_ACE_input' else source
        checks[row['id']+'_'+row['variant']+'_hash_shape_input']=sha(p)==row['sha256'] and a.shape==read(source).shape and np.array_equal(read(OUT/'inputs'/row['variant']/row['id']/(c['page']+'.png')),read(input_source))
        ms=ROOT/c['v2_dir'] if row['variant']=='v2_masks' else md
        checks[row['id']+'_'+row['variant']+'_masks_logs']=all(sha(ms/n)==sha(OUT/'masks'/row['variant']/c['page']/n) for n in ['02_text_black.png','03_text_red.png']) and '[INFO] 完了。' in (OUT/'logs'/(row['id']+'_'+row['variant']+'.log')).read_text() and 'Traceback' not in (OUT/'logs'/(row['id']+'_'+row['variant']+'.log')).read_text()
        if row['variant']=='saved_ACE_input':
            b=read(ROOT/c['folder']/'output/baseline'/c['page']/'result'/(c['page']+'.png'));red=read(md/'03_text_red.png')[:,:,0]<127;changed=np.any(a!=b,axis=2)
            checks['ACE_input_only_red_changes_black_restoration_exact']=np.array_equal(changed,red) and int(changed.sum())==169325
    for c in cases.values():
        md=(ROOT/c['source']['npj_black']).parent
        checks[c['id']+'_four_regenerated_masks_class_membership_pixel_exact']=all(np.array_equal(read(md/n)[:,:,0]<127 if key in ['black','red'] else read(md/n)[:,:,0]>127,read(OUT/'mask_audit'/c['id']/(key+'_regenerated.png'))[:,:,0]>127) for n,key in [('02_text_black.png','black'),('03_text_red.png','red'),('04_degradation.png','degradation'),('05_background.png','background')])
        ace=Path(str(ROOT/c['source']['original']).replace('split_dataset/','split_dataset_csc/'))
        checks[c['id']+'_stored_classifier_input_equals_decoded_ACE']=np.array_equal(read(md/'00_input.png'),read(ace))
    for fig in json.loads((OUT/'figures/rendering.json').read_text()):
        c=cases[fig['id']];s=fig['integer_scale'];x,y,w,h=fig['ROI'];roi=np.s_[y:y+h,x:x+w];canvas=read(OUT/fig['path']);md=(ROOT/c['source']['npj_black']).parent
        if 'sources' in fig:arr=[read(Path(p))[roi] for p in fig['sources']]
        else:
            raw=read(ROOT/c['source']['original']);ace=read(md/'00_input.png');arr=[raw[roi],ace[roi]]
            for folder in [md,ROOT/c['v2_dir']]:
                a=raw.copy();a[read(folder/'02_text_black.png')[:,:,0]<127]=(20,50,210);a[read(folder/'03_text_red.png')[:,:,0]<127]=(220,25,150);arr.append(a[roi])
        checks[fig['path']+'_panels_exact']=all(np.array_equal(canvas[py:py+ph,px:px+pw],np.repeat(np.repeat(a,s,0),s,1)) for (px,py,pw,ph),a in zip(fig['panels'],arr))
    checks['all_relative_report_links_resolve']=all((OUT/p).exists() for f in ['再現監査報告.md','前回報告への訂正.md'] for p in re.findall(r'\]\(([^)]+)\)',(OUT/f).read_text()) if not p.startswith('https://'))
    checks['eight_completed_conditions_three_pages']=len(complete['results'])==8 and len(cases)==3
    save(OUT/'verification.json',dict(checks=checks,passed=sum(bool(v) for v in checks.values()),total=len(checks),authentic_paper_replay=False,verification_scope='File integrity, input/mask identity, native dimensions, saved-ACE control and figure fidelity. Does not establish fidelity to the paper or reading accuracy.'))
    assert all(checks.values()),[k for k,v in checks.items() if not v]

    pages=[];summary=Image.new('RGB',(1500,1060),'#faf9f5');d=ImageDraw.Draw(summary)
    lines=[('NPJ再現監査：比較基準を見直す',36),('論文の再現が十分とは確認できない。過去の再現も正解扱いしない。',25),('1. 学習ログ：3,878画像・条件なし。論文：4,829ページ・クラス条件付き。',25),('2. 旧ローカル重みは残るが、論文との対応を認証できない。',25),('3. 重みだけの変更で黒い塊は残る。既存v2マスクで大きく減る。',25),('4. 保存日時だけでは、記憶されている過去の結果を特定できなかった。',25),('3ページの旧マスク再生成：4クラスすべて全画素一致。',25),('8条件の原寸全ページ推論。新たな学習は行っていない。',25),('再学習より先に、掲載例・マスク生成・コードと重みの来歴を確定する。',25),('前回の「NPJ原法の失敗／新規性」という解釈は撤回・保留する。',25),('v2でも文字を読みづらくする残存例を、次の研究対象として選び直す。',25),('注意：v2も後年のローカル変更。文字保持と一般化は未検証。',25),('比較画像の原寸PNG、実験条件、証拠は同梱の再現監査報告.mdを参照。',23),('資料：nature.com/articles/s40494-025-02214-8（2025年12月6日公開）',22),('2026年10月7日 / ローカル再現監査 / 出版社の図との直接再現は未達',22)]
    y=40
    for text,size in lines:d.text((35,y),text,font=ImageFont.truetype(FONT,size),fill='#20313b');y+=64
    summary.save(OUT/'figures/監査要点.png');pages.append(summary)
    for id in ['N01','T01','N02']:
        for suffix in ['factor_controls','masks']:
            im=Image.open(OUT/'figures'/(id+'_'+suffix+'.png')).convert('RGB');page=Image.new('RGB',(1500,1060),'#faf9f5');im.thumbnail((1470,940),Image.Resampling.LANCZOS);page.paste(im,((1500-im.width)//2,(1040-im.height)//2));ImageDraw.Draw(page).text((20,1015),'全ページを推論後、同一座標を表示。原寸の確認は対応するPNGを参照。',font=ImageFont.truetype(FONT,21),fill='#53636d');pages.append(page)
    pages[0].save(OUT/'NPJ再現監査.pdf',save_all=True,append_images=pages[1:],resolution=120)
    files=[OUT/f for f in ['再現監査報告.md','前回報告への訂正.md','NPJ再現監査.pdf','verification.json','protocol.json','complete.json','weight_audit.json','weight_dates.json','training_provenance.json','saved_image_dates.json','historical_outputs.json','controlled_runs.py','mask_audit.py','render_audit.py','match_paper_pages.py','saved_image_dates.py','finalize_audit.py']]+list((OUT/'figures').glob('*'))+[OUT/'mask_audit/summary.json']+[OUT/'paper'/n for n in ['sources.json','page_matches_all.json','matching_review.json']]+list((OUT/'logs').glob('*.json'))
    save(OUT/'deliverable_manifest.json',{str(p.relative_to(OUT)):sha(p) for p in files});files.append(OUT/'deliverable_manifest.json')
    with zipfile.ZipFile(OUT/'NPJ再現監査_資料.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(OUT))
    print(json.dumps(dict(passed=sum(bool(v) for v in checks.values()),total=len(checks),pdf_pages=len(pages),zip_bytes=(OUT/'NPJ再現監査_資料.zip').stat().st_size),ensure_ascii=False))
if __name__=='__main__':main()
