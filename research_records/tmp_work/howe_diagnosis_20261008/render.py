from pathlib import Path
import json, csv, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import experiment as e

H=e.HERE;R=e.ROOT
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def txt(d,xy,s,n=19):d.text(xy,s,font=ImageFont.truetype(FONT,n),fill='#203442')
def gray_rgb(a):return np.repeat(a[...,None],3,axis=2)
def binary(a):return gray_rgb(np.where(a,0,255).astype('uint8'))
def pad(a):return np.pad(a,((0,1),(0,1)),mode='edge')
FIGS=[]
def sheet(name,case,panels,ncol=3,detail=False,footer=None):
    x,y,w,h=case['detail_xywh'] if detail else case['roi_xywh'];scale=2 if detail else 1
    pw,ph=w*scale,h*scale;gap=20;head=98;label=42;nrow=(len(panels)+ncol-1)//ncol
    canvas=Image.new('RGB',(ncol*pw+(ncol+1)*gap,head+nrow*(ph+label+gap)+75),'#faf9f5');d=ImageDraw.Draw(canvas)
    txt(d,(15,12),name,24);txt(d,(15,49),case['page']+f' / xywh={x},{y},{w},{h} / 全ページ処理後の同位置',17)
    positions=[]
    for i,(key,title,a) in enumerate(panels):
        tile=a[y:y+h,x:x+w];tile=np.repeat(np.repeat(tile,scale,0),scale,1)
        px=gap+(i%ncol)*(pw+gap);py=head+(i//ncol)*(ph+label+gap)
        txt(d,(px,py),title,18);canvas.paste(Image.fromarray(tile),(px,py+label));positions.append(dict(key=key,xywh=[px,py+label,pw,ph],tile=tile))
    txt(d,(15,canvas.height-61),'Howe：著者Alg3の選択規則を移植。Cannyは近似で、原実装との一致は未検証。',16)
    txt(d,(15,canvas.height-36),footer or '条件を一つずつ変更した原因診断。精度や総合順位を示す実験ではない。',16)
    dest=H/'figures'/(name+'.png');dest.parent.mkdir(exist_ok=True);canvas.save(dest)
    saved=e.read(dest)
    for p in positions:
        xx,yy,ww,hh=p['xywh'];assert np.array_equal(saved[yy:yy+hh,xx:xx+ww],p.pop('tile'))
    FIGS.append(dict(path=str(dest.relative_to(H)),case=case['id'],roi_xywh=[x,y,w,h],scale=scale,panels=positions,sha256=e.sha(dest)))

def main():
    cases=json.loads((H/'protocol.json').read_text())['cases'];results={p.stem:json.loads(p.read_text()) for p in (H/'results').glob('*.json')}
    probe=json.loads((H/'contact_stroke_probe.json').read_text());pts=np.array(probe['points_xy'])
    stats=[];independent=[]
    for page in sorted(results):
        cc=[c for c in cases if c['page']==page];c=cc[0];raw=e.read(R/c['original']);g=e.deep.base.gray(raw)
        folder=H/'masks'/page;keys=['sauvola','fixed','auto','no_post','no_bias','no_edges','c0','c40','c80','c320','thi025_c160']
        masks={k:e.mask(folder/(k+'.png')) for k in keys};ims={k:binary(v) for k,v in masks.items()}
        rr=results[page];t,_=e.terms(g,thi=rr['thi']);pre,flow=e.deep.cut(t[0],t[1],rr['c']*t[2],rr['c']*t[3]);auto_ind,_=e.post(pre)
        assert np.array_equal(auto_ind,masks['auto']);independent.append(dict(page=page,auto_matches_independent_graph=True,source_unchanged=e.sha(R/c['original'])==c['source_sha256']))
        for c in cc:
            x,y,w,h=c['roi_xywh'];sl=np.s_[y:y+h,x:x+w]
            for k,m in masks.items():
                stats.append(dict(case=c['id'],page=page,method=k,roi_black_pixels=int(m[sl].sum()),changed_from_fixed=int((m[sl]!=masks['fixed'][sl]).sum()),removed_by_post=int((masks['no_post'][sl]&~masks['fixed'][sl]).sum())))
            for detail in [False,True]:
                sheet(c['id'][:2]+'_自動調整比較'+('_詳細2倍' if detail else ''),c,[('original','原画像',raw),('sauvola','Sauvola w61 k=.2',ims['sauvola']),('fixed','Howe 固定 c160 / thi=.4',ims['fixed']),('original_repeat','原画像（同一）',raw),('auto',f'Howe Alg3 / c={rr["c"]:.1f}, thi={rr["thi"]}',ims['auto']),('low','Howe c160 / thi=.25',ims['thi025_c160'])],detail=detail)
            sheet(c['id'][:2]+'_原因の切り分け',c,[('original','原画像',raw),('fixed','固定 c160 / thi=.4',ims['fixed']),('no_post','後処理の直前',ims['no_post']),('no_bias','背景への強制判定なし',ims['no_bias']),('c40','重み c40 / thi=.4',ims['c40']),('no_edges','エッジによる境界保護なし',ims['no_edges'])])
        if page=='hnsd007_042':
            c=cc[0];t,aux=e.terms(g);_,auxlow=e.terms(g,thi=.25)
            auxdir=H/'intermediates'/c['id'];auxdir.mkdir(exist_ok=True,parents=True)
            # Data cost prefers ink at positive Laplacian, not an ink label GT.
            lap=pad(aux['lapraw']);v=np.clip(lap/25,-1,1);colors=np.full((*lap.shape,3),245.,dtype='float32')
            pos=v>0;neg=~pos;colors[pos]=245-(v[pos,None])*np.array([180,65,10]);colors[neg]=245-(-v[neg,None])*np.array([0,155,190]);colors=colors.astype('uint8')
            disagreement=raw.copy();lost=masks['sauvola']&~masks['fixed'];added=masks['fixed']&~masks['sauvola'];disagreement[lost]=[230,45,160];disagreement[added]=[0,175,210]
            sheet('03_エネルギーと境界',c,[('original','原画像',raw),('lap','分類根拠：青=墨側 / 赤=背景側',colors),('bias','背景への強制判定（黒）',binary(aux['confident'])),('edges','エッジ thi=.4（黒）',binary(aux['edges'])),('edgeslow','エッジ thi=.25（黒）',binary(auxlow['edges'])),('difference','桃=Sauvolaのみ / 水=Howeのみ',disagreement)],detail=True,footer='色は分類コストと手法間の差を示す。文字・汚れの正解ではない。')
            # A constrained corridor prevents connections through remote dirt.
            path=np.zeros(g.shape,'uint8')
            for a,b in zip(pts[:-1],pts[1:]):e.cv2.line(path,tuple(a),tuple(b),1,1)
            corridor=ndi.binary_dilation(path,iterations=8)
            for k,m in masks.items():
                labels,n=ndi.label(m&corridor,np.ones((3,3)))
                def endpoint_labels(p):
                    xx,yy=p;return set(np.unique(labels[yy-2:yy+3,xx-2:xx+3]))-{0}
                same=bool(endpoint_labels(pts[0])&endpoint_labels(pts[-1]))
                stats.append(dict(case='contact_stroke_probe',page=page,method=k,points_retained=int(m[pts[:,1],pts[:,0]].sum()),points_total=len(pts),path_pixel_coverage=float(m[path>0].mean()),endpoints_connected_in_8px_corridor=same))
            gap=(1085,495,65,57);xx,yy,ww,hh=gap
            cause=dict(gap_xywh=gap,baseline_edges_in_gap=int(aux['edges'][yy:yy+hh,xx:xx+ww].sum()),low_threshold_edges_in_gap=int(auxlow['edges'][yy:yy+hh,xx:xx+ww].sum()),background_forced_pixels_in_gap=int(aux['confident'][yy:yy+hh,xx:xx+ww].sum()),probe_laplacian=[float(aux['lapraw'][y,x]) for x,y in pts],probe_background_forced=[bool(aux['confident'][y,x]) for x,y in pts],probe_removed_by_post=int((masks['no_post']&~masks['fixed'])[pts[:,1],pts[:,0]].sum()))
            e.dump(H/'contact_cause.json',cause)
        if page=='hnsd001_011':
            for k,m in masks.items():stats.append(dict(case='isolated_spot_probe',page=page,method=k,foreground_pixels=int(m[590:650,460:550].sum()),area=5400))
    e.dump(H/'measurements.json',stats);e.dump(H/'figure_verification.json',dict(figures=FIGS,exact_panels=sum(len(f['panels']) for f in FIGS),selected_auto_independent_solve=independent))
    print('figures',len(FIGS),'verified panels',sum(len(f['panels']) for f in FIGS))
    for r in stats:
        if r['case'] in ['contact_stroke_probe','isolated_spot_probe']:print(r)

if __name__=='__main__':main()
