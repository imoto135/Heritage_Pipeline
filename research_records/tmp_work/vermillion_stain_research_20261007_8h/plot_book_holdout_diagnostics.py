"""Publication-style artifact of tentative-point diagnostics, never GT accuracy."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from synthetic_layers import OUT,file_hash

def main():
    dest=OUT/'book_holdout_overview';dest.mkdir(exist_ok=True)
    sources={
        '保存済みCSC v2':('sparse_scores/initial_development/scores.json',['saved_csc_v2_red'],'新規の暫定点で学習せず'),
        'CPU ACE＋CSC v2':('sparse_scores/initial_development/scores.json',['csc_v2_red'],'新規の暫定点で学習せず'),
        '点と紙で学習したCNN':('sparse_scores/weak_texture_development/scores.json',['texture_paper_texture_sparse'],'同じ3冊を学習に使用'),
        '当該冊なしのCNN':('sparse_scores/leave_book_out_texture/scores.json',['texture_paper_texture_sparse_exclude'+b for b in ['200020019','200021086','200021763']],'冊ごとに学習から除外'),
        '当該冊なしの森分類':('sparse_scores/leave_book_out_region_forest/scores.json',['region_spatial_forest_exclude'+b for b in ['200020019','200021086','200021763']],'冊ごとに学習から除外')}
    results=[];points_hash=None
    for label,(path,methods,scope) in sources.items():
        p=OUT/path;data=json.loads(p.read_text())
        if points_hash is None:points_hash=data['points_sha256']
        assert points_hash==data['points_sha256']
        rows=[r for r in data['records'] if r['method'] in methods and r['case'] in ['dev_06','dev_08','dev_10']]
        pos=[r for r in rows if r['label']=='observed_red_mark'];neg=[r for r in rows if r['label']=='paper_or_brown_nonmark']
        assert sum(r['points'] for r in pos)==66;assert sum(r['points'] for r in neg)==26
        results.append(dict(label=label,scope=scope,selected_red=sum(r['selected_center'] for r in pos),red_points=66,
            selected_paper=sum(r['selected_center'] for r in neg),paper_points=26,source=path,source_sha256=file_hash(p)))
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size=11)
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8));ys=list(range(len(results)));labels=[r['label'] for r in results]
    for ax,key,total,title,color in [(axes[0],'selected_red',66,'暫定朱点で候補に入った数（多いほど保存）','#b6493c'),(axes[1],'selected_paper',26,'暫定紙点で候補に入った数（少ないほど誤保持なし）','#426b86')]:
        vals=[r[key] for r in results];ax.barh(ys,vals,color=[color if 'なし' in r['label'] or 'CSC' in r['label'] else '#999999' for r in results])
        ax.set_xlim(0,total+7);ax.set_yticks(ys);ax.set_yticklabels(labels if ax is axes[0] else [],fontproperties=font);ax.invert_yaxis();ax.set_title(title,fontproperties=font,pad=12)
        for y,val in zip(ys,vals):ax.text(val+.5,y,f'{val}/{total}',va='center',fontsize=10)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('同じ3冊・同じ暫定点：学習適合と、冊を外した検証の違い',fontproperties=font,y=.98)
    fig.text(.02,.03,'灰色は同じ冊での学習適合。原画像のみから選んだ92点の診断であり、画素精度・読解精度ではない。',fontproperties=font,fontsize=10)
    fig.tight_layout(rect=[0,.1,1,.93]);fig.savefig(dest/'暫定点_学習適合と冊外し.png',dpi=180);plt.close(fig)
    (dest/'results.json').write_text(json.dumps(dict(rows=results,points_sha256=points_hash,source_sha256=file_hash(__file__),expert_GT=False,representative_pixel_accuracy=False),ensure_ascii=False,indent=2))
    with (dest/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print([(r['label'],r['selected_red'],r['selected_paper']) for r in results])

if __name__=='__main__':main()
