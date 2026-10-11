from common import *
import os,csv,collections
os.environ['MPLCONFIGDIR']=str(HERE/'plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont(FONT);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#faf9f5','axes.facecolor':'#faf9f5','savefig.facecolor':'#faf9f5'})
names={'input':'入力のまま','residual':'RGB差分','gain_od_regression':'濃度教師＋明るさ制約','gain_augmented':'汚れ種類・薄線を追加','stain':'StainRestorer'}
colors={'input':'#7f8790','residual':'#276b9a','gain_od_regression':'#bb6d2f','gain_augmented':'#8061a5','stain':'#498270'}
domains=['identity','id','shape','ring','crease','sharp','color','gray'];labels=['無追加\n正常対照','茶色\n学習形状','不規則形','輪状','細い跡','鋭い境界','別の色','灰色']
out=HERE/'figures';out.mkdir(exist_ok=True);summaries=[]
def grouped(rs,key='mae'):
 by=collections.defaultdict(list)
 for r in rs:
  if r.get(key) is not None:by[r['book']].append(r[key])
 values=[float(np.mean(v)) for v in by.values()];return dict(mean=float(np.mean(values)),std=float(np.std(values)),books=len(values),min=min(values),max=max(values))
records={mode:json.loads((HERE/'synthetic_eval'/mode/'records.json').read_text()) for mode in ['residual','gain_od_regression','gain_augmented','stain']}
records['input']=[dict(r,mae=r['input_mae']) for r in records['residual']]
for mode,rs in records.items():
 for domain in domains:
  rows=[r for r in rs if r['domain']==domain];summaries.append(dict(method=mode,domain=domain,patches=len(rows),**grouped(rows)))
dump(HERE/'synthetic_eval/book_macro_summary.json',summaries)
with (HERE/'synthetic_eval/book_macro_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
fig,ax=plt.subplots(figsize=(12,5.5));x=np.arange(8)
for mode in ['input','residual','gain_od_regression','gain_augmented','stain']:
 values=[next(r['mean'] for r in summaries if r['method']==mode and r['domain']==d) for d in domains];ax.plot(x,values,'o-',label=names[mode],color=colors[mode],lw=2,ms=5)
ax.set_xticks(x,labels);ax.set_ylabel('RGB平均絶対誤差（0〜1、低いほど良い）');ax.set_title('合成汚れ：7資料群を同じ重みで平均');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9,ncol=3,loc='upper left');ax.set_ylim(0,.092)
fig.text(.05,.015,'資料群分離：学習11／検証4／評価7。各条件230パッチ。正解は人工汚れ追加前の画像で、元からの劣化は残る。',fontsize=10)
fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(out/'01_synthetic_domains.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5.6))
for mode in ['input','residual','gain_od_regression','gain_augmented','stain']:
 xx=next(r['mean'] for r in summaries if r['method']==mode and r['domain']=='identity');yy=next(r['mean'] for r in summaries if r['method']==mode and r['domain']=='id')
 ax.scatter(xx,yy,s=85,color=colors[mode]);ax.annotate(names[mode],(xx,yy),xytext=(8,8 if mode!='gain_od_regression' else -16),textcoords='offset points',fontsize=10)
ax.set_xlabel('無追加の正常対照に作った誤差（低いほど良い）');ax.set_ylabel('茶色い合成汚れの残る誤差（低いほど良い）');ax.set_title('汚れの改善と正常部分の保持を別に評価');ax.set_xlim(-.002,.025);ax.set_ylim(0,.065);ax.grid(alpha=.2)
fig.text(.06,.02,'左下が望ましい。正常保持だけの最良値は「何もしない」0。これは読解・真の筆画保持の指標ではない。',fontsize=10);fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(out/'02_preservation_tradeoff.png',dpi=170);plt.close(fig)
rs=json.loads((HERE/'controlled_strokes/records.json').read_text());extra=HERE/'controlled_strokes_augmented/records.json'
if extra.exists():rs+=json.loads(extra.read_text())
fig,axs=plt.subplots(1,2,figsize=(12,5));contrasts=[.06,.12,.25,.5]
for method,color in [('sauvola','#b77229'),('docres','#276b9a')]:
 for domain,ls in [('identity','-'),('id','--')]:
  v=[float(np.mean([r['known_line_recall'] for r in rs if r['method']==method and r['color']=='gray' and r['domain']==domain and r['contrast']==c])) for c in contrasts]
  axs[0].plot(np.array(contrasts)*100,v,'o',ls=ls,color=color,label=method+(' 汚れなし' if domain=='identity' else ' 茶色汚れあり'))
axs[0].set_ylim(-.04,1.08);axs[0].set_xlabel('人工線の元コントラスト（%）');axs[0].set_ylabel('既知線領域の検出割合');axs[0].set_title('二値化：薄い線が消える条件');axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
for method in ['residual','gain_od_regression','gain_augmented','stain']:
 if not any(r['method']==method for r in rs):continue
 v=[float(np.mean([r['paired_contrast_response'] for r in rs if r['method']==method and r['color']=='brown' and r['domain']=='identity' and r['contrast']==c])) for c in contrasts]
 axs[1].plot(np.array(contrasts)*100,v,'o-',label=names[method],color=colors[method])
axs[1].axhline(1,color='#7f8790',ls='--');axs[1].set_xlabel('茶色い人工線の元コントラスト（%）');axs[1].set_ylabel('線を追加したときの出力応答／正解応答');axs[1].set_title('RGB補正：線の濃さも変わる');axs[1].legend(fontsize=8);axs[1].grid(alpha=.2)
fig.text(.04,.01,'手続き生成した1.25px曲線、7資料群、112条件。実文字ではない。右図は1が忠実、1超は強調であり保存の成功とは限らない。',fontsize=9);fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(out/'03_controlled_strokes.png',dpi=170);plt.close(fig)
stress=json.loads((HERE/'stress_eval/records.json').read_text());sd=['chromatic_field','micro_speckles','opaque_overlay'];sl=['位置で色が変わる汚れ','1〜3pxの微小斑点','インクも明るくする不透明混合'];ss=[]
fig,axs=plt.subplots(1,3,figsize=(12,5))
for ax,domain,label in zip(axs,sd,sl):
 for j,mode in enumerate(['input','residual','gain_od_regression','gain_augmented']):
  rr=[r for r in stress if r['domain']==domain and r['method']==mode];value=grouped(rr);ss.append(dict(method=mode,domain=domain,**value));ax.bar(j,value['mean'],color=colors[mode])
 ax.set_xticks(range(4),['入力','RGB差分','濃度教師','種類追加'],rotation=30);ax.set_title(label,fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_ylabel('RGB平均絶対誤差')
fig.text(.04,.015,'学習・チェックポイント選択に使わない追加合成条件。元の7評価資料群を使用。微小斑点では補正が入力より悪い。',fontsize=10);fig.tight_layout(rect=[0,.075,1,1]);fig.savefig(out/'04_stress_domains.png',dpi=170);plt.close(fig)
dump(HERE/'stress_eval/book_macro_summary.json',ss);print('research plots complete',flush=True)
