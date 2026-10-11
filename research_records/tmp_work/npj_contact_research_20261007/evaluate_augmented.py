from common import *
import torch,time
import evaluate_pilot as ev
from apply_real_models import selection,infer_pilot
def load(mode,device):
 ck=HERE/'learned'/mode/'best.pt';state=torch.load(ck,map_location=device,weights_only=True);model=ev.Pilot('gain').to(device)
 model.load_state_dict(state['state']);model.eval();return model,state,ck
while not (HERE/'learned/gain_augmented/complete.json').exists():time.sleep(5)
ev.load=load;ev.main('gain_augmented','cuda:0')
p=HERE/'synthetic_eval/gain_augmented/protocol.json';v=json.loads(p.read_text());v.update(load_wrapper_sha256=sha(__file__),old_test_results_informed_design=True,exploratory=True);dump(p,v)
model,_,ck=load('gain_augmented','cuda:0');rows=[]
for c in selection():
 id=c['id'];a=read(HERE/'real_models'/id/'original.png');output,field=infer_pilot(model,a,'cuda:0');folder=HERE/'real_augmented'/id
 write(folder/'output.png',output);np.save(folder/'field.npy',field)
 x,y,w,h=c['detail_xywh'];sl=np.s_[y:y+h,x:x+w]
 arrays=[a,read(HERE/'real_models'/id/'docres.png'),read(HERE/'real_models'/id/'residual.png'),read(HERE/'real_models'/id/'gain_od_regression.png'),output]
 card(folder/'comparison.png',[im[sl] for im in arrays],['原画像','DocRes','RGB差分学習','茶色中心の濃度学習','種類・薄線を増やした濃度学習'],id+' '+c['page'],
  '人工汚れで学習、実画像は学習・チェックポイント選択に不使用。\n改良の設計は以前の合成評価を見てから。独立した一般化の証明ではない。',size=420)
 rows.append(dict(id=id,changed_fraction=float(np.any(output!=a,axis=2).mean()),mean_abs_difference=float(np.abs(output.astype('float32')-a).mean()/255),field_mean=float(field.mean())))
dump(HERE/'real_augmented/complete.json',dict(records=rows,checkpoint_sha256=sha(ck),wrapper_sha256=sha(__file__)))
ref=json.loads((HERE/'synthetic_eval/residual/records.json').read_text());rs=json.loads((HERE/'synthetic_eval/gain_augmented/records.json').read_text())
for a,b in zip(ref,rs):assert a['index']==b['index'] and a['domain']==b['domain'] and abs(a['input_mae']-b['input_mae'])<1e-8
print('augmented evaluation and 25 real frames complete',flush=True)
