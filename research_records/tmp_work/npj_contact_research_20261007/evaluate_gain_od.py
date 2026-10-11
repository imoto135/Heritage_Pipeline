from common import *
import time,torch
import evaluate_pilot as ev
def load(mode,device):
 ck=HERE/'learned'/mode/'best.pt';state=torch.load(ck,map_location=device,weights_only=True)
 model=ev.Pilot('gain').to(device);model.load_state_dict(state['state']);model.eval();return model,state,ck
while not (HERE/'learned/gain_od_regression/complete.json').exists():time.sleep(5)
ev.load=load;ev.main('gain_od_regression','cuda:1')
p=HERE/'synthetic_eval/gain_od_regression/protocol.json';v=json.loads(p.read_text());v['load_wrapper_sha256']=sha(__file__);v['training_loss_changed']=True;dump(p,v)
ref=json.loads((HERE/'synthetic_eval/residual/records.json').read_text());rs=json.loads((HERE/'synthetic_eval/gain_od_regression/records.json').read_text())
assert len(ref)==len(rs)
for a,b in zip(ref,rs):assert a['index']==b['index'] and a['domain']==b['domain'] and abs(a['input_mae']-b['input_mae'])<1e-8
print('gain OD evaluation input matching verified',flush=True)
