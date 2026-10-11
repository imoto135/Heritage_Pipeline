"""Three official test pairs: own-domain colour/geometry sanity, not our heritage task."""
import argparse,json,time
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,file_hash
from run_direct_literature_baselines import load,invoke,tiles

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');a=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(2);d=OUT/'stain_official_pair_probe';model,resources=load('stain',a.device);font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17);records=[]
 for n in ['1','10','1006']:
  src=d/('input_'+n+'.jpg');tar=d/('target_'+n+'.jpg');inp=np.asarray(Image.open(src).convert('RGB'));gt=np.asarray(Image.open(tar).convert('RGB'));assert inp.shape==gt.shape;h,w=inp.shape[:2];panels=[('公式の汚れ入力',inp),('公式の対応画像',gt)]
  for side in [256,512]:
   bgr=cv2.resize(inp[:,:,::-1].copy(),(side,side),interpolation=cv2.INTER_LINEAR);target=cv2.resize(gt,(side,side),interpolation=cv2.INTER_LINEAR);v=invoke(model,bgr,'stain',a.device);pred=np.rint(v[:,:,::-1]*255).astype(np.uint8);Image.fromarray(pred).save(d/('output_'+n+'_'+str(side)+'.png'));out=cv2.resize(pred,(w,h),interpolation=cv2.INTER_LINEAR);panels.append(('StainRestorer '+str(side)+'全体',out))
   # Evaluate same resized input/target/model dimensions to avoid scoring upsizing as recovered detail.
   inputs=bgr[:,:,::-1].astype(np.float32);targetf=target.astype(np.float32);predf=pred.astype(np.float32);paper=(target.min(2)>=240)&(cv2.Laplacian(cv2.cvtColor(target,cv2.COLOR_RGB2GRAY),cv2.CV_32F)**2<25)
   rec=dict(test_file=n,side=side,source_sha256=file_hash(src),target_sha256=file_hash(tar),mean_error_input_RGB=(inputs-targetf).mean((0,1)).tolist(),mean_error_prediction_RGB=(predf-targetf).mean((0,1)).tolist(),MSE_input=float(((inputs-targetf)**2).mean()),MSE_prediction=float(((predf-targetf)**2).mean()),MAE_input=float(np.abs(inputs-targetf).mean()),MAE_prediction=float(np.abs(predf-targetf).mean()),bright_target_paper_pixels=int(paper.sum()),mean_error_prediction_on_bright_target_RGB=(predf[paper]-targetf[paper]).mean(0).tolist() if paper.any() else None,own_domain_sanity_not_task_aligned=True,known_pair_resize=True);records.append(rec)
  sheet=Image.new('RGB',(4*400,600),'#faf9f5');draw=ImageDraw.Draw(sheet)
  for i,(name,array) in enumerate(panels):
   draw.text((i*400+5,5),name,font=font,fill='#20313a');im=Image.fromarray(array);im.thumbnail((390,535));sheet.paste(im,(i*400+5,45))
  draw.text((5,576),'公式テスト対：古文書の薄い朱を保存する正解ではない。256/512全体縮小は細部を失う。',font=font,fill='#6b4a32');sheet.save(d/('公式対_'+n+'_比較.png'))
 (d/'control_results.json').write_text(json.dumps(dict(records=records,source_sha256=file_hash(__file__),adapter_sha256=file_hash(OUT/'run_direct_literature_baselines.py'),resources={str(f):file_hash(f) for f in resources},geometry='Official config256 / diagnostic512 whole-page bilinear; final output head; RGB/255',dataset='Official public StainDoc test pairs, first three lexical files registered before output',scope='Own-domain small sanity control; not author benchmark reproduction, representative test accuracy, or red-preservation training'),ensure_ascii=False,indent=2));print(records,flush=True)
if __name__=='__main__':main()
