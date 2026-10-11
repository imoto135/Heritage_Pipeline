"""Small official public test-pair control; no broad dataset download or training."""
import requests,json,zipfile,io,hashlib,datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from synthetic_layers import OUT,file_hash
base='https://www.kaggle.com/api/v1/datasets/download/xuhangc/wacv2025-staindoc/'
d=OUT/'stain_official_pair_probe';d.mkdir(exist_ok=True)
# Predetermined first lexical test files from public metadata, before viewing results.
items=[(kind,n) for n in ['1','10','1006'] for kind in ['input','target']]
def run(item):
 kind,n=item;remote='WACV2025-StainDoc/StainDoc/test/'+kind+'/'+n+'.jpg';u=base+quote(remote,safe='');p=d/(kind+'_'+n+'.jpg')
 if p.exists():return dict(remote=remote,url=u,file=p.name,sha256=file_hash(p),bytes=p.stat().st_size)
 r=requests.get(u,timeout=60);rec=dict(remote=remote,url=u,status=r.status_code,type=r.headers.get('Content-Type'),bytes=len(r.content))
 if r.status_code==200:
  content=r.content
  if content[:2]==b'PK':
   z=zipfile.ZipFile(io.BytesIO(content));assert len(z.namelist())==1;content=z.read(z.namelist()[0])
  if content[:2]==b'\xff\xd8':p.write_bytes(content);rec.update(file=p.name,sha256=file_hash(p))
  else:rec['error_preview']=r.text[:300]
 return rec
with ThreadPoolExecutor(max_workers=3) as e:records=list(e.map(run,items))
(d/'downloads.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source='Official StainRestorer README Kaggle link',records=records,purpose='Colour/geometry own-domain sanity control only; no task-aligned benchmark or training',selector_outputs_used=False),indent=2));print(records,flush=True)
