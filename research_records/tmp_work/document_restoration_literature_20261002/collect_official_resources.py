"""Archive official README/config only; no code execution or weight download."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,json
out=Path(__file__).resolve().parent
repos=['CXH-Research/StainRestorer','ZZZHANG-jx/DocRes','beargolden/H-DIBCO-2018','FuxiJia/DocumentBinarizationSSP','Royalvice/DocDiff','xiaomore/MMDIR']
def fetch(repo):
 row=dict(repository=repo,url='https://github.com/'+repo,files=[],status='not_downloaded')
 p=out/'official_resources'/repo.replace('/','__');p.mkdir(parents=True,exist_ok=True)
 for branch in ['main','master']:
  u='https://raw.githubusercontent.com/'+repo+'/'+branch+'/README.md'
  r=requests.get(u,timeout=30)
  if r.ok:
   (p/'README.md').write_bytes(r.content);row.update(branch=branch,status='downloaded');row['files'].append(dict(file='README.md',url=u))
   for filename in ['config.yml','requirements.txt']:
    u='https://raw.githubusercontent.com/'+repo+'/'+branch+'/'+filename;v=requests.get(u,timeout=30)
    if v.ok:(p/filename).write_bytes(v.content);row['files'].append(dict(file=filename,url=u))
   break
 if repo=='CXH-Research/StainRestorer':row['pretrained_release_confirmed_via_browser']='https://github.com/CXH-Research/StainRestorer/releases/tag/Weights'
 return row
with ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(fetch,repos))
(out/'official_resources.json').write_text(json.dumps(dict(api_attempt='GitHub API returned rate-limit 403; raw author README used instead',resources=rows),ensure_ascii=False,indent=2)+'\n')
print(json.dumps([dict(repository=r['repository'],status=r['status']) for r in rows]),flush=True)
