from pathlib import Path
import os,json,collections,datetime
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def main():
 rows=[];summaries=[];skip={'.git','__pycache__','wandb','node_modules','plot_cache'}
 for base in [ROOT/'results',ROOT/'output',ROOT/'modules',ROOT/'tmp_work']:
  for parent,dirs,files in os.walk(base,followlinks=False):
   dirs[:]=[d for d in dirs if d not in skip and Path(parent)/d!=OUT and d not in ['model','experiments','src','codes','datasets','functions','runners','scripts']]
   for name in files:
    p=Path(parent)/name
    if p.suffix.lower() not in ['.png','.jpg','.jpeg','.tif','.tiff']:continue
    st=p.stat();rel=str(p.relative_to(ROOT));stamp=datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat()
    rows.append(dict(path=rel,mtime_utc=stamp,mtime=st.st_mtime,ctime=st.st_ctime,bytes=st.st_size,possibly_restoration_output=any(k in rel.lower() for k in ['result/','restored','_fin.','turezuregusa','tureduregusa','makura','heritage_result'])))
 rows.sort(key=lambda v:v['mtime']);groups=collections.defaultdict(list)
 for r in rows:
  ps=Path(r['path']).parts;key='/'.join(ps[:2]) if ps[0] in ['tmp_work','results','output'] else '/'.join(ps[:4]);groups[key].append(r)
 for key,rr in groups.items():summaries.append(dict(folder=key,images=len(rr),potential_restoration_images=sum(r['possibly_restoration_output'] for r in rr),earliest_utc=rr[0]['mtime_utc'],latest_utc=rr[-1]['mtime_utc'],earliest_example=rr[0]['path']))
 summaries.sort(key=lambda v:v['earliest_utc']);before=[r for r in rows if r['mtime_utc']<'2025-12-07']
 (OUT/'saved_image_dates.json').write_text(json.dumps(dict(scope='Workspace output/results/modules and historical tmp_work; excludes data sources, weights, source code, wandb and current audit; does not traverse symlinks. mtime is evidence, not authenticated creation time.',total_images=len(rows),images_mtime_before_publication=len(before),oldest100=rows[:100],folder_summary=summaries,prepublication=before),ensure_ascii=False,indent=2))
 (OUT/'saved_image_dates_rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 print('saved-image date inventory',len(rows),'prepublication mtime',len(before),flush=True)
 for r in summaries[:18]:print(r,flush=True)
if __name__=='__main__':main()
