from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,subprocess,json,hashlib
out=Path(__file__).resolve().parent
sources=[('StainRestorer_WACV2025','https://openaccess.thecvf.com/content/WACV2025/papers/Li_High-Fidelity_Document_Stain_Removal_via_A_Large-Scale_Real-World_Dataset_and_WACV_2025_paper.pdf'),('DocRes_CVPR2024','https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_DocRes_A_Generalist_Model_Toward_Unifying_Document_Image_Restoration_Tasks_CVPR_2024_paper.pdf'),('DocDiff_2023','https://arxiv.org/pdf/2305.03892'),('DR_Mamba_2026','https://arxiv.org/pdf/2606.22625'),('Uni_DocDiff_2025','https://arxiv.org/pdf/2508.04055')]
def fetch(row):
 name,url=row
 try:
  r=requests.get(url,timeout=45);r.raise_for_status();assert r.content.startswith(b'%PDF'),r.headers.get('content-type')
  p=out/'papers'/(name+'.pdf');p.write_bytes(r.content)
  subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True)
  return dict(name=name,url=url,pdf=str(p),sha256=hashlib.sha256(r.content).hexdigest(),status='downloaded')
 except Exception as e:return dict(name=name,url=url,status='failed',reason=str(e))
with ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(fetch,sources))
(out/'source_downloads.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([dict(name=r['name'],status=r['status']) for r in rows]),flush=True)
