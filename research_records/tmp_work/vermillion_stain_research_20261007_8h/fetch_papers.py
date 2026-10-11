"""Download publicly accessible primary papers into the project for close reading."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,subprocess
from datetime import datetime,timezone
import requests

OUT=Path(__file__).resolve().parent/'literature'
SOURCES=[
    ('sauvolanet_2021','https://arxiv.org/pdf/2105.05521v1','https://arxiv.org/abs/2105.05521',2021),
    ('stainrestorer_2025','https://arxiv.org/pdf/2410.22922v1','https://arxiv.org/abs/2410.22922',2025),
    ('docres_2024','https://arxiv.org/pdf/2405.04408v1','https://arxiv.org/abs/2405.04408',2024),
    ('sage_restore_2026_preprint','https://arxiv.org/pdf/2609.36243v1','https://arxiv.org/abs/2609.36243',2026),
    ('prism_2025','https://openaccess.thecvf.com/content/ICCV2025W/VisionDocs/papers/Declercq_PRISM_Pruning_for_Rank-adaptive_Interpretable_Segmentation_Model_with_Application_to_ICCVW_2025_paper.pdf','https://openaccess.thecvf.com/content/ICCV2025W/VisionDocs/papers/Declercq_PRISM_Pruning_for_Rank-adaptive_Interpretable_Segmentation_Model_with_Application_to_ICCVW_2025_paper.pdf',2025),
    ('npj_2025','https://www.nature.com/articles/s40494-025-02214-8.pdf','https://www.nature.com/articles/s40494-025-02214-8',2025),
]

def fetch(row):
    name,url,landing,year=row
    record=dict(id=name,url=landing,pdf_url=url,year=year,read_status='download_only_not_yet_read',
        at=datetime.now(timezone.utc).isoformat())
    try:
        r=requests.get(url,timeout=45)
        r.raise_for_status()
        assert r.content[:5]==b'%PDF-', 'Not a PDF response'
        path=OUT/(name+'.pdf')
        path.write_bytes(r.content)
        dest=OUT/(name+'.txt')
        subprocess.run(['pdftotext','-layout',str(path),str(dest)],check=True)
        record.update(status='downloaded',bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest(),
            text_characters=len(dest.read_text()))
    except Exception as e:
        record.update(status='failed',error=str(e))
    print(name,record['status'],flush=True)
    return record

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(fetch,SOURCES))
    (OUT/'downloads.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
