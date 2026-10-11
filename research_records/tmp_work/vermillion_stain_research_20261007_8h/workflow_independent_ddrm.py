import json,time,subprocess,sys
from synthetic_layers import OUT
while not (OUT/'runtime_logs/independent_direct_complete.json').exists() or not (OUT/'independent_predictions/classical_complete.json').exists():time.sleep(15)
pages=sorted({r['page'] for r in json.loads((OUT/'independent_cases_v2.json').read_text())['cases'] if not r['page'].startswith('external_')})
subprocess.run([sys.executable,'-u',str(OUT/'run_independent_ddrm.py'),'--device','cuda:0','--pages',*pages],check=True,cwd=OUT)
(OUT/'runtime_logs/independent_ddrm_complete.json').write_text(json.dumps(dict(pages=pages)))
