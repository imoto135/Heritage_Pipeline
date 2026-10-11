import json,time,subprocess,sys,os
from synthetic_layers import OUT
while not (OUT/'runtime_logs/native_photometry_complete.json').exists():time.sleep(15)
for method in ['docres_appearance_author','stain','docres_binarization','docres_appearance_native','stain_global256','stain_global512']:
 cmd=[sys.executable,'-u',str(OUT/'run_independent_direct.py'),'--method',method,'--device','cuda:0'];print('RUN',cmd,flush=True);subprocess.run(cmd,check=True,cwd=OUT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
(OUT/'runtime_logs/independent_direct_complete.json').write_text(json.dumps(dict(methods=6)))
