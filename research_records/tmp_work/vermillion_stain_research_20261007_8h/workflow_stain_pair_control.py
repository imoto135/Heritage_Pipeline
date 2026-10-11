import time,subprocess,sys,json
from synthetic_layers import OUT
while not (OUT/'independent_predictions/queue_complete.json').exists():time.sleep(15)
subprocess.run([sys.executable,'-u',str(OUT/'run_stain_pair_control.py'),'--device','cuda:1'],check=True,cwd=OUT)
(OUT/'runtime_logs/stain_pair_control_complete.json').write_text(json.dumps(dict(pairs=3)))
