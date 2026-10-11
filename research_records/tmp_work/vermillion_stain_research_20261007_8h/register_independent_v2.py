"""Append prospective evaluation protocol without replacing v1."""
import json,datetime
from synthetic_layers import OUT,ROOT,file_hash
v1=OUT/'independent_evaluation_protocol_v1.json';p=json.loads(v1.read_text());p['version']=2;p['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p['predecessor_sha256']=file_hash(v1)
ck=OUT/'native_partial_trials/dense_native_photometry_seed715/last.pth';assert (ck.parent/'complete.json').exists()
p['model_resources']['partial_dense_native_photometry']=dict(path=str(ck.relative_to(ROOT)),sha256=file_hash(ck),step=8000)
p['photometry_extension']='Registered before any independent output inspection; fixed LAST8000 and .5 as all partial models; source-only shared input/paper gain augmentation, not a new physical model'
p['prospective_cases']='prospective_cases/cases_frozen.json';p['prospective_points']='prospective_cases/points_frozen.json';p['prospective_cases_sha256']=file_hash(OUT/p['prospective_cases']);p['prospective_points_sha256']=file_hash(OUT/p['prospective_points']);p['prospective_status']='Original-only selection and labels fixed before inference; no prospective adaptation'
cs=[]
for r in json.loads((OUT/'cases_frozen.json').read_text())['cases']:
 if r['role'] in p['local_roles']:cs.append({**r,'case_id':r.get('case_id',r.get('id',r.get('case'))),'source_manifest':'cases_frozen.json'})
for r in json.loads((OUT/p['external_cases']).read_text())['cases']:
 cs.append({**r,'page':'external_'+r['page'],'original':str((OUT/r['original']).relative_to(ROOT)),'ROI':r.get('ROI',r.get('xywh_original',r.get('xywh'))),'case_id':r.get('case_id',r.get('id',r.get('case'))),'source_manifest':p['external_cases']})
for r in json.loads((OUT/p['prospective_cases']).read_text())['cases']:cs.append({**r,'source_manifest':p['prospective_cases']})
for r in cs:
 assert r['ROI'] is not None and r['case_id'] is not None,r
 assert file_hash(ROOT/r['original'])==r['source_sha256']
q=OUT/'independent_cases_v2.json';assert not q.exists();q.write_text(json.dumps(dict(cases=cs,utc=p['utc'],protocol='independent_evaluation_protocol_v2.json',scope='This fit only; prior project book use not universally ruled out'),ensure_ascii=False,indent=2));p['unified_cases']='independent_cases_v2.json';p['unified_cases_sha256']=file_hash(q)
v2=OUT/'independent_evaluation_protocol_v2.json';assert not v2.exists();v2.write_text(json.dumps(p,ensure_ascii=False,indent=2));print('Registered independent cases',len(cs),'pages',len({r['page'] for r in cs}))
