"""Two genuinely unused book controls, same frozen transfer protocol."""
import argparse
import json
from pathlib import Path
import shutil

import validate_npj_dirt_transfer as runner
from build_npj_v6_comparison import ROOT, dump, sha

runner.OUT=ROOT/'tmp_work/npj_dirt_unseen_books_20261004'
runner.SPECS=[
    dict(id='unseen_lower_frame_mark',page='200004107_00023_1',xywh=[1150,2181,768,768],
         reason='Monochrome dark fragments contacting lower writing frame; physical material unconfirmed',
         role='monochrome_damage_candidate_and_text',new_v6_book=True),
    dict(id='unseen_lower_binding_mark',page='200003967_00043_1',xywh=[1185,2187,768,768],
         reason='Monochrome dark mass outside lower frame near binding; physical material unconfirmed',
         role='monochrome_damage_candidate_and_text',new_v6_book=True),
]


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','infer','classify','compare'])
    stage=parser.parse_args().stage
    if stage=='prepare':
        runner.prepare()
        p=runner.OUT/'selection.json';selection=json.loads(p.read_text())
        selection['dependencies'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
        selection['unused_book_audit']='Absent from prior baseline-review book sets and both warm-v6 manifests, checked before inference; original screening only in October 1 ranking'
        selection['illustration_review']='Both complete originals visually reviewed as text pages; monochrome scan defects are not pixel GT'
        dump(p,selection);shutil.copyfile(__file__,runner.OUT/'wrapper_snapshot.py')
    else:
        {'infer':runner.infer,'classify':runner.classify,'compare':runner.compare}[stage]()
