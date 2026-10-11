"""Freeze small, original-image-only visual probes for known development cases.

The pages and contact points were previously studied. These labels are exploratory,
not independent or expert ground truth. Run once, before scoring this frozen set.
"""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE.parent


def main():
    target = HERE / 'frozen_probes.json'
    if target.exists():
        raise FileExistsError(target)
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    cases = {c['id'][:2]: c for c in protocol['cases']}
    contact = json.loads((SOURCE / 'contact_stroke_probe.json').read_text())
    probes = [
        dict(id='thin_01', case='01', role='visibly thin original stroke',
             points_xy=[[1407, 600], [1401, 613], [1398, 625],
                        [1404, 640], [1409, 652], [1411, 662]],
             radius=2, corridor_radius=5,
             note='Approximate centerline traced on 01_original_grid.png; faint lower continuation excluded.'),
        dict(id='contact_03', case='03', role='visible stroke crossing the stained region',
             points_xy=contact['points_xy'], radius=contact['probe_radius'],
             corridor_radius=contact['corridor_radius'],
             note='Reuses the previously frozen contact-stroke probe; not new independent annotation.'),
        dict(id='spot_04', case='04', role='isolated nontext brown spot',
             box_xywh=[462, 592, 78, 48],
             note='Rectangular probe around the isolated spot on 04_original_grid.png; includes paper around spot, excludes nearby writing.'),
    ]
    sources = {}
    for key in ['01', '03', '04']:
        c = cases[key]
        source = ROOT / c['original']
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == c['source_sha256']
        sources[key] = dict(page=c['page'], original=c['original'], sha256=digest)
    record = dict(
        purpose='Exploratory fixed probes on already studied development pages',
        provenance='Positions selected from original-only coordinate grids; contact points pre-existing; prior method outputs had already been viewed.',
        caveat='Not blinded, dense pixel GT, ink material GT, expert transcription, or independent generalization evidence.',
        unknown_case='02 water-stained faint characters remain visually ambiguous; no positive labels asserted here.',
        source_files=sources, probes=probes)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
