"""Check final documentation and hashes without repeating unchanged image tests."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((OUT / 'manifest.json').read_text())
assert sha(ROOT / 'scripts/benchmark/refine_warm_ink_hold.py') == manifest['source_sha256']
for rel, expected in manifest['dependencies'].items():
    assert sha(ROOT / rel) == expected, rel
complete = json.loads((OUT / 'complete.json').read_text())
for rel, expected in complete['outputs'].items():
    assert sha(OUT / rel) == expected, rel


class Links(HTMLParser):
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value and not value.startswith('#'):
                assert (OUT / value).is_file(), value


Links().feed((OUT / 'index.html').read_text())
documents = [OUT / 'visual_findings.md', OUT / 'report.md',
             ROOT / 'tmp_work/preprocessing_current_status_20260930.md']
for path in documents:
    for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
        if not target.startswith(('http://', 'https://', '#')):
            assert (path.parent / target).is_file(), (path, target)
subprocess.run(['node', '--check', str(OUT / 'viewer.js')], check=True, cwd=ROOT)
verification = json.loads((OUT / 'verification.json').read_text())
assert verification['status'] == 'passed'
verification['output_hashes_checked'] = len(complete['outputs'])
verification['final_documentation_hashes_and_links_checked'] = True
verification['unchanged_native_image_checks_repeated'] = False
(OUT / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
(OUT / 'progress.json').write_text(json.dumps(dict(status='verified', completed_pages=21, total_pages=21)) + '\n')
print(json.dumps(dict(status='passed', output_hashes_checked=len(complete['outputs']),
                      dependencies_checked=len(manifest['dependencies']), documentation_links_checked=True,
                      native_image_verification='Previously passed; unchanged output hashes verified'), indent=2))
