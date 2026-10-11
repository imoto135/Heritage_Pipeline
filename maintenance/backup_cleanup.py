"""Prepare and apply the explicitly authorized 2026-10-11 backup cleanup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / 'research_records'
PLAN = ROOT / 'maintenance/cleanup_plan_20261011.json'
RAW = ['data/train', 'data/val', 'data/test', 'data/split_dataset_jogai',
       'data/makura_kiridashi']
PROCESSED = ['data/split_dataset_csc', 'data/split_dataset_csc_mask',
             'data/split_dataset_csc_mask_v2', 'data/makura_csc_mask',
             'data/masks_compare']
WANDB = ['modules/ddrm_npj/wandb', 'modules/ddrm_npj_v2/wandb',
         'modules/stain_removal/wandb']
TEXT = {'.md', '.txt', '.csv', '.tsv', '.json', '.jsonl', '.yaml', '.yml',
        '.py', '.sh', '.cpp', '.h', '.hpp', '.c', '.m', '.java', '.tex',
        '.bib', '.html', '.css', '.js', '.patch', '.cfg', '.ini', '.toml',
        '.xml', '.pdf', '.arpa'}
MODELS = {'.pth', '.pt', '.ckpt', '.pkl', '.onnx', '.engine'}
GENERATED = {'.png', '.jpg', '.jpeg', '.npy', '.npz', '.bin', '.zip',
             '.log', '.out', '.wandb', '.gz'}
KEEP_EPOCHS = {
    'modules/stain_removal/model/model_page': {100, 400, 500},
    'modules/stain_removal/experiments/model_char/model_char': {100, 350, 450, 500},
    'modules/stain_removal/experiments/model_char_csc/model_char_csc': {450, 500},
    'modules/stain_removal/experiments/model_page_csc/model_page_csc': {450, 500},
}
SKIP = {'.git', '__pycache__', 'node_modules', '.venv'}


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args]).decode()


def files(parent):
    for d, dirs, names in os.walk(ROOT / parent, followlinks=False):
        dirs[:] = [x for x in dirs if x not in SKIP]
        for name in names:
            yield Path(d) / name


def entry(p):
    s = p.lstat()
    return dict(path=str(p.relative_to(ROOT)), bytes=s.st_size,
                allocated_bytes=s.st_blocks * 512, mtime_ns=s.st_mtime_ns,
                inode=s.st_ino, symlink=p.is_symlink())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def linked_assets():
    found = set()
    for p in files('tmp_work'):
        if p.suffix != '.md' or p.is_symlink():
            continue
        for value in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', p.read_text(errors='replace')):
            value = value.split('#')[0].strip('<>')
            if not value or '://' in value:
                continue
            q = (p.parent / value).resolve()
            if q.is_relative_to(ROOT) and q.is_file() and q.suffix.lower() in {'.png', '.jpg', '.jpeg', '.pdf'}:
                found.add(q)
    return found


def protect_tmp(p, linked):
    if p.resolve() in linked or p.suffix.lower() in TEXT | MODELS:
        return True
    # Keep source photographs, annotations, external originals, and frozen inputs.
    if p.suffix.lower() in {'.jpg', '.jpeg'} or p.name == 'graph_output.bin':
        return True
    return any(re.search(r'original|annotat|label|ground.?truth|manual|external|source|input|frozen', part, re.I)
               for part in p.relative_to(ROOT).parts[2:-1])


def prepare(preserve=True):
    tracked = set(git('ls-files', '-z').split('\0'))
    linked = linked_assets()
    selected = set(linked)
    for parent in ['tmp_work', 'data', 'results', *WANDB]:
        for p in files(parent):
            if not p.is_symlink() and p.suffix.lower() in TEXT and str(p.relative_to(ROOT)) not in tracked:
                selected.add(p)
    selected.update(ROOT.glob('*.pdf'))
    manifest = []
    for p in sorted(selected) if preserve else []:
        dest = RECORDS / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        source_hash = sha(p)
        if not dest.exists() or sha(dest) != source_hash:
            shutil.copy2(p, dest)
        if sha(dest) != source_hash:
            raise RuntimeError('Preservation failed: ' + str(p))
        manifest.append(dict(source=str(p.relative_to(ROOT)),
                             preserved_path=str(dest.relative_to(ROOT)),
                             bytes=p.stat().st_size, sha256=source_hash))
    if preserve:
        (RECORDS / 'preservation_manifest.json').write_text(json.dumps(
            dict(date='2026-10-11', files=manifest), ensure_ascii=False, indent=2) + '\n')
    else:
        manifest = json.loads((RECORDS / 'preservation_manifest.json').read_text())['files']

    deletions = []
    protected = {str(p.relative_to(ROOT)): entry(p) for parent in RAW for p in files(parent)}
    for parent in PROCESSED + WANDB + ['results', 'tmp_work']:
        for p in files(parent):
            rel = str(p.relative_to(ROOT))
            if rel in tracked:
                continue
            if p.is_symlink():
                # Remove only aliases whose derived target will be removed later.
                continue
            if parent == 'tmp_work':
                keep = protect_tmp(p, linked) or p.suffix.lower() not in GENERATED
            else:
                keep = p.resolve() in linked or p.suffix.lower() in TEXT | MODELS
            if keep:
                protected[rel] = entry(p)
            else:
                row = entry(p)
                row['group'] = parent
                deletions.append(row)
    for parent, epochs in KEEP_EPOCHS.items():
        for p in files(parent):
            match = re.fullmatch(r'ckpt_epoch(\d+)\.pth', p.name)
            if match and int(match.group(1)) not in epochs and str(p.relative_to(ROOT)) not in tracked:
                row = entry(p)
                row['group'] = 'unused_training_checkpoints'
                deletions.append(row)
    deleted_paths = {x['path'] for x in deletions}
    for parent in ['modules/stain_removal/model', 'modules/stain_removal/experiments', 'tmp_work']:
        for p in files(parent):
            rel = str(p.relative_to(ROOT))
            if p.suffix.lower() in MODELS and rel not in deleted_paths and not p.is_symlink():
                protected[rel] = entry(p)
    for parent in PROCESSED + WANDB + ['results', 'tmp_work']:
        for p in files(parent):
            if p.is_symlink() and str(p.relative_to(ROOT)) not in tracked:
                resolved = p.resolve()
                if resolved.is_relative_to(ROOT) and str(resolved.relative_to(ROOT)) in deleted_paths:
                    row = entry(p)
                    row['group'] = parent
                    deletions.append(row)
    assert not deleted_paths.intersection(protected)
    totals = defaultdict(lambda: [0, 0])
    for row in deletions:
        totals[row['group']][0] += 1
        totals[row['group']][1] += row['allocated_bytes']
    plan = dict(date='2026-10-11', authorization='Keep research restart assets; delete others and push.',
                deletions=deletions, protected=list(protected.values()), group_totals=dict(totals),
                archived_files=len(manifest), archived_bytes=sum(x['bytes'] for x in manifest))
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: dict(files=n, GiB=round(b/1024**3, 3)) for k, (n, b) in totals.items()}, ensure_ascii=False), flush=True)
    print('Archived files:', len(manifest), 'MiB:', round(plan['archived_bytes']/1024**2, 2), flush=True)


def check(row):
    p = ROOT / row['path']
    now = entry(p)
    for key in ['bytes', 'mtime_ns', 'inode', 'symlink']:
        if now[key] != row[key]:
            raise RuntimeError('File changed since preparation: ' + row['path'])


def apply():
    head = git('rev-parse', 'HEAD').strip()
    remote = git('ls-remote', 'origin', 'refs/heads/main').split()[0]
    if head != remote:
        raise RuntimeError('Refusing deletion before the preservation commit is pushed.')
    if git('status', '--porcelain', '--untracked-files=no').strip():
        raise RuntimeError('Tracked changes must be committed before cleanup.')
    plan = json.loads(PLAN.read_text())
    tracked = set(git('ls-files', '-z').split('\0'))
    for row in plan['protected'] + plan['deletions']:
        check(row)
    for row in plan['deletions']:
        rel = row['path']
        if rel in tracked or any(rel.startswith(x + '/') for x in RAW) or '..' in Path(rel).parts:
            raise RuntimeError('Protected deletion path: ' + rel)
    for n, row in enumerate(plan['deletions'], 1):
        check(row)
        (ROOT / row['path']).unlink()
        if n % 10000 == 0:
            print('Deleted', n, '/', len(plan['deletions']), flush=True)
    for parent in PROCESSED + WANDB + ['results', 'tmp_work']:
        for d, dirs, names in os.walk(ROOT / parent, topdown=False, followlinks=False):
            p = Path(d)
            if p != ROOT / parent and not p.is_symlink():
                try:
                    p.rmdir()
                except OSError:
                    pass
    for row in plan['protected']:
        check(row)
    remaining = [x['path'] for x in plan['deletions'] if os.path.lexists(ROOT / x['path'])]
    if remaining:
        raise RuntimeError('Planned files remain: ' + str(remaining[:5]))
    result = dict(date='2026-10-11', deleted_files=len(plan['deletions']),
                  deleted_allocated_GiB=round(sum(x['allocated_bytes'] for x in plan['deletions'])/1024**3, 3),
                  protected_files_verified=len(plan['protected']), preservation_commit=head,
                  git_tracked_files_deleted=0)
    (ROOT / 'maintenance/cleanup_result_20261011.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'plan', 'apply'])
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'apply':
        apply()
    else:
        prepare(preserve=args.action == 'prepare')
