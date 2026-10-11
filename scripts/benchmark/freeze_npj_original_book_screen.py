"""Record original-only book contact sheets; no restoration claims."""
from pathlib import Path
import shutil
from build_npj_v6_comparison import ROOT,sha,dump

OUT=ROOT/'tmp_work/npj_next_book_original_screen_20261004'


def main():
    assert not (OUT/'complete.json').exists(), 'Already frozen'
    pages=sorted((ROOT/'data/train/200021925').glob('*.jpg'))
    assert len(pages)==45
    figures=sorted(OUT.glob('*.png'));assert len(figures)==6
    dump(OUT/'manifest.json',dict(book='200021925',original_only=True,
        original_files={str(p.relative_to(ROOT)):sha(p) for p in pages},
        selected_writing_pages=['200021925_00004_1','200021925_00007_1'],
        illustration_cover_decorated_title_excluded=True,
        not_model_training_independence=True,no_npj_output_or_repair_here=True,
        source_recorded_at_freeze=True,comparison_selected_before_inference=True,
        selection_record='tmp_work/npj_dirt_new_book_20261004/selection.json'))
    text=['# 原画像だけで本文ページを選ぶ', '',
          '2026-10-04。200021925の45原JPEGを一覧にした。本文・挿絵・表紙・装飾扉が混在する冊。', '',
          '00004_1と00007_1を本文のみのページとして比較へ選び、挿絵・表紙・装飾扉は使わない。'
          'この一覧は原画像だけで、npjや修復結果を含まない。手法間の比較PNGや除去成功例として数えない。'
          '「実験登録にない冊」は一般の原画像一覧やモデル学習から独立という意味ではない。', '',
          '[選定後の固定条件比較](../npj_dirt_new_book_20261004/report.md)／'
          '[全変更成分の回帰比較・報告の数え方訂正](../npj_contact_component_veto_20261004/report.md)', '']
    for p in figures:text.append('- [%s](%s)'%(p.name,p.name))
    (OUT/'report.md').write_text('\n'.join(text)+'\n')
    shutil.copyfile(__file__,OUT/'freeze_source_snapshot.py')
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},
        original_only_png=6,book_originals=45,no_model_inference=True))


if __name__=='__main__':main()
