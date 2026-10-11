"""Freeze/check the seven user-approved exploratory image selections.

The lock is independent of build.py: rebuilding the catalog never updates it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK = HERE / "selection_lock.json"
FIELDS = ("id", "page", "original", "source_sha256", "roi_xywh",
          "detail_xywh", "role", "challenge", "judgment")


def digest(data: object) -> str:
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_selection() -> list[dict]:
    cases = json.loads((HERE / "catalog.json").read_text())["cases"]
    result = [{key: row[key] for key in FIELDS} for row in cases]
    assert len(result) == 7
    assert [row["id"] for row in result] == [f"{i:02d}" for i in range(1, 8)]
    for row in result:
        source = ROOT / row["original"]
        assert source.exists(), source
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source_sha256"], source
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Create lock once; refuses overwrite")
    parser.add_argument("--revise", metavar="REASON", help="Archive v1 and freeze the user-requested revision")
    args = parser.parse_args()
    cases = current_selection()
    assert not (args.write and args.revise)
    if args.write:
        assert not LOCK.exists(), f"Selection is already frozen: {LOCK}"
        record = {"schema_version": 1, "frozen_date": "2026-10-09",
                  "purpose": "User-approved exploratory development cases; not an independent test set",
                  "cases_sha256": digest(cases), "cases": cases}
        with LOCK.open("x") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"Created {LOCK}")
    elif args.revise:
        archive = HERE / "selection_lock_v1_20261009.json"
        assert LOCK.exists() and not archive.exists(), "Revision already archived or no current lock"
        previous = json.loads(LOCK.read_text())
        assert previous["cases_sha256"] == digest(previous["cases"]), "Old lock checksum changed"
        assert previous["cases"] != cases, "No selection change to revise"
        record = {"schema_version": 1, "revision": 2, "frozen_date": "2026-10-09",
                  "revision_reason": args.revise,
                  "previous_cases_sha256": previous["cases_sha256"],
                  "purpose": "User-approved exploratory development cases; not an independent test set",
                  "cases_sha256": digest(cases), "cases": cases}
        archive.write_text(json.dumps(previous, ensure_ascii=False, indent=2)+"\n")
        LOCK.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
        print(f"Revised {LOCK}; archived {archive}")
    else:
        locked = json.loads(LOCK.read_text())
        assert locked["cases_sha256"] == digest(locked["cases"]), "Lock checksum changed"
        assert locked["cases"] == cases, "Catalog selection differs from lock"
        print(f"Selection matches lock: {locked['cases_sha256']}")


if __name__ == "__main__":
    main()
