"""Extract FM-side identity (FM UID -> name, DOB, nationality) from the Kaggle FM CSVs into
source_identity (under a synthetic 'fm-uid' source). The FM UID is the shared key across
fminside/futek/kaggle, so resolving ESPN->FM-UID links to every FM grade source at once.
UID/Name/DoB/Nat are FRONT columns (before the money-comma corruption) so front-anchored
parsing is safe. Usage: python D:/Programming/claude/FM/src/extract_identity_kaggle.py
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db

DATASET = "furkanuluta/football-manager-22-complete-player-dataset"


def parse_dob(s):
    # "9/5/1992 (28 years old)" -> 1992-05-09 ; FM CSV is d/m/Y
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", s or "")
    if not m:
        return None
    d, mo, y = m.groups()
    return f"{y}-{int(mo):02d}-{int(d):02d}"


def main():
    import kagglehub
    root = Path(kagglehub.dataset_download(DATASET))
    con = db.connect()
    src = db.source_id(con, "fm-uid", "FM in-game UID identity")
    total = 0
    for p in sorted(root.rglob("*.csv")):
        with open(p, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            header = [str(c).strip() for c in next(reader)]
            idx = {h.lower(): i for i, h in enumerate(header)}
            iu = idx.get("uid", 0)
            ina = next((idx[k] for k in ("name", "player name") if k in idx), 1)
            idob = idx.get("dob")
            inat = idx.get("nat")
            if idob is None:
                print(f"  {p.name}: no DoB col — skip"); continue
            n = 0
            for fields in reader:
                if len(fields) <= max(iu, ina, idob, inat or 0):
                    continue
                uid = fields[iu].strip()
                if not uid or not uid.isdigit():
                    continue
                con.execute(
                    """INSERT OR IGNORE INTO source_identity
                       (source_id, source_player_id, name, dob, nationality)
                       VALUES (?,?,?,?,?)""",
                    (src, uid, fields[ina].strip(), parse_dob(fields[idob]),
                     fields[inat].strip() if inat is not None else None))
                n += 1
                if n % 5000 == 0:
                    con.commit()
            con.commit()
            print(f"  {p.name}: {n:,} rows")
            total += n
    got = con.execute("SELECT COUNT(*) FROM source_identity WHERE source_id=? AND dob IS NOT NULL",
                      (src,)).fetchone()[0]
    print(f"DONE. FM-UID identities with DOB: {got:,}")
    con.close()


if __name__ == "__main__":
    main()
