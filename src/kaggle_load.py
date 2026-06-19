"""Bulk-load FM20/21/22/23 player grades from the Kaggle 'football-manager complete' dataset
(furkanuluta), replacing the slow fminside scrape for the THREE OLDER editions across ALL leagues.

Stored under the SAME fm_version identity as the fminside ones (FM21=21.0.0 etc.) with
source='kaggle', so build_dataset uses them interchangeably. Only players already in our DB
(i.e. who appear in a lineup) are loaded, by normalized-name join.

Robust parsing: these FM CSV exports have UNQUOTED commas in money fields (e.g. "£1,234,567"),
which shifts columns. We anchor the NAME from the front (before money cols) and the 47 ATTRIBUTE
values from the BACK (they're the trailing block of clean integers) — immune to mid-row comma drift.

Usage: python D:/Programming/claude/FM/src/kaggle_load.py [--reset]
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db

DATASET = "furkanuluta/football-manager-22-complete-player-dataset"

# full FM short-code (normalized) -> canonical attribute name + category.
# NB: 'nat' = Natural Fitness here; the nationality 'Nat' column is at the FRONT and excluded by
# right-anchoring, so it never reaches this map.
FM_CODE = {
    # technical
    "cro": ("crossing", "technical"), "dri": ("dribbling", "technical"),
    "fin": ("finishing", "technical"), "fir": ("first-touch", "technical"),
    "hea": ("heading", "technical"), "lon": ("long-shots", "technical"),
    "mar": ("marking", "technical"), "pas": ("passing", "technical"),
    "tck": ("tackling", "technical"), "tec": ("technique", "technical"),
    # set pieces
    "cor": ("corners", "set_pieces"), "fre": ("free-kick-taking", "set_pieces"),
    "lth": ("long-throws", "set_pieces"), "pen": ("penalty-taking", "set_pieces"),
    # mental
    "agg": ("aggression", "mental"), "ant": ("anticipation", "mental"),
    "bra": ("bravery", "mental"), "cmp": ("composure", "mental"),
    "cnt": ("concentration", "mental"), "dec": ("decisions", "mental"),
    "det": ("determination", "mental"), "fla": ("flair", "mental"),
    "ldr": ("leadership", "mental"), "otb": ("off-the-ball", "mental"),
    "pos": ("positioning", "mental"), "tea": ("teamwork", "mental"),
    "vis": ("vision", "mental"), "wor": ("work-rate", "mental"),
    # physical
    "acc": ("acceleration", "physical"), "agi": ("agility", "physical"),
    "bal": ("balance", "physical"), "jum": ("jumping-reach", "physical"),
    "nat": ("natural-fitness", "physical"), "nat1": ("natural-fitness", "physical"),
    "pac": ("pace", "physical"),
    "sta": ("stamina", "physical"), "str": ("strength", "physical"),
    # goalkeeping
    "aer": ("aerial-reach", "goalkeeping"), "cmd": ("command-of-area", "goalkeeping"),
    "com": ("communication", "goalkeeping"), "ecc": ("eccentricity", "goalkeeping"),
    "han": ("handling", "goalkeeping"), "kic": ("kicking", "goalkeeping"),
    "1v1": ("one-on-ones", "goalkeeping"), "pun": ("punching-tendency", "goalkeeping"),
    "ref": ("reflexes", "goalkeeping"), "tro": ("rushing-out-tendency", "goalkeeping"),
    "thr": ("throwing", "goalkeeping"),
}

EDITIONS = {
    "fm21data": ("FM21", "21.0.0", "2020-11-01"),
    "fm22data": ("FM22", "22.1.0", "2021-11-01"),
    "fm2023": ("FM23", "23.4.0", "2023-02-01"),
    "fm20data": ("FM20", "20.0.0", "2019-11-01"),
}


def norm_code(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def parse_dob(raw):
    """kaggle DoB is 'D/M/YYYY (NN years old)' -> ISO 'YYYY-MM-DD'."""
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", str(raw))
    if not m:
        return None
    d, mo, y = m.groups()
    return f"{y}-{int(mo):02d}-{int(d):02d}"


def analyze_header(header):
    """Return (name_idx, K, attr_names, club_idx, dob_idx). attr_names are the canonical names of the
    trailing K attribute columns. club_idx/dob_idx are FRONT columns (before the comma-drifting money
    fields) so they're safe to read by absolute index — these carry the club discriminator + DOB that
    let us disambiguate common names (Gabriel/Lucas/Silva) in kaggle-only leagues (Brazil/Portugal/...)."""
    cols = [str(c).strip() for c in header]
    name_idx = next((i for i, c in enumerate(cols) if c.lower() == "name"), 1)
    club_idx = next((i for i, c in enumerate(cols) if c.lower() == "club"), None)
    dob_idx = next((i for i, c in enumerate(cols) if c.lower() in ("dob", "date of birth")), None)
    attr_names = []
    for c in reversed(cols):
        code = norm_code(c)
        if code in FM_CODE:
            attr_names.append(FM_CODE[code][0])
        else:
            break
    attr_names.reverse()
    return name_idx, len(attr_names), attr_names, club_idx, dob_idx


def download():
    import kagglehub
    return Path(kagglehub.dataset_download(DATASET))


def load_edition(con, csv_path, game, db_version, date, known):
    src = db.source_id(con, "kaggle", "kaggle:" + DATASET)
    fmv = db.fm_version_id(con, game, db_version, date)
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        name_idx, K, attr_names, club_idx, dob_idx = analyze_header(header)
        if K < 30:
            print(f"  {csv_path.name}: only {K} trailing attr cols — skip"); return 0, 0
        cat_of = {v[0]: v[1] for v in FM_CODE.values()}  # canonical attr name -> category
        saved = skipped = seen = 0
        for fields in reader:
            seen += 1
            if len(fields) < K + name_idx + 1:
                continue
            name = fields[name_idx].strip()
            if not name:
                continue
            nn = db.norm(name)
            if nn not in known:
                continue
            attr_vals = fields[-K:]
            attrs = {}
            for an, raw in zip(attr_names, attr_vals):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    attrs[an] = (cat_of[an], float(raw))
                except ValueError:
                    continue
            if len(attrs) < 20:
                continue
            club = fields[club_idx].strip() if club_idx is not None and len(fields) > club_idx else ""
            dob = parse_dob(fields[dob_idx]) if dob_idx is not None and len(fields) > dob_idx else None
            cid = db.club_id(con, club) if club else None
            pid = db.player_id(con, name, dob=dob, src=src, src_player_id=fields[0].strip() or None, grade_uid=True)
            sid = db.save_snapshot(con, pid=pid, src=src, fmv=fmv, cid=cid,
                                   snapshot_date=date, attrs=attrs, meta={})
            if sid:
                saved += 1
            else:
                skipped += 1
            if (saved + skipped) % 2000 == 0:
                con.commit()
        con.commit()
        return saved, skipped


def main():
    con = db.connect()
    con.execute("PRAGMA synchronous=NORMAL")   # WAL + NORMAL: cheap writes (no per-row fsync). Single writer.
    if "--reset" in sys.argv:
        s = con.execute("SELECT source_id FROM source WHERE name='kaggle'").fetchone()
        if s:
            con.execute("DELETE FROM player_attribute WHERE snapshot_id IN "
                        "(SELECT snapshot_id FROM player_snapshot WHERE source_id=?)", (s[0],))
            con.execute("DELETE FROM player_snapshot WHERE source_id=?", (s[0],))
            con.commit()
            print("reset: cleared existing kaggle snapshots")
    known = {r[0] for r in con.execute(
        "SELECT DISTINCT p.norm_name FROM player p JOIN match_player mp USING(player_id)")}
    print(f"known lineup players (join target): {len(known):,}")
    root = download()
    csvs = {p.stem.lower(): p for p in root.rglob("*.csv")}
    total = 0
    for stem, (game, dbv, date) in EDITIONS.items():
        p = csvs.get(stem)
        if not p:
            print(f"  {stem}: file not found"); continue
        print(f"== {game} <- {p.name} ==", flush=True)
        saved, skipped = load_edition(con, p, game, dbv, date, known)
        db.log(con, "kaggle", game, "ok", f"saved={saved} skipped={skipped}")
        print(f"  {game}: saved={saved:,} skipped={skipped:,}", flush=True)
        total += saved
    print(f"TOTAL kaggle snapshots saved: {total:,}")
    con.close()


if __name__ == "__main__":
    main()
