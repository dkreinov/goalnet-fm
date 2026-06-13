"""League registry + season<->FM-database mapping for the multi-league pipeline.

Each league row carries the codes every source uses. fminside league strings and the
nationality collision-filter were verified live against the club filter (db5/FM24).

Add a league: append a LEAGUES entry; verify its fminside string returns ~18-20 clubs
(see scripts/probe via update_filter.php) before trusting enumeration.
"""

# season label -> (fminside db id, FM game, FM db_version, snapshot date used for that season)
SEASON_DB = {
    "2020-21": (1, "FM21", "21.0.0", "2020-11-01"),
    "2021-22": (2, "FM22", "22.1.0", "2021-11-01"),
    "2022-23": (3, "FM23", "23.4.0", "2023-02-01"),
    "2023-24": (5, "FM24", "24.3.0", "2024-02-26"),
    "2024-25": (6, "FMU25", "24.4.0-community", "2024-10-01"),
    "2025-26": (7, "FM26", "26.2.0", "2026-03-01"),
}
SEASONS = list(SEASON_DB)

# football-data.co.uk season file code per season label
FD_SEASON = {"2020-21": "2021", "2021-22": "2122", "2022-23": "2223",
             "2023-24": "2324", "2024-25": "2425", "2025-26": "2526"}
# ESPN scoreboard date window per season. Must span < ~330 days (ESPN 400s on wider ranges).
ESPN_WINDOW = {"2020-21": "20200801-20210615", "2021-22": "20210801-20220615",
               "2022-23": "20220801-20230615", "2023-24": "20230801-20240615",
               "2024-25": "20240801-20250615", "2025-26": "20250801-20260615"}
# understat uses the start year
UNDERSTAT_YEAR = {"2020-21": "2020", "2021-22": "2021", "2022-23": "2022",
                  "2023-24": "2023", "2024-25": "2024", "2025-26": "2025"}

# rank = global priority (1 highest). fm_league/fm_nat verified live unless noted.
# fd = football-data.co.uk code (results+odds) or None; espn = ESPN league code;
# understat = understat league name or None.
LEAGUES = [
    dict(rank=1,  name="England Premier League", country="England", tier=1,
         fm_league="Premier League", fm_nat="England", fd="E0", espn="eng.1", understat="EPL"),
    dict(rank=2,  name="Spain LaLiga", country="Spain", tier=1,
         fm_league="LaLiga", fm_nat="Spain", fd="SP1", espn="esp.1", understat="La_liga"),
    dict(rank=3,  name="Italy Serie A", country="Italy", tier=1,
         fm_league="Serie A", fm_nat="Italy", fd="I1", espn="ita.1", understat="Serie_A"),
    dict(rank=4,  name="Germany Bundesliga", country="Germany", tier=1,
         fm_league="Bundesliga", fm_nat="Germany", fd="D1", espn="ger.1", understat="Bundesliga"),
    dict(rank=5,  name="France Ligue 1", country="France", tier=1,
         fm_league="Ligue 1", fm_nat="France", fd="F1", espn="fra.1", understat="Ligue_1"),
    dict(rank=6,  name="England Championship", country="England", tier=2,
         fm_league="Championship", fm_nat="England", fd="E1", espn="eng.2", understat=None),
    dict(rank=7,  name="Netherlands Eredivisie", country="Netherlands", tier=1,
         fm_league="Eredivisie", fm_nat="Netherlands", fd="N1", espn="ned.1", understat=None),
    dict(rank=8,  name="Portugal Primeira Liga", country="Portugal", tier=1,
         fm_league="Primeira Liga", fm_nat="Portugal", fd="P1", espn="por.1", understat=None),
    # --- next tier: fminside string still needs live verification before enabling ---
    dict(rank=9,  name="Belgium Pro League", country="Belgium", tier=1,
         fm_league="Pro League", fm_nat="Belgium", fd="B1", espn="bel.1", understat=None, verify=True),
    dict(rank=10, name="Turkey Super Lig", country="Turkey", tier=1,
         fm_league="Super Lig", fm_nat="Turkey", fd="T1", espn="tur.1", understat=None, verify=True),
    dict(rank=11, name="Scotland Premiership", country="Scotland", tier=1,
         fm_league="Premiership", fm_nat="Scotland", fd="SC0", espn="sco.1", understat=None, verify=True),
    dict(rank=12, name="Germany 2. Bundesliga", country="Germany", tier=2,
         fm_league="2. Bundesliga", fm_nat="Germany", fd="D2", espn="ger.2", understat=None, verify=True),
    dict(rank=13, name="Italy Serie B", country="Italy", tier=2,
         fm_league="Serie B", fm_nat="Italy", fd="I2", espn="ita.2", understat=None, verify=True),
    dict(rank=14, name="Spain LaLiga 2", country="Spain", tier=2,
         fm_league="LaLiga 2", fm_nat="Spain", fd="SP2", espn="esp.2", understat=None, verify=True),
    dict(rank=15, name="France Ligue 2", country="France", tier=2,
         fm_league="Ligue 2", fm_nat="France", fd="F2", espn="fra.2", understat=None, verify=True),
]

BY_NAME = {l["name"]: l for l in LEAGUES}


def enabled(include_verify=False):
    return [l for l in LEAGUES if include_verify or not l.get("verify")]
