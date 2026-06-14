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
    # English lower tiers (FM scouts the English pyramid deeply); football-data odds available
    dict(rank=40, name="England League One", country="England", tier=3,
         fm_league="League One", fm_nat="England", fd="E2", espn="eng.3", understat=None, verify=True),
    dict(rank=41, name="England League Two", country="England", tier=4,
         fm_league="League Two", fm_nat="England", fd="E3", espn="eng.4", understat=None, verify=True),
    dict(rank=42, name="England National League", country="England", tier=5,
         fm_league="National League", fm_nat="England", fd="EC", espn="eng.5", understat=None, verify=True),
]

# UEFA club cups — cross-league competitions (clubs from many leagues). ESPN-primary; players
# join to FM grades via their domestic clubs (no separate fminside scrape needed for most).
UEFA_CUPS = [
    dict(rank=43, name="UEFA Champions League", country="Europe", tier=None, espn="uefa.champions", season_type="euro"),
    dict(rank=44, name="UEFA Europa League", country="Europe", tier=None, espn="uefa.europa", season_type="euro"),
    dict(rank=45, name="UEFA Conference League", country="Europe", tier=None, espn="uefa.europa.conf", season_type="euro"),
]

BY_NAME = {l["name"]: l for l in LEAGUES}


def enabled(include_verify=False):
    return [l for l in LEAGUES if include_verify or not l.get("verify")]


# --- ESPN-primary leagues (no football-data odds): results+lineups+context created from ESPN,
# FM grades from fminside later. fm_league strings are best-guess, verified at grade-scrape time.
# Continuous ~165-day windows tile the whole calendar so calendar-year leagues (Brazil/MLS/
# Scandinavia/Japan, which play through summer) aren't missed by the European Aug-Jun windows.
EXTRA_LEAGUES = [
    dict(rank=16, name="Brazil Serie A", country="Brazil", tier=1, espn="bra.1",
         fm_league="Brazilian National First Division", fm_nat="Brazil", season_type="calendar"),
    dict(rank=17, name="Argentina Liga Profesional", country="Argentina", tier=1, espn="arg.1",
         fm_league="Primera Division", fm_nat="Argentina", season_type="calendar"),
    dict(rank=18, name="USA MLS", country="USA", tier=1, espn="usa.1",
         fm_league="Major League Soccer", fm_nat="United States", season_type="calendar"),
    dict(rank=19, name="Saudi Pro League", country="Saudi Arabia", tier=1, espn="ksa.1",
         fm_league="Saudi Pro League", fm_nat="Saudi Arabia", season_type="euro"),
    dict(rank=20, name="Mexico Liga MX", country="Mexico", tier=1, espn="mex.1",
         fm_league="Liga MX", fm_nat="Mexico", season_type="split"),
    dict(rank=21, name="Russia Premier League", country="Russia", tier=1, espn="rus.1",
         fm_league="Russian Premier Division", fm_nat="Russia", season_type="euro"),
    dict(rank=22, name="Austria Bundesliga", country="Austria", tier=1, espn="aut.1",
         fm_league="Austrian Bundesliga", fm_nat="Austria", season_type="euro"),
    dict(rank=23, name="Switzerland Super League", country="Switzerland", tier=1, espn="sui.1",
         fm_league="Swiss Super League", fm_nat="Switzerland", season_type="euro"),
    dict(rank=24, name="Greece Super League", country="Greece", tier=1, espn="gre.1",
         fm_league="Super League", fm_nat="Greece", fd="G1", season_type="euro"),
    dict(rank=25, name="Denmark Superliga", country="Denmark", tier=1, espn="den.1",
         fm_league="Danish Superliga", fm_nat="Denmark", season_type="euro"),
    dict(rank=26, name="Norway Eliteserien", country="Norway", tier=1, espn="nor.1",
         fm_league="Eliteserien", fm_nat="Norway", season_type="calendar"),
    dict(rank=27, name="Sweden Allsvenskan", country="Sweden", tier=1, espn="swe.1",
         fm_league="Allsvenskan", fm_nat="Sweden", season_type="calendar"),
    dict(rank=28, name="Australia A-League", country="Australia", tier=1, espn="aus.1",
         fm_league="A-League", fm_nat="Australia", season_type="euro"),
    dict(rank=29, name="Colombia Primera A", country="Colombia", tier=1, espn="col.1",
         fm_league="Primera A", fm_nat="Colombia", season_type="split"),
    dict(rank=30, name="Israel Ligat haAl", country="Israel", tier=1, espn="isr.1",
         fm_league="Ligat ha'Al", fm_nat="Israel", season_type="euro", partial_lineups=True),
    dict(rank=31, name="Japan J1 League", country="Japan", tier=1, espn="jpn.1",
         fm_league="J1 League", fm_nat="Japan", season_type="calendar"),
    # second discovery batch (ESPN full lineups verified)
    dict(rank=32, name="Chile Primera Division", country="Chile", tier=1, espn="chi.1",
         fm_league="Primera Division", fm_nat="Chile", season_type="calendar"),
    dict(rank=33, name="China Super League", country="China", tier=1, espn="chn.1",
         fm_league="Chinese Super League", fm_nat="China", season_type="calendar"),
    dict(rank=34, name="Ecuador LigaPro", country="Ecuador", tier=1, espn="ecu.1",
         fm_league="Ecuadorian Serie A", fm_nat="Ecuador", season_type="calendar"),
    dict(rank=35, name="India Super League", country="India", tier=1, espn="ind.1",
         fm_league="Indian Super League", fm_nat="India", season_type="euro"),
    dict(rank=36, name="Paraguay Primera Division", country="Paraguay", tier=1, espn="par.1",
         fm_league="Paraguayan Primera Division", fm_nat="Paraguay", season_type="split"),
    dict(rank=37, name="Peru Liga 1", country="Peru", tier=1, espn="per.1",
         fm_league="Peruvian Primera Division", fm_nat="Peru", season_type="calendar"),
    dict(rank=38, name="South Africa Premiership", country="South Africa", tier=1, espn="rsa.1",
         fm_league="Premier Soccer League", fm_nat="South Africa", season_type="euro"),
]
EXTRA_BY_NAME = {l["name"]: l for l in EXTRA_LEAGUES + UEFA_CUPS}


def espn_windows():
    """Continuous ~165-day windows tiling Jan 2020 -> Jun 2026 (overlap ok; matches dedupe)."""
    wins = []
    for y in range(2020, 2027):
        wins.append(f"{y}0101-{y}0615")
        wins.append(f"{y}0601-{y}1215")
    return wins
