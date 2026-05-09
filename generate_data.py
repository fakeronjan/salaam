"""
generate_data.py — turns SALAAM ratings + standings + cached games into the
JSON the SALAAM web frontend reads. Run after salaam.py. Outputs to docs/data/.

CFB-specific tweaks vs DILLON:
  - Per-(team, season) conference (handles realignment + rebrands)
  - P5 + Big East + Other bucketing for the toggle
  - Phase 1: CFP-era championship attribution (2014+) only
  - Postseason week labels derived from game notes
"""

import pandas as pd
import json
import os
import re
from pathlib import Path
from bisect import bisect_right
from datetime import datetime, timezone

DATA_DIR = Path(__file__).parent / 'data'
OUT_DIR  = Path(__file__).parent / 'docs' / 'data'

CFP_FIRST_SEASON = 2014  # Phase 1 championship attribution starts here

# BCS era (1998-2013). Hardcoded because CFBD only tags "BCS championship" in
# game notes for 2006+; 1998-2005 used a rotating BCS bowl whose notes just say
# "Fiesta Bowl" / "Sugar Bowl" / etc. without mentioning BCS championship.
BCS_CHAMPIONS = {
    1998: {'champion': 'Tennessee',     'runner_up': 'Florida State', 'final_score': '23-16'},
    1999: {'champion': 'Florida State', 'runner_up': 'Virginia Tech', 'final_score': '46-29'},
    2000: {'champion': 'Oklahoma',      'runner_up': 'Florida State', 'final_score': '13-2'},
    2001: {'champion': 'Miami',         'runner_up': 'Nebraska',      'final_score': '37-14'},
    2002: {'champion': 'Ohio State',    'runner_up': 'Miami',         'final_score': '31-24'},   # 2OT
    2003: {'champion': 'LSU',           'runner_up': 'Oklahoma',      'final_score': '21-14'},   # AP gave to USC — handled in Track B (poll era)
    2004: {'champion': 'USC',           'runner_up': 'Oklahoma',      'final_score': '55-19'},   # vacated by NCAA but still BCS-recorded champion
    2005: {'champion': 'Texas',         'runner_up': 'USC',           'final_score': '41-38'},
    2006: {'champion': 'Florida',       'runner_up': 'Ohio State',    'final_score': '41-14'},
    2007: {'champion': 'LSU',           'runner_up': 'Ohio State',    'final_score': '38-24'},
    2008: {'champion': 'Florida',       'runner_up': 'Oklahoma',      'final_score': '24-14'},
    2009: {'champion': 'Alabama',       'runner_up': 'Texas',         'final_score': '37-21'},
    2010: {'champion': 'Auburn',        'runner_up': 'Oregon',        'final_score': '22-19'},
    2011: {'champion': 'Alabama',       'runner_up': 'LSU',           'final_score': '21-0'},
    2012: {'champion': 'Alabama',       'runner_up': 'Notre Dame',    'final_score': '42-14'},
    2013: {'champion': 'Florida State', 'runner_up': 'Auburn',        'final_score': '34-31'},
}

# AP poll override: years where AP awarded the title to a different team than
# the BCS title game winner. For 2003, AP went USC; BCS/Coaches went LSU.
AP_OVERRIDES = {
    2003: 'USC',
}

# Poll era (1982-1997). AP and Coaches sometimes agreed, sometimes split.
# Coaches Poll = UPI through 1990, then USA Today/CNN from 1991 onward.
# Splits in this window: 1990 (CU/GT), 1991 (Miami/UW), 1997 (Michigan/Nebraska).
POLL_CHAMPIONS = {
    1982: {'AP': 'Penn State',     'Coaches': 'Penn State'},
    1983: {'AP': 'Miami',          'Coaches': 'Miami'},
    1984: {'AP': 'BYU',            'Coaches': 'BYU'},
    1985: {'AP': 'Oklahoma',       'Coaches': 'Oklahoma'},
    1986: {'AP': 'Penn State',     'Coaches': 'Penn State'},
    1987: {'AP': 'Miami',          'Coaches': 'Miami'},
    1988: {'AP': 'Notre Dame',     'Coaches': 'Notre Dame'},
    1989: {'AP': 'Miami',          'Coaches': 'Miami'},
    1990: {'AP': 'Colorado',       'Coaches': 'Georgia Tech'},   # split
    1991: {'AP': 'Miami',          'Coaches': 'Washington'},     # split
    1992: {'AP': 'Alabama',        'Coaches': 'Alabama'},
    1993: {'AP': 'Florida State',  'Coaches': 'Florida State'},
    1994: {'AP': 'Nebraska',       'Coaches': 'Nebraska'},
    1995: {'AP': 'Nebraska',       'Coaches': 'Nebraska'},
    1996: {'AP': 'Florida',        'Coaches': 'Florida'},
    1997: {'AP': 'Michigan',       'Coaches': 'Nebraska'},       # split
}

# Conference bucketing for the site toggle. 'Big 8' is the Big 12 lineage,
# 'Pac-10' is the Pac-12 lineage, 'Big East' surfaces only while it existed
# as a football conference (1991-2012).
CONF_LINEAGE = {
    'Big 8':   'Big 12',
    'Pac-10':  'Pac-12',
}
P5_OR_PRE_CFP = {'ACC', 'Big Ten', 'Big 12', 'Pac-12', 'SEC', 'Big East'}
BIG_EAST_LAST_SEASON = 2012


def conf_bucket(conf, season):
    if not conf:
        return 'Other'
    bucket = CONF_LINEAGE.get(conf, conf)
    if bucket == 'Big East' and int(season) > BIG_EAST_LAST_SEASON:
        return 'Other'
    if bucket in P5_OR_PRE_CFP:
        return bucket
    return 'Other'


# ── Setup output dirs ──────────────────────────────────────────────────────────
os.makedirs(OUT_DIR / 'teams',   exist_ok=True)
os.makedirs(OUT_DIR / 'seasons', exist_ok=True)


# ── Load ratings ──────────────────────────────────────────────────────────────
print('Reading ratings...')
df = pd.read_csv('salaam_ratings_with_standings.csv')

# ── Load games (for date lookups + championship attribution) ──────────────────
print('Reading games...')
games = pd.read_csv('all_NCAA_games.csv')
games['date'] = pd.to_datetime(games['date'], errors='coerce')

_sw_to_date = (
    games.dropna(subset=['date'])
    .groupby('season_week')['date']
    .max()
    .dt.date
    .to_dict()
)


def sw_to_date_str(sw):
    d = _sw_to_date.get(float(sw))
    return str(d) if d else ''


df['date'] = df['season_week'].apply(sw_to_date_str)


# ── Per-(team, season) conference from cached teams_*.json ─────────────────────
print('Loading per-season team conferences...')
team_conf_raw = {}    # (team, season) → raw CFBD conference string
for f in sorted((DATA_DIR / 'teams').glob('teams_*.json')):
    year = int(f.stem.split('_')[1])
    for t in json.loads(f.read_text()):
        team_conf_raw[(t['school'], year)] = t.get('conference') or 'Other'


def conf_raw(team, season):
    return team_conf_raw.get((team, int(season)), 'Other')


def conf(team, season):
    return conf_bucket(conf_raw(team, season), season)


# ── CFP + Conference championship attribution ────────────────────────────────
print('Building championship attribution...')
cfp_outcomes = {}        # season → {champion, runner_up, final_score, appearances}
conf_champs = {}         # (team, season) → conference name (e.g., 'SEC')

# Tier-based postseason week labels (matches salaam.py tier numbering).
# Week 100 covers all conference championship games (regardless of whether
# CFBD classified them as postseason or regular). Bowls + CFP First Round
# collapse into 101; later CFP rounds get their own slot.
POSTSEASON_LABELS = {
    100: 'Conference Championships',
    101: 'Bowls / CFP 1st Round',
    102: 'CFP Quarterfinals',
    103: 'CFP Semifinals',
    104: 'National Championship',
}

# Conferences we attribute champions for — both modern P5 buckets and their
# historical predecessors (Big 8 → Big 12, Pac-10 → Pac-12), plus Big East and
# the dissolved Southwest Conference.
HISTORICAL_P5 = {'ACC', 'Big Ten', 'Big 12', 'Big 8', 'Pac-10', 'Pac-12',
                 'SEC', 'Big East', 'Southwest'}


def _conf_record_champions(year_games, members):
    """Compute team(s) with the best regular-season conference record. Returns
    list (multiple = ties / co-champions). Min 4 conf games to count."""
    record = {team: {'w': 0, 'l': 0, 't': 0} for team in members}
    for g in year_games:
        if not g.get('conferenceGame') or g.get('seasonType') != 'regular' or not g.get('completed'):
            continue
        h, a = g['homeTeam'], g['awayTeam']
        if h not in members or a not in members:
            continue
        hp, ap = g.get('homePoints'), g.get('awayPoints')
        if hp is None or ap is None:
            continue
        if hp > ap:   record[h]['w'] += 1; record[a]['l'] += 1
        elif hp < ap: record[a]['w'] += 1; record[h]['l'] += 1
        else:         record[h]['t'] += 1; record[a]['t'] += 1

    def wp(r):
        n = r['w'] + r['l'] + r['t']
        return -1.0 if n < 4 else (r['w'] + 0.5 * r['t']) / n

    if not record:
        return []
    best = max(wp(r) for r in record.values())
    if best <= 0:
        return []
    return sorted([t for t, r in record.items() if wp(r) == best])


def _detect_title_game(year_games, members):
    """Return the title-game winner if a championship game can be identified.
    Layered detection:
      1) Game notes contain "championship" (CFBD has these for 2022+).
      2) Neutral-site conference game between two members in the standard
         CCG window (mid-Nov through mid-Dec). seasonType varies (regular
         in modern years, postseason in some older years like 1994 SEC),
         and CFBD's neutralSite flag is unreliable pre-2003 — so we just
         take the latest qualifying game in the window. The Dec 15 cutoff
         keeps BCS/CFP National Championships out (those are Jan 1+).
    Returns team name (string) or None."""
    # CFBD notes
    for g in year_games:
        if not g.get('completed'):
            continue
        if g.get('homeTeam') not in members or g.get('awayTeam') not in members:
            continue
        notes = (g.get('notes') or '').lower()
        if 'championship' not in notes:
            continue
        hp, ap = g.get('homePoints'), g.get('awayPoints')
        if hp is None or ap is None:
            continue
        return g['homeTeam'] if hp >= ap else g['awayTeam']

    # Neutral-site heuristic — date-window based (handles year-by-year
    # inconsistency in seasonType + week numbering).
    cands = []
    for g in year_games:
        if not g.get('completed') or not g.get('conferenceGame') or not g.get('neutralSite'):
            continue
        if g.get('homeTeam') not in members or g.get('awayTeam') not in members:
            continue
        if g.get('homePoints') is None or g.get('awayPoints') is None:
            continue
        date_str = (g.get('startDate') or '')[:10]
        if not date_str:
            continue
        # CCG window: Nov 15 → Dec 20 (inclusive). The Dec 20 cutoff catches
        # COVID-shifted dates (2020 Big 12 CCG was Dec 19) and any future year
        # where weather or scheduling pushes a CCG late. Bowls + CFP First
        # Round happen in this window too, but they're inter-conference so
        # the same-conf members filter excludes them.
        month_day = date_str[5:10]
        if not ('11-15' <= month_day <= '12-20'):
            continue
        cands.append((date_str, g))
    if cands:
        cands.sort(key=lambda x: x[0])
        _, g = cands[-1]
        return g['homeTeam'] if g['homePoints'] >= g['awayPoints'] else g['awayTeam']

    return None


for f in sorted((DATA_DIR / 'games').glob('games_*.json')):
    year = int(f.stem.split('_')[1])
    raw = json.loads(f.read_text())

    # Build per-conference team rosters for this year
    teams_year = json.loads((DATA_DIR / 'teams' / f'teams_{year}.json').read_text())
    by_conf = {}
    for t in teams_year:
        c = t.get('conference')
        if c in HISTORICAL_P5:
            by_conf.setdefault(c, set()).add(t['school'])

    for conf_name, members in by_conf.items():
        title_winner = _detect_title_game(raw, members)
        if title_winner:
            # Title game determined — single official champion
            conf_champs[(title_winner, year)] = conf_name
        else:
            # No title game (or pre-CCG era) — best conf record, ties = co-champions
            for team in _conf_record_champions(raw, members):
                conf_champs[(team, year)] = conf_name

    post = [g for g in raw if g.get('seasonType') == 'postseason' and g.get('completed')]
    if not post or year < CFP_FIRST_SEASON:
        continue

    title_game = None
    for g in post:
        notes = (g.get('notes') or '').lower()
        if 'national championship' in notes:
            title_game = g
            break
    if title_game is None:
        continue

    if title_game['homePoints'] > title_game['awayPoints']:
        champ, ru = title_game['homeTeam'], title_game['awayTeam']
        score = f"{int(title_game['homePoints'])}-{int(title_game['awayPoints'])}"
    else:
        champ, ru = title_game['awayTeam'], title_game['homeTeam']
        score = f"{int(title_game['awayPoints'])}-{int(title_game['homePoints'])}"

    cfp_appearances = set()
    for g in post:
        notes = (g.get('notes') or '').lower()
        if 'college football playoff' in notes or 'cfp ' in notes:
            cfp_appearances.add(g['homeTeam'])
            cfp_appearances.add(g['awayTeam'])

    cfp_outcomes[year] = {
        'era':          'CFP',
        'champion':     champ,
        'runner_up':    ru,
        'final_score':  score,
        'appearances':  cfp_appearances,
    }


# Layer in BCS-era champions from the hardcoded table. CFBD's notes-based
# detection is unreliable pre-2006 because the BCS title rotated through
# Fiesta/Sugar/Rose/Orange — those games' notes don't mention "BCS championship".
# Also include co_champion when AP picked a different team than the BCS title game.
for year, info in BCS_CHAMPIONS.items():
    co_champ = AP_OVERRIDES.get(year)  # AP picked a different team
    cfp_outcomes[year] = {
        'era':            'BCS',
        'champion':       info['champion'],
        'co_champion':    co_champ,
        'runner_up':      info['runner_up'],
        'final_score':    info['final_score'],
        'appearances':    set(),
        'selectors':      {info['champion']: ['BCS', 'Coaches']},  # primary
    }
    if co_champ:
        cfp_outcomes[year]['selectors'][co_champ] = ['AP']

# Add CFP selectors so per-team badges render uniformly
for year, out in cfp_outcomes.items():
    if out['era'] == 'CFP' and 'selectors' not in out:
        out['selectors'] = {out['champion']: ['CFP']}

# Poll era (1982-1997). Each year has AP champion + Coaches champion (often the
# same team). When they agree, the team gets selectors=['AP','Coaches']. When
# they split, each team gets only their own selector.
for year, polls in POLL_CHAMPIONS.items():
    ap_champ      = polls['AP']
    coaches_champ = polls['Coaches']
    selectors = {}
    selectors.setdefault(ap_champ,      []).append('AP')
    selectors.setdefault(coaches_champ, []).append('Coaches')
    is_split = ap_champ != coaches_champ
    cfp_outcomes[year] = {
        'era':            'Poll',
        'champion':       ap_champ,                       # AP gets the primary slot
        'co_champion':    coaches_champ if is_split else None,
        'runner_up':      None,                           # no title game in poll era
        'final_score':    '',
        'appearances':    set(),
        'selectors':      selectors,
    }


def cfp_status(team, season):
    """0 = none, 1 = runner-up, 2 = champion (or co-champion). Works across CFP/BCS/Poll."""
    out = cfp_outcomes.get(int(season))
    if not out:
        return 0
    if team == out['champion'] or team == out.get('co_champion'):
        return 2
    if team == out.get('runner_up'):
        return 1
    return 0


def cfp_appearance(team, season):
    out = cfp_outcomes.get(int(season))
    if not out:
        return 0
    return 1 if team in out['appearances'] else 0


def champ_era(season):
    """Returns 'CFP', 'BCS', 'Poll', or '' (no champion attribution available)."""
    out = cfp_outcomes.get(int(season))
    return out['era'] if out else ''


def title_selectors(team, season):
    """Returns the list of selector(s) (e.g. ['AP'], ['AP','Coaches'], ['BCS','Coaches'])
    that named this team national champion in this season. Empty list if not a champion."""
    out = cfp_outcomes.get(int(season))
    if not out:
        return []
    return out.get('selectors', {}).get(team, [])


def conf_champ(team, season):
    """Returns the conference name string if team won their conference championship game that year, else ''."""
    return conf_champs.get((team, int(season)), '')


# ── Postseason week labels (tier-based, era-agnostic) ─────────────────────────
def week_label(season, wk):
    wk = int(wk)
    if wk < 100:
        return f'Week {wk}'
    return POSTSEASON_LABELS.get(wk, 'Postseason')


def snapshot_label(season, wk, flag):
    base = week_label(season, wk)
    if flag == 1:
        return f'{base} · End of regular season'
    if flag == 2:
        return f'{base} · End of season'
    return base


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean(val):
    if pd.isna(val):
        return ''
    return str(val)


def slug(name):
    return re.sub(r'[^\w]', '_', name).strip('_')


df['is_game_day']      = (df['lastgame'] != 'Bye / No Game').astype(int)
df['is_end_of_season'] = df['season_flag'].isin([1, 2]).astype(int)


# Per-(team, ranking_id) conference rank — position within the team's
# conference at that snapshot, sorted by overall REACT rank. Lets the
# Standings / Champions tabs show CONF # alongside OVR # (mirrors ZIDANE).
print('Computing conference ranks...')
conf_rank_map = {}
for ranking_id, group in df.groupby('ranking_id'):
    season = int(group['season'].iloc[0])
    counter = {}
    for _, r in group.sort_values('rank').iterrows():
        c = conf(r['name'], season)
        counter[c] = counter.get(c, 0) + 1
        conf_rank_map[(r['name'], int(ranking_id))] = counter[c]


def conf_rank(team, ranking_id):
    return conf_rank_map.get((team, int(ranking_id)))


# Per-(team, season) forward-filled last game. Keying by season prevents
# cross-season carry-forward — at the start of a new season, teams that
# haven't played yet correctly show empty rather than their previous-season
# bowl/playoff result.
_last_game_history = {}
for (team, season), tdf in df[df['is_game_day'] == 1].sort_values('season_week').groupby(['name', 'season']):
    _last_game_history[(team, int(season))] = (
        list(tdf['season_week']),
        list(tdf['lastgame']),
        list(tdf['date']),
    )


def last_game_as_of(team, sw, season):
    entry = _last_game_history.get((team, int(season)))
    if not entry:
        return ''
    sws, games_list, _ = entry
    idx = bisect_right(sws, sw) - 1
    return games_list[idx] if idx >= 0 else ''


def last_game_date_as_of(team, sw, season):
    entry = _last_game_history.get((team, int(season)))
    if not entry:
        return ''
    sws, _, dates = entry
    idx = bisect_right(sws, sw) - 1
    return dates[idx] if idx >= 0 else ''


# Per-season last regular-season week
_rs_end_sw = (
    df[df['season_flag'] == 1]
    .groupby('season')['season_week']
    .max()
    .to_dict()
)


def is_postseason(season, sw):
    rs_end = _rs_end_sw.get(season)
    if rs_end is None:
        return False
    return sw > rs_end


# Regular-season-end record per (team, season)
_reg_record_lookup = {
    (row['name'], int(row['season'])): row['record']
    for _, row in df[df['season_flag'] == 1].iterrows()
}


def _parse_record(rec):
    if not rec or pd.isna(rec):
        return None
    parts = str(rec).split('-')
    try:
        if len(parts) == 2:
            return int(parts[0]), int(parts[1]), 0
        if len(parts) == 3:
            return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    return None


def playoff_record(full_record, regular_record):
    f = _parse_record(full_record)
    r = _parse_record(regular_record)
    if not f or not r:
        return ''
    pw = f[0] - r[0]
    pl = f[1] - r[1]
    if pw < 0 or pl < 0:
        return ''
    return f'{pw}-{pl}'


# ── 1. Current standings ──────────────────────────────────────────────────────
print('Writing current_standings.json...')
latest_id = int(df['ranking_id'].max())
latest = df[df['ranking_id'] == latest_id].sort_values('rank').copy()
latest_date = str(latest['date'].iloc[0]) if not latest.empty else ''
latest_season = int(latest['season'].iloc[0]) if not latest.empty else None

standings_data = {
    'updated': latest_date,
    'season':  latest_season,
    'teams': [
        {
            'rank':            int(r['rank']),
            'conf_rank':       conf_rank(r['name'], r['ranking_id']),
            'team':            r['name'],
            'conference':      conf(r['name'], r['season']),
            'conference_raw':  conf_raw(r['name'], r['season']),
            'rating':          round(float(r['rating']), 3),
            'record':          clean(r['record']),
            'last_match':      clean(r['lastgame']) if r['lastgame'] != 'Bye / No Game' else last_game_as_of(r['name'], r['season_week'], r['season']),
            'cfp_status':       cfp_status(r['name'], r['season']),
            'cfp_appearance':   cfp_appearance(r['name'], r['season']),
            'champ_era':        champ_era(r['season']),
            'title_selectors':  title_selectors(r['name'], r['season']),
            'conference_champ': conf_champ(r['name'], r['season']),
        }
        for _, r in latest.iterrows()
    ],
}
with open(OUT_DIR / 'current_standings.json', 'w') as f:
    json.dump(standings_data, f, separators=(',', ':'))

# ── 2. GOAT table ─────────────────────────────────────────────────────────────
print('Writing goat_teams.json...')
eos_all = df[df['season_flag'] == 2].copy()

# Trophy gate: a team must have actually WON something — either their
# conference championship, or some share of the national championship —
# to qualify for the all-time list. Mirrors ZIDANE's "League or CL win"
# eligibility rule. Cleans up the COVID-2020 cluster (Iowa State, Buffalo,
# Ball State, etc. ranked high but didn't win anything; excluded). Teams
# that did win something in 2020 (Alabama, Oklahoma) stay — the COVID
# inline tag still flags them on each row.
def _qualifies_for_goat(row):
    if cfp_status(row['name'], row['season']) == 2:   # national champion (CFP/BCS/AP/Coaches)
        return True
    if conf_champ(row['name'], row['season']):        # conference champion
        return True
    return False

eos_qualified = eos_all[eos_all.apply(_qualifies_for_goat, axis=1)].copy()
eos_top = eos_qualified.sort_values('rating', ascending=False).head(50).reset_index(drop=True)

goat_data = []
for i, (_, r) in enumerate(eos_top.iterrows()):
    reg = _reg_record_lookup.get((r['name'], int(r['season'])), '')
    goat_data.append({
        'rank':            i + 1,
        'team':            r['name'],
        'conference':      conf(r['name'], r['season']),
        'conference_raw':  conf_raw(r['name'], r['season']),
        'season':          int(r['season']),
        'rating':          round(float(r['rating']), 3),
        'record':          clean(r['record']),
        'regular_record':  reg,
        'playoff_record':  playoff_record(r['record'], reg),
        'cfp_status':       cfp_status(r['name'], r['season']),
        'cfp_appearance':   cfp_appearance(r['name'], r['season']),
        'champ_era':        champ_era(r['season']),
        'title_selectors':  title_selectors(r['name'], r['season']),
        'conference_champ': conf_champ(r['name'], r['season']),
    })
with open(OUT_DIR / 'goat_teams.json', 'w') as f:
    json.dump(goat_data, f, separators=(',', ':'))

# ── 3. Per-team JSON files ────────────────────────────────────────────────────
print('Writing per-team JSON files...')
team_data = df[(df['is_game_day'] == 1) | (df['is_end_of_season'] == 1)].copy()
team_data = team_data.sort_values(['name', 'season', 'season_week'])

all_teams = sorted(df['name'].unique())
teams_index = []

for team in all_teams:
    tdf = team_data[team_data['name'] == team]
    if len(tdf) == 0:
        continue

    team_slug = slug(team)
    most_recent_season = int(tdf['season'].max())
    # Every conference bucket the team has been in across all seasons
    # (lets the conference pill filter on Team Summary catch teams that
    # have switched leagues — USC pre-2024 Pac-12 vs USC 2024+ Big Ten).
    team_seasons_played = sorted(tdf['season'].unique())
    all_confs = sorted(set(conf(team, s) for s in team_seasons_played))
    teams_index.append({
        'name':            team,
        'conference':      conf(team, most_recent_season),
        'all_conferences': all_confs,
        'slug':            team_slug,
    })

    seasons = {}
    for season, sdf in tdf.groupby('season'):
        rs_end = _rs_end_sw.get(season)
        final_reg = _reg_record_lookup.get((team, int(season)))
        entries = []
        for _, r in sdf.sort_values('season_week').iterrows():
            in_post = (rs_end is not None) and (r['season_week'] > rs_end) and (final_reg is not None)
            if in_post:
                reg = final_reg
                po  = playoff_record(r['record'], final_reg)
            else:
                reg = clean(r['record'])
                po  = ''
            entries.append({
                'date':              clean(r['date']),
                'season_week':       float(r['season_week']),
                'week':              int(r['week']),
                'week_label':        week_label(season, r['week']),
                'rating':            round(float(r['rating']), 3),
                'rank':              int(r['rank']),
                'conf_rank':         conf_rank(team, r['ranking_id']),
                'record':            clean(r['record']),
                'regular_record':    reg,
                'playoff_record':    po,
                'last_match':        clean(r['lastgame']) if r['lastgame'] != 'Bye / No Game' else last_game_as_of(team, r['season_week'], season),
                'is_end_of_season':  int(r['is_end_of_season']),
                'season_flag':       int(r['season_flag']),
                'is_playoff':        int(is_postseason(season, r['season_week'])),
                'cfp_status':        cfp_status(team, season),
                'cfp_appearance':    cfp_appearance(team, season),
                'champ_era':         champ_era(season),
                'title_selectors':   title_selectors(team, season),
                'conference_champ':  conf_champ(team, season),
                'conference':        conf(team, season),
                'conference_raw':    conf_raw(team, season),
            })
        seasons[int(season)] = entries

    with open(OUT_DIR / 'teams' / f'{team_slug}.json', 'w') as f:
        json.dump({
            'team':       team,
            'conference': conf(team, most_recent_season),
            'seasons':    seasons,
        }, f, separators=(',', ':'))

teams_index.sort(key=lambda x: x['name'])
with open(OUT_DIR / 'teams_index.json', 'w') as f:
    json.dump(teams_index, f, separators=(',', ':'))

# ── 4. Season standings files ─────────────────────────────────────────────────
print('Writing season standings files...')
all_seasons = sorted(df['season'].unique())

for season in all_seasons:
    sdf = df[df['season'] == season]
    snapshots = []
    for ranking_id, rdf in sdf.groupby('ranking_id'):
        rdf = rdf.sort_values('rank')
        snap_sw   = rdf['season_week'].iloc[0]
        snap_date = clean(rdf['date'].iloc[0])
        wk        = int(rdf['week'].iloc[0])
        flag      = int(rdf['season_flag'].iloc[0])
        label     = snapshot_label(season, wk, flag)

        rs_end = _rs_end_sw.get(season)
        in_post = (rs_end is not None) and (snap_sw > rs_end)

        teams_snap = []
        for _, r in rdf.iterrows():
            if in_post:
                reg = _reg_record_lookup.get((r['name'], int(season)), r['record'])
                po  = playoff_record(r['record'], reg)
            else:
                reg = clean(r['record'])
                po  = ''
            played_today = r['lastgame'] != 'Bye / No Game'
            teams_snap.append({
                'rank':            int(r['rank']),
                'conf_rank':       conf_rank(r['name'], r['ranking_id']),
                'team':            r['name'],
                'conference':      conf(r['name'], season),
                'conference_raw':  conf_raw(r['name'], season),
                'rating':          round(float(r['rating']), 3),
                'record':          clean(r['record']),
                'regular_record':  reg,
                'playoff_record':  po,
                'last_match':      clean(r['lastgame']) if played_today else last_game_as_of(r['name'], snap_sw, season),
                'last_match_date': snap_date if played_today else last_game_date_as_of(r['name'], snap_sw, season),
                'cfp_status':       cfp_status(r['name'], season),
                'cfp_appearance':   cfp_appearance(r['name'], season),
                'champ_era':        champ_era(season),
                'title_selectors':  title_selectors(r['name'], season),
                'conference_champ': conf_champ(r['name'], season),
            })
        snapshots.append({
            'date':        snap_date,
            'season_week': float(snap_sw),
            'week':        wk,
            'label':       label,
            'teams':       teams_snap,
        })

    snapshots.sort(key=lambda x: x['season_week'])
    with open(OUT_DIR / 'seasons' / f'{int(season)}.json', 'w') as f:
        json.dump({'season': int(season), 'snapshots': snapshots}, f, separators=(',', ':'))

seasons_meta = {
    'seasons':    [int(s) for s in reversed(all_seasons)],
    'first_date': str(games['date'].min().date()),
    'last_date':  str(games['date'].max().date()),
    'generated_at': datetime.now(timezone.utc).isoformat(),
}
with open(OUT_DIR / 'seasons_index.json', 'w') as f:
    json.dump(seasons_meta, f, separators=(',', ':'))

# ── 5. Champions table (CFP + BCS + Poll eras) ────────────────────────────────
print('Writing champions.json...')


def _team_block(team_name, season, sdf, selectors):
    """Build a {team, conference, rating, ...} dict for a champion or RU.
    Returns None if the team has no flag=2 row in this season's data (e.g.,
    1982 warm-up, missing data)."""
    rows = sdf[sdf['name'] == team_name]
    reg = _reg_record_lookup.get((team_name, season), '')
    base = {
        'team':             team_name,
        'conference':       conf(team_name, season),
        'conference_raw':   conf_raw(team_name, season),
        'conference_champ': conf_champ(team_name, season),
        'selectors':        selectors,           # ['AP'], ['AP','Coaches'], etc.
        'regular_record':   reg,
    }
    if rows.empty:
        # No rating snapshot available (warm-up window). Surface the team without metrics.
        base.update({'rating': None, 'rank': None, 'conf_rank': None, 'record': '', 'playoff_record': ''})
        return base
    r = rows.iloc[0]
    base.update({
        'rating':         round(float(r['rating']), 3),
        'rank':           int(r['rank']),
        'conf_rank':      conf_rank(team_name, r['ranking_id']),
        'record':         clean(r['record']),
        'playoff_record': playoff_record(r['record'], reg),
    })
    return base


champions = []
for season in sorted(cfp_outcomes.keys(), reverse=True):
    out = cfp_outcomes[season]
    sdf = df[(df['season'] == season) & (df['season_flag'] == 2)]
    selectors_map = out.get('selectors', {})

    champ = _team_block(out['champion'], season, sdf, selectors_map.get(out['champion'], []))
    co_champ = (
        _team_block(out['co_champion'], season, sdf, selectors_map.get(out['co_champion'], []))
        if out.get('co_champion') else None
    )
    ru = (
        _team_block(out['runner_up'], season, sdf, [])
        if out.get('runner_up') else None
    )

    # Skip the row only if we can't even name a champion (shouldn't happen)
    if champ is None:
        continue

    champions.append({
        'season':       season,
        'era':          out['era'],
        'final_score':  out['final_score'],
        'champion':     champ,
        'co_champion':  co_champ,
        'runner_up':    ru,
    })

# Running title counts — count BOTH champion and co_champion (split years
# count both teams). RU counts apply to teams that lost a title game.
_champ_count = {}
_ru_count    = {}
for entry in reversed(champions):
    ct = entry['champion']['team']
    _champ_count[ct] = _champ_count.get(ct, 0) + 1
    entry['champion']['title_count'] = _champ_count[ct]
    if entry.get('co_champion'):
        cct = entry['co_champion']['team']
        _champ_count[cct] = _champ_count.get(cct, 0) + 1
        entry['co_champion']['title_count'] = _champ_count[cct]
    if entry.get('runner_up'):
        rt = entry['runner_up']['team']
        _ru_count[rt] = _ru_count.get(rt, 0) + 1
        entry['runner_up']['runner_up_count'] = _ru_count[rt]

with open(OUT_DIR / 'champions.json', 'w') as f:
    json.dump({'CFB': champions}, f, separators=(',', ':'))

print(f'\nDone. {len(teams_index)} teams, {len(standings_data["teams"])} in current standings.')
print(f'Wrote {len(all_seasons)} season files. Standings date: {latest_date}')
_by_era = {}
for c in champions:
    _by_era[c['era']] = _by_era.get(c['era'], 0) + 1
print(f'Champions: {len(champions)} entries — ' + ', '.join(f'{n} {e}' for e, n in sorted(_by_era.items())))
