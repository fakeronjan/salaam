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

# Tier-based postseason week labels (matches salaam.py tier numbering)
POSTSEASON_LABELS = {
    101: 'Bowls / CFP 1st Round',
    102: 'CFP Quarterfinals',
    103: 'CFP Semifinals',
    104: 'National Championship',
}

for f in sorted((DATA_DIR / 'games').glob('games_*.json')):
    year = int(f.stem.split('_')[1])
    raw = json.loads(f.read_text())

    # Conference championships from regular-season games tagged "championship".
    # Each conference has at most one championship game per year — winner = champ.
    reg = [g for g in raw if g.get('seasonType') == 'regular' and g.get('completed')
           and g.get('homeClassification') == 'fbs' and g.get('awayClassification') == 'fbs']
    for g in reg:
        notes = (g.get('notes') or '').lower()
        if 'championship' not in notes:
            continue
        # Winner of the championship game = conference champion
        if g['homePoints'] is None or g['awayPoints'] is None:
            continue
        winner = g['homeTeam'] if g['homePoints'] >= g['awayPoints'] else g['awayTeam']
        # Tag with the team's actual conference for this season (resolved later)
        conf_champs[(winner, year)] = team_conf_raw.get((winner, year), 'Conference')

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
for year, info in BCS_CHAMPIONS.items():
    cfp_outcomes[year] = {
        'era':          'BCS',
        'champion':     info['champion'],
        'runner_up':    info['runner_up'],
        'final_score':  info['final_score'],
        'appearances':  set(),  # BCS had no playoff bracket — only the title game
    }


def cfp_status(team, season):
    """0 = none, 1 = runner-up, 2 = champion. Era-agnostic — works for CFP + BCS."""
    out = cfp_outcomes.get(int(season))
    if not out:
        return 0
    if team == out['champion']:
        return 2
    if team == out['runner_up']:
        return 1
    return 0


def cfp_appearance(team, season):
    out = cfp_outcomes.get(int(season))
    if not out:
        return 0
    return 1 if team in out['appearances'] else 0


def champ_era(season):
    """Returns 'CFP', 'BCS', or '' (no champion attribution available)."""
    out = cfp_outcomes.get(int(season))
    return out['era'] if out else ''


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


# Per-team forward-filled last game
_last_game_history = {}
for team, tdf in df[df['is_game_day'] == 1].sort_values('season_week').groupby('name'):
    _last_game_history[team] = (
        list(tdf['season_week']),
        list(tdf['lastgame']),
        list(tdf['date']),
    )


def last_game_as_of(team, sw):
    entry = _last_game_history.get(team)
    if not entry:
        return ''
    sws, games_list, _ = entry
    idx = bisect_right(sws, sw) - 1
    return games_list[idx] if idx >= 0 else ''


def last_game_date_as_of(team, sw):
    entry = _last_game_history.get(team)
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
            'team':            r['name'],
            'conference':      conf(r['name'], r['season']),
            'conference_raw':  conf_raw(r['name'], r['season']),
            'rating':          round(float(r['rating']), 3),
            'record':          clean(r['record']),
            'last_match':      clean(r['lastgame']) if r['lastgame'] != 'Bye / No Game' else last_game_as_of(r['name'], r['season_week']),
            'cfp_status':       cfp_status(r['name'], r['season']),
            'cfp_appearance':   cfp_appearance(r['name'], r['season']),
            'champ_era':        champ_era(r['season']),
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
eos_top = eos_all.sort_values('rating', ascending=False).head(50).reset_index(drop=True)

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
                'record':            clean(r['record']),
                'regular_record':    reg,
                'playoff_record':    po,
                'last_match':        clean(r['lastgame']) if r['lastgame'] != 'Bye / No Game' else last_game_as_of(team, r['season_week']),
                'is_end_of_season':  int(r['is_end_of_season']),
                'season_flag':       int(r['season_flag']),
                'is_playoff':        int(is_postseason(season, r['season_week'])),
                'cfp_status':        cfp_status(team, season),
                'cfp_appearance':    cfp_appearance(team, season),
                'champ_era':         champ_era(season),
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
                'team':            r['name'],
                'conference':      conf(r['name'], season),
                'conference_raw':  conf_raw(r['name'], season),
                'rating':          round(float(r['rating']), 3),
                'record':          clean(r['record']),
                'regular_record':  reg,
                'playoff_record':  po,
                'last_match':      clean(r['lastgame']) if played_today else last_game_as_of(r['name'], snap_sw),
                'last_match_date': snap_date if played_today else last_game_date_as_of(r['name'], snap_sw),
                'cfp_status':       cfp_status(r['name'], season),
                'cfp_appearance':   cfp_appearance(r['name'], season),
                'champ_era':        champ_era(season),
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
}
with open(OUT_DIR / 'seasons_index.json', 'w') as f:
    json.dump(seasons_meta, f, separators=(',', ':'))

# ── 5. Champions table (CFP era only — Phase 1) ───────────────────────────────
print('Writing champions.json...')

champions = []
for season in sorted(cfp_outcomes.keys(), reverse=True):
    out = cfp_outcomes[season]
    sdf = df[(df['season'] == season) & (df['season_flag'] == 2)]
    if sdf.empty:
        continue
    champ_row = sdf[sdf['name'] == out['champion']]
    ru_row    = sdf[sdf['name'] == out['runner_up']]
    if champ_row.empty or ru_row.empty:
        continue

    cr = champ_row.iloc[0]
    rr = ru_row.iloc[0]
    champ_reg = _reg_record_lookup.get((cr['name'], season), '')
    ru_reg    = _reg_record_lookup.get((rr['name'], season), '')

    champions.append({
        'season':       season,
        'era':          out['era'],
        'final_score':  out['final_score'],
        'champion': {
            'team':           cr['name'],
            'conference':     conf(cr['name'], season),
            'conference_raw': conf_raw(cr['name'], season),
            'conference_champ': conf_champ(cr['name'], season),
            'rating':         round(float(cr['rating']), 3),
            'rank':           int(cr['rank']),
            'record':         clean(cr['record']),
            'regular_record': champ_reg,
            'playoff_record': playoff_record(cr['record'], champ_reg),
        },
        'runner_up': {
            'team':           rr['name'],
            'conference':     conf(rr['name'], season),
            'conference_raw': conf_raw(rr['name'], season),
            'conference_champ': conf_champ(rr['name'], season),
            'rating':         round(float(rr['rating']), 3),
            'rank':           int(rr['rank']),
            'record':         clean(rr['record']),
            'regular_record': ru_reg,
            'playoff_record': playoff_record(rr['record'], ru_reg),
        },
    })

# Running title counts (Phase 1 — CFP era only)
_champ_count = {}
_ru_count    = {}
for entry in reversed(champions):
    ct = entry['champion']['team']
    rt = entry['runner_up']['team']
    _champ_count[ct] = _champ_count.get(ct, 0) + 1
    _ru_count[rt]    = _ru_count.get(rt, 0) + 1
    entry['champion']['title_count']      = _champ_count[ct]
    entry['runner_up']['runner_up_count'] = _ru_count[rt]

with open(OUT_DIR / 'champions.json', 'w') as f:
    json.dump({'CFB': champions}, f, separators=(',', ':'))

print(f'\nDone. {len(teams_index)} teams, {len(standings_data["teams"])} in current standings.')
print(f'Wrote {len(all_seasons)} season files. Standings date: {latest_date}')
_by_era = {}
for c in champions:
    _by_era[c['era']] = _by_era.get(c['era'], 0) + 1
print(f'Champions: {len(champions)} entries — ' + ', '.join(f'{n} {e}' for e, n in sorted(_by_era.items())))
