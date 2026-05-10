# =========================================================
# SALAAM NCAA FOOTBALL POWER RATINGS
# Named for Rashaan Salaam (1994 Heisman winner, Colorado).
# Data: cached JSON from CollegeFootballData.com (see fetch_data.py).
# =========================================================

import json
from pathlib import Path
import pandas as pd
import numpy as np
# rankit==0.2 uses deprecated numpy aliases (np.int, np.float, np.bool) removed in numpy 1.24+.
if not hasattr(np, 'int'):   np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'):  np.bool = bool
from datetime import datetime
import warnings
import rankit  # pip install rankit
from rankit.Table import Table
from rankit.Ranker import MasseyRanker

try:
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
except AttributeError:
    pass


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path(__file__).parent / 'data'
GAMES_DIR = DATA_DIR / 'games'
TEAMS_DIR = DATA_DIR / 'teams'

MIN_SEASON           = 1982   # NCAA Division I-A formalized
WEEKS_REACT          = 20     # rolling window for REACT ratings (long-view)
HOME_FIELD_ADVANTAGE = 0.5
MARGIN_CAP           = 35

# Postseason weeks get shifted into the 100-block to keep them sortable
# after regular-season weeks within the same season.
POSTSEASON_WEEK_OFFSET = 100


# =========================================================
# LOAD GAMES FROM CACHED JSON
# =========================================================

def load_games(min_season=MIN_SEASON):
    """Load every cached games_YYYY.json into a single DataFrame."""
    rows = []
    for f in sorted(GAMES_DIR.glob('games_*.json')):
        year = int(f.stem.split('_')[1])
        if year < min_season:
            continue
        rows.extend(json.loads(f.read_text()))
    df = pd.DataFrame(rows)
    print(f'Loaded {len(df):,} raw games from cache, seasons {min_season}-{int(df["season"].max())}')
    return df


# =========================================================
# GAME DATA PREPARATION
# =========================================================

def prepare_game_data(raw_df):
    """
    Filter to completed FBS-vs-FBS games, convert home/away to winner/loser
    frame, attach margins, week IDs, and result strings.
    """
    df = raw_df.copy()

    # FBS-vs-FBS only (per ZIDANE-style filter — exclude FCS opponents entirely)
    df = df[(df['homeClassification'] == 'fbs') & (df['awayClassification'] == 'fbs')].copy()

    # Completed games with valid scores only
    df = df[df['completed'] == True].copy()
    df = df.dropna(subset=['homePoints', 'awayPoints']).copy()
    df['homePoints'] = pd.to_numeric(df['homePoints']).astype(int)
    df['awayPoints'] = pd.to_numeric(df['awayPoints']).astype(int)

    # Date
    df['date'] = pd.to_datetime(df['startDate'], utc=True).dt.tz_convert(None)

    # Winner/loser frame (home team wins ties on points; ties handled via is_tie)
    df['home_won'] = df['homePoints'] >= df['awayPoints']
    df['winner']   = np.where(df['home_won'], df['homeTeam'], df['awayTeam'])
    df['loser']    = np.where(df['home_won'], df['awayTeam'], df['homeTeam'])
    df['ptsw']     = np.where(df['home_won'], df['homePoints'], df['awayPoints']).astype(int)
    df['ptsl']     = np.where(df['home_won'], df['awayPoints'], df['homePoints']).astype(int)

    df['marginw'] = df['ptsw'] - df['ptsl']
    df['marginl'] = -df['marginw']
    df['is_tie']  = (df['ptsw'] == df['ptsl']).astype(int)
    df['winw']    = np.where(df['ptsw'] > df['ptsl'], 1, 0.5)
    df['winl']    = 1 - df['winw']

    # Home-field signal from winner's perspective
    df['neutralSite'] = df['neutralSite'].fillna(False)
    df['home'] = np.where(
        df['neutralSite'], 0.0,
        np.where(df['winner'] == df['homeTeam'],
                 HOME_FIELD_ADVANTAGE,
                 -HOME_FIELD_ADVANTAGE)
    )
    df['adjmarginw'] = (df['marginw'] + df['home']).clip(upper=MARGIN_CAP)
    df['adjmarginl'] = -df['adjmarginw']

    df['week'] = pd.to_numeric(df['week']).astype(int)

    # CFBD lumps Week 0 games (Hawai'i openers, neutral-site kickoffs in
    # Ireland/Dublin/Australia) into week=1. Detect them: within each season's
    # week=1 set, any game more than 5 days before the median is Week 0.
    reg_w1 = df[(df['seasonType'] == 'regular') & (df['week'] == 1)]
    for season, sub in reg_w1.groupby('season'):
        cutoff = sub['date'].median() - pd.Timedelta(days=5)
        df.loc[sub[sub['date'] < cutoff].index, 'week'] = 0

    # === Postseason taxonomy (overrides CFBD's date-based week numbering) ===
    # College football is structurally awkward — CFBD inconsistently flags
    # Conference Championships (sometimes seasonType=postseason, sometimes
    # =regular at "neutral" sites whose flag is unreliable for older years).
    # We enforce ONE explicit taxonomy for all eras:
    #   100 = Conference Championships (CCGs)
    #   101 = Bowl Games + CFP First Round
    #   102 = CFP Quarterfinals
    #   103 = CFP Semifinals
    #   104 = National Championship
    # Bowls collapse together (per user spec — elongated bowl calendar would
    # otherwise inflate weight on late bowls). CCGs get their own slot so
    # they aren't conflated with bowls (1996 Florida bug: SEC CG + Sugar
    # Bowl both ended up in the same row).

    # Step 1: Conference Championship detection. Catch them across all
    # CFBD classifications (postseason w/o notes, regular w/ neutral site).
    # CCG window: Nov 28 → Dec 20. The Dec 20 cutoff is intentionally generous
    # to catch COVID-shifted dates (2020 Big 12 CCG was Dec 19) and any future
    # year-end-week scheduling. Bowls and CFP First Round happen in this same
    # window, but they're inter-conference, so the same-conf filter below
    # excludes them.
    in_ccg_window = (
        ((df['date'].dt.month == 11) & (df['date'].dt.day >= 28))
        | ((df['date'].dt.month == 12) & (df['date'].dt.day <= 20))
    )
    notes_lower      = df['notes'].fillna('').str.lower()
    has_champ_note   = notes_lower.str.contains('championship', regex=False)
    is_post          = df['seasonType'] == 'postseason'
    is_conf          = df['conferenceGame'].fillna(False).astype(bool)
    is_neutral       = df['neutralSite'].fillna(False).astype(bool)
    ccg_mask = in_ccg_window & (
        has_champ_note
        | (is_post & is_conf)
        | (is_neutral & is_conf & (df['week'] >= 13))
    )
    df.loc[ccg_mask, 'week'] = POSTSEASON_WEEK_OFFSET  # week 100

    # Step 2: Tier-classify remaining postseason games (101-104).
    post_mask = (df['seasonType'] == 'postseason') & ~ccg_mask
    if post_mask.any():
        def postseason_tier(notes):
            n = '' if not isinstance(notes, str) else notes.lower()
            if 'national championship' in n or 'bcs championship' in n:
                return 4
            if 'semifinal' in n:
                return 3
            if 'quarterfinal' in n:
                return 2
            return 1
        df.loc[post_mask, 'week'] = (
            POSTSEASON_WEEK_OFFSET + df.loc[post_mask, 'notes'].apply(postseason_tier)
        )

    # Result strings for last-game display
    df['winner_marker'] = np.where(
        df['neutralSite'], ' vs. (N) ',
        np.where(df['winner'] == df['homeTeam'], ' vs. ', ' @ ')
    )
    df['loser_marker'] = np.where(
        df['neutralSite'], ' vs. (N) ',
        np.where(df['winner'] == df['homeTeam'], ' @ ', ' vs. ')
    )

    df['winner_last_game'] = np.where(
        df['is_tie'] == 1,
        'T ' + df['ptsw'].astype(str) + '-' + df['ptsl'].astype(str) + df['winner_marker'] + df['loser'],
        'W ' + df['ptsw'].astype(str) + '-' + df['ptsl'].astype(str) + df['winner_marker'] + df['loser']
    )
    df['loser_last_game'] = np.where(
        df['is_tie'] == 1,
        'T ' + df['ptsl'].astype(str) + '-' + df['ptsw'].astype(str) + df['loser_marker'] + df['winner'],
        'L ' + df['ptsl'].astype(str) + '-' + df['ptsw'].astype(str) + df['loser_marker'] + df['winner']
    )

    # IDs
    df = df.sort_values(['season', 'week', 'date']).reset_index(drop=True)
    df['season_week']  = df['season'] + df['week'] / 1000
    df['cume_week_id'] = df.groupby(['season_week']).ngroup() + 1

    df['winner_season'] = df['winner'] + ' - ' + df['season'].astype(str)
    df['loser_season']  = df['loser']  + ' - ' + df['season'].astype(str)

    df = df.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
    df.to_csv('all_NCAA_games.csv', index=False)
    print(f'Prepared {len(df):,} FBS-vs-FBS games across {df["cume_week_id"].max()} game-weeks')
    return df


# =========================================================
# MASSEY RATINGS (REACT / HOTTAKE windows)
# =========================================================

def compute_ratings(master_df, existing_ratings_df, window, label):
    """
    Compute Massey ratings using a rolling `window`-game-week window.
    Skips ranking_ids already cached, but always recomputes the latest
    one (in case mid-week games arrive between runs).
    """
    max_date_id = int(master_df['cume_week_id'].max())
    min_date_id = window

    if len(existing_ratings_df) > 0 and 'ranking_id' in existing_ratings_df.columns:
        max_ranked = int(existing_ratings_df['ranking_id'].max())
        min_ranked = int(existing_ratings_df['ranking_id'].min())
    else:
        max_ranked = -1
        min_ranked = -1

    print(f'Running {label} ratings ({window}-week window)...')
    new_frames = []

    for i in range(min_date_id, max_date_id + 1):
        if min_ranked <= i < max_ranked:
            continue

        win = master_df[
            (master_df['cume_week_id'] >= i - (window - 1)) &
            (master_df['cume_week_id'] <= i)
        ].copy()

        win['date_weight']     = (win['cume_week_id'] - i + window) / window
        win['weightedmarginl'] = win['adjmarginl'] * win['date_weight']
        win['weightedmarginw'] = -win['weightedmarginl']

        current_week = win['season_week'].max()
        season       = int(win['season'].max())

        ncaa_table = Table(win, ['loser', 'winner', 'weightedmarginl', 'weightedmarginw'])
        ranked = MasseyRanker(ncaa_table).rank()
        ranked['season_week'] = current_week
        ranked['ranking_id']  = i
        ranked['season']      = season
        new_frames.append(ranked)

    df = pd.concat([existing_ratings_df] + new_frames, axis=0, sort=False).reset_index(drop=True)
    df['week'] = (df['season_week'] - df['season']) * 1000
    df.sort_values(['ranking_id', 'name'], inplace=True)
    df.drop_duplicates(subset=['ranking_id', 'name'], keep='last', inplace=True)
    print(f'CSV of {label} ratings is ready!')
    return df


# =========================================================
# STANDINGS (cumulative season W-L-T per game-week)
# =========================================================

def _make_pivot(df, value_col, index_col, new_value_name, aggfunc=np.sum):
    pivot = pd.pivot_table(df, values=value_col, index=[index_col], aggfunc=aggfunc)
    return pivot.fillna(0).reset_index().rename(columns={value_col: new_value_name, index_col: 'name'})


def compute_standings(master_df, existing_standings_df):
    """
    Cumulative season W-L-T standings per cume_week_id.
    Ties existed in CFB through 1995 (OT introduced 1996), so we keep the
    DILLON-style tie tracking for historical accuracy.
    """
    df_for_calc = master_df[['season', 'season_week', 'cume_week_id',
                             'winner', 'loser', 'is_tie']].copy()
    df_for_calc['winner_real_win']  = np.where(df_for_calc['is_tie'] == 0, 1, 0)
    df_for_calc['loser_real_loss']  = np.where(df_for_calc['is_tie'] == 0, 1, 0)
    df_for_calc['winner_tie']       = df_for_calc['is_tie']
    df_for_calc['loser_tie']        = df_for_calc['is_tie']

    max_date_id = int(master_df['cume_week_id'].max())
    min_date_id = WEEKS_REACT

    if len(existing_standings_df) > 0 and 'ranking_id' in existing_standings_df.columns:
        max_ranked = int(existing_standings_df['ranking_id'].max())
        min_ranked = int(existing_standings_df['ranking_id'].min())
    else:
        max_ranked = -1
        min_ranked = -1

    print('Producing standings...')
    new_frames = []

    for i in range(min_date_id, max_date_id + 1):
        if min_ranked <= i < max_ranked:
            continue

        slicer = df_for_calc[df_for_calc['cume_week_id'] <= i]
        season = int(slicer['season'].max())
        slicer = slicer[slicer['season'] == season]
        ranking_week = slicer['season_week'].max()

        wins_w = _make_pivot(slicer, 'winner_real_win', 'winner', 'wins_as_winner')
        loss_l = _make_pivot(slicer, 'loser_real_loss', 'loser',  'losses_as_loser')
        tie_w  = _make_pivot(slicer, 'winner_tie',      'winner', 'ties_as_winner')
        tie_l  = _make_pivot(slicer, 'loser_tie',       'loser',  'ties_as_loser')

        merged = (wins_w.merge(loss_l, on='name', how='outer')
                        .merge(tie_w,  on='name', how='outer')
                        .merge(tie_l,  on='name', how='outer')
                        .fillna(0))

        merged['wins']   = merged['wins_as_winner'].astype(int)
        merged['losses'] = merged['losses_as_loser'].astype(int)
        merged['ties']   = (merged['ties_as_winner'] + merged['ties_as_loser']).astype(int)

        def _fmt(row):
            if row['ties'] > 0:
                return f"{row['wins']}-{row['losses']}-{row['ties']}"
            return f"{row['wins']}-{row['losses']}"
        merged['record'] = merged.apply(_fmt, axis=1)

        merged = merged[['name', 'wins', 'losses', 'ties', 'record']]
        merged['ranking_id']   = i
        merged['season_week']  = ranking_week
        merged['season']       = season
        new_frames.append(merged)

    df = pd.concat([existing_standings_df] + new_frames, axis=0, sort=False).reset_index(drop=True)
    df.sort_values(['ranking_id', 'name'], inplace=True)
    df.drop_duplicates(subset=['ranking_id', 'name'], keep='last', inplace=True)
    print('CSV of standings is ready!')
    return df


# =========================================================
# FINAL ASSEMBLY
# =========================================================

def assemble_final(master_df, react_df, standings_df):
    """Merge REACT ratings + standings, add week/season flags."""
    print('Final step — merging SALAAM ratings and standings...')

    final_df = pd.merge(react_df, standings_df, how='left', on=['ranking_id', 'name'])
    final_df.rename(columns={'season_week_x': 'season_week', 'season_x': 'season'}, inplace=True)
    final_df['season'] = final_df['season'].round(0).astype(int)
    final_df['record'] = final_df['record'].fillna('0-0')

    final_df['week']        = ((final_df['season_week'] - final_df['season']) * 1000).round(0).astype(int)
    final_df['name_season'] = final_df['name'] + ' - ' + final_df['season'].map(str)

    latest_week_id = final_df['ranking_id'].max()
    final_df['most_recent_week'] = (final_df['ranking_id'] == latest_week_id).astype(int)

    # season_flag: only populated for fully-complete seasons.
    # CFB season YYYY ends with the CFP/title game in mid-Jan of YYYY+1; safe
    # to consider "fully complete" once today is past Feb 1 of YYYY+1.
    today = datetime.now().date()
    def season_is_fully_complete(season):
        return today > datetime(int(season) + 1, 2, 1).date()

    seasons = sorted(final_df['season'].unique())

    # Last regular-season week marker (always populated, in-progress seasons too)
    final_df['last_week_of_regular_season'] = 0
    for s in seasons:
        season_rows = final_df[final_df['season'] == s]
        reg = season_rows[season_rows['week'] < POSTSEASON_WEEK_OFFSET]
        if reg.empty:
            continue
        max_reg_week = reg['season_week'].max()
        final_df.loc[final_df['season_week'] == max_reg_week, 'last_week_of_regular_season'] = 1

    # season_flag: 0=regular, 1=last regular-season week, 2=postseason terminal week
    # Only set for fully-complete seasons.
    final_df['season_flag'] = 0
    for s in seasons:
        if not season_is_fully_complete(s):
            continue
        season_rows = final_df[final_df['season'] == s]
        reg = season_rows[season_rows['week'] < POSTSEASON_WEEK_OFFSET]
        if not reg.empty:
            max_reg_week = reg['season_week'].max()
            final_df.loc[
                (final_df['season'] == s) & (final_df['season_week'] == max_reg_week),
                'season_flag'
            ] = 1
        post = season_rows[season_rows['week'] >= POSTSEASON_WEEK_OFFSET]
        if not post.empty:
            max_post_week = post['season_week'].max()
            final_df.loc[
                (final_df['season'] == s) & (final_df['season_week'] == max_post_week),
                'season_flag'
            ] = 2

    # Last game info per (season, week, team). Pre-aggregate so a team that
    # plays multiple games in the same collapsed postseason week (e.g.,
    # 1996 Florida won SEC Championship + Sugar Bowl, both tier-1 here)
    # produces ONE row in the merge — joining their game strings with ' · '
    # — instead of duplicating the whole standings row downstream.
    _SEP = ' · '
    lastgamew = (master_df[['season', 'week', 'winner', 'winner_last_game', 'loser']]
                 .rename(columns={'winner': 'name'})
                 .groupby(['season', 'week', 'name'], as_index=False)
                 .agg({'winner_last_game': lambda s: _SEP.join(s),
                       'loser':            lambda s: _SEP.join(s)}))
    lastgamel = (master_df[['season', 'week', 'loser', 'loser_last_game', 'winner']]
                 .rename(columns={'loser': 'name'})
                 .groupby(['season', 'week', 'name'], as_index=False)
                 .agg({'loser_last_game': lambda s: _SEP.join(s),
                       'winner':          lambda s: _SEP.join(s)}))
    final_df = final_df.merge(lastgamew, how='left', on=['season', 'week', 'name'])
    final_df = final_df.merge(lastgamel, how='left', on=['season', 'week', 'name'])

    for col in ['winner_last_game', 'loser_last_game', 'winner', 'loser']:
        final_df[col] = final_df[col].fillna('')

    # Combine winner-side + loser-side strings; insert separator between them
    # only when both exist (otherwise we'd get ' · L 0-3 vs X' or 'W 7-3 vs Y · ').
    final_df['lastgame'] = final_df.apply(
        lambda r: _SEP.join(p for p in [r['winner_last_game'], r['loser_last_game']] if p),
        axis=1,
    )
    final_df['opponent'] = final_df['loser'] + final_df['winner']

    final_df = final_df[final_df['record'] != '0-0']

    final_df = final_df[[
        'ranking_id', 'season_week', 'season', 'week', 'name', 'name_season',
        'rating', 'rank',
        'record', 'most_recent_week', 'last_week_of_regular_season',
        'season_flag', 'lastgame', 'opponent'
    ]]

    final_df.to_csv('salaam_ratings_with_standings.csv', index=False)
    print(f'CSV of everything is ready! ({len(final_df):,} rows)')
    return final_df


# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':
    # 1. Load cached games
    raw_df = load_games(MIN_SEASON)

    # 2. Prepare
    master_df = prepare_game_data(raw_df)

    # 3. REACT ratings — drop the current season from the cache before
    # recomputing. cume_week_ids are assigned via groupby().ngroup() over
    # all (season, season_week) pairs, so adding/reclassifying a game in
    # the current season can shift its cume_week_ids and leave cached
    # ranking_ids pointing at stale weeks. Past seasons are stable
    # (CFBD doesn't backfill historical games), so their cache stays valid.
    current_season = int(master_df['season'].max())
    try:
        existing_react = pd.read_csv('salaam_react_ratings.csv')
        existing_react = existing_react[existing_react['season'] != current_season].reset_index(drop=True)
    except FileNotFoundError:
        existing_react = pd.DataFrame()
    react_df = compute_ratings(master_df, existing_react, WEEKS_REACT, 'REACT')
    react_df.to_csv('salaam_react_ratings.csv', index=False)

    # 4. Standings (same cache-trim rationale)
    try:
        existing_standings = pd.read_csv('weekly_standings.csv')
        existing_standings = existing_standings[existing_standings['season'] != current_season].reset_index(drop=True)
    except FileNotFoundError:
        existing_standings = pd.DataFrame()
    standings_df = compute_standings(master_df, existing_standings)
    standings_df.to_csv('weekly_standings.csv', index=False)

    # 5. Final assembly
    assemble_final(master_df, react_df, standings_df)
