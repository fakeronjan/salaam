"""Weekly Matchups (port of DILLON's weekly_matchups.py): every FBS vs FBS
game of a week, previewed from the ratings going into it - win probability,
SALAAM's line and O/U (projected total), and the stakes (each team's CFP and
title odds with a win vs with a loss). After the games: the final score and
whether SALAAM's pick was right. 2014 on (the CFP sim's first season).

Weeks: the regular season, Conference Championships (SALAAM week 100), Bowls
(every non-CFP bowl: win probability and line, no stakes), then each CFP
round. Stakes come from the season sim (playoff_sim.py): 10k simulations
from the snapshot before the week's first game, split by each game's
simulated result. 10k vs 100k on 2025: pick % within 0.4 pts on average,
W/L odds within 0.1 pts, Stakes rank Spearman 0.99 (no top-quartile game
moved more than 10 ranks). Larger runs go in batches of BATCH (one 100k run
peaks at ~4 GB).

Line and O/U (fit on 2001-2025 FBS games, leave-one-season-out; 2014-25
margin MAE 13.1 vs 16.7 naive, total MAE 13.4 vs 14.3):
    margin = LINE_LAM * (home rating - away rating + home edge)
        home edge = the sim's era home-field value (0 at neutral sites)
    total  = 2 * FBS points per team-game last season
             + TOTAL_B * (both offenses - both defenses)
"""
import hashlib
import json
import multiprocessing as _mp
import os
import pickle

import numpy as np
import pandas as pd
from scipy.special import ndtr

import playoff_sim

N_SIMS = 10_000                   # plenty at every stage (user call, 2026-10-02)
BATCH = 25_000
LINE_LAM = 0.88
TOTAL_B = 0.51
FIRST_SEASON = playoff_sim.FIRST_SEASON
WEEK_CCG, WEEK_BOWLS = 100, 101
ROUND_WEEK = {'R1': 102, 'QF': 103, 'SF': 104, 'F': 105}
WEEK_LABELS = {WEEK_CCG: 'Conference Championships', WEEK_BOWLS: 'Bowls',
               102: 'CFP First Round', 103: 'CFP Quarterfinals', 104: 'CFP Semifinals',
               105: 'CFP National Championship'}
HERE = os.path.dirname(os.path.abspath(__file__))
_MIN = pd.Timedelta(minutes=1)
_COLS = ['id', 'week', 'seasonType', 'homeTeam', 'awayTeam', 'neutralSite', 'homePoints', 'awayPoints',
         'startDate', 'startTimeTBD']


def week_label(w):
    return WEEK_LABELS.get(w, f'Week {w}')


def _int(v):
    return None if v is None or pd.isna(v) else int(v)


def _record(w, l, t):
    return f"{w}-{l}" + (f"-{t}" if t else "")


def load_ratings():
    """The engine's full rating pool. salaam_ratings_with_standings.csv
    drops 0-0 rows (week 0 has only the teams that played), but its ranks
    come from this pool, so ranks here match the site's."""
    rf = pd.read_csv(os.path.join(HERE, 'salaam_react_ratings.csv'),
                     usecols=['season', 'week', 'name', 'rating', 'rank', 'rating_o', 'rating_d'])
    rf['week'] = rf['week'].round().astype(int)
    return rf


def _snapshots(season, rf):
    """{week: {team: (rating, o, d)}} carried forward like the sim's
    season_inputs (week -1 = last season's final ratings), and {week: {team:
    rank}} for the teams rated that week."""
    cols = ['name', 'rating', 'rating_o', 'rating_d']
    prev = rf[rf['season'] == season - 1]
    base, rk = {}, {}
    if len(prev):
        last = prev[prev['week'] == prev['week'].max()]
        base = {n: (r, o, d) for n, r, o, d in last[cols].itertuples(index=False)}
        rk[-1] = dict(zip(last['name'], last['rank']))
    snaps = {-1: dict(base)}
    for w, x in rf[rf['season'] == season].groupby('week'):
        base = {**base, **{n: (r, o, d) for n, r, o, d in x[cols].itertuples(index=False)}}
        snaps[int(w)] = dict(base)
        rk[int(w)] = dict(zip(x['name'], x['rank']))
    return snaps, rk


def _season_games(season, gs, current_season):
    """The season's games (played, plus unplayed ones in the live season),
    with Pacific-local 'date' like the sim."""
    t = gs[_COLS].copy()
    t['date'] = pd.to_datetime(gs['date'])
    if season == current_season:
        raw = json.load(open(os.path.join(HERE, 'data', 'games', f'games_{season}.json')))
        un = pd.DataFrame([x for x in raw if not x.get('completed') and x.get('homeTeam') and x.get('awayTeam')])
        if len(un):
            un = un[~un['id'].isin(t['id'])].copy()
            un['date'] = pd.to_datetime(un['startDate'], utc=True).dt.tz_convert('America/Los_Angeles').dt.tz_localize(None)
            un.loc[un['seasonType'] == 'postseason', 'week'] = WEEK_BOWLS
            un['homePoints'] = np.nan
            un['awayPoints'] = np.nan
            t = pd.concat([t, un[_COLS + ['date']]], ignore_index=True)
    t['neutralSite'] = t['neutralSite'].fillna(False).astype(bool)
    return t.sort_values('date').reset_index(drop=True)


def _kickoff(start, tbd):
    """(UTC ISO kickoff, None), or (None, date) when the time is TBD. CFBD
    stores TBD games at midnight Eastern, so the date is the Eastern day."""
    if not isinstance(start, str) or not start:
        return None, None
    t = pd.Timestamp(start)
    if bool(tbd) and tbd == tbd:
        return None, t.tz_convert('America/New_York').strftime('%Y-%m-%d')
    return t.tz_convert('UTC').strftime('%Y-%m-%dT%H:%MZ'), None


class _Acc:
    """Per-game sums over simulation batches: home wins, and each team's
    field / title counts split by its result."""

    def __init__(self):
        self.d = {}

    def add(self, key, h, a, hw, made, champ):
        s = self.d.setdefault(key, {'n': 0, 'hw': 0, h: [0, 0, 0, 0, 0], a: [0, 0, 0, 0, 0]})
        s['n'] += len(hw)
        s['hw'] += int(hw.sum())
        for t, won in ((h, hw), (a, ~hw)):
            it = t[1]
            x = s[t]
            x[0] += int(won.sum())
            x[1] += int(made[won, it].sum()); x[2] += int(made[~won, it].sum())
            x[3] += int((champ[won] == it).sum()); x[4] += int((champ[~won] == it).sum())

    def result(self, key, h, a, playoff=False):
        s = self.d.get(key)
        if s is None or not s['n']:
            return None, None
        stakes = {}
        for t in (h, a):
            nw, mw, ml, cw, cl = s[t]
            nl = s['n'] - nw
            stakes[t[0]] = {
                'cfp_win': None if playoff or not nw else round(mw / nw, 4),
                'cfp_loss': None if playoff or not nl else round(ml / nl, 4),
                'title_win': round(cw / nw, 4) if nw else None,
                'title_loss': 0.0 if playoff else (round(cl / nl, 4) if nl else None),
            }
        return s['hw'] / s['n'], stakes


def _run(sim, wk, d, n_sims, ids=(), on_batch=None):
    """n_sims at snapshot wk as of d, in batches; on_batch(cap, made, champ)."""
    T = len(sim.teams)
    for b in range(1, max(n_sims // BATCH, 1) + 1):
        cap = {'ids': set(ids)}
        sim.odds_at(wk, n_sims=min(BATCH, n_sims), d=d, capture=cap, seed=b)
        seeds = np.asarray(cap['seeds'])
        made = np.zeros((seeds.shape[0], T), bool)
        made[np.arange(seeds.shape[0])[:, None], seeds] = True
        on_batch(cap, made, np.asarray(cap['champ']))


def build_season(season, g, r, rf, cfp, current_season, n_sims=N_SIMS, log=print):
    """Weeks for one season (list of {week, label, games})."""
    teams, gs, ratings, sched, final_ranking = playoff_sim.season_inputs(season, g, r, cfp, current_season)
    if not ratings:
        return []
    sim = playoff_sim.SeasonSim(season, gs, teams, ratings, sched, final_ranking)
    snaps, snap_ranks = _snapshots(season, rf)
    sim.ratings[-1] = {t: v[0] for t, v in snaps[-1].items()}
    A, hp = sim.A, sim.hp
    games = _season_games(season, gs, current_season)
    games = games[games['homeTeam'].isin(sim.idx) | games['awayTeam'].isin(sim.idx)]
    played = games[games['homePoints'].notna()]
    fbs = games[games['homeTeam'].isin(sim.idx) & games['awayTeam'].isin(sim.idx)]
    prev = g[(g['season'] == season - 1) & (g['homeClassification'] == 'fbs') & (g['awayClassification'] == 'fbs')]
    mu = float(pd.concat([prev['homePoints'], prev['awayPoints']]).mean()) if len(prev) else 28.0
    wdate = sim.week_date

    def snapshot(cutoff, pre_selection):
        """Latest ratings week finished before cutoff (pre-selection: a
        regular-season week, so the sim plays out the season)."""
        ks = [k for k in sim.ratings if (k < WEEK_CCG) == pre_selection]
        ks = [k for k in ks if k == -1 or wdate.get(k, pd.Timestamp.max) < cutoff]
        return max(ks) if ks else (-1 if pre_selection else WEEK_CCG)

    def records(cutoff):
        rec = {}
        for h, a, hpts, apts in played.loc[played['date'] < cutoff, ['homeTeam', 'awayTeam', 'homePoints', 'awayPoints']].itertuples(index=False):
            for t, pf, pa in ((h, hpts, apts), (a, apts, hpts)):
                w, l, tt = rec.get(t, (0, 0, 0))
                rec[t] = (w + (pf > pa), l + (pf < pa), tt + (pf == pa))
        return rec

    def ranks(wk):
        return snaps.get(wk, snaps[-1]), snap_ranks.get(wk, {})

    def game_row(x, wk, cutoff, p_home, stakes):
        sn, rk = ranks(wk)
        rec = records(cutoff)
        blank = (np.nan, np.nan, np.nan)
        rh, ra = sn.get(x.homeTeam, blank), sn.get(x.awayTeam, blank)
        rh = rh if np.isfinite(rh[0]) else (sim.rating_vec(wk)[1], 0.0, 0.0)
        ra = ra if np.isfinite(ra[0]) else (sim.rating_vec(wk)[1], 0.0, 0.0)
        margin = LINE_LAM * (rh[0] - ra[0] + (0.0 if x.neutralSite else hp))
        total = 2 * mu + TOTAL_B * ((rh[1] + ra[1]) - (rh[2] + ra[2]))
        line = round(float(margin) * 2) / 2
        if line and (line > 0) != (p_home >= 0.5):
            # Coin flip the sim tips the other way (2014-26: all within 1.5
            # pts, the sim at 50-54%; early weeks, where the drift lean pulls
            # ratings to the middle but not home edge): Pick'em.
            line = 0.0
        kick, date = _kickoff(x.startDate, x.startTimeTBD)
        game = {
            'id': int(x.id), 'home': x.homeTeam, 'away': x.awayTeam, 'neutral': bool(x.neutralSite),
            'home_record': _record(*rec.get(x.homeTeam, (0, 0, 0))), 'away_record': _record(*rec.get(x.awayTeam, (0, 0, 0))),
            'home_rating': round(float(rh[0]), 2), 'away_rating': round(float(ra[0]), 2),
            'home_rank': _int(rk.get(x.homeTeam)), 'away_rank': _int(rk.get(x.awayTeam)),
            'p_home': round(float(p_home), 4),
            'line': line,                              # home by this many (negative = away)
            'total': round(float(total) * 2) / 2,      # SALAAM O/U: projected combined points
            'stakes': stakes, 'result': None, 'kickoff': kick, 'date': date,
        }
        if not np.isnan(x.homePoints):
            game['result'] = {'home': int(x.homePoints), 'away': int(x.awayPoints)}
        return game

    weeks_out = []

    def emit(week, rows):
        weeks_out.append({'week': int(week), 'label': week_label(int(week)), 'games': rows})
        log(f'  {season} {week_label(int(week))}: {len(rows)} games')
        return any(r['result'] is None for r in rows)    # live: stop after the current week

    # Regular season + Conference Championships: the sim plays out the rest.
    for week, wg in fbs[fbs['week'] <= WEEK_CCG].groupby('week', sort=True):
        cutoff = games.loc[games['week'] == week, 'date'].min()
        wk = snapshot(cutoff, pre_selection=True)
        acc = _Acc()
        idx = sim.idx

        def on_batch(cap, made, champ):
            ccg = {}
            for _, ta, tb, a_wins in cap['ccg']:
                if (ta == ta[0]).all() and (tb == tb[0]).all():
                    ccg[frozenset((sim.teams[ta[0]], sim.teams[tb[0]]))] = (sim.teams[ta[0]], a_wins)
            for x in wg.itertuples(index=False):
                hw = cap['hw'].get(x.id)
                if hw is None:
                    c = ccg.get(frozenset((x.homeTeam, x.awayTeam)))
                    if c is None:
                        continue
                    hw = c[1] if c[0] == x.homeTeam else ~c[1]
                acc.add(x.id, (x.homeTeam, idx[x.homeTeam]), (x.awayTeam, idx[x.awayTeam]), hw, made, champ)

        _run(sim, wk, cutoff - _MIN, n_sims, ids=wg['id'], on_batch=on_batch)
        rows = []
        for x in wg.itertuples(index=False):
            p, st = acc.result(x.id, (x.homeTeam, idx[x.homeTeam]), (x.awayTeam, idx[x.awayTeam]))
            if p is not None:
                rows.append(game_row(x, wk, cutoff, p, st))
        if rows and emit(week, rows):
            return weeks_out

    if final_ranking is None:
        return weeks_out

    # Postseason. The field: one cheap run from selection day.
    ccg_dates = [pd.Timestamp(v[4]) for v in sim.ccg_actual.values()]
    sel = (max(ccg_dates) if ccg_dates else sim.snap_date(WEEK_CCG)).normalize() + pd.Timedelta(days=2) - _MIN
    sim.odds_at(snapshot(sel, False), n_sims=1, d=sel)
    field = set(sim.seeds)
    post = fbs[fbs['week'] > WEEK_CCG]
    is_cfp = post['homeTeam'].isin(field) & post['awayTeam'].isin(field)
    cfp_games = post[is_cfp]

    # Bowls: no stakes, win probability straight from the ratings.
    bowls = post[~is_cfp]
    if len(bowls):
        cutoff = bowls['date'].min()
        wk = snapshot(cutoff, False)
        sn, _ = ranks(wk)
        rows = []
        for x in bowls.itertuples(index=False):
            if x.homeTeam not in sn or x.awayTeam not in sn:
                continue
            p = ndtr(A * (sn[x.homeTeam][0] - sn[x.awayTeam][0] + (0.0 if x.neutralSite else hp)))
            rows.append(game_row(x, wk, cutoff, p, None))
        emit(WEEK_BOWLS, rows)

    # CFP rounds, each from the snapshot before its first game.
    remaining = cfp_games
    for rnd in playoff_sim.round_names(season)[1]:
        rnd = 'F' if rnd == 'NC' else rnd
        if not len(remaining):
            break
        cutoff = remaining['date'].min()
        wk = snapshot(cutoff, False)
        acc = _Acc()
        pair_of = {frozenset((x.homeTeam, x.awayTeam)): x for x in remaining.itertuples(index=False)}
        idx = sim.idx

        def on_batch(cap, made, champ):
            for r_, ta, tb, a_wins in cap['ps_games']:
                x = pair_of.get(frozenset((ta, tb)))
                if r_ != rnd or x is None:
                    continue
                hw = a_wins if ta == x.homeTeam else ~a_wins
                acc.add(x.id, (x.homeTeam, idx[x.homeTeam]), (x.awayTeam, idx[x.awayTeam]), hw, made, champ)

        _run(sim, wk, cutoff - _MIN, n_sims, on_batch=on_batch)
        rows, done_ids = [], []
        for x in pair_of.values():
            p, st = acc.result(x.id, (x.homeTeam, idx[x.homeTeam]), (x.awayTeam, idx[x.awayTeam]), playoff=True)
            if p is not None:
                rows.append(game_row(x, wk, cutoff, p, st))
                done_ids.append(x.id)
        if not rows:
            break
        remaining = remaining[~remaining['id'].isin(done_ids)]
        if emit(ROUND_WEEK[rnd], rows):
            break
    weeks_out.sort(key=lambda w: w['week'])
    return weeks_out


# ── Juice: Quality x Stakes (same recipe as DILLON) ──────────────────────────
# Quality = 35% the worse team's rating, 35% the better team's, 30% closeness
#           (how near a toss-up);
# Stakes  = both teams' CFP-odds swing + K x their title-odds swing, K
#           ramping 4 -> 8 over the regular season; Conference Championships
#           K = 8; CFP games: title swing only, K = 8. Bowls have no stakes
#           (Quality only, no Juice).
# Each is ranked (0-100) against every game in the pool, then
# Juice = sqrt(Quality x Stakes): a game has to deliver on both.
TITLE_K_START, TITLE_K_END = 4.0, 8.0


def add_juice(seasons):
    """seasons: {season: weeks}. Adds 'quality' to every game and
    'stakes_score' and 'juice' (0-100) to every game with stakes."""
    qrows, srows = [], []
    for season, weeks in seasons.items():
        rs = [w['week'] for w in weeks if w['week'] < WEEK_CCG]
        first, last = (min(rs), max(rs)) if rs else (1, 15)
        for w in weeks:
            wk = w['week']
            k = TITLE_K_END if wk >= WEEK_CCG else TITLE_K_START + (TITLE_K_END - TITLE_K_START) * (wk - first) / max(last - first, 1)
            for g in w['games']:
                qrows.append((g, min(g['home_rating'], g['away_rating']), max(g['home_rating'], g['away_rating']),
                              1 - abs(2 * g['p_home'] - 1)))
                if g['stakes'] is None:
                    g['stakes_score'] = g['juice'] = None
                    continue
                st = list(g['stakes'].values())
                po = sum((s['cfp_win'] or 0) - (s['cfp_loss'] or 0) for s in st)
                sb = sum((s['title_win'] or 0) - (s['title_loss'] or 0) for s in st)
                srows.append((g, po + k * sb))
    if not qrows:
        return
    df = pd.DataFrame([r[1:] for r in qrows], columns=['qmin', 'qmax', 'close'])
    q = (0.35 * df['qmin'].rank(pct=True) + 0.35 * df['qmax'].rank(pct=True)
         + 0.3 * df['close'].rank(pct=True)).rank(pct=True) * 100
    for (g, *_), qq in zip(qrows, q):
        g['quality'] = int(round(qq))
        g['_q'] = qq
    s = pd.Series([r[1] for r in srows]).rank(pct=True) * 100
    for (g, _), ss in zip(srows, s):
        g['stakes_score'] = int(round(ss))
        g['juice'] = int(round(np.sqrt(g['_q'] * ss)))
    for g, *_ in qrows:
        del g['_q']


# ── Per-season cache ─────────────────────────────────────────────────────────
# Finished seasons never change unless the engine or their inputs do: cache
# each season under a fingerprint of this file + the CFP sim's own season
# fingerprint (engine files, games, ratings, teams, committee rankings, live
# schedule) + the O/D ratings (3dp) and last season's games (scoring baseline).
CACHE_DIR = os.path.join(HERE, 'matchups_cache')
_JOB = {}


def _fingerprint(season, g, r, rf, cfp, current_season):
    h = hashlib.sha256()
    h.update(open(os.path.join(HERE, 'weekly_matchups.py'), 'rb').read())
    h.update(playoff_sim._fingerprint(season, g, r, cfp, current_season).encode())
    x = rf[rf['season'].isin([season - 1, season])].sort_values(['season', 'week', 'name']).copy()
    for c in ('rating', 'rating_o', 'rating_d'):
        x[c] = x[c].round(3)
    h.update(x.to_csv(index=False).encode())
    p = g[g['season'] == season - 1]
    h.update(p[['id', 'homePoints', 'awayPoints']].sort_values('id').to_csv(index=False).encode())
    return h.hexdigest()


def _one(season):
    j = _JOB
    return season, build_season(season, j['g'], j['r'], j['rf'], j['cfp'], j['current'], log=lambda *_: None)


def build_cached(g, r, cfp, current_season, workers=None, log=print):
    """{season: weeks} for 2014 through the current season, reusing cached
    seasons and building the rest in parallel."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    rf = load_ratings()
    seasons = [s for s in range(FIRST_SEASON, current_season + 1) if (r['season'] == s).any()]
    out, todo, sigs = {}, [], {}
    for s in seasons:
        sigs[s] = _fingerprint(s, g, r, rf, cfp, current_season)
        path = os.path.join(CACHE_DIR, f'{s}.pkl')
        if os.path.exists(path):
            try:
                sig, weeks = pickle.load(open(path, 'rb'))
                if sig == sigs[s]:
                    out[s] = weeks
                    continue
            except Exception:
                pass
        todo.append(s)
    log(f'  weekly matchups: {len(out)} seasons from cache, computing {len(todo)}: {todo}')
    if todo:
        _JOB.update(g=g, r=r, rf=rf, cfp=cfp, current=current_season)
        ctx = _mp.get_context('fork')
        with ctx.Pool(workers or min(4, os.cpu_count())) as pool:
            for s, weeks in pool.imap_unordered(_one, todo):
                out[s] = weeks
                pickle.dump((sigs[s], weeks), open(os.path.join(CACHE_DIR, f'{s}.pkl'), 'wb'))
                log(f'  {s} done')
    return {s: out[s] for s in seasons}
