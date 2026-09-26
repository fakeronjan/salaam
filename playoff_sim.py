"""SALAAM title odds: Monte Carlo of the rest of the college football season,
the conference title games, the CFP selection committee and the playoff.

For every rating snapshot from 2014 on (the CFP era):
  1. Simulate the remaining regular-season games. Played games are fixed.
  2. Conference title games: division winners, or the top two in the
     standings (ties: head-to-head among the tied teams, then rating). Once a
     conference's regular season is over the real matchup is used.
  3. The committee (committee_model.py) scores every team's final résumé and
     ranks them with Gumbel noise, the same noise its fit measured.
  4. Select and seed the field under that season's rules, then play it.
Once the committee's final ranking is out, the real field and seeds are used.

Game model (probit on FBS-vs-FBS games 2000-2025, pre-game snapshot ratings):
    P(home win) = Phi(A * (rating_home - rating_away + home_pts)), neutral: no home_pts
    P(FBS team beats a non-FBS team) = Phi(FCS_C0 + FCS_C1 * rating)
Ratings drift for the rest of the season: each simulation offsets every
team's rating by N(0, drift_sd(share of season left)), the rating error
measured against how the rest of past seasons actually went.
The offset rating also stands in for the rating the committee sees.

Output per (snapshot, team): conference title, selection, bye, each round
reached and the title.
"""
import hashlib
import json
import multiprocessing as _mp
import os
import pickle

import numpy as np
import pandas as pd
from scipy.special import ndtr

from committee_model import MODEL_JSON, RANKINGS_JSON, conf_champions, final_key, is_power

HERE = os.path.dirname(os.path.abspath(__file__))
FIRST_SEASON = 2014
N_SIMS = 10_000
N_SIMS_PLAYOFFS = 100_000

# (first season, A, home points), fit per era.
ERA_PARAMS = [(2000, 0.0575, 4.16), (2010, 0.0586, 3.25), (2020, 0.0523, 3.32)]
FCS_C0, FCS_C1 = 1.351, 0.0310
# Rating error for the rest of the season, by share of the season left:
# the per-team rating SD that best explains 2005-2025 results after each
# week, given that week's ratings (beyond the game model's own noise).
# SALAAM's long rating window carries last season into September, so early
# ratings are far less reliable than how far they later move suggests.
DRIFT_LEFT = [0.0, 0.22, 0.37, 0.53, 0.69, 0.84, 0.96, 1.0]
DRIFT_SD = [0.0, 0.0, 0.57, 4.93, 7.70, 10.60, 12.50, 12.50]


def drift_sd(frac_left):
    return float(np.interp(frac_left, DRIFT_LEFT, DRIFT_SD))

POWER4 = ('SEC', 'Big Ten', 'Big 12', 'ACC')
INDEPENDENT = 'FBS Independents'
# Title games not at a neutral site are hosted by the team with the better
# conference record.
NEUTRAL_CCG = set(POWER4) | {'Pac-12', 'Mid-American'}


def era_params(season):
    a, h = ERA_PARAMS[0][1:]
    for start, aa, hh in ERA_PARAMS:
        if season >= start:
            a, h = aa, hh
    return a, h


def fmt(season):
    """'four' (2014-23), 'twelve2024' (5 best conference champions in, top 4
    champions get the byes), 'twelve2025' (same bids, byes = top 4 overall),
    'twelve2026' (Power 4 champions + best Group of 6 team + Notre Dame if
    top 12, byes = top 4 overall)."""
    if season < 2024:
        return 'four'
    return {2024: 'twelve2024', 2025: 'twelve2025'}.get(season, 'twelve2026')


def round_names(season):
    """Playoff rounds (names, short). A team's adv[] holds its chance to get
    past each round, the last being the title."""
    if fmt(season) == 'four':
        return ['Semifinals', 'National Championship'], ['SF', 'NC']
    return (['First Round', 'Quarterfinals', 'Semifinals', 'National Championship'],
            ['R1', 'QF', 'SF', 'NC'])


def entry_rounds(season):
    """{seed: round the team enters} (1-based; byes enter round 2)."""
    if fmt(season) == 'four':
        return {k: 1 for k in range(1, 5)}
    return {k: (2 if k <= 4 else 1) for k in range(1, 13)}


_MODEL = None


def committee():
    global _MODEL
    if _MODEL is None:
        m = json.load(open(MODEL_JSON))
        _MODEL = (m['feats'], np.array(m['mu']), np.array(m['sd']), np.array(m['beta']))
    return _MODEL


def select(fmt_, pos, champ, conf_tier, is_nd):
    """Field and seeds from committee order. pos: (n, T) rank position (0 =
    best); champ: (n, T) bool; conf_tier: (T,) 'p4'/'g6'/'ind'. Returns
    seeds (n, 4 or 12) of team indices, seed 1 first."""
    n, T = pos.shape
    big = T + 1000
    if fmt_ == 'four':
        return np.argsort(pos, axis=1)[:, :4]
    sel = np.zeros((n, T), bool)
    rows = np.arange(n)[:, None]
    if fmt_ in ('twelve2024', 'twelve2025'):
        auto = np.argsort(np.where(champ, pos, big), axis=1)[:, :5]
        ok = np.take_along_axis(champ, auto, 1)
        sel[np.broadcast_to(rows, auto.shape)[ok], auto[ok]] = True
    else:
        p4c = champ & (conf_tier == 'p4')[None, :]
        sel |= p4c
        g6 = np.argmin(np.where((conf_tier == 'g6')[None, :], pos, big), axis=1)
        sel[np.arange(n), g6] = True
        sel |= is_nd[None, :] & (pos < 12)
    left = 12 - sel.sum(1)
    rest = np.argsort(np.where(sel, big, pos), axis=1)
    take = np.arange(T)[None, :] < left[:, None]
    sel[np.broadcast_to(rows, rest.shape)[take], rest[take]] = True
    key = np.where(sel, pos, big).astype(float)
    if fmt_ == 'twelve2024':
        top_champs = np.argsort(np.where(sel & champ, pos, big), axis=1)[:, :4]
        key[np.broadcast_to(rows, top_champs.shape), top_champs] -= big
    return np.argsort(key, axis=1)[:, :12]


class SeasonSim:
    def __init__(self, season, games, teams, ratings, schedule=None, final_ranking=None):
        """games: the season's played games (all_NCAA_games.csv rows, SALAAM
        weeks). teams: CFBD FBS teams for the season (school, conference,
        division). ratings: {week: {team: rating}}. schedule: unplayed games
        (homeTeam, awayTeam, neutralSite, conferenceGame). final_ranking:
        committee's final [(rank, team)] once published."""
        self.season = season
        self.fmt = fmt(season)
        self.A, self.hp = era_params(season)
        self.ratings = ratings
        g = games.copy()
        g['date'] = pd.to_datetime(g['date'])
        champs, ccg = conf_champions(g[g['week'] <= 100])
        self.ccg_ids = set(ccg['id'])
        self.actual_champs = champs

        fbs = teams[teams['classification'] == 'fbs'] if 'classification' in teams else teams
        self.teams = sorted(fbs['school'])
        self.idx = {t: i for i, t in enumerate(self.teams)}
        T = len(self.teams)
        self.conf = np.array([dict(zip(fbs['school'], fbs['conference'])).get(t) for t in self.teams], dtype=object)
        div = dict(zip(fbs['school'], fbs['division'] if 'division' in fbs else [None] * len(fbs)))
        self.div = np.array([div.get(t) or '' for t in self.teams], dtype=object)
        self.tier = np.array(['ind' if c == INDEPENDENT else ('p4' if is_power(c, season) else 'g6')
                              for c in self.conf], dtype=object)
        self.is_nd = np.array([t == 'Notre Dame' for t in self.teams])

        # Regular season (+ week-100 games that aren't title games), played and scheduled.
        rs = g[(g['week'] <= 100) & ~g['id'].isin(self.ccg_ids)]
        rs = pd.DataFrame({'home': rs['homeTeam'], 'away': rs['awayTeam'], 'date': rs['date'],
                           'hpts': rs['homePoints'], 'apts': rs['awayPoints'],
                           'neutral': rs['neutralSite'].fillna(False).astype(bool),
                           'conf_game': rs['conferenceGame'].fillna(False).astype(bool)})
        if schedule is not None and len(schedule):
            sc = pd.DataFrame({'home': schedule['homeTeam'], 'away': schedule['awayTeam'],
                               'date': pd.to_datetime(schedule['startDate']).dt.tz_localize(None),
                               'hpts': np.nan, 'apts': np.nan,
                               'neutral': schedule['neutralSite'].fillna(False).astype(bool),
                               'conf_game': schedule['conferenceGame'].fillna(False).astype(bool)})
            rs = pd.concat([rs, sc], ignore_index=True)
        rs = rs[(rs['hpts'] != rs['apts']) | rs['hpts'].isna()]
        rs = rs[rs['home'].isin(self.idx) | rs['away'].isin(self.idx)].sort_values('date').reset_index(drop=True)
        rs['h'] = rs['home'].map(self.idx).fillna(T).astype(int)   # T = non-FBS opponent
        rs['a'] = rs['away'].map(self.idx).fillna(T).astype(int)
        same = (rs['h'] < T) & (rs['a'] < T)
        same &= self.conf[np.minimum(rs['h'], T - 1)] == self.conf[np.minimum(rs['a'], T - 1)]
        army_navy = rs['home'].isin(['Army', 'Navy']) & rs['away'].isin(['Army', 'Navy'])  # CFBD flags it AAC; it isn't
        rs['cg'] = rs['conf_game'] & same & ~army_navy & (self.conf[np.minimum(rs['h'], T - 1)] != INDEPENDENT)
        self.rs = rs

        # Conferences with a title game, and the real matchups.
        members = pd.Series(self.conf).value_counts()
        if len(ccg):
            self.ccg_confs = sorted(set(ccg['homeConference']) & set(self.conf))
        elif schedule is not None:   # season not reached title games yet
            self.ccg_confs = sorted(c for c, n in members.items() if n >= 8 and c != INDEPENDENT)
        else:
            self.ccg_confs = []
        self.ccg_actual = {c.homeConference: (c.homeTeam, c.awayTeam, c.homePoints, c.awayPoints, c.date)
                           for c in ccg.itertuples()}
        self.all_confs = sorted(c for c in set(self.conf) if c != INDEPENDENT)

        # Postseason games actually played, by pair. Two playoff teams can't
        # meet twice in one postseason or play in another bowl, so the pair
        # is enough (SALAAM files 2015-16's un-noted semifinals as bowls).
        post = g[g['week'].between(101, 104)]
        self.ps = {frozenset((x.homeTeam, x.awayTeam)): x.winner for x in post.itertuples()}
        self.ps_date = {frozenset((x.homeTeam, x.awayTeam)): x.date for x in post.itertuples()}
        self.ps_scores = {frozenset((x.homeTeam, x.awayTeam)): (x.homeTeam, x.awayTeam, x.homePoints, x.awayPoints)
                          for x in post.itertuples()}
        self.final_ranking = final_ranking
        self.used_actual = 0

        # Snapshot dates: the last game date in each SALAAM week.
        self.week_date = g.groupby('week')['date'].max().to_dict()

    # ── helpers ─────────────────────────────────────────────────────────
    def snap_date(self, w):
        ds = [d for wk, d in self.week_date.items() if wk <= w]
        return max(ds) if ds else pd.Timestamp(f'{self.season}-08-01')

    def rating_vec(self, w):
        rt = self.ratings.get(w, {})
        R = np.array([rt.get(t, np.nan) for t in self.teams])
        lo = np.nanmin(R) if np.isfinite(R).any() else 0.0
        return np.where(np.isnan(R), lo, R), lo

    def real_selection(self, R):
        """Field + seeds from the committee's real final ranking."""
        T = len(self.teams)
        ranked = {t: k for k, t in self.final_ranking}
        rk = np.array([ranked.get(t, 1000) for t in self.teams], float)
        rk = np.where(rk >= 1000, 1000 - R / 100.0, rk)   # unranked: by rating
        pos = np.argsort(np.argsort(rk)).astype(float)[None, :]
        champ = np.array([any(t in v for v in self.actual_champs.values()) for t in self.teams])[None, :]
        return select(self.fmt, pos, champ, self.tier, self.is_nd)[0]

    # ── one snapshot ────────────────────────────────────────────────────
    def odds_at(self, w, n_sims=N_SIMS, d=None):
        """Odds with week w's ratings, as of date d (default: end of week w)."""
        T = len(self.teams)
        rng = np.random.default_rng(self.season * 1000 + int(w) + (0 if d is None else d.dayofyear * 7))
        R, Rmin = self.rating_vec(w)
        A, hp = self.A, self.hp
        d = self.snap_date(w) if d is None else d
        selection_known = self.final_ranking is not None and w >= 100
        out = {k: np.zeros(T) for k in ('conf', 'field', 'bye', 'QF', 'SF', 'F', 'champ')}
        rs = self.rs
        played = rs['hpts'].notna() & (rs['date'] <= d)
        frac_left = 1.0 - played.mean() if len(rs) else 0.0
        sd = drift_sd(frac_left)
        self.matchups = []

        if selection_known:
            n = n_sims
            seeds = np.broadcast_to(self.real_selection(R), (n, 4 if self.fmt == 'four' else 12))
            Fr = np.broadcast_to(R, (n, T))
            for c in self.all_confs:
                for t in self.actual_champs.get(c, []):
                    if t in self.idx:
                        out['conf'][self.idx[t]] = 1.0
            self.seeds = {self.teams[t]: i + 1 for i, t in enumerate(seeds[0])}
        else:
            n = n_sims
            E = rng.normal(0.0, sd, (n, T)) if sd > 0 else np.zeros((1, T))
            Fr = R[None, :] + E                                  # (n or 1, T)
            Fx = np.concatenate([np.broadcast_to(Fr, (n, T)), np.full((n, 1), Rmin)], axis=1)
            h = rs['h'].to_numpy(); a = rs['a'].to_numpy()
            G = len(rs)
            HW = np.empty((n, G), np.float32)
            pl = played.to_numpy()
            HW[:, pl] = (rs['hpts'][pl] > rs['apts'][pl]).to_numpy()
            up = ~pl
            if up.any():
                hu, au = h[up], a[up]
                both = (hu < T) & (au < T)
                edge = np.where(rs['neutral'].to_numpy()[up], 0.0, hp)
                p = np.empty((n, up.sum()))
                p[:, both] = ndtr(A * (Fx[:, hu[both]] - Fx[:, au[both]] + edge[both]))
                hf = (hu < T) & ~both                              # home FBS vs non-FBS
                p[:, hf] = ndtr(FCS_C0 + FCS_C1 * Fx[:, hu[hf]])
                af = (au < T) & ~both
                p[:, af] = 1 - ndtr(FCS_C0 + FCS_C1 * Fx[:, au[af]])
                HW[:, up] = rng.random((n, up.sum())) < p
            Hm = np.zeros((G, T + 1), np.float32); Hm[np.arange(G), h] = 1
            Am = np.zeros((G, T + 1), np.float32); Am[np.arange(G), a] = 1
            W = HW @ Hm + (1 - HW) @ Am
            L = (1 - HW) @ Hm + HW @ Am

            # Conference standings -> title games
            cg = rs['cg'].to_numpy()
            cW = HW[:, cg] @ Hm[cg] + (1 - HW[:, cg]) @ Am[cg]
            cN = (Hm[cg] + Am[cg]).sum(0)
            pct = cW[:, :T] / np.maximum(cN[:T], 1)
            h2h = np.zeros((n, T))
            for gi in np.flatnonzero(cg):
                i, j = h[gi], a[gi]
                tie = pct[:, i] == pct[:, j]
                hw = HW[:, gi]
                h2h[:, i] += tie * hw; h2h[:, j] += tie * (1 - hw)
            key = pct * 1000 + h2h + Fx[:, :T] * 1e-4
            champ = np.zeros((n, T), bool)
            ccg_games = []                                        # (conf, a (n,), b (n,), a_wins (n,))
            for c in self.all_confs:
                mem = np.flatnonzero(self.conf == c)
                if not len(mem):
                    continue
                if c not in self.ccg_confs:
                    best = pct[:, mem].max(1, keepdims=True)
                    champ[:, mem] |= pct[:, mem] == best      # co-champions
                    continue
                conf_games = cg & ((self.conf[np.minimum(h, T - 1)] == c) & (h < T))
                conf_done = bool(pl[conf_games].all()) if conf_games.any() else False
                act = self.ccg_actual.get(c)
                if act is not None and (conf_done or pd.Timestamp(act[4]) <= d):
                    ia, ib = self.idx.get(act[0], mem[0]), self.idx.get(act[1], mem[-1])
                    ta = np.full(n, ia); tb = np.full(n, ib)
                else:
                    divs = sorted(set(self.div[mem]) - {''})
                    k = key[:, mem]
                    if len(divs) >= 2:
                        d0 = mem[self.div[mem] == divs[0]]; d1 = mem[self.div[mem] == divs[1]]
                        ta = d0[np.argmax(key[:, d0], 1)]; tb = d1[np.argmax(key[:, d1], 1)]
                    else:
                        o = np.argsort(-k, axis=1)
                        ta = mem[o[:, 0]]; tb = mem[o[:, 1]]
                if act is not None and pd.Timestamp(act[4]) <= d:
                    a_wins = np.full(n, act[2] > act[3]) if self.idx.get(act[0]) == ta[0] else np.full(n, act[3] > act[2])
                else:
                    host = np.where(pct[np.arange(n), ta] >= pct[np.arange(n), tb], 1.0, -1.0)
                    edge = 0.0 if c in NEUTRAL_CCG else hp
                    pa = ndtr(A * (Fx[np.arange(n), ta] - Fx[np.arange(n), tb] + edge * host))
                    a_wins = rng.random(n) < pa
                win = np.where(a_wins, ta, tb); lose = np.where(a_wins, tb, ta)
                champ[np.arange(n), win] = True
                ccg_games.append((ta, tb, a_wins))
                np.add.at(W, (np.arange(n), win), 1); np.add.at(L, (np.arange(n), lose), 1)
            out['conf'] = champ.mean(0)

            # Résumés -> committee ranking
            order = np.argsort(-Fx[:, :T], axis=1)
            rk = np.empty((n, T), int); rk[np.arange(n)[:, None], order] = np.arange(T)[None, :]
            top25 = np.concatenate([rk < 25, np.zeros((n, 1), bool)], 1).astype(np.float32)
            no50 = np.concatenate([rk >= 50, np.ones((n, 1), bool)], 1).astype(np.float32)
            qw = (HW * top25[:, a]) @ Hm + ((1 - HW) * top25[:, h]) @ Am
            bad = ((1 - HW) * no50[:, a]) @ Hm + (HW * no50[:, h]) @ Am
            sos = Fx[:, a] @ Hm + Fx[:, h] @ Am
            ng = (Hm + Am).sum(0)[None, :].repeat(n, 0)
            for ta, tb, a_wins in ccg_games:
                ar = np.arange(n)
                for x, y, xw in ((ta, tb, a_wins), (tb, ta, ~a_wins)):
                    np.add.at(sos, (ar, x), Fx[ar, y]); np.add.at(ng, (ar, x), 1)
                    np.add.at(qw, (ar, x), xw * top25[ar, y])
                    np.add.at(bad, (ar, x), (~xw) * no50[ar, y])
            W, L, qw, bad, sos, ng = W[:, :T], L[:, :T], qw[:, :T], bad[:, :T], sos[:, :T], ng[:, :T]
            feats, mu, sdv, beta = committee()
            p5nd = ((self.tier == 'p4') | self.is_nd).astype(float)[None, :]
            g5 = (self.tier == 'g6').astype(float)[None, :]
            F = {'rating': Fx[:, :T], 'L': L, 'wpct': W / np.maximum(W + L, 1),
                 'sos': sos / np.maximum(ng, 1), 'qw': qw, 'bad_l': bad,
                 'p5nd': np.broadcast_to(p5nd, (n, T)), 'champ_p5nd': champ * p5nd, 'champ_g5': champ * g5}
            score = sum(beta[i] * (F[f] - mu[i]) / sdv[i] for i, f in enumerate(feats))
            score = score + rng.gumbel(size=(n, T))
            order = np.argsort(-score, axis=1)
            pos = np.empty((n, T), float); pos[np.arange(n)[:, None], order] = np.arange(T)[None, :]
            seeds = select(self.fmt, pos, champ, self.tier, self.is_nd)
            Fr = Fx[:, :T]
            self.seeds = None

        self._bracket(seeds, Fr, out, rng, d)
        return pd.DataFrame(out, index=self.teams)

    # ── bracket ─────────────────────────────────────────────────────────
    def _bracket(self, seeds, Fr, out, rng, d):
        n = seeds.shape[0]
        A, hp = self.A, self.hp
        ar = np.arange(n)
        np.add.at(out['field'], seeds.ravel(), 1)

        def play(a, b, rnd, home_a=None):
            fixed = np.all(a == a[0]) and np.all(b == b[0])
            key = frozenset((self.teams[a[0]], self.teams[b[0]]))
            done = fixed and key in self.ps and self.ps_date[key] <= d
            if fixed:
                self.matchups.append((rnd, self.teams[a[0]], self.teams[b[0]],
                                      self.ps_scores[key] if done else None, self.ps[key] if done else None))
            if done:
                self.used_actual += 1
                return np.where(self.ps[key] == self.teams[a[0]], a, b)
            edge = 0.0 if home_a is None else hp * home_a
            pa = ndtr(A * (Fr[ar, a] - Fr[ar, b] + edge))
            return np.where(rng.random(n) < pa, a, b)

        if self.fmt == 'four':
            s = seeds
            f1 = play(s[:, 0], s[:, 3], 'SF'); f2 = play(s[:, 1], s[:, 2], 'SF')
            np.add.at(out['F'], f1, 1); np.add.at(out['F'], f2, 1)
            c = play(f1, f2, 'F')
        else:
            s = seeds
            np.add.at(out['bye'], s[:, :4].ravel(), 1)
            np.add.at(out['QF'], s[:, :4].ravel(), 1)
            r1 = [play(s[:, 4 + k], s[:, 11 - k], 'R1', home_a=1.0) for k in range(4)]  # 5v12 6v11 7v10 8v9
            for x in r1:
                np.add.at(out['QF'], x, 1)
            q = [play(s[:, 0], r1[3], 'QF'), play(s[:, 1], r1[2], 'QF'),
                 play(s[:, 2], r1[1], 'QF'), play(s[:, 3], r1[0], 'QF')]
            for x in q:
                np.add.at(out['SF'], x, 1)
            f1 = play(q[0], q[3], 'SF'); f2 = play(q[1], q[2], 'SF')
            np.add.at(out['F'], f1, 1); np.add.at(out['F'], f2, 1)
            c = play(f1, f2, 'F')
        np.add.at(out['champ'], c, 1)
        for k in ('field', 'bye', 'QF', 'SF', 'F', 'champ'):
            out[k] /= n


# ── Driver ───────────────────────────────────────────────────────────────
def load_inputs():
    g = pd.read_csv(os.path.join(HERE, 'all_NCAA_games.csv'), low_memory=False)
    r = pd.read_csv(os.path.join(HERE, 'salaam_ratings_with_standings.csv'),
                    usecols=['ranking_id', 'season', 'week', 'name', 'rating'])
    cfp = json.load(open(RANKINGS_JSON)) if os.path.exists(RANKINGS_JSON) else {}
    return g, r, cfp


def season_inputs(season, g, r, cfp, current_season):
    teams = pd.DataFrame(json.load(open(os.path.join(HERE, 'data', 'teams', f'teams_{season}.json'))))
    gs = g[g['season'] == season]
    # Week-0 snapshots only rate the few teams that played; carry everyone's
    # latest rating forward (from last season's final snapshot on).
    prev = r[r['season'] == season - 1]
    base = dict(zip(*prev[prev['week'] == prev['week'].max()][['name', 'rating']].T.values)) if len(prev) else {}
    ratings = {}
    for w, x in r[r['season'] == season].groupby('week'):
        base = {**base, **dict(zip(x['name'], x['rating']))}
        ratings[int(w)] = dict(base)
    sched = None
    if season == current_season:
        raw = json.load(open(os.path.join(HERE, 'data', 'games', f'games_{season}.json')))
        sched = pd.DataFrame([x for x in raw if not x.get('completed') and x.get('seasonType') == 'regular'])
    weeks = cfp.get(str(season), {})
    fk = final_key(weeks, season)
    final_ranking = [tuple(x) for x in weeks[fk]] if fk and field_announced(season, gs, current_season) else None
    return teams, gs, ratings, sched, final_ranking


def field_announced(season, gs, current_season):
    """Past seasons: the latest committee ranking is the final one. Current
    season: only once CFBD lists the playoff games with teams in them."""
    if season < current_season:
        return True
    raw = json.load(open(os.path.join(HERE, 'data', 'games', f'games_{season}.json')))
    return any('College Football Playoff' in (x.get('notes') or '') and x.get('homeTeam') and x.get('awayTeam')
               and x.get('seasonType') == 'postseason' for x in raw)


def compute(g, r, cfp, current_season, seasons=None, log=print):
    out, brackets = [], {}
    for season in range(FIRST_SEASON, current_season + 1):
        if seasons is not None and season not in seasons:
            continue
        teams, gs, ratings, sched, final_ranking = season_inputs(season, g, r, cfp, current_season)
        if not ratings:
            continue
        sim = SeasonSim(season, gs, teams, ratings, sched, final_ranking)
        for w in sorted(ratings):
            n = N_SIMS_PLAYOFFS if (final_ranking is not None and w >= 100) else N_SIMS
            o = sim.odds_at(w, n_sims=n)
            o.index.name = 'team'
            o = o.reset_index()
            o['season'] = season
            o['week'] = w
            o['n_sims'] = n
            out.append(o)
        if final_ranking is not None:
            brackets[season] = playoff_snapshots(sim)
        log(f'  {season}: {len(ratings)} snapshots, used_actual={sim.used_actual}')
    return pd.concat(out, ignore_index=True), brackets


def playoff_snapshots(sim):
    """Bracket snapshots for the Playoff tab: selection day (the day after
    the title games), then the end of every day with playoff games. SALAAM's
    weeks don't line up with the rounds (bowl week runs past the
    quarterfinals), so these use real dates, each with the latest ratings.
    {date: (seeds, matchups, n_sims, odds DataFrame, ratings week)}"""
    # Game times are UTC; a US night game belongs to the day before.
    local = lambda t: (pd.Timestamp(t) - pd.Timedelta(hours=6)).normalize()
    day_end = lambda day: day + pd.Timedelta(days=1, hours=6) - pd.Timedelta(minutes=1)   # in UTC
    ccg_dates = [pd.Timestamp(v[4]) for v in sim.ccg_actual.values()]
    sel = local(max(ccg_dates) if ccg_dates else sim.snap_date(100)) + pd.Timedelta(days=1)
    sim.odds_at(100, n_sims=1, d=day_end(sel))
    field = set(sim.seeds)
    game_days = sorted({local(dt) for pair, dt in sim.ps_date.items() if pair <= field})
    out = {}
    for day in [sel] + game_days:
        d = day_end(day)
        wk = max([w for w in sim.ratings if w >= 100 and sim.week_date.get(w, pd.Timestamp.max) <= d] or [100])
        o = sim.odds_at(wk, n_sims=N_SIMS_PLAYOFFS, d=d)
        o.index.name = 'team'
        out[day.date().isoformat()] = (dict(sim.seeds), list(sim.matchups), N_SIMS_PLAYOFFS, o.reset_index(), wk)
    return out
