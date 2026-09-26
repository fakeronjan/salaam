"""CFP selection committee model.

Fits a Plackett-Luce ranking model to the committee's final top 25 in every
CFP season: each team gets a score from its résumé, and the committee's
ranking is that score plus Gumbel noise. playoff_sim.py samples the noise to
turn résumés into selection odds. Refit by hand once a season ends:

    python committee_model.py          # -> data/cfp_committee_model.json

Résumé features (final week, incl. conference title games):
    rating   SALAAM rating
    L, wpct  losses and win % (all games, FCS included)
    sos      mean opponent rating (FCS opponents = the lowest FBS rating)
    qw       wins over SALAAM top-25 teams
    bad_l    losses to teams outside the SALAAM top 50 (FCS included)
    p5nd     Power conference member (Pac-12 through 2023) or Notre Dame
    champ_p5nd / champ_g5  conference champion, by tier
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(HERE, 'data', 'cfp_committee_model.json')
RANKINGS_JSON = os.path.join(HERE, 'data', 'cfp_rankings.json')
FEATS = ['rating', 'L', 'wpct', 'sos', 'qw', 'bad_l', 'p5nd', 'champ_p5nd', 'champ_g5']
POWER = {'SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12'}


def is_power(conf, season):
    return conf in POWER and not (conf == 'Pac-12' and season >= 2024)


def conf_champions(g):
    """{conference: [champion(s)]} for one season's games (SALAAM weeks).
    Title game = the conference's last December week-100 game between two
    members (Army-Navy excluded); no title game = best conference record."""
    x = g.copy()
    x['d'] = pd.to_datetime(x['date'])
    same = ((x.homeConference == x.awayConference) & x.homeConference.notna()
            & (x.homeConference != 'FBS Independents'))
    army_navy = x.homeTeam.isin(['Army', 'Navy']) & x.awayTeam.isin(['Army', 'Navy'])
    ccg = x[same & (x.week == 100) & ~army_navy & (x.d.dt.month == 12)]
    ccg = ccg.sort_values('d').groupby('homeConference').tail(1)
    champs = {c.homeConference: [c.winner] for c in ccg.itertuples()}
    cg = x[same & (x.week < 100) & (x.conferenceGame == True)]
    for conf in set(cg.homeConference.dropna()) - set(champs):
        y = cg[cg.homeConference == conf]
        n = pd.concat([y.winner, y.loser]).value_counts()
        pct = y.winner.value_counts().reindex(n.index, fill_value=0) / n
        champs[conf] = sorted(pct[pct == pct.max()].index)
    return champs, ccg


def resume(season, g, rating, conf_of, champs):
    """Résumé features for every rated team, from games g (already cut to
    the snapshot) and a {team: rating} Series."""
    order = rating.sort_values(ascending=False).index
    top25, top50 = set(order[:25]), set(order[:50])
    rows = []
    for t in rating.index:
        mine = g[(g.homeTeam == t) | (g.awayTeam == t)]
        wins, losses = mine[mine.winner == t], mine[mine.loser == t]
        opp = np.where(mine.homeTeam == t, mine.awayTeam, mine.homeTeam)
        W, L = len(wins), len(losses)
        p5nd = float(is_power(conf_of.get(t), season) or t == 'Notre Dame')
        champ = float(t in champs)
        rows.append(dict(team=t, rating=rating[t], W=W, L=L, wpct=W / max(W + L, 1),
                         sos=np.mean([rating.get(o, rating.min()) for o in opp]) if len(opp) else 0.0,
                         qw=wins.loser.isin(top25).sum(),
                         bad_l=(~losses.winner.isin(top50)).sum(),
                         p5nd=p5nd, champ_p5nd=champ * p5nd,
                         champ_g5=champ * (1 - p5nd) * float(conf_of.get(t) != 'FBS Independents')))
    return pd.DataFrame(rows)


# CFBD files 2020's final ranking (title games pushed to Dec 18-19) under
# postseason week 1; every other year it's the last regular-season week.
FINAL_KEY_OVERRIDE = {2020: 'postseason-1'}


def final_key(weeks, season=None):
    if season in FINAL_KEY_OVERRIDE and FINAL_KEY_OVERRIDE[season] in weeks:
        return FINAL_KEY_OVERRIDE[season]
    regs = [k for k in weeks if k.startswith('regular')]
    return max(regs, key=lambda k: int(k.split('-')[1])) if regs else None


def build():
    r = pd.read_csv(os.path.join(HERE, 'salaam_ratings_with_standings.csv'))
    g = pd.read_csv(os.path.join(HERE, 'all_NCAA_games.csv'), low_memory=False)
    cfp = json.load(open(RANKINGS_JSON))
    out = []
    for season, weeks in cfp.items():
        season = int(season)
        fk = final_key(weeks, season)
        if fk is None or season >= 2026:
            continue
        gs = g[(g.season == season) & (g.week <= 100)]
        champs, _ = conf_champions(gs)
        champ_set = {t for v in champs.values() for t in v}
        rs = r[(r.season == season) & (r.week == 100)].set_index('name').rating
        conf_of = pd.concat([gs.set_index('homeTeam').homeConference,
                             gs.set_index('awayTeam').awayConference])
        conf_of = conf_of[~conf_of.index.duplicated(keep='last')]
        f = resume(season, gs, rs, conf_of, champ_set)
        ranked = {s: k for k, s in weeks[fk]}
        f['rank'] = f.team.map(ranked)
        f['season'] = season
        out.append(f)
    return pd.concat(out, ignore_index=True)


def _nll(b, groups):
    tot = 0.0
    for Xg, order in groups:
        u = Xg @ b
        alive = np.ones(len(u), bool)
        for i in order:
            m = u[alive].max()
            tot -= u[i] - (m + np.log(np.exp(u[alive] - m).sum()))
            alive[i] = False
    return tot


def fit(df):
    mu, sd = df[FEATS].mean(), df[FEATS].std()
    groups = []
    for _, x in df.groupby('season'):
        x = x.reset_index(drop=True)
        groups.append((((x[FEATS] - mu) / sd).to_numpy(), x['rank'].dropna().sort_values().index.to_numpy()))
    b = minimize(_nll, np.zeros(len(FEATS)), args=(groups,), method='L-BFGS-B').x
    return dict(feats=FEATS, mu=mu.round(6).tolist(), sd=sd.round(6).tolist(), beta=b.round(6).tolist(),
                seasons=sorted(int(s) for s in df.season.unique()))


if __name__ == '__main__':
    m = fit(build())
    json.dump(m, open(MODEL_JSON, 'w'), indent=1)
    print(dict(zip(m['feats'], m['beta'])))
