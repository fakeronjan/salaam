#!/usr/bin/env python3
"""
fetch_data.py — pull NCAA FBS football data from CollegeFootballData.com
and cache as JSON files in data/.

Reads API key from $CFBD_API_KEY (env) or ~/.cfbd_api_key (file).

Default behavior: backfills any missing years, refreshes the current season,
and skips already-cached prior seasons. Use --force to re-fetch everything,
or --year YYYY for a single season.
"""

import argparse
import datetime
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'
GAMES_DIR = DATA_DIR / 'games'
TEAMS_DIR = DATA_DIR / 'teams'
CONFERENCES_FILE = DATA_DIR / 'conferences.json'
META_FILE = DATA_DIR / '_meta.json'

START_SEASON = 1982  # NCAA Division I-A formalized — 24 programs (Ivy + old SoCon) reclassified down
API_BASE = 'https://api.collegefootballdata.com'
API_KEY_FILE = Path.home() / '.cfbd_api_key'
RATE_LIMIT_SLEEP = 0.4


def load_key():
    key = os.environ.get('CFBD_API_KEY')
    if key:
        return key.strip()
    if API_KEY_FILE.exists():
        return API_KEY_FILE.read_text().strip()
    raise SystemExit('Set $CFBD_API_KEY or create ~/.cfbd_api_key')


def current_season():
    today = datetime.date.today()
    return today.year if today.month >= 8 else today.year - 1


def fetch(path, key):
    req = urllib.request.Request(
        f'{API_BASE}{path}',
        headers={'Authorization': f'Bearer {key}'},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch_year_games(year, key):
    reg = fetch(f'/games?year={year}&seasonType=regular', key)
    time.sleep(RATE_LIMIT_SLEEP)
    try:
        post = fetch(f'/games?year={year}&seasonType=postseason', key)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            post = []
        else:
            raise
    games = reg + post
    return [g for g in games if g.get('homeClassification') == 'fbs'
            or g.get('awayClassification') == 'fbs']


def fetch_year_teams(year, key):
    return fetch(f'/teams/fbs?year={year}', key)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Re-fetch all years even if cached')
    parser.add_argument('--year', type=int, help='Fetch only this single season')
    parser.add_argument('--start', type=int, default=START_SEASON)
    args = parser.parse_args()

    key = load_key()
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    TEAMS_DIR.mkdir(parents=True, exist_ok=True)

    cur = current_season()
    print(f'Current season detected: {cur}')

    print('Fetching /conferences...')
    confs = fetch(f'/conferences', key)
    write_json(CONFERENCES_FILE, confs)
    print(f'  {len(confs)} conferences')
    time.sleep(RATE_LIMIT_SLEEP)

    years = [args.year] if args.year else list(range(args.start, cur + 1))

    log = {}
    for year in years:
        games_file = GAMES_DIR / f'games_{year}.json'
        teams_file = TEAMS_DIR / f'teams_{year}.json'
        is_current = (year == cur)

        if args.force or is_current or not games_file.exists():
            print(f'  games {year}...', end=' ', flush=True)
            games = fetch_year_games(year, key)
            write_json(games_file, games)
            print(f'{len(games)} FBS-involved games')
            log[f'games_{year}'] = {
                'count': len(games),
                'fetched_at': datetime.datetime.utcnow().isoformat(timespec='seconds'),
            }
            time.sleep(RATE_LIMIT_SLEEP)
        else:
            print(f'  games {year}: cached')

        if args.force or is_current or not teams_file.exists():
            print(f'  teams {year}...', end=' ', flush=True)
            teams = fetch_year_teams(year, key)
            write_json(teams_file, teams)
            print(f'{len(teams)} teams')
            log[f'teams_{year}'] = {
                'count': len(teams),
                'fetched_at': datetime.datetime.utcnow().isoformat(timespec='seconds'),
            }
            time.sleep(RATE_LIMIT_SLEEP)
        else:
            print(f'  teams {year}: cached')

    meta = {}
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text())
    meta.setdefault('runs', []).append({
        'run_at': datetime.datetime.utcnow().isoformat(timespec='seconds'),
        'years': [args.year] if args.year else [years[0], years[-1]],
        'updates': log,
    })
    write_json(META_FILE, meta)
    print(f'\nDone. {len(log)} files updated this run.')


if __name__ == '__main__':
    main()
