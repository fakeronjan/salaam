# SALAAM - NCAA Football Power Rankings

fakeronjan WLS power ratings for FBS college football, 1982–present.

**Live site:** https://fakeronjan.github.io/salaam

Named for Rashaan Salaam (Colorado, 1994 Heisman).

## How it works

- **Data**: cached from [CollegeFootballData.com](https://collegefootballdata.com) (one JSON file per season, FBS games only)
- **Ratings**: homebrew fakeronjan WLS solver (custom weighted-least-squares regression with per-game date weights, margin cap, zero-sum anchor), two windows
  - **REACT** - 20-week rolling window (long view)
  - **HOTTAKE** - 10-week rolling window (recent form)
- **Scope**: 1982 onward (NCAA Division I-A formalized; 24 programs reclassified down that year)
- **Coverage**: 44 seasons, ~30,000 FBS-vs-FBS games, 136+ teams

## Files

| File | Purpose |
|---|---|
| `fetch_data.py` | Pulls game/team JSON from CFBD into `data/` (idempotent) |
| `salaam.py` | Builds REACT + HOTTAKE fakeronjan WLS ratings + standings |
| `generate_data.py` | Turns ratings into per-team / per-season JSON for the website |
| `data/` | Cached source data from CFBD |
| `docs/` | Static site (deployed to GitHub Pages) |

## Local setup

```bash
pip install -r requirements.txt
echo "YOUR_CFBD_KEY" > ~/.cfbd_api_key   # or set $CFBD_API_KEY env var
python fetch_data.py
python salaam.py
python generate_data.py
```

## Phasing

- **Phase 1** (current): CFP era championship attribution (2014–present)
- **Phase 2** (planned): BCS era backfill (1998–2013)
- **Phase 3** (planned): Pre-BCS poll-era backfill (1982–1997), with split-championship handling

## Sibling rating sites

Portal: [fakeronjan.com/sports](https://fakeronjan.com/sports) ([repo](https://github.com/fakeronjan/sports-ratings))

**Basketball**
- [DUNCAN](https://github.com/fakeronjan/duncan) - NBA
- [LOBO](https://github.com/fakeronjan/lobo) - WNBA
- [CARMELO](https://github.com/fakeronjan/carmelo) - Men's international basketball

**Football**
- [DILLON](https://github.com/fakeronjan/dillon) - NFL

**Soccer**
- [ZIDANE](https://github.com/fakeronjan/zidane) - European club soccer
- [COBI](https://github.com/fakeronjan/cobi) - MLS
- [MESSI](https://github.com/fakeronjan/messi) - Men's international soccer

**Baseball**
- [GRIFFEY](https://github.com/fakeronjan/griffey) - MLB
- [ICHIRO](https://github.com/fakeronjan/ichiro) - Men's international baseball

**Hockey**
- [SAKIC](https://github.com/fakeronjan/sakic) - NHL
- [FORSBERG](https://github.com/fakeronjan/forsberg) - Men's international hockey

**Tennis**
- [Tennis](https://github.com/fakeronjan/tennis) - Open-era Grand Slams (data-viz, no rating model)

**Reality TV**
- [LAVIN](https://github.com/fakeronjan/lavin) - The Challenge
