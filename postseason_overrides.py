"""Hardcoded postseason game overrides for CFBD data gaps (2001-2007).

CFBD inconsistently classifies older postseason games. Two gaps need patching:

  1. Conference Championship Games, 2001-2007. CFBD tags these as regular-season
     games (seasonType=regular) at sites whose neutralSite flag is False/missing,
     and with empty notes (championship notes only start ~2022). So the three
     heuristic prongs in salaam.py (postseason+conf, neutral+conf, notes) all
     miss them. CFBD tagged CCGs as seasonType=postseason through ~2000 and set
     a reliable neutralSite flag from 2008 on, so only the 2001-2007 window is
     affected. These exact game IDs (verified by matchup + final score across
     SEC, Big 12, ACC, Conference USA and the MAC) force them to week 100.

  2. The 2007 BCS National Championship (LSU 38, Ohio State 24, Jan 8 2008) has
     an empty CFBD notes field, so the notes-based title detector collapses it
     into the bowl tier (101). 2006 and 2008-2023 title games carry proper
     notes and are detected correctly. This one game ID forces it to week 104.

The window is closed history, so this table needs no maintenance. Imported by
both salaam.py (week tiering) and generate_data.py (conference-champ attribution).
"""

# Conference Championship Games, 2001-2007 (week 100).
CCG_OVERRIDE_IDS = {
    213420099, 213350251, 213342649,  # 2001: SEC, Big 12, MAC
    223410008, 223410038, 223410276,  # 2002: SEC, Big 12, MAC
    233400061, 233400201, 233380189,  # 2003: SEC, Big 12, MAC
    243390002, 243390038, 243370193,  # 2004: SEC, Big 12, MAC
    253370099, 253370251, 253370259, 253372116, 253352459,  # 2005: SEC, Big 12, ACC, C-USA, MAC
    263360057, 263360201, 263360059, 263350248, 263340195,  # 2006: SEC, Big 12, ACC, C-USA, MAC
    273350099, 273350142, 273350103, 273352116, 273352117,  # 2007: SEC, Big 12, ACC, C-USA, MAC
}

# Standalone title games with empty/undetectable CFBD notes (week 104).
TITLE_OVERRIDE_IDS = {
    280070194,  # 2007 BCS National Championship: LSU 38, Ohio State 24
}
