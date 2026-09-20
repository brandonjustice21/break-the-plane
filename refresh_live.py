"""
Break the Plane -- hourly live refresh.

Replicates RotoBombs' hourly-cron pattern for the Anytime TD tool: no
RotoWire scraping, no re-fitting anything. Every run:

  1. Pulls ESPN's public injuries endpoint (all 32 teams, no key needed) --
     replaces the RotoWire injury scrape as the injury *source*, and lets
     newly-out players get zeroed the moment ESPN posts them instead of
     waiting for the next manual weekly rebuild.
  2. Pulls live NFL game lines (spread/total) and Anytime TD Scorer odds
     from OpticOdds (needs ODDSJAM_KEY) -- replaces the static
     schedules_2026.csv snapshot as the *line* source, so a line move (like
     the DET@BUF 52.5/-3 -> 55/-5.5 move handled manually on 2026-09-17)
     flows through automatically.
  3. Recomputes the full formula chain -- lambda_team (team environment) ->
     pace_scalar -> bottom-up (rate x volume) -> top-down (share of team
     lambda) -> 55/45 blend -- using the PERSISTED, "reused not refit"
     regression constants in model_constants_2026.json. Nothing here ever
     refits intercept/w_implied/w_trailing/pts_to_td_ratio/league_rush_share;
     those only change when a human re-runs build_model_constants.py at the
     start of a new season.
  4. Writes data.json -- the file break_the_plane_v5_live.html now fetches
     at runtime (see the __DATA_JSON__ -> fetch('data.json') refactor).

Scope boundary, on purpose: this script zeroes a newly-sidelined player's
own volume/share the moment ESPN posts "Out" or "Injured Reserve", but it
does NOT reassign that share to a specific "next man up" teammate -- that
beneficiary assignment (damping-fraction role bumps, rank-based eligibility)
is a judgment-heavy, low-frequency computation that still belongs to the
weekly pipeline (scripts 32/34/35, run manually when real roster news
breaks). Running hourly, this script keeps the numbers honest ("don't bet
the O'd-out player") without silently guessing who inherits the touches.
A returning/upgraded player (Questionable/Doubtful/Active per ESPN) is
always restored from the untouched raw slate, so a false-positive zero
never lingers.

Usage:
    ODDSJAM_KEY=... python3 refresh_live.py --week 2

Safe without a key: OpticOdds calls are skipped and the script falls back
to the static schedules_2026.csv lines and whatever odds are already in
data.json, so the injury refresh (which needs no key) still runs.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

# Overridable via env vars so this same script runs unchanged in the GitHub
# Actions checkout (repo-relative paths) and in this sandbox (absolute paths).
ND = os.environ.get("BTP_ND", "/home/claude/nflverse_data")
OUT = os.environ.get("BTP_OUT", "/home/claude/model_build")
# Where data.json lives -- the repo ROOT, next to index.html (the page does a
# same-directory fetch('data.json')), which is a different directory than OUT
# (the static per-week model inputs, e.g. a data/ subfolder) in the shipped repo.
SITE = os.environ.get("BTP_SITE", OUT)
SEASON = 2026

TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
WONT_PLAY_ESPN = {"Out", "Injured Reserve"}  # mirrors script 21/34's WONT_PLAY tier, ESPN's smaller vocabulary
SKILL_POS_ABBR = {"RB", "WR", "TE", "FB"}

BOOK_BATCHES = [
    ["DraftKings", "FanDuel", "BetMGM", "Caesars", "theScore Bet"],
    ["Fanatics", "Bally Bet", "Hard Rock Bet", "BetRivers", "Bet365"],
    ["Novig", "Kalshi", "PrizePicks", "Underdog", "ProphetX"],
    ["Pinnacle"],
]
TD_MARKETS = ["Anytime Touchdown Scorer", "Any Time Touchdown Scorer", "To Score A Touchdown"]
TD_MARKETS_L = [m.lower() for m in TD_MARKETS]


def log(msg):
    print(f"[refresh_live] {msg}", flush=True)


def norm_name(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace(".", "").replace("'", "").strip().lower()
    s = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def get_json(url, timeout=20, headers=None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "break-the-plane-refresh/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# =====================================================================
# LIVE FETCH 1 -- ESPN injuries (no key required)
# =====================================================================
def fetch_espn_injuries():
    """Returns a DataFrame: espn_id, team, name, status, body_part, updated -- for
    every player leaguewide currently listed on ESPN's injury report."""
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
    try:
        data = get_json(url, headers={"User-Agent": "curl/8.5.0"})
    except Exception as e:
        log(f"WARNING: ESPN injuries fetch failed ({e}); no live injury update this run.")
        return pd.DataFrame(columns=["espn_id", "team", "name", "status", "body_part", "updated"])

    rows = []
    for team_block in data.get("injuries", []):
        team_abbr = TEAM_NAME_TO_ABBR.get(team_block.get("displayName"))
        for inj in team_block.get("injuries", []):
            athlete = inj.get("athlete", {}) or {}
            pos = (athlete.get("position") or {}).get("abbreviation")
            rows.append({
                "espn_id": str(athlete.get("id")) if athlete.get("id") else None,
                "team": team_abbr,
                "name": athlete.get("displayName"),
                "pos_abb": pos,
                "status": inj.get("status"),
                "body_part": (inj.get("details") or {}).get("type"),
                "updated": inj.get("date"),
            })
    df = pd.DataFrame(rows)
    log(f"ESPN injuries: {len(df)} listed leaguewide, "
        f"{(df['status'].isin(WONT_PLAY_ESPN)).sum() if len(df) else 0} in the won't-play tier (Out/IR).")
    return df


# =====================================================================
# LIVE FETCH 2 -- OpticOdds game lines + Anytime TD odds (needs ODDSJAM_KEY)
# =====================================================================
def fetch_opticodds(week, key, sched_wk):
    """Returns (fixtures, lines_by_game_id, td_odds_by_norm_name). Any piece
    that can't be confidently parsed comes back empty rather than guessed,
    so callers fall back to the static/last-known value for that piece only.

    Fixture matching is done by TEAM PAIR against our own known schedule
    (sched_wk), not by trusting OpticOdds' own `season_week`/`season_year`
    fixture fields -- a first live run (Sept 20, 2026, first time this ran
    with a real key) came back with 100 active NFL fixtures leaguewide but
    zero matching `season_week==2, season_year==2026`, meaning those field
    names/values are not what this script originally assumed. Team-pair
    matching against sched_wk (which we already trust -- it's the same
    schedules_2026.csv the rest of the pipeline uses) sidesteps that
    entirely: this week's 16 games are known in advance from OUR data,
    we just need to find each one's fixture id in OpticOdds' feed."""
    empty = ([], {}, {})
    if not key:
        log("No ODDSJAM_KEY set -- skipping OpticOdds fetch (game lines + odds stay at their last-known values).")
        return empty

    base = "https://api.opticodds.com/api/v3"

    def qs(params):
        return urllib.parse.urlencode(params, doseq=True)

    try:
        fx = get_json(f"{base}/fixtures/active?{qs({'key': key, 'sport': 'football', 'league': 'nfl'})}")
    except Exception as e:
        log(f"WARNING: OpticOdds fixtures/active failed ({e}); skipping live odds this run.")
        return empty

    all_fixtures = fx.get("data", [])
    known_pairs = set(zip(sched_wk["home_team"], sched_wk["away_team"]))
    wk_fixtures = []
    for f in all_fixtures:
        try:
            home = f["home_competitors"][0]["abbreviation"]
            away = f["away_competitors"][0]["abbreviation"]
        except (KeyError, IndexError):
            continue
        if (home, away) in known_pairs:
            wk_fixtures.append(f)
    log(f"OpticOdds active NFL fixtures: {len(all_fixtures)}; matched to Week {week} {SEASON}'s "
        f"{len(known_pairs)} known games by team pair: {len(wk_fixtures)}")
    if not wk_fixtures and all_fixtures:
        sample = all_fixtures[0]
        sample_keys = sorted(sample.keys())
        sample_teams = None
        try:
            sample_teams = (sample["home_competitors"][0]["abbreviation"],
                             sample["away_competitors"][0]["abbreviation"])
        except (KeyError, IndexError):
            pass
        log(f"DIAGNOSTIC (no team-pair match found): sample fixture keys={sample_keys}; "
            f"sample fixture teams={sample_teams}; our known Week {week} pairs (first 5)="
            f"{list(known_pairs)[:5]}")
    if not wk_fixtures:
        return empty
    fixture_ids = [f["id"] for f in wk_fixtures]
    fixture_teams = {}  # fixture id -> (home_abbr, away_abbr)
    for f in wk_fixtures:
        try:
            fixture_teams[f["id"]] = (
                f["home_competitors"][0]["abbreviation"], f["away_competitors"][0]["abbreviation"])
        except (KeyError, IndexError):
            pass

    lines_by_fixture = {}  # fixture id -> {"total": x, "spread_home": y}  (spread from HOME perspective)
    td_players = {}
    per_book_td = {}
    odds_seen = 0

    for i in range(0, len(fixture_ids), 5):
        batch = fixture_ids[i:i + 5]
        for books in BOOK_BATCHES:
            base_params = [("key", key)]
            base_params += [("fixture_id", fid) for fid in batch]
            base_params += [("sportsbook", b) for b in books]

            # ---- Pass 1: Anytime TD odds, SERVER-side market filter ----
            # Restores script 33/api_odds.ts's proven-safe pattern: pass
            # `market=` explicitly so OpticOdds itself only returns the
            # confirmed anytime-TD markets. This is the fix for a real bug
            # (shipped, then caught live on 2026-09-20): when this call
            # shared the unfiltered game-lines response below and matched
            # client-side on a bare "touchdown" substring, it also picked up
            # unrelated long-shot side markets (e.g. "to score 2+ TDs",
            # "first/last touchdown scorer") for the same player. The
            # best-price logic (highest American odds wins) then let one of
            # those side-market prices silently overwrite the correct
            # primary-market price -- e.g. Derrick Henry showing +4499
            # (~2% implied) against a 71% model projection. Filtering the
            # market server-side removes the ambiguity entirely.
            td_params = list(base_params) + [("market", m) for m in TD_MARKETS]
            td_url = f"{base}/fixtures/odds?{qs(td_params)}"
            try:
                td_oj = get_json(td_url)
            except Exception as e:
                log(f"  TD odds fetch failed for batch {batch} / {books}: {e}")
                td_oj = {"data": []}
            for g in td_oj.get("data", []):
                for o in g.get("odds", []):
                    odds_seen += 1
                    market = str(o.get("market") or o.get("market_id") or "")
                    market_l = market.lower()
                    # Belt-and-suspenders client check mirroring script 33 --
                    # the server-side `market=` param is the real filter, this
                    # just guards against an unexpected response shape.
                    if not any(m.lower() in market_l for m in TD_MARKETS_L):
                        continue
                    line = str(o.get("selection_line") or "").lower()
                    if line in ("no", "under"):
                        continue
                    player = str(o.get("selection") or "").strip()
                    if not player:
                        player = re.sub(r"\s+(yes|no|over|under).*$", "", str(o.get("name") or ""), flags=re.I).strip()
                    try:
                        price = float(o.get("price"))
                    except (TypeError, ValueError):
                        continue
                    if not player or not price:
                        continue
                    k = norm_name(player)
                    if k not in td_players or price > td_players[k]["price"]:
                        td_players[k] = {"price": price, "book": str(o.get("sportsbook") or ""), "player_display": player}
                    pb = per_book_td.setdefault(k, {})
                    book = str(o.get("sportsbook") or "")
                    if book not in pb or price > pb[book]:
                        pb[book] = price

            # ---- Pass 2: game lines (spread/total), NO market filter ----
            # OpticOdds' exact NFL spread/total market-name strings are still
            # unverified in this environment, so this pass pulls every market
            # for the batch and filters client-side by substring -- but it is
            # now used ONLY for spread/total extraction, never for anytime-TD
            # odds, so it can no longer contaminate player prices above.
            lines_url = f"{base}/fixtures/odds?{qs(base_params)}"
            try:
                lines_oj = get_json(lines_url)
            except Exception as e:
                log(f"  game-line odds fetch failed for batch {batch} / {books}: {e}")
                continue
            for g in lines_oj.get("data", []):
                fid = g.get("id") or g.get("fixture_id")
                if fid is None:
                    continue
                for o in g.get("odds", []):
                    market = str(o.get("market") or o.get("market_id") or "")
                    market_l = market.lower()
                    # Same root cause as the TD-odds bug, applied to game lines:
                    # a bare "total"/"spread" substring also matches derivative
                    # sub-markets (team total, 1st-half total, quarter spread,
                    # alternate lines) that carry much smaller numbers than the
                    # real full-game line. Caught live on 2026-09-20 -- with no
                    # exclusion, BAL@NO's real 46.5 full-game total was getting
                    # overwritten by what looks like a team-total market (19.5),
                    # and GB@NYJ / KC@IND's real ~42.5/47.5 totals were getting
                    # overwritten by what looks like a quarter-total market
                    # (9.5, identical for both games -- too coincidental to be
                    # real full-game numbers). Deny-list the sub-market terms,
                    # since the exact primary-market name is still unverified.
                    if any(x in market_l for x in ("half", "quarter", "team", "alt", "1q", "2q", "3q", "4q", "1h", "2h")):
                        continue
                    is_total = "total" in market_l or "over/under" in market_l or "over under" in market_l
                    is_spread = "spread" in market_l or "point spread" in market_l or "handicap" in market_l
                    if not (is_total or is_spread):
                        continue
                    points = o.get("points")
                    if points is None:
                        continue
                    try:
                        points = float(points)
                    except (TypeError, ValueError):
                        continue
                    # Plausibility guard (belt-and-suspenders on top of the
                    # deny-list): real NFL full-game totals and spreads never
                    # fall outside these ranges, but a mis-tagged sub-market
                    # line often does.
                    if is_total and not (30 <= points <= 75):
                        continue
                    if is_spread and not (-35 <= points <= 35):
                        continue
                    sel_line = str(o.get("selection_line") or "").lower()
                    entry = lines_by_fixture.setdefault(fid, {})
                    if is_total and sel_line in ("over", ""):
                        entry["total"] = points
                    if is_spread:
                        home_abbr = fixture_teams.get(fid, (None, None))[0]
                        selection = str(o.get("selection") or "")
                        # keep the HOME team's spread number specifically
                        if home_abbr and home_abbr.lower() in selection.lower():
                            entry["spread_home"] = points
                        elif sel_line == "home":
                            entry["spread_home"] = points

    def implied(american):
        return 100 / (american + 100) if american > 0 else -american / (-american + 100)

    for k in td_players:
        prices = sorted(implied(v) for v in per_book_td.get(k, {}).values())
        if prices:
            n = len(prices)
            mid = prices[(n - 1) // 2] if n % 2 else (prices[n // 2 - 1] + prices[n // 2]) / 2
            td_players[k]["consensus"] = round(mid, 4)
            td_players[k]["n"] = n
        td_players[k]["books"] = [{"book": b, "price": p} for b, p in per_book_td.get(k, {}).items()]

    n_lines = sum(1 for v in lines_by_fixture.values() if "total" in v and "spread_home" in v)
    log(f"OpticOdds: odds rows scanned {odds_seen}; anytime-TD players matched {len(td_players)}; "
        f"complete game lines (total+spread) parsed for {n_lines}/{len(wk_fixtures)} fixtures.")
    return wk_fixtures, lines_by_fixture, td_players


def build_live_game_lines(week, wk_fixtures, lines_by_fixture, sched_wk):
    """Per-team live (total_line, spread_line, is_home) dict, falling back to the
    static schedules_2026.csv line for any game OpticOdds didn't give us."""
    fixture_by_teams = {}
    for f in wk_fixtures:
        try:
            home = f["home_competitors"][0]["abbreviation"]
            away = f["away_competitors"][0]["abbreviation"]
            fixture_by_teams[(home, away)] = f["id"]
        except (KeyError, IndexError):
            continue

    out = {}
    for _, g in sched_wk.iterrows():
        home, away = g["home_team"], g["away_team"]
        total_line, spread_line = g["total_line"], g["spread_line"]
        fid = fixture_by_teams.get((home, away))
        live = lines_by_fixture.get(fid, {}) if fid else {}
        if "total" in live:
            total_line = live["total"]
        if "spread_home" in live:
            spread_line = live["spread_home"]
        out[home] = {"total_line": total_line, "spread_line": spread_line, "is_home": True, "game_id": g["game_id"]}
        # away side implied uses the SAME total/spread pair, just the away formula (see recompute below)
        out[away] = {"total_line": total_line, "spread_line": spread_line, "is_home": False, "game_id": g["game_id"]}
    return out


# =====================================================================
# RECOMPUTE -- lambda_team -> pace_scalar -> bottom-up -> top-down -> blend
# =====================================================================
def recompute(week, constants, lam_static, slate_raw, profile, calib, rec_def, rush_def,
              qb_influence_by_team, team_pace, sched_wk, live_lines, live_injuries):
    intercept = constants["intercept"]
    w_implied = constants["w_implied"]
    w_trailing = constants["w_trailing"]
    pts_to_td_ratio = constants["pts_to_td_ratio"]

    # ---- team-level lambda, using LIVE implied_total + STATIC trailing history ----
    lam = lam_static.copy()
    lam["implied_total"] = lam["team"].map(lambda t: live_lines[t]["total_line"] / 2 +
                                            (live_lines[t]["spread_line"] / 2 if live_lines[t]["is_home"]
                                             else -live_lines[t]["spread_line"] / 2))
    lam["implied_tds"] = lam["implied_total"] / pts_to_td_ratio
    lam["lambda_team"] = np.maximum(0.1, intercept + w_implied * lam["implied_tds"] + w_trailing * lam["trailing_off_tds"])
    lam["lambda_rush"] = lam["lambda_team"] * lam["rush_share"]
    lam["lambda_rec"] = lam["lambda_team"] * (1 - lam["rush_share"])

    # ---- game/team context for pace_scalar ----
    league_avg_pace = team_pace["pace_plays_per_game"].mean()
    league_avg_total_2026 = sched_wk.assign(
        total_line=sched_wk["game_id"].map(lambda gid: next(
            (live_lines[t]["total_line"] for t, v in live_lines.items() if v["game_id"] == gid), None))
    )["total_line"].astype(float).mean()

    slate = slate_raw.copy()
    # "position" (not "pos_abb") is the cleaned RB/WR/TE/QB grouping the weekly
    # pipeline uses -- pos_abb is None for QBs and splits fullbacks out to "FB"
    # separately, which would silently drop QBs from the rush-share pool below.
    slate = slate[slate["position"].isin(SKILL_POS_ABBR | {"QB"})].copy()

    game_map = sched_wk.set_index("game_id")
    home_map = sched_wk.set_index("home_team")["game_id"]
    away_map = sched_wk.set_index("away_team")["game_id"]

    def opp_of(team):
        if team in home_map.index:
            gid = home_map[team]
            return game_map.loc[gid, "away_team"], gid, True, game_map.loc[gid, "roof"], game_map.loc[gid, "div_game"]
        if team in away_map.index:
            gid = away_map[team]
            return game_map.loc[gid, "home_team"], gid, False, game_map.loc[gid, "roof"], game_map.loc[gid, "div_game"]
        return None, None, None, None, None

    opp_info = slate["team"].map(opp_of)
    slate["opponent_team"] = opp_info.map(lambda x: x[0])
    slate["game_id"] = opp_info.map(lambda x: x[1])
    slate["is_home"] = opp_info.map(lambda x: x[2])
    slate["roof"] = opp_info.map(lambda x: x[3])
    slate["div_game"] = opp_info.map(lambda x: x[4])
    slate = slate.dropna(subset=["game_id"])

    # =================================================================
    # Injury, two layers:
    #
    #   1. KNOWN (static, from the last weekly pipeline run): the
    #      already-computed zero-out + damping-fraction role bump for
    #      whoever was Out/IR as of that build -- injury_sidelined_matched
    #      / injury_role_bumps. This is real "next man up" reassignment
    #      (rank-based eligibility, damping fractions) and must stay
    #      applied every hour, or an hourly run would silently regress
    #      an already-correct role bump back to its un-bumped baseline
    #      the moment it re-reads the raw (pre-injury) slate.
    #   2. LIVE DELTA (this run only): anyone ESPN newly lists as Out/IR
    #      who ISN'T already in that known list gets their own volume
    #      zeroed immediately -- no beneficiary reassignment yet (that
    #      judgment call still needs a weekly-pipeline re-run), but the
    #      number stops overstating a player who is out.
    #
    # A player ESPN no longer lists as Out/IR only "returns" once a new
    # weekly pipeline run drops them from the known list -- this script
    # never un-applies a KNOWN role bump on its own.
    # =================================================================
    try:
        known_sidelined = pd.read_csv(f"{OUT}/injury_sidelined_matched_2026wk{week}.csv")
    except FileNotFoundError:
        known_sidelined = pd.DataFrame(columns=["gsis_id"])
    known_sidelined_ids = set(known_sidelined["gsis_id"].dropna())

    sidelined_info = known_sidelined.drop_duplicates("gsis_id").set_index("gsis_id")[["injury", "status"]].rename(
        columns={"injury": "sidelined_injury", "status": "sidelined_status"}) if len(known_sidelined) else pd.DataFrame()
    if len(sidelined_info):
        slate = slate.merge(sidelined_info, left_on="gsis_id", right_index=True, how="left")
    slate["is_sidelined_known"] = slate["gsis_id"].isin(known_sidelined_ids)
    zero_cols = ["final_L4_targets", "final_L4_target_share", "final_L4_carries", "final_rush_share"]
    slate.loc[slate["is_sidelined_known"], zero_cols] = 0.0

    try:
        bumps = pd.read_csv(f"{OUT}/injury_role_bumps_2026wk{week}.csv")
    except FileNotFoundError:
        bumps = pd.DataFrame(columns=["beneficiary_gsis_id", "target_share_bump", "rush_share_bump"])
    bumps_by_id = bumps.drop_duplicates("beneficiary_gsis_id").set_index("beneficiary_gsis_id") if len(bumps) else pd.DataFrame()
    if len(bumps_by_id):
        slate = slate.merge(
            bumps_by_id[["target_share_bump", "rush_share_bump"]], left_on="gsis_id", right_index=True, how="left")
    else:
        slate["target_share_bump"] = np.nan
        slate["rush_share_bump"] = np.nan
    slate["target_share_bump"] = slate["target_share_bump"].fillna(0.0)
    slate["rush_share_bump"] = slate["rush_share_bump"].fillna(0.0)

    _pre_target_share = slate["final_L4_target_share"].replace(0, np.nan)
    _pre_rush_share = slate["final_rush_share"].replace(0, np.nan)
    target_scale = (1 + slate["target_share_bump"] / _pre_target_share).fillna(1.0).clip(upper=4.0)
    rush_scale = (1 + slate["rush_share_bump"] / _pre_rush_share).fillna(1.0).clip(upper=4.0)
    slate.loc[(slate["target_share_bump"] > 0) & _pre_target_share.isna(), "final_L4_target_share"] = slate["target_share_bump"]
    slate.loc[(slate["rush_share_bump"] > 0) & _pre_rush_share.isna(), "final_rush_share"] = slate["rush_share_bump"]
    has_share = _pre_target_share.notna()
    slate.loc[has_share, "final_L4_target_share"] = slate.loc[has_share, "final_L4_target_share"] + slate.loc[has_share, "target_share_bump"]
    slate.loc[has_share, "final_L4_targets"] = slate.loc[has_share, "final_L4_targets"] * target_scale[has_share]
    has_rush = _pre_rush_share.notna()
    slate.loc[has_rush, "final_rush_share"] = slate.loc[has_rush, "final_rush_share"] + slate.loc[has_rush, "rush_share_bump"]
    slate.loc[has_rush, "final_L4_carries"] = slate.loc[has_rush, "final_L4_carries"] * rush_scale[has_rush]

    # ---- live delta: newly Out/IR per ESPN, not yet in the known list ----
    sidelined_now = set(live_injuries.loc[
        live_injuries["status"].isin(WONT_PLAY_ESPN), "espn_id"].dropna())
    slate["is_sidelined_live_new"] = slate["espn_id"].isin(sidelined_now) & ~slate["is_sidelined_known"]
    slate.loc[slate["is_sidelined_live_new"], zero_cols] = 0.0
    slate["is_sidelined_live"] = slate["is_sidelined_known"] | slate["is_sidelined_live_new"]
    if slate["is_sidelined_live_new"].any():
        log(f"NEW live injury zero-out (ESPN Out/IR, not yet in the known weekly list): "
            f"{', '.join(slate.loc[slate['is_sidelined_live_new'], 'full_name'])}")

    # ---- QB influence (static per team for the week; already fitted) ----
    slate["qb_influence_on_rec_td"] = slate["team"].map(qb_influence_by_team).fillna(0.0)

    # ---- bottom-up rate x volume ----
    slate = slate.merge(profile[["gsis_id", "rec_raw_partial", "rush_raw_partial"]], on="gsis_id", how="left")
    slate["has_rec_profile"] = slate["rec_raw_partial"].notna()
    slate["has_rush_profile"] = slate["rush_raw_partial"].notna()
    slate["opp_rec_def_influence"] = slate["opponent_team"].map(rec_def)
    slate["opp_rush_def_influence"] = slate["opponent_team"].map(rush_def)

    rec_int, rec_mult = calib.loc["ReceivingTDs", ["intercept", "multiplier"]]
    rush_int, rush_mult = calib.loc["RushingTDs", ["intercept", "multiplier"]]
    slate["rec_td_rate"] = (rec_int + rec_mult * (
        slate["rec_raw_partial"].fillna(0) + slate["opp_rec_def_influence"].fillna(0)
        + slate["qb_influence_on_rec_td"])).clip(lower=0.0)
    slate["rush_td_rate"] = (rush_int + rush_mult * (
        slate["rush_raw_partial"].fillna(0) + slate["opp_rush_def_influence"].fillna(0))).clip(lower=0.0)
    slate.loc[~slate["has_rec_profile"], "rec_td_rate"] = np.nan
    slate.loc[~slate["has_rush_profile"], "rush_td_rate"] = np.nan

    slate["team_total_line"] = slate["team"].map(lambda t: live_lines[t]["total_line"])
    slate["pace_scalar"] = (slate["team"].map(team_pace.set_index("team")["pace_plays_per_game"]) / league_avg_pace) * \
        (slate["team_total_line"] / league_avg_total_2026)
    slate["pace_scalar"] = slate["pace_scalar"].fillna(1.0)
    slate["proj_targets"] = slate["final_L4_targets"] * slate["pace_scalar"]
    slate["proj_carries"] = slate["final_L4_carries"] * slate["pace_scalar"]

    def prob_at_least_one(rate, volume):
        rate = rate.fillna(0.0).clip(0, 1)
        volume = volume.fillna(0.0).clip(lower=0)
        return 1 - (1 - rate) ** volume

    slate["p_rec_td"] = prob_at_least_one(slate["rec_td_rate"], slate["proj_targets"])
    slate["p_rush_td"] = prob_at_least_one(slate["rush_td_rate"], slate["proj_carries"])
    slate["p_anytime_td_bottomup"] = 1 - (1 - slate["p_rec_td"]) * (1 - slate["p_rush_td"])

    # ---- top-down: team lambda x normalized share ----
    slate = slate.merge(lam[["team", "lambda_team", "rush_share", "lambda_rush", "lambda_rec"]], on="team", how="left")
    team_wk_target_sum = slate.groupby("team")["final_L4_target_share"].transform("sum")
    team_wk_rush_sum = slate.groupby("team")["final_rush_share"].transform("sum")
    slate["target_share_norm"] = (slate["final_L4_target_share"] / team_wk_target_sum).where(team_wk_target_sum > 0, 0.0)
    slate["rush_share_norm"] = (slate["final_rush_share"] / team_wk_rush_sum).where(team_wk_rush_sum > 0, 0.0)
    slate["indiv_lambda_rec_topdown"] = slate["lambda_rec"] * slate["target_share_norm"]
    slate["indiv_lambda_rush_topdown"] = slate["lambda_rush"] * slate["rush_share_norm"]
    slate["indiv_lambda_topdown"] = slate["indiv_lambda_rec_topdown"].fillna(0) + slate["indiv_lambda_rush_topdown"].fillna(0)
    slate["p_anytime_td_topdown"] = 1 - np.exp(-slate["indiv_lambda_topdown"])

    W_TOPDOWN = 0.55
    slate["p_final_blend"] = W_TOPDOWN * slate["p_anytime_td_topdown"] + (1 - W_TOPDOWN) * slate["p_anytime_td_bottomup"]
    # a live-Out player never shows a live number above a floor, whatever the (now-zeroed) math implies
    slate.loc[slate["is_sidelined_live"], ["p_final_blend", "p_anytime_td_topdown", "p_anytime_td_bottomup"]] = 0.01

    return slate, lam


def attach_live_odds(slate, td_odds):
    slate = slate.copy()
    slate["name_norm"] = slate["full_name"].map(norm_name)
    if not td_odds:
        return slate  # keep whatever odds columns already existed upstream (none here -- patched onto dashboard instead)
    rows = []
    for k, d in td_odds.items():
        rows.append({"name_norm": k, "market_price": d.get("price"), "market_book": d.get("book"),
                     "market_consensus": d.get("consensus"), "market_n_books": d.get("n")})
    odds_df = pd.DataFrame(rows)
    slate = slate.merge(odds_df, on="name_norm", how="left")
    slate["has_market_odds"] = slate["market_consensus"].notna()
    slate["plus_ev_edge"] = slate["p_final_blend"] - slate["market_consensus"]
    return slate


# =====================================================================
# PATCH data.json (the file break_the_plane_v5_live.html fetches at runtime)
# =====================================================================
def patch_dashboard(dashboard, recomputed, lam):
    by_id = recomputed.set_index("gsis_id")
    lam_by_team = lam.set_index("team")
    updated_players = []
    for p in dashboard["players"]:
        pid = p["id"]
        if pid not in by_id.index:
            updated_players.append(p)  # not in the live-recomputed skill-position universe (shouldn't happen); keep as-is
            continue
        row = by_id.loc[pid]
        p = dict(p)
        p["p_final"] = round(float(row["p_final_blend"]), 4)
        p["p_topdown"] = round(float(row["p_anytime_td_topdown"]), 4)
        p["p_bottomup"] = round(float(row["p_anytime_td_bottomup"]), 4)
        p["lambda_team"] = round(float(lam_by_team.loc[row["team"], "lambda_team"]), 4) if row["team"] in lam_by_team.index else p.get("lambda_team")
        p["total"] = float(row["team_total_line"]) if pd.notna(row.get("team_total_line")) else p.get("total")
        if "market_price" in row and pd.notna(row.get("market_price")):
            p["market_price"] = row["market_price"]
            p["market_book"] = row["market_book"]
            p["market_consensus"] = row["market_consensus"]
            p["market_n_books"] = int(row["market_n_books"]) if pd.notna(row["market_n_books"]) else p.get("market_n_books")
            p["has_market_odds"] = True
            p["plus_ev_edge"] = round(float(row["plus_ev_edge"]), 4) if pd.notna(row.get("plus_ev_edge")) else p.get("plus_ev_edge")
        if bool(row.get("is_sidelined_live")):
            p["liveOut"] = True
        elif "liveOut" in p:
            p["liveOut"] = False
        updated_players.append(p)

    # re-rank by the freshly blended probability, skill positions and QBs alike
    updated_players.sort(key=lambda p: p.get("p_final", 0.0), reverse=True)
    for i, p in enumerate(updated_players, start=1):
        p["rank"] = i

    dashboard = dict(dashboard)
    dashboard["players"] = updated_players
    dashboard["updated_at"] = pd.Timestamp.utcnow().isoformat() + "Z"
    return dashboard


def sanitize_for_json(obj):
    """Recursively replace NaN/Infinity with None. Python's json module writes
    bare `NaN`/`Infinity` tokens by default (valid Python float repr, invalid
    JSON) -- the OLD build-time __DATA_JSON__ = {...} bake-in got away with
    this because that's raw JS source (NaN is a real identifier there), but
    the new runtime fetch('data.json') does a strict JSON.parse, which throws
    on the first bare NaN it hits. This is a genuine latent bug in the export
    pipeline's scoreBreakdown math (a rookie/no-track-record player's
    "opportunity"/"composite" score dividing by a zero sample size) -- it
    predates this refresh script and previously did no visible harm.
    Sanitizing here fixes the immediate breakage without touching script 37;
    the underlying scoreBreakdown NaN is worth a real fix in that script."""
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, default=2)
    args = ap.parse_args()
    week = args.week

    key = os.environ.get("ODDSJAM_KEY")

    constants = json.load(open(f"{OUT}/model_constants_2026.json"))
    lam_static = pd.read_csv(f"{OUT}/lambda_2026wk{week}.csv")[["team", "trailing_off_tds", "rush_share"]]
    slate_raw = pd.read_parquet(f"{OUT}/slate_2026wk{week}_players_raw.parquet")
    profile = pd.read_csv(f"{OUT}/player_efficiency_profile.csv")
    calib = pd.read_csv(f"{OUT}/models_calibration.csv").set_index("stat")
    rec_def = pd.read_csv(f"{OUT}/defteam_rec_td_influence.csv", index_col=0)["Influence"]
    rush_def = pd.read_csv(f"{OUT}/defteam_rush_td_influence.csv", index_col=0)["Influence"]
    team_pace = pd.read_csv(f"{OUT}/team_pace_extracted.csv")

    # starting-QB influence, static for the week -- a small extracted CSV
    # (built once by the weekly pipeline) rather than depending on the full
    # final parquet, which this script's whole job is to stop being the
    # source of truth for.
    qb_influence_by_team = (
        pd.read_csv(f"{OUT}/qb_influence_2026wk{week}.csv")
        .set_index("team")["qb_influence_on_rec_td"].to_dict())

    sched_2026 = pd.read_csv(f"{ND}/schedules_2026.csv")
    sched_wk = sched_2026[(sched_2026["season"] == SEASON) & (sched_2026["week"] == week)].copy()

    live_injuries = fetch_espn_injuries()
    wk_fixtures, lines_by_fixture, td_odds = fetch_opticodds(week, key, sched_wk)
    live_lines = build_live_game_lines(week, wk_fixtures, lines_by_fixture, sched_wk)

    recomputed, lam = recompute(
        week, constants, lam_static, slate_raw, profile, calib, rec_def, rush_def,
        qb_influence_by_team, team_pace, sched_wk, live_lines, live_injuries)
    recomputed = attach_live_odds(recomputed, td_odds)

    dashboard_path = f"{OUT}/dashboard_2026wk{week}.json"
    data_json_path = f"{SITE}/data.json"
    seed_path = data_json_path if os.path.exists(data_json_path) else dashboard_path
    dashboard = json.load(open(seed_path))

    patched = sanitize_for_json(patch_dashboard(dashboard, recomputed, lam))
    with open(data_json_path, "w") as f:
        json.dump(patched, f)
    log(f"Wrote {data_json_path}: {len(patched['players'])} players, updated_at={patched['updated_at']}")


if __name__ == "__main__":
    main()
