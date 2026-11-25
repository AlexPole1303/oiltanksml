"""Verify the price basis of the eight archived equity marks needed by holdout windows."""
from contextlib import closing
from datetime import datetime
import json
import math
from zoneinfo import ZoneInfo

from build_market_event_panel import CORE
from build_study_execution_inputs import BENCHMARK, VENUE
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import SOURCE, export, read_only
from collect_equity_reference import chart_exchange_daily_rows, yahoo_rows
from collect_public import RAW, ROOT

STEM = "cushing_holdout_equity_price_bridges"
TWS_BASIS = "ibkr_trades_split_adjusted"
EASTERN = ZoneInfo("America/New_York")
FIELDS = ("open","high","low","close")
TAIL = "2026-11-11"
OKE = "2026-09-10"


def timestamps(db):
    return {d:(datetime.fromisoformat(o),datetime.fromisoformat(c)) for d,o,c in db.execute("""
        SELECT session_date,open_utc,close_utc FROM equity_session_calendar WHERE venue=? ORDER BY session_date""",(VENUE,))}


def tail_bridge(symbol, sessions, actions):
    prior = RAW/f"tws/{symbol}/TRADES_2026-11-10.json"
    connector = RAW/f"equity_reference/ibkr_tail/{symbol}_{TAIL}.json"
    independent = RAW/f"equity_reference/yahoo_tail/{symbol}_{TAIL}.json"
    tws,web,yahoo = (json.loads(p.read_bytes()) for p in (prior,connector,independent))
    contract, request, bars = tws["contract"],web["request"],web["response"]
    if (contract["symbol"]!=symbol or contract["currency"]!="USD" or contract["conId"]!=request["contract_id"]
        or request["security_type"]!="STK" or request["outside_rth"] or tws["what_to_show"]!="TRADES"
        or not tws["regular_hours"] or bars["chart_step"]!=86400 or bars["source"]!="Last"
        or datetime.fromisoformat(web["downloaded_at_utc"])<=sessions[TAIL][1]):
        raise ValueError(f"Wrong chart identity, field type, or capture time for {symbol}")
    times = [datetime.fromisoformat(t) for t in bars["time"]]
    days = [t.astimezone(EASTERN).date().isoformat() for t in times]
    calendar = list(sessions)
    end = calendar.index(TAIL)
    if (days!=calendar[end-3:end+1] or any(t!=sessions[d][0] for t,d in zip(times,days,strict=True))
        or any(len(bars[k])!=len(days) for k in (*FIELDS,"volume"))):
        raise ValueError(f"Wrong regular-session chart boundary for {symbol}")
    history = {r["date"]:r for r in tws["bars"]}
    if any(not math.isclose(bars[k][i],history[d][k],abs_tol=1e-9,rel_tol=0) for i,d in enumerate(days[:-1]) for k in FIELDS):
        raise ValueError(f"Connector OHLC does not reproduce TWS for {symbol}")
    result, = yahoo["chart"]["result"]
    meta = result["meta"]
    if (meta["symbol"],meta["currency"],meta["exchangeTimezoneName"])!=(symbol,"USD","America/New_York"):
        raise ValueError(f"Independent chart identity differs for {symbol}")
    dates = [datetime.fromtimestamp(t,EASTERN).date().isoformat() for t in result["timestamp"]]
    quote, = result["indicators"]["quote"]
    if result.get("events",{}).get("splits") or any(days[0]<=d<=TAIL for d in actions):
        raise ValueError(f"Share basis changed on the quote bridge for {symbol}")
    j = dates.index(TAIL)
    if any(abs(bars[k][-1]-quote[k][j])>.005 for k in ("open","close")):
        raise ValueError(f"Independent opening or closing price differs for {symbol}")
    values = tuple(bars[k][-1] for k in (*FIELDS,"volume"))
    return evidence(symbol,TAIL,"ibkr_connector_tail",connector,independent,prior,contract["conId"],days[:-1],
                    "matched_conid_and_three_tws_ohlc_bars_independent_open_and_close",values)


def evidence(symbol,day,provider,raw,independent,tws,conid,overlap,check,values):
    op,high,low,close,volume = values
    if not all(math.isfinite(v) for v in values) or not 0<low<=min(op,close)<=max(op,close)<=high or volume<=0:
        raise ValueError(f"Invalid market observation at {symbol}/{day}")
    return dict(instrument=symbol,session_date=day,provider=provider,source_file=str(raw.relative_to(ROOT)),
        independent_source_file=str(independent.relative_to(ROOT)),tws_source_file=str(tws.relative_to(ROOT)),
        tws_conid=conid,overlapping_tws_sessions="|".join(overlap),check=check),values


def oke_bridge(sessions, actions):
    root = RAW/"equity_reference"
    raw,independent,tws = root/"oke_yahoo_history_2016_2026.html",root/"oke_chartexchange_daily.html",RAW/"tws/OKE/TRADES_2026-11-10.json"
    yahoo,cash = yahoo_rows(raw.read_text())
    adjacent = list(sessions)
    i = adjacent.index(OKE)
    before,after = adjacent[i-1],adjacent[i+1]
    checked,_ = chart_exchange_daily_rows(independent,{before,OKE,after})
    t = json.loads(tws.read_bytes())
    history = {r["date"]:r for r in t["bars"]}
    if (t["contract"]["symbol"]!="OKE" or t["contract"]["currency"]!="USD" or not t["regular_hours"]
        or t["what_to_show"]!="TRADES" or any(before<=d<=after for d in actions) or OKE in cash
        or yahoo[OKE][:4]!=checked[OKE][:4]
        or any(float(yahoo[d][3])!=history[d]["close"] or float(checked[d][3])!=history[d]["close"] for d in (before,after))
        or any(float(yahoo[OKE][j])!=history[OKE][k] for j,k in enumerate(FIELDS[:-1]))
        or history[OKE]["close"]!=history[before]["close"] or history[OKE]["close"]==float(yahoo[OKE][3])):
        raise ValueError("The independently verified ONEOK close no longer reproduces")
    return evidence("OKE",OKE,"yahoo_finance",raw,independent,tws,t["contract"]["conId"],[before,after],
                    "two_independent_ohlc_bars_agree_tws_close_repeats_prior_session",tuple(map(float,yahoo[OKE])))


def bridges():
    rows = read_csv("study_equity_action_events")
    splits = {s:{r["first_relevant_date"] for r in rows if r["instrument"]==s and r["event_type"]=="share_split"}
              for s in (*CORE,BENCHMARK)}
    with closing(read_only(SOURCE)) as db:
        sessions = timestamps(db)
        evidence_rows = [oke_bridge(sessions,splits["OKE"])]
        evidence_rows.extend(tail_bridge(s,sessions,splits[s]) for s in sorted(splits))
        for r,values in evidence_rows:
            key = r["instrument"],r["session_date"]
            stored = db.execute("""SELECT provider,source_file,open,high,low,close,volume FROM market_study_quote
                WHERE instrument=? AND session_date=? AND currency='USD' AND venue=?""",(*key,VENUE)).fetchone()
            if stored is None or stored[:2]!=(r["provider"],r["source_file"]) or any(
                not math.isclose(a,b,rel_tol=0,abs_tol=1e-8) for a,b in zip(values,stored[2:],strict=True)):
                raise ValueError(f"Certified quote does not match the assembled quote database at {key}")
            yield r


def effective_basis(certificate,provider,reference,basis):
    if certificate:
        if (provider,reference)!=(certificate["provider"],certificate["source_file"]):
            raise ValueError("Quote certificate does not match the archived source")
        return TWS_BASIS
    return basis


def load_bridges():
    return {(r["instrument"],r["session_date"]):r for r in read_csv(STEM)}


def main():
    rows = list(bridges())
    export(STEM,rows)
    print(f"Verified {len(rows)} equity price-basis bridges across {len({r['session_date'] for r in rows})} sessions")


if __name__=="__main__":
    main()
