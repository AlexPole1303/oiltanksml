"""Break the research leader's payoff around the following EIA publication."""

import json
import math
from collections import defaultdict
from datetime import datetime

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_before_next_release import components
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT

STEM = "cushing_cl_rollover_attribution"
MODELS = (PUBLIC,SUPPLEMENT)


def replay():
    labels = {f"{m}_conditional":m for m in MODELS}
    rows = {(r["release_date"],labels[r["method"]]):r for r in read(f"{OUTER}_trades") if r["method"] in labels}
    trade_rows, attributions = [], []
    for r,_ in inputs():
        issue = r["release_date"]
        if (issue,PUBLIC) not in rows:
            continue
        boundary = r["next_report_clock"]
        if not boundary:
            raise ValueError(f"Missing following report clock on {issue}")
        index = r["last_before_next_release"]
        for method in MODELS:
            known = rows[issue,method]
            if (any(known[k]!=r[k] for k in ("near_month","far_month")) or any(
                datetime.fromisoformat(known[k])!=r[field] for k,field in (
                    ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at")))):
                raise ValueError(f"Different dated spread on {issue}/{method}")
            predicted = float(known["expected_long_move_usd"])
            full = trade(r,f"{method}_full",predicted,3)
            before = trade(r,f"{method}_before_next_report",predicted if index else 0.,index or 0)
            trade_rows.extend((full,before))
            if (full["gross_usd"]!=int(known["gross_usd"]) or full["near_side"]!=int(known["near_side"])
                    or full["net_170_usd"]!=int(known["net_170_usd"]) or before["near_side"] and before["crosses_next_eia"]):
                raise ValueError(f"Payoff or cutoff differs on {issue}/{method}")
            segments = components(r,full["near_side"])
            attributions.append(dict(release_date=issue,study_phase=r["study_phase"],method=method,
                near_month=r["near_month"],far_month=r["far_month"],near_side=full["near_side"],
                next_report_at_utc=boundary.isoformat(),entry_at_utc=full["entry_at_utc"],
                exit_at_utc=full["exit_at_utc"],previous_possible_exit_at_utc=before["exit_at_utc"] if index else None,
                held_through_next_report=bool(full["near_side"] and full["crosses_next_eia"]),
                net_170_usd=full["net_170_usd"],before_report_net_170_usd=before["net_170_usd"],
                incremental_net_170_usd=full["net_170_usd"]-before["net_170_usd"],**segments))
    return trade_rows,attributions


def main():
    trades, attribution = replay()
    groups = defaultdict(list)
    for r in trades:
        groups[r["study_phase"],r["method"]].append(r)
    report = dict(status="exploratory_on_reviewed_history",phases={})
    for p in PHASES:
        methods = report["phases"][p] = {}
        for m in MODELS:
            full,before = (groups[p,f"{m}_{suffix}"] for suffix in ("full","before_next_report"))
            traces = [r for r in attribution if r["study_phase"]==p and r["method"]==m]
            if [r["release_date"] for r in full]!=[r["release_date"] for r in before] or len(traces)!=len(full):
                raise ValueError(f"Different calendars on {p}/{m}")
            perf = {name:robustness(group) | daily_risk(group) | dict(net_340_usd=sum(
                r["gross_usd"]-340*abs(r["near_side"]) for r in group))
                for name,group in (("full",full),("before_report",before))}
            costs = [a["net_170_usd"]-b["net_170_usd"] for a,b in zip(full,before,strict=True)]
            methods[m] = perf | dict(positions_held_through_next_report=sum(r["held_through_next_report"] for r in traces),
                incremental_net_170_usd=sum(costs),paired_increment_95_usd=block_bootstrap_mean(costs),
                gross_by_segment_usd={field:sum(r[field] for r in traces) for field in (
                    "before_next_report","straddling_next_report","after_next_report")})
        print(p,{m:(v["full"]["net_170_usd"],v["before_report"]["net_170_usd"]) for m,v in methods.items()})
    export(f"{STEM}_trades",trades)
    export(f"{STEM}_segments",attribution)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
