"""Compare the five-session CL signal with and without Cushing-specific information."""

import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta

from backtest_wti_cushing_spread import COSTS, statistics
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS, inputs, predictions
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed

STEM = "cushing_regional_cl"
HURDLE = 170
INVENTORIES = ("cushing_crude", "padd2_commercial_crude", "us_commercial_crude")
PHASES = ("development", "validation", "holdout")
MODELS = ("regional", "regional_cushing", "regional_optical", "cushing_only", "always_long_spread", "no_trade")


def inventory_features():
    vintage = defaultdict(dict)
    for r in read("eia_control_vintage"):
        if r["metric"] not in INVENTORIES:
            continue
        sample = vintage[r["release_date"],r["report_date"]]
        if r["metric"] in sample or r["unit"]!="thousand_bbl" or r["table_number"]!="4":
            raise ValueError("Invalid Table 4 stock series")
        sample[r["metric"]] = (float(r["value"]),r["source_url"])
    for issue, clock in indexed("eia_release_clock").items():
        latest = clock["latest_week_in_issue"]
        # The July 2019 PDF recovery uses the same date for both clock fields.
        # Both original stock rows are present under the exact observation weeks.
        prior_week = (date.fromisoformat(latest)-timedelta(days=7)).isoformat()
        current, previous = vintage[issue,latest], vintage[issue,prior_week]
        if current.keys()!=set(INVENTORIES) or previous.keys()!=set(INVENTORIES):
            raise ValueError(f"Missing original stock levels on {issue}")
        sources = {v[1] for v in (*current.values(),*previous.values())}
        if len(sources)!=1:
            raise ValueError(f"Mix of Table 4 vintages on {issue}")
        cushing, midwest, country = [(current[name][0]-previous[name][0])/1000 for name in INVENTORIES]
        yield issue, (cushing,midwest-cushing,country-midwest,sources.pop(),
                      datetime.fromisoformat(clock["conservative_postrelease_from_utc"]))


def replay():
    book = dict(inventory_features())
    optical = indexed("cushing_coverage_nowcast_forecasts")
    seen = []
    for r in inputs():
        issue = r["release_date"]
        published, midwest, rest_us, vintage, available = book[issue]
        info = optical[issue]
        if (not math.isclose(published,r["x"][0],abs_tol=1e-10) or available>r["decision"]
                or datetime.fromisoformat(info["forecast_cutoff_utc"])>=available):
            raise ValueError(f"Release chronology or Cushing label differs on {issue}")
        carry, carry_change = r["x"][2:4]
        adjustment = float(info["terminal_coverage_ml"])-float(info["public_ensemble"])
        regional = (carry,carry_change,midwest,rest_us)
        cushing = (published,r["x"][1],r["x"][4])
        photo = (adjustment,adjustment*carry)
        current = r | {"x":(*regional,*cushing,*photo), "cushing_only_features":r["x"][:5]}
        train = [q for q in seen if q["exit_at"]<r["decision"]][-HISTORY:]
        seen.append(current)
        if len(train)<MIN_WINDOWS:
            continue
        fits = [predictions(train,current,n) for n in (4,7,9)]
        old_train = [q | {"x":q["cushing_only_features"]} for q in train]
        expected = predictions(old_train,r,5)
        for method, estimate in zip(MODELS,(*fits,expected,None,None),strict=True):
            side = int(estimate>HURDLE)-int(estimate<-HURDLE) if estimate is not None else int(method=="always_long_spread")
            gross = r["long_gross"]*side
            yield {k:r[k] for k in ("study_phase","release_date","entry_session","exit_session","near_month","far_month","source")} | dict(
                method=method, near_side=side, expected_long_gross_usd=estimate, gross_usd=gross,
                decision_at_utc=r["decision"].isoformat(), latest_training_exit=train[-1]["exit_at"].isoformat(),
                inventory_source=vintage, cushing_million_bbl=published, midwest_ex_cushing_million_bbl=midwest,
                us_ex_midwest_million_bbl=rest_us, optical_revision_million_bbl=adjustment,
                **{f"cost_{cost}_usd":cost*abs(side) for cost in COSTS},
                **{f"net_{cost}_usd":gross-cost*abs(side) for cost in COSTS})


def summarize(rows):
    stress = [r["net_170_usd"] for r in rows]
    return statistics(rows) | dict(
        stressed_mean_net_ci_95=block_bootstrap_mean(stress),
        net_excluding_largest_win_at_170_usd=sum(stress)-max(0,max(stress)),
        net_excluding_2026_04_01_at_170_usd=sum(r["net_170_usd"] for r in rows if r["release_date"]!="2026-04-01"))


def main():
    rows = list(replay())
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"],r["method"]].append(r)
    benchmark = {(r["study_phase"],r["release_date"]):r for r in read("cushing_cl_regime_strategy_trades")
                 if r["method"]=="public" and r["hurdle_usd"]==str(HURDLE)}
    rebuilt = [r for r in rows if r["method"]=="cushing_only"]
    if len(rebuilt)!=len(benchmark) or any(r["net_170_usd"]!=int(benchmark[r["study_phase"],r["release_date"]]["net_170_usd"]) for r in rebuilt):
        raise ValueError("Regional comparison changed the public Cushing-only benchmark")
    paired = {}
    for p in PHASES:
        paired[p] = {}
        for comparison, control in (("regional_cushing","regional"),("regional_optical","regional_cushing")):
            a,b = groups[p,comparison],groups[p,control]
            if [r["release_date"] for r in a]!=[r["release_date"] for r in b]:
                raise ValueError("Paired comparison uses different market windows")
            delta = [x["net_170_usd"]-y["net_170_usd"] for x,y in zip(a,b,strict=True)]
            paired[p][f"{comparison}_vs_{control}"] = dict(net_difference_usd=sum(delta), mean_difference_ci_95=block_bootstrap_mean(delta))
    metrics = {p:{m:summarize(groups[p,m]) for m in MODELS} for p in PHASES}
    result = dict(status="exploratory_replay_on_previously_examined_history", phases=metrics, paired_stress=paired)
    export(f"{STEM}_trades",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    for p, values in metrics.items():
        print(p,{m:(r["trades"],r["net_50_usd"],r["net_170_usd"],r["net_excluding_largest_win_at_170_usd"]) for m,r in values.items()})


if __name__=="__main__":
    main()
