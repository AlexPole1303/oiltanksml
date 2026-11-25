"""Freeze the historical CL winner's expectation and score it against public controls."""

import argparse
import json
import math
from datetime import date, datetime, timezone

from threadpoolctl import threadpool_limits

from audit_cushing_forward_wti_coverage import archives
from build_s1_reference_matches import DERIVED, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write
from evaluate_cushing_forward_public_cl import (
    CARRY_INTERACTION_METHOD, ENSEMBLE_INTERACTION_METHOD, REGISTERED, registered_decision,
)
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, risk_prediction
from fit_cushing_delayed_calendar import replay as market_calendar
from fit_cushing_outer_capture_nowcast import PUBLIC, STEM, SUPPLEMENT, replay, run_forecast
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from register_cushing_prerelease_forecast import digest, freeze, score as score_inventory
from score_cushing_forward_pair import ROUND_TRIP_COSTS, score as score_spread

POLICIES = (SUPPLEMENT, PUBLIC)
RULE = dict(entry_full_session=2, exit_full_session=5, months_apart=3,
            near_contracts=1, far_contracts=1, entry_hurdle_usd=HURDLE,
            scenario_round_trip_costs_usd=list(ROUND_TRIP_COSTS), overlapping_windows=False)


def inventory(issue, path):
    prefix = f"cushing_prerelease_forecast_{issue}"
    original_path = DERIVED/f"{prefix}.json"
    recovered_path = DERIVED/f"{prefix}_recovered_optical.json"
    original, recovered = (json.loads(p.read_bytes()) for p in (original_path,recovered_path))
    lineage = ("release_date", "week_ending", "information_cutoff_utc", "registration_deadline_utc")
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in lineage[-2:])
    if path.exists():
        record = json.loads(path.read_bytes())
        if (any(record[k] != original[k] for k in lineage) or record["rule"] != RULE
                or not cutoff <= datetime.fromisoformat(record["recorded_at_utc"]) < deadline
                or not datetime.fromisoformat(record["latest_training_label_utc"]) < cutoff
                or any(digest(ROOT/name) != checksum for name,checksum in record["sources_sha256"].items())):
            raise ValueError("The registered strategy or its source hashes differ")
        return record
    now = datetime.now(timezone.utc)
    if (not cutoff <= now < deadline or issue in indexed("eia_wpsr_forward_releases")
            or any(original[k] != recovered[k] for k in lineage)
            or not datetime.fromisoformat(recovered["recorded_at_utc"]) <= now
            or any(digest(DERIVED/name) != checksum for name,checksum in recovered["source_sha256"].items())):
        raise ValueError("The point forecast must be registered with matching inputs before publication")
    tiles = f"cushing_forward_optical_coverage_{issue}_tiles"
    for row in read(tiles):
        ready = row["l1c_with_detector_available_by_utc"]
        if (row["segment"] == "after_target" and int(row["clear_roofs"]) and ready
                and datetime.fromisoformat(ready) < cutoff):
            raise ValueError("Fit the eligible spanning pairs before using this optical panel")
    native_path = DERIVED/f"cushing_forward_native_optical_{issue}_summary.json"
    area = float(json.loads(native_path.read_bytes())["dated_roof_area_m2"])
    observed = [r | dict(area=r["volume_per_m"]*BARREL_M3*1e6, available_utc=r["source_available_utc"])
                for r in original["measured_pairs"]] + recovered["recovered_roofs"]
    if (not 0 < len(observed) == recovered["expanded_roofs"]
            or len({r["candidate_id"] for r in observed}) != len(observed)
            or any(datetime.fromisoformat(r["available_utc"]) >= cutoff for r in observed)):
        raise ValueError("The frozen optical panel has late or repeated observations")
    public = original["public_ensemble_kbbl"]/1000
    row = dict(cutoff=cutoff,known=deadline,public=public,week_ending=original["week_ending"],observed=observed,
               hub_area=area,area_fraction=math.fsum(r["area"] for r in observed)/area)
    state = {}
    with threadpool_limits(limits=1):
        for _ in replay(state=state,before=cutoff):
            pass
        expected, last_optical = run_forecast(state["history"][SUPPLEMENT],state["errors"],row,False,update=False)
    sources = [original_path,recovered_path,native_path,DERIVED/f"{tiles}.csv",
               *(DERIVED/f"{stem}.csv" for stem in (
                   f"{STEM}_forecasts",f"{STEM}_trades","cushing_outer_capture_pairs_supplement_fits",
               )), *(ROOT/"scripts"/name for name in (
                   "fit_cushing_outer_capture_nowcast.py", "fit_cushing_cl_conditional_response.py",
                   "fit_cushing_cl_volatility.py", "fit_cushing_regional_balance.py",
               ))]
    latest_training = max(r["known"] for r in state["errors"])
    now = datetime.now(timezone.utc)
    if not latest_training < cutoff <= now < deadline:
        raise ValueError("The prerelease deadline passed while building the forecast")
    record = {k:original[k] for k in lineage} | dict(recorded_at_utc=now.isoformat(),
        latest_training_label_utc=latest_training.isoformat(),latest_optical_training_label_utc=last_optical,
        estimates_kbbl={PUBLIC:1000*public,SUPPLEMENT:1000*expected},measured_roofs=len(observed),
        optical_input_policy="frozen_panel_without_eligible_after_target_observations",rule=RULE,
        sources_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources})
    freeze(path,record)
    return record


def estimate(history, current, method):
    train = [r | dict(x=replace_expectation(r,r[method])) for r in history][-HISTORY:]
    design = replace_expectation(current,current[method])
    point = risk_prediction(train,current | dict(x=design),len(design))
    context = [r | dict(base_estimate=r[f"{method}_base"],response_features=features(
        r["x"][2],r["volatility_usd"],r["x"][0]-r[method])) for r in train if f"{method}_base" in r]
    correction, support = conditional_residual(context,current | dict(response_features=features(
        current["x"][2],current["volatility_usd"],current["x"][0]-current[method]))) if len(context) >= MIN_WINDOWS else (0.,0.)
    return point, correction, support


def history():
    forecasts = indexed(f"{STEM}_forecasts")
    base = {(r["release_date"],r["method"]):r for r in read(f"{STEM}_trades") if r["method"] in POLICIES}
    _, _, cases = market_calendar()
    for r in cases:
        issue = r["release_date"]
        f = forecasts[issue]
        if (not datetime.fromisoformat(f["cutoff_utc"]) < datetime.fromisoformat(f["known_utc"]) <= r["decision"]
                or float(f["actual"]) != r["x"][0]):
            raise ValueError(f"Report clocks differ on {issue}")
        points = {m:base[issue,m] for m in POLICIES if (issue,m) in base}
        for q in points.values():
            if datetime.fromisoformat(q["settled_training_label_utc"]) >= r["decision"] or any(
                    datetime.fromisoformat(q[k]) != r[field] for k,field in (
                        ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at"))):
                raise ValueError(f"Market clocks differ on {issue}")
        yield r | {m:float(f[m]) for m in POLICIES} | {f"{m}_base":float(q["expected_long_move_usd"]) for m,q in points.items()}


def decisions(issue, record, path):
    now = datetime.now(timezone.utc)
    ledger = json.loads((DERIVED/"cushing_forward_public_cl_summary.json").read_bytes())["release_rows"]
    public, curve = (registered_decision(r,register=False,now=now) for method in
        (ENSEMBLE_INTERACTION_METHOD,CARRY_INTERACTION_METHOD)
        for r in ledger if r["release_date"] == issue and r["method"] == method)
    decision, deadline = (datetime.fromisoformat(public[k]) for k in ("decision_at_utc","registration_deadline_utc"))
    fields = ("release_date","decision_at_utc","entry_at","exit_at","near_month","far_month")
    if (any(public[k] != curve[k] for k in fields) or not public["eligible_nonoverlap"]
            or not curve["eligible_nonoverlap"] or not public["recorded_at_utc"] or not curve["recorded_at_utc"]
            or public["week_ending"] != record["week_ending"]):
        raise ValueError("The forward calendar has no matching timely public and curve controls")
    if not path.exists():
        if not decision <= now < deadline:
            raise ValueError("The second-session decision window is closed")
        published = indexed("eia_wpsr_forward_releases")[issue]
        if float(published["cushing_change_kbbl"])/1000 != public["actual_million_bbl"]:
            raise ValueError("The public decision uses a different inventory outcome")
        settled = [r for r in history() if r["exit_at"] < decision][-HISTORY:]
        if len(settled) < MIN_WINDOWS:
            raise ValueError("Missing settled training windows")
        actual, carry, change, sigma = (public[k] for k in ("actual_million_bbl","carry","carry_change","volatility_usd"))
        current = dict(decision=decision,x=(actual,0.,carry,change,0.,0.,0.),volatility_usd=sigma,
                       **{m:record["estimates_kbbl"][m]/1000 for m in POLICIES})
        stamp = datetime.now(timezone.utc)
        rows = {}
        for method in POLICIES:
            point, correction, support = estimate(settled,current,method)
            total = point+correction
            rows[method] = {k:public[k] for k in fields} | dict(recorded_at_utc=stamp.isoformat(),
                expected_long_move_usd=total,near_side=int(total>HURDLE)-int(total<-HURDLE),
                base_estimate_usd=point,correction_usd=correction,similar_report_weight=support)
        if datetime.now(timezone.utc) >= deadline:
            raise ValueError("The decision finished after the entry mark")
        freeze(path,dict(decisions=rows,rule=RULE,inventory_sha256=digest(path.with_name(f"{path.stem}_inventory.json")),
                         latest_training_exit_utc=settled[-1]["exit_at"].isoformat(),
                         training_windows=len(settled),first_release_sha256=published["source_sha256"],
                         control_sha256={m:digest(REGISTERED/f"{issue}_{m}.json")
                                         for m in (ENSEMBLE_INTERACTION_METHOD,CARRY_INTERACTION_METHOD)}))
    result = json.loads(path.read_bytes())
    if (result["rule"] != RULE or not datetime.fromisoformat(result["latest_training_exit_utc"]) < decision
            or result["inventory_sha256"] != digest(path.with_name(f"{path.stem}_inventory.json"))
            or result["first_release_sha256"] != indexed("eia_wpsr_forward_releases")[issue]["source_sha256"]
            or set(result["decisions"]) != set(POLICIES)
            or any(digest(REGISTERED/f"{issue}_{m}.json") != checksum for m,checksum in result["control_sha256"].items())):
        raise ValueError("Decision receipt differs from the registered strategy or report")
    for r in result["decisions"].values():
        value = r["expected_long_move_usd"]
        if (any(r[k] != public[k] for k in fields)
                or not decision <= datetime.fromisoformat(r["recorded_at_utc"]) < deadline
                or r["near_side"] != int(value>HURDLE)-int(value<-HURDLE)):
            raise ValueError("Decision receipt contains a different calendar or a late timestamp")
    return result["decisions"] | {"curve":curve}


def run(issue, *, scoring):
    path = DERIVED/f"cushing_forward_cl_satellite_{issue}.json"
    record = inventory(issue,path.with_name(f"{path.stem}_inventory.json"))
    original_path = DERIVED/f"cushing_prerelease_forecast_{issue}.json"
    outcome = score_inventory(original_path,json.loads(original_path.read_bytes()))
    result = dict(release_date=issue,status="awaiting_first_release",estimates_kbbl=record["estimates_kbbl"])
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(status="awaiting_registered_market_controls",actual_eia_change_kbbl=actual,
                      inventory_absolute_error_kbbl={m:abs(actual-v) for m,v in record["estimates_kbbl"].items()},
                      first_release_sha256=outcome["first_release_sha256"])
    controls_ready = all((REGISTERED/f"{issue}_{m}.json").is_file()
                         for m in (ENSEMBLE_INTERACTION_METHOD,CARRY_INTERACTION_METHOD))
    if controls_ready and outcome["status"] == "scored_first_release":
        result["status"] = "awaiting_market_window"
        models = decisions(issue,record,path)
        now = datetime.now(timezone.utc)
        _, marks = archives(as_of=now)
        calendar = {issue:[r for r in read("cushing_forward_wti_coverage_bars")
                            if r["release_date"] == issue and r["session_index"] != "1"]}
        for comparator in (PUBLIC,"curve"):
            comparison, = score_spread({issue:{m:models[m] for m in (SUPPLEMENT,comparator)}},calendar,marks,now,
                                      methods=(SUPPLEMENT,comparator))
            result[comparator] = comparison
        if result[PUBLIC]["market_window_complete"]:
            result["status"] = "window_scored"
    if scoring:
        write(path.with_name(f"{path.stem}_score.json"),(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    return result if scoring or outcome["status"] == "scored_first_release" else record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",required=True,type=date.fromisoformat)
    parser.add_argument("--score",action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.release.isoformat(),scoring=args.score),indent=2,allow_nan=False))
