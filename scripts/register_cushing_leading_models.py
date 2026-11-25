"""Register the best historical inventory and trading models for a paired forward test."""

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from threadpoolctl import threadpool_limits

from audit_cushing_forward_wti_coverage import archives
from audit_cushing_spot_vintages import inspect
from build_s1_reference_matches import DERIVED, read
from collect_public import ROOT, write
from evaluate_cushing_forward_public_cl import ENSEMBLE_INTERACTION_METHOD, REGISTERED, registered_decision
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE
from fit_cushing_late_pair_recovery import COMBINED
from fit_cushing_outer_capture_nowcast import PUBLIC, SUPPLEMENT, INTERNAL_CALIBRATION
from fit_cushing_public_state import basis, released_balances, spot_reference
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_storage_state import regional_adjustment, storage_coordinates
from fit_cushing_terminal_weekly_transfers import LIMITED_MARKET
from fit_cushing_volatility_neighbors import REGIONAL, STEM as VOLATILITY, archived_coordinates, inventory, volatility_forecast
from fit_cushing_volatility_neighbors import median_residual
from publication_clocks import EASTERN
from register_cushing_cl_satellite import RULE, decisions as satellite_decisions, history as market_history, inventory as satellite_inventory
from register_cushing_prerelease_forecast import digest, freeze, score as score_inventory
from register_cushing_public_fallback import latest_forward_balances
from score_cushing_forward_pair import score as score_spread

STEM = "cushing_forward_leading_models"
VOLATILITY_METHOD = f"{SUPPLEMENT}:volatility"
LINEAGE = ("release_date","week_ending","information_cutoff_utc","registration_deadline_utc")


def checked_hashes(proofs):
    if any(digest(ROOT/name)!=checksum for name,checksum in proofs.items()):
        raise ValueError("A frozen source differs from its recorded checksum")


def register(issue, path, original_path, trade_path, trade_record):
    original = json.loads(original_path.read_bytes())
    cutoff,deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if not cutoff<=datetime.now(timezone.utc)<deadline or issue in indexed("eia_wpsr_forward_releases"):
        raise ValueError("The leading models must be registered before the EIA release")
    prefix = original_path.stem
    calibrated = DERIVED/f"{prefix}_internal_calibration.json"
    recovered = DERIVED/f"{prefix}_recovered_optical.json"
    tile_path = DERIVED/f"cushing_forward_optical_coverage_{issue}_tiles.csv"
    optical = json.loads(calibrated.read_bytes())
    recovered_data = json.loads(recovered.read_bytes())
    checked_hashes(optical["source_sha256"])
    if any(r[k]!=original[k] for r in (optical,recovered_data,trade_record) for k in LINEAGE):
        raise ValueError("The registered models used different issue clocks")
    observations = original["measured_pairs"]+recovered_data["recovered_roofs"]
    terminals = {q["terminal_id"] for q in observations}
    if len(terminals)<=3 or len(observations)!=optical["measured_roofs"] or len({q["candidate_id"] for q in observations})!=len(observations):
        raise ValueError("Recompute the current release's terminal adjustment before registering")
    # This forward panel has two observation passes split across adjacent
    # tiles. A new eligible pass requires recalculating the multiframe term.
    captures = set()
    for q in read(tile_path.stem):
        stamp = q["l1c_with_detector_available_by_utc"]
        if int(q["clear_roofs"]) and stamp and datetime.fromisoformat(stamp)<cutoff:
            captures.add((q["segment"],q["captured_at_utc"][:16]))
    if len(captures)!=2 or {segment for segment,_ in captures}!={"preceding_week","target_week"}:
        raise ValueError("Compute the additional optical passes before registration")
    week = original["week_ending"]
    new, evidence = latest_forward_balances(cutoff,week,original["prior_release"])
    old = [r for r in released_balances() if r["known"]<cutoff and r["week"]<week]
    if old[-1]["week"]>=new[0]["week"] or new[-1]["week"]!=(date.fromisoformat(week)-timedelta(days=7)).isoformat():
        raise ValueError("Published regional ledgers have inconsistent dates")
    storage = storage_coordinates(old+new)
    days,marks = spot_reference()
    local_day = cutoff.astimezone(EASTERN).date()
    reference = ((local_day-timedelta(days=4)).isoformat() if days[-1]<(local_day-timedelta(days=4)).isoformat()
                 else basis(cutoff,days,marks)[1])
    archived = inspect(dict(release_date=issue,study_phase="forward",cutoff_utc=cutoff.isoformat(),spot_reference_day=reference))
    evidence.update({archived[f"{k}_path"]:archived[f"{k}_sha256"] for k in ("wti","brent")})
    base = optical["estimates_kbbl"][INTERNAL_CALIBRATION]/1000
    current = dict(cutoff=cutoff, inventory_coordinates=archived_coordinates(archived,storage["state"])) | storage
    with threadpool_limits(limits=1):
        past = [r for r in inventory() if r["known"]<cutoff][-HISTORY:]
        if len(past)<MIN_WINDOWS:
            raise ValueError("Insufficient settled historical inventory forecasts")
        region = regional_adjustment(past,current)
        error_median = median_residual(past,current,2)
    sources = (original_path,trade_path,calibrated,recovered,tile_path,DERIVED/f"{VOLATILITY}_forecasts.csv",
               DERIVED/f"{VOLATILITY}_trades.csv",DERIVED/"cushing_storage_state_forecasts.csv",DERIVED/"eia_control_vintage.csv",
               *(ROOT/"scripts"/name for name in ("fit_cushing_storage_state.py","fit_cushing_volatility_neighbors.py",
                   "fit_cushing_cl_conditional_response.py","fit_cushing_terminal_weekly_transfers.py","register_cushing_cl_satellite.py")),
               Path(__file__).resolve())
    evidence.update({str(p.relative_to(ROOT)):digest(p) for p in sources})
    stamp = datetime.now(timezone.utc)
    if not past[-1]["known"]<cutoff<=stamp<deadline:
        raise ValueError("The winning models missed the forecast deadline")
    projection = {PUBLIC:trade_record["estimates_kbbl"][PUBLIC],SUPPLEMENT:trade_record["estimates_kbbl"][SUPPLEMENT],
                  LIMITED_MARKET:1000*(base+region+error_median)}
    record = {k:original[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(),estimates_kbbl=projection,
        inventory_components_kbbl=dict(optical_base=1000*base,regional_levels=1000*region,basis_error_median=1000*error_median,
                                       additional_passes=0.,terminal_limited_coverage=0.),
        measured_roofs=len(observations),measured_terminals=len(terminals),past_forecasts=len(past),
        latest_training_label_utc=past[-1]["known"].isoformat(),latest_balance_known_at_utc=new[-1]["known"].isoformat(),
        archived_spot_vintage=archived["vintage_date"],archived_spot_reference_date=archived["archived_reference_date"],
        market_method=VOLATILITY_METHOD,market_rule=RULE,sources_sha256=evidence)
    freeze(path,record)
    return record


def prediction(settled, calibration, current):
    sample = [r | dict(x=replace_expectation(r,r[SUPPLEMENT])) for r in settled]
    design = replace_expectation(current,current[SUPPLEMENT])
    point = volatility_forecast(sample,current | dict(x=design))
    source = features(current["x"][2],current["volatility_usd"],current["x"][0]-current[SUPPLEMENT])
    context = current | dict(response_features=source)
    residual,_ = conditional_residual(calibration,context) if len(calibration)>=MIN_WINDOWS else (0.,0.)
    return point,residual


def register_decision(issue, forecast_path, stamp, registered_public, trade_record, path):
    decision,deadline = (datetime.fromisoformat(registered_public[k]) for k in ("decision_at_utc","registration_deadline_utc"))
    public_receipt = REGISTERED/f"{issue}_{ENSEMBLE_INTERACTION_METHOD}.json"
    fields = ("release_date","decision_at_utc","entry_at","exit_at","near_month","far_month")
    if not path.exists():
        if not decision<=stamp<deadline:
            raise ValueError("The preregistered weighted decision missed its execution window")
        published = indexed("eia_wpsr_forward_releases")[issue]
        actual = float(published["cushing_change_kbbl"])/1000
        if actual!=registered_public["actual_million_bbl"] or digest(ROOT/published["source_path"])!=published["source_sha256"]:
            raise ValueError("The strategy would use a different or revised publication")
        carry,change,sigma = (float(registered_public[k]) for k in ("carry","carry_change","volatility_usd"))
        current = dict(decision=decision,volatility_usd=sigma,x=(actual,0.,carry,change,0.,0.,0.),
                       **{SUPPLEMENT:trade_record["estimates_kbbl"][SUPPLEMENT]/1000})
        past = [r for r in market_history() if r["exit_at"]<decision]
        settled = past[-HISTORY:]
        if len(settled)<MIN_WINDOWS:
            raise ValueError("Insufficient settled dated-contract trades")
        weighted = {r["release_date"]:r for r in read(f"{VOLATILITY}_trades") if r["method"]==VOLATILITY_METHOD}
        calibration = [r | dict(base_estimate=float(weighted[r["release_date"]]["base_estimate_usd"]),
            response_features=features(r["x"][2],r["volatility_usd"],r["x"][0]-r[SUPPLEMENT]))
            for r in past if r["release_date"] in weighted][-HISTORY:]
        with threadpool_limits(limits=1):
            point,residual = prediction(settled,calibration,current)
        recorded = datetime.now(timezone.utc)
        if not decision<=recorded<deadline:
            raise ValueError("The weighted model finished after its execution deadline")
        total = point+residual
        result = {k:registered_public[k] for k in fields} | dict(recorded_at_utc=recorded.isoformat(),
            near_side=int(total>HURDLE)-int(total<-HURDLE),expected_long_move_usd=total,point_usd=point,residual_usd=residual,
            forecast_sha256=digest(forecast_path),public_receipt_sha256=digest(public_receipt),
            first_release_sha256=published["source_sha256"],latest_training_exit_utc=settled[-1]["exit_at"].isoformat(),rule=RULE)
        freeze(path,result)
    result = json.loads(path.read_bytes())
    estimate = result["expected_long_move_usd"]
    if (not datetime.fromisoformat(result["latest_training_exit_utc"])<decision<=datetime.fromisoformat(result["recorded_at_utc"])<deadline
            or any(result[k]!=registered_public[k] for k in fields) or result["rule"]!=RULE
            or result["forecast_sha256"]!=digest(forecast_path) or result["public_receipt_sha256"]!=digest(public_receipt)
            or result["first_release_sha256"]!=indexed("eia_wpsr_forward_releases")[issue]["source_sha256"]
            or result["near_side"]!=int(estimate>HURDLE)-int(estimate<-HURDLE)):
        raise ValueError("The weighted market decision receipt does not verify")
    return result


def run(issue, *, scoring):
    prefix = DERIVED/f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    original_path = DERIVED/f"cushing_prerelease_forecast_{issue}.json"
    satellite_path = DERIVED/f"cushing_forward_cl_satellite_{issue}.json"
    trade_path = satellite_path.with_name(f"{satellite_path.stem}_inventory.json")
    trade_record = satellite_inventory(issue,trade_path)
    record = json.loads(path.read_bytes()) if path.exists() else register(issue,path,original_path,trade_path,trade_record)
    checked_hashes(record["sources_sha256"])
    now = datetime.now(timezone.utc)
    cutoff,stamp,deadline,known = (datetime.fromisoformat(record[k]) for k in (
        "information_cutoff_utc","recorded_at_utc","registration_deadline_utc","latest_training_label_utc"))
    if (not known<cutoff<=stamp<deadline or any(record[k]!=trade_record[k] for k in LINEAGE)
            or record["market_rule"]!=RULE or record["market_method"]!=VOLATILITY_METHOD):
        raise ValueError("The prerelease registration does not verify")
    if not scoring:
        return record
    outcome = score_inventory(original_path,json.loads(original_path.read_bytes()))
    result = dict(release_date=issue,status=outcome["status"],estimates_kbbl=record["estimates_kbbl"],registration_sha256=digest(path))
    if outcome["status"]=="scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_eia_change_kbbl=actual,first_release_sha256=outcome["first_release_sha256"],
                      inventory_absolute_error_kbbl={m:abs(v-actual) for m,v in record["estimates_kbbl"].items()},
                      status="awaiting_registered_market_controls")
        ledger = json.loads((DERIVED/"cushing_forward_public_cl_summary.json").read_bytes())["release_rows"]
        public = next((registered_decision(q,register=False,now=now) for q in ledger if q["release_date"]==issue
                       and q["method"]==ENSEMBLE_INTERACTION_METHOD),None)
        if public and public["recorded_at_utc"] and public["eligible_nonoverlap"]:
            baseline = satellite_decisions(issue,trade_record,satellite_path)
            fresh = register_decision(issue,path,now,public,trade_record,prefix.with_name(f"{prefix.name}_decision.json"))
            calendar = {issue:[r for r in read("cushing_forward_wti_coverage_bars") if r["release_date"]==issue and r["session_index"]!="1"]}
            _,marks = archives(as_of=now)
            for m in (PUBLIC,SUPPLEMENT,"curve"):
                comparison, = score_spread({issue:{VOLATILITY_METHOD:fresh,m:baseline[m]}},calendar,marks,now,methods=(VOLATILITY_METHOD,m))
                result[m] = comparison
            result["status"] = "window_scored" if result[PUBLIC]["market_window_complete"] else "awaiting_market_window"
    write(prefix.with_name(f"{prefix.name}_score.json"),(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",required=True,type=date.fromisoformat)
    parser.add_argument("--score",action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.release.isoformat(),scoring=args.score),indent=2,allow_nan=False))
