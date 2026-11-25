"""Freeze the inventory forecasts for a release with and without optical coverage."""

import argparse
import json
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cushing_holdout_eia_vintages import archive
from audit_cushing_spot_vintages import inspect
from build_s1_reference_matches import DERIVED, read
from collect_public import RAW, ROOT, write
from evaluate_cushing_forward_public_cl import forward_report_vintage
from fit_cushing_public_state import OPTICAL, PUBLIC, STATE, adjust, basis, released_balances, source_rows, spot_reference, state
from publication_clocks import EASTERN
from register_cushing_prerelease_forecast import digest, freeze, score

TABLES = (1,2,4,9)
FALLBACK = "balance_and_basis_without_optics"
BALANCE = "balance_without_optics"


def latest_forward_balances(cutoff, week, prior_issue):
    directory = RAW/"eia"/"wpsr_forward"
    available, sources = [], {}
    for r in read("eia_wpsr_forward_releases"):
        issue = r["release_date"]
        known = datetime.fromisoformat(r["conservative_postrelease_from_utc"])
        if not known<cutoff or not r["latest_week_in_issue"]<week:
            continue
        for table in TABLES:
            _,_,proof = archive(issue,table,directory=directory)
            path = ROOT/proof["source_path"]
            if digest(path)!=proof["source_sha256"]:
                raise ValueError(f"Source archive changed for {issue}/{table}")
            sources[proof["source_path"]] = proof["source_sha256"]
        dates = (r["latest_week_in_issue"],r["previous_week_ending"])
        vintage = defaultdict(dict)
        forward_report_vintage(issue,*dates,vintage)
        values = np.array([[vintage[issue,when][key]/1000 for key in STATE] for when in dates])
        if not np.isfinite(values).all():
            raise ValueError(f"Incomplete archived inventory balance on {issue}")
        values[:,1] -= values[:,0]
        available.append(dict(issue=issue,week=dates[0],known=known,current=values[0],delta=values[0]-values[1]))
    if not available or available[-1]["issue"] != prior_issue:
        raise ValueError("Forward releases do not include the registered prior WPSR")
    return available,sources


def create(issue, original, optical):
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in ("information_cutoff_utc","registration_deadline_utc"))
    week = original["week_ending"]
    if not cutoff <= datetime.now(timezone.utc) < deadline:
        raise ValueError("Registration is outside the report's forecast window")
    forward,proof = latest_forward_balances(cutoff,week,original["prior_release"])
    if proof.get(f"data/raw/eia/wpsr_forward/{original['prior_release'].replace('-','_')}_table4.csv") != original["prior_release_sha256"]:
        raise ValueError("The preceding stocks report no longer matches the saved projection")
    previous = [q for q in released_balances() if q["known"]<cutoff and q["week"]<week]
    if previous[-1]["week"]>=forward[0]["week"] or forward[-1]["week"]!=(date.fromisoformat(week)-timedelta(days=7)).isoformat():
        raise ValueError("The previous survey week is not aligned")
    balance = state(previous+forward,week)
    days,marks = spot_reference()
    legacy_cutoff = cutoff.astimezone(EASTERN).date()
    # The historic spot book is only used to reproduce the reference-date
    # cap. If that book ends earlier, use the archived vintage's own dates.
    if days[-1] < (legacy_cutoff-timedelta(days=4)).isoformat():
        reference = (legacy_cutoff-timedelta(days=4)).isoformat()
    else:
        _,reference,_ = basis(cutoff,days,marks)
    archival = inspect(dict(release_date=issue,study_phase="forward",
                           cutoff_utc=cutoff.isoformat(),spot_reference_day=reference))
    for k in ("wti","brent"):
        proof[archival[f"{k}_path"]] = archival[f"{k}_sha256"]
    relative = tuple(archival[f"archived_basis_{k}"] for k in ("level","move_5","move_20"))
    current = dict(cutoff=cutoff,balance=balance,basis=(*balance,*relative))
    history = [q for q in source_rows(use_archived_basis=True) if q["known"]<cutoff][-156:]
    if len(history)<52:
        raise ValueError("Too few settled forecast errors")
    with threadpool_limits(limits=1):
        adjustments = {k:1000*adjust(history,current,k) for k in ("balance","basis")}
    published_public = original["public_ensemble_kbbl"]
    available_roofs = optical["measured_roofs"] if optical else original["measured_roofs"]
    if not 0<=available_roofs or (optical and not np.isclose(published_public,optical["estimates_kbbl"][PUBLIC])):
        raise ValueError("Forecasts had different optical panels or public anchors")
    corrected = {BALANCE:published_public+adjustments["balance"],FALLBACK:published_public+adjustments["basis"]}
    observed = optical["estimates_kbbl"][OPTICAL] if optical else published_public
    stamp = datetime.now(timezone.utc)
    if not cutoff <= stamp < deadline:
        raise ValueError("The new forecast missed its deadline")
    return dict(release_date=issue,week_ending=week,recorded_at_utc=stamp.isoformat(),
        information_cutoff_utc=cutoff.isoformat(),registration_deadline_utc=deadline.isoformat(),
        latest_training_label_utc=history[-1]["known"].isoformat(),trained_releases=len(history),
        latest_balance_label_utc=forward[-1]["known"].isoformat(),
        spot_vintage=archival["vintage_date"],spot_reference_date=archival["archived_reference_date"],
        baseline_public_kbbl=published_public,measured_roofs=available_roofs,conditional_fallbacks_kbbl=corrected,
        primary_kbbl=observed if available_roofs else corrected[FALLBACK],
        primary_method=OPTICAL if available_roofs else FALLBACK,branch_activated=not bool(available_roofs),
        forward_evidence_sha256=proof)


def run(issue, *, scoring):
    original_path = DERIVED/f"cushing_prerelease_forecast_{issue}.json"
    optical_path = original_path.with_name(f"{original_path.stem}_internal_calibration.json")
    path = original_path.with_name(f"{original_path.stem}_public_fallback.json")
    original = json.loads(original_path.read_bytes())
    optical = json.loads(optical_path.read_bytes()) if optical_path.is_file() else None
    fields = ("release_date","week_ending","information_cutoff_utc","registration_deadline_utc")
    if optical and any(original[key]!=optical[key] for key in fields):
        raise ValueError("The two original forecast registrations do not match")
    if not path.exists() and not scoring:
        if optical is None and original["measured_roofs"]:
            raise ValueError("Register the calibrated optical forecast first for a measured release")
        record = create(issue,original,optical)
        source_paths = (original_path,*((optical_path,) if optical else ()),
                        DERIVED/"cushing_public_state_forecasts.csv",DERIVED/"cushing_spot_vintages.csv",
                        DERIVED/"eia_control_vintage.csv",ROOT/"scripts/fit_cushing_public_state.py",Path(__file__).resolve())
        record["input_sha256"] = {str(p.relative_to(ROOT)):digest(p) for p in source_paths}
        freeze(path,record)
    record = json.loads(path.read_bytes())
    trained, balance, cutoff, written, deadline = (datetime.fromisoformat(record[k]) for k in (
        "latest_training_label_utc","latest_balance_label_utc","information_cutoff_utc","recorded_at_utc","registration_deadline_utc"))
    if (not max(trained,balance)<cutoff<=written<deadline or any(record[k]!=original[k] for k in fields)
            or any(digest(p)!=record["input_sha256"][str(p.relative_to(ROOT))]
                for p in (original_path,optical_path) if str(p.relative_to(ROOT)) in record["input_sha256"])
            or any(digest(ROOT/k)!=value for k,value in record["forward_evidence_sha256"].items())):
        raise ValueError("The new receipt differs from the original forecast information set")
    if not scoring:
        return record
    outcome = score(original_path,original)
    result = dict(status=outcome["status"],release_date=issue,receipt_sha256=digest(path),primary_method=record["primary_method"],
                  branch_activated=record["branch_activated"])
    if outcome["status"]=="scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_eia_change_kbbl=actual,first_release_sha256=outcome["first_release_sha256"],
                      primary_absolute_error_kbbl=abs(record["primary_kbbl"]-actual),
                      public_absolute_error_kbbl=abs(record["baseline_public_kbbl"]-actual),
                      counterfactual_no_optical_errors_kbbl={k:abs(v-actual) for k,v in record["conditional_fallbacks_kbbl"].items()})
    write(path.with_name(f"{path.stem}_score.json"),(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register-release",type=date.fromisoformat)
    mode.add_argument("--score-release",type=date.fromisoformat)
    args = parser.parse_args()
    issue = args.register_release or args.score_release
    print(json.dumps(run(issue.isoformat(),scoring=args.score_release is not None),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
