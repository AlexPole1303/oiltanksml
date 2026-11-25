"""Join terminal measurements to EIA development releases and compare fixed rules."""
import json
import math
import sqlite3
from collections import defaultdict
from contextlib import closing
from statistics import fmean

from build_terminal_development_measurements import DATED, MODES, RETROSPECTIVE, STUDY, TERMINALS
from build_weekly_shadow_development_check import ACTUAL, BASELINES, SEASONAL_FIELD, numeric, read_csv, scores
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write

STEM = "weekly_shadow_terminal_eia"
LAST = "last_published_change_kbbl"
LEGACY = "existing_visible_change_kbbl"
COMBINED = "combined_visible_change_kbbl"
LEGACY_PLUS = "last_plus_existing_kbbl"
COMBINED_PLUS = "last_plus_combined_kbbl"
FALLBACK = "last_plus_new_when_available_kbbl"
SCORE_FIELDS = ("weeks", "mean_abs_error_kbbl", "median_abs_error_kbbl", "mean_error_kbbl", "sign_agreement", "correlation")


def index(rows, fields):
    result = {}
    for row in rows:
        identity = tuple(row[field] for field in fields)
        if identity in result:
            raise ValueError(f"Duplicate row for {fields}: {identity}")
        result[identity] = row
    return result


def eia_releases():
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute(f"""SELECT p.week_ending,p.release_date,p.cushing_kbbl,p.{ACTUAL},
            p.latest_weekly_change_known_before_release,p.latest_vintage_known_before_release,
            p.prior_vintage_for_latest_known_change,p.source_url,w.inputs_available_before_utc
            FROM eia_inventory_release_panel p JOIN eia_satellite_window w USING(week_ending,release_date)
            WHERE p.study_phase='development' AND w.study_phase='development' AND w.endpoint='current'
            ORDER BY p.week_ending""")]


def volume(row):
    value = numeric(row["resolved_visible_change_equivalent_bbl"])
    return value / 1000 if value is not None else None


def add_when_available(a, b):
    return a + b if a is not None and b is not None else None


def signal_columns(label):
    return f"{label}_visible_change_kbbl", f"last_plus_{label}_kbbl", f"seasonal_plus_{label}_kbbl"


def weekly_rows(releases, base, new, terminals):
    for release in releases:
        week, issue = release["week_ending"], release["release_date"]
        reference = base[week,]
        last = release["latest_weekly_change_known_before_release"]
        if (reference["release_date"] != issue or numeric(reference[ACTUAL]) != release[ACTUAL]
                or numeric(reference[LAST]) != last or reference["eia_source_url"] != release["source_url"]):
            raise ValueError(f"Reference rows disagree with the first EIA release for {week}")
        old_issue = release["latest_vintage_known_before_release"]
        prior_issue = release["prior_vintage_for_latest_known_change"]
        seasonal_issues = reference["seasonal_source_releases"].split("|") if reference["seasonal_source_releases"] else []
        seasonal_weeks = reference["seasonal_source_weeks"].split("|") if reference["seasonal_source_weeks"] else []
        if (not all(d and d < issue for d in (old_issue, prior_issue, *seasonal_issues))
                or not all(d < week for d in seasonal_weeks)
                or not len(seasonal_issues) == len(seasonal_weeks) == int(reference["seasonal_observations"])):
            raise ValueError(f"A reference calculation uses a date outside its historical window for {week}")
        common = {field: numeric(reference[field]) for field in BASELINES}
        existing = numeric(reference["visible_resolved_change_kbbl"])
        for mode in MODES:
            measurement = new[week, mode]
            terminal_rows = [terminals[week, mode, t] for t in TERMINALS]
            for row in (measurement, *terminal_rows):
                if (row["release_date"], row["inputs_available_before_utc"]) != (issue, release["inputs_available_before_utc"]):
                    raise ValueError(f"Release cutoff differs from the measurement panel for {week} {mode}")
            new_value = volume(measurement)
            by_terminal = [value for r in terminal_rows if (value := volume(r)) is not None]
            if ((new_value is None) != (not by_terminal)
                    or (new_value is not None and not math.isclose(new_value, sum(by_terminal), abs_tol=1e-9))
                    or int(measurement["resolved_efr_pairs"]) != sum(int(r["resolved_efr_pairs"]) for r in terminal_rows)):
                raise ValueError(f"Terminal totals disagree with pooled measurements for {week} {mode}")
            combined = add_when_available(existing, new_value)
            row = dict(release, geometry_mode=mode, **common,
                seasonal_source_weeks=reference["seasonal_source_weeks"],
                seasonal_source_releases=reference["seasonal_source_releases"],
                existing_resolved_pairs=int(reference["resolved_efr_pairs"]),
                existing_resolved_area_m2=float(reference["reference_efr_area_m2"]) * float(reference["resolved_reference_area_fraction"]),
                existing_visible_change_kbbl=existing, combined_visible_change_kbbl=combined,
                last_plus_existing_kbbl=add_when_available(last, existing),
                last_plus_combined_kbbl=add_when_available(last, combined),
                last_plus_new_when_available_kbbl=last if new_value is None else add_when_available(last, new_value))
            for label, data in (("new", measurement), *zip(TERMINALS, terminal_rows, strict=True)):
                value = volume(data)
                raw, last_plus, seasonal_plus = signal_columns(label)
                row.update({f"{label}_input_pairs": int(data["input_efr_pairs"]),
                    f"{label}_resolved_pairs": int(data["resolved_efr_pairs"]),
                    f"{label}_resolved_area_m2": float(data["resolved_circle_area_m2"]),
                    raw: value, last_plus: add_when_available(last, value),
                    seasonal_plus: add_when_available(common[SEASONAL_FIELD], value)})
            yield row


def shared_geometry_rows(releases, rows):
    pairs = index(rows, ("week_ending", "candidate_id", "geometry_mode"))
    groups = defaultdict(list)
    for (week, candidate, mode), pair in pairs.items():
        if mode != DATED or pair["both_shadow_diagnostics_pass"] != "True":
            continue
        alternate = pairs[week, candidate, RETROSPECTIVE]
        for field in ("previous_version", "current_version", "previous_canonical_l2a_id", "current_canonical_l2a_id"):
            if pair[field] != alternate[field]:
                raise ValueError(f"Different scenes for shared geometry comparison in {week} for {candidate}")
        if alternate["both_shadow_diagnostics_pass"] == "True":
            groups[week].append((pair, alternate))
    for week, matches in sorted(groups.items()):
        release = releases[week,]
        dated, retrospective = (sum(float(pair["optical_change_equivalent_bbl"]) for pair in mode_pairs) / 1000
                                for mode_pairs in zip(*matches, strict=True))
        yield dict(week_ending=week, release_date=release["release_date"], **{ACTUAL:release[ACTUAL]},
            eia_source_url=release["source_url"], shared_resolved_pairs=len(matches),
            shared_candidate_ids="|".join(sorted(p[0]["candidate_id"] for p in matches)),
            dated_visible_change_kbbl=dated, retrospective_visible_change_kbbl=retrospective,
            **{LAST:release["latest_weekly_change_known_before_release"]},
            last_plus_dated_kbbl=release["latest_weekly_change_known_before_release"] + dated,
            last_plus_retrospective_kbbl=release["latest_weekly_change_known_before_release"] + retrospective)


def comparison_rows(sample_name, mode, rows, fields, reference):
    matched = [r for r in rows if all(r[field] is not None for field in (*fields, ACTUAL))]
    for name in fields:
        errors = [abs(r[name] - r[ACTUAL]) - abs(r[reference] - r[ACTUAL]) for r in matched]
        metrics = scores(matched, name)
        yield dict(sample=sample_name, geometry_mode=mode, rule=name, reference_rule=reference,
            **{field: metrics.get(field) for field in SCORE_FIELDS},
            first_week=min((r["week_ending"] for r in matched), default=None),
            last_week=max((r["week_ending"] for r in matched), default=None),
            delta_mae_vs_reference_kbbl=fmean(errors) if errors else None,
            weeks_better=sum(d < -1e-9 for d in errors), weeks_tied=sum(abs(d) <= 1e-9 for d in errors),
            weeks_worse=sum(d > 1e-9 for d in errors))


def comparisons(weekly, shared):
    for mode in MODES:
        rows = [r for r in weekly if r["geometry_mode"] == mode]
        raw, last_plus, _ = signal_columns("new")
        yield from comparison_rows("new_and_last_measured_weeks", mode, rows, (LAST, raw, last_plus), LAST)
        for label in ("new", *TERMINALS):
            raw, last_plus, seasonal_plus = signal_columns(label)
            yield from comparison_rows(f"{label}_measured_weeks", mode, rows,
                (*BASELINES, raw, last_plus, seasonal_plus), LAST)
        yield from comparison_rows("existing_and_new_measured_weeks", mode, rows,
            (LAST, LEGACY, COMBINED, LEGACY_PLUS, COMBINED_PLUS), LEGACY_PLUS)
        yield from comparison_rows("all_development_weeks", mode, rows, (LAST, FALLBACK), LAST)
    yield from comparison_rows("shared_resolved_pairs", "same_tanks_same_scenes", shared,
        (LAST, "dated_visible_change_kbbl", "retrospective_visible_change_kbbl",
         "last_plus_dated_kbbl", "last_plus_retrospective_kbbl"), LAST)


def main():
    releases = eia_releases()
    release_index = index(releases, ("week_ending",))
    base = index(read_csv("weekly_shadow_development_check"), ("week_ending",))
    new = index(read_csv(f"{STUDY}_weeks"), ("week_ending", "geometry_mode"))
    terminals = index(read_csv(f"{STUDY}_terminals"), ("week_ending", "geometry_mode", "terminal_id"))
    new_pairs = read_csv(f"{STUDY}_pairs")
    prior_pairs = read_csv("weekly_shadow_development_pairs")
    new_candidates = {p["candidate_id"] for p in new_pairs}
    if new_candidates & {p["candidate_id"] for p in prior_pairs}:
        raise ValueError("New terminal reservoirs overlap the existing development panel")
    if (release_index.keys() != base.keys()
            or new.keys() != {(r["week_ending"], mode) for r in releases for mode in MODES}
            or terminals.keys() != {(*key, terminal) for key in new for terminal in TERMINALS}
            or any((p["week_ending"],) not in release_index for p in new_pairs)):
        raise ValueError("Measurements differ from the development release calendar")
    weekly = list(weekly_rows(releases, base, new, terminals))
    shared = list(shared_geometry_rows(release_index, new_pairs))
    results = list(comparisons(weekly, shared))
    export(f"{STEM}_development", weekly)
    export(f"{STEM}_geometry_matched", shared)
    export(f"{STEM}_scores", results)
    result = dict(development_weeks=len(releases), weekly_rows=len(weekly), comparison_rows=len(results),
        first_week=releases[0]["week_ending"], last_week=releases[-1]["week_ending"],
        new_reservoirs=len(new_candidates), same_tank_and_scene_weeks=len(shared),
        same_tank_and_scene_resolved_pairs=sum(r["shared_resolved_pairs"] for r in shared),
        comparison_scope="Fixed unit weights. No training, scaling to hub capacity, return outcomes, or validation/holdout EIA outcomes. "
                         "An empty measurement stays missing. The explicitly labelled fallback rule uses last published change in unmeasured weeks. "
                         "Geometry vintage labels describe engineering sources, with shared retrospective image screening and optical procedure.",
        pooled_terminal_last_plus_new=[
            r for r in results if r["sample"] == "new_and_last_measured_weeks" and r["rule"] in (LAST, "last_plus_new_kbbl")],
        existing_panel_addition=[
            r for r in results if r["sample"] == "existing_and_new_measured_weeks" and r["rule"] in (LEGACY_PLUS, COMBINED_PLUS)])
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
