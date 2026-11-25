"""Join the audited release, optics, WTI and equity calendars for review."""
import json
from collections import Counter
from datetime import date, datetime
from statistics import median

from build_holdout_optical_inputs import indexed
from build_market_event_panel import CORE
from build_operator_development_inputs import group, one_per_key
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_public import write, write_csv

STEM = "cushing_holdout_readiness"
OPTICAL = ("b04_status","baseline_b04_tanks","baseline_b04_area_fraction","baseline_b04_space_time_fraction",
    "area_weighted_capture_interval_hours","previous_l1c_archive_products","current_l1c_archive_products",
    "previous_l1c_screened_products","current_l1c_screened_products","landsat_15m_paired_tanks",
    "sentinel1_slc_common_relative_track","sentinel1_grd_common_relative_track")
GAP_FIELDS = ("release_date","instrument","entry_session","fifth_session","accounting_input_gaps","measured_tanks")


def yes(r,k):
    if r[k] not in ("True","False"):
        raise ValueError(f"Invalid Boolean {k} at {r.get('release_date')}: {r[k]}")
    return r[k]=="True"


def build():
    releases,optics,views,wti = (indexed(s) for s in ("cushing_holdout_release_index","holdout_optical_weeks",
        "holdout_optical_expanded_view_features","cushing_holdout_market_feature_plan"))
    eia = indexed("cushing_holdout_eia_vintage_audit_releases")
    cftc = indexed("cushing_holdout_cftc_audit_releases")
    recovery = one_per_key(read_csv("cushing_holdout_optical_recovery_weeks"),"week_ending")
    radar_slc,radar_grd = (one_per_key(read_csv(s),"week_ending") for s in
        ("s1_holdout_gap_burst_weeks","s1_grd_holdout_gap_coverage_weeks"))
    rtc = one_per_key(read_csv("cushing_s1_rtc_weeks"),"week_ending")
    ceda = group(read_csv("ceda_s1_slc_windows_pairs"),"week_ending")
    ceda_regional_path = DERIVED / "ceda_s1_slc_regional_weekly_pairs.csv"
    ceda_regional = group(read_csv(ceda_regional_path.stem),"week_ending") if ceda_regional_path.is_file() else {}
    opera = one_per_key(read_csv("opera_cslc_holdout_weeks"),"week_ending")
    landsat = one_per_key(read_csv("landsat_c2_holdout_gap_weeks"),"week_ending")
    landsat_qa = one_per_key(read_csv("landsat_c2_b8_stage_weeks"),"week_ending")
    landsat_footprint = one_per_key(read_csv("landsat_c2_b8_measurements_footprint_qa_weeks"),"week_ending")
    landsat_pairs = one_per_key(read_csv("landsat_c2_b8_weekly_weeks"),"week_ending")
    landsat_context = one_per_key(read_csv("landsat_c2_context_registration_weeks"),"week_ending")
    landsat_esri = one_per_key(read_csv("landsat_esri_b8_controls_weeks"),"week_ending")
    esri_registration = one_per_key(read_csv("landsat_esri_b8_registration_weeks"),"week_ending")
    esri_refits = one_per_key(read_csv("landsat_esri_b8_registered_changes_weeks"),"week_ending")
    b8_timing = one_per_key(read_csv("cushing_b8_hls_lineage_weeks"),"week_ending")
    usgs_b8_metadata = one_per_key(read_csv("cushing_usgs_b8_metadata_weeks"),"week_ending")
    modern = one_per_key(read_csv("cushing_modern_lidar_optical_weeks"),"week_ending")
    wider = one_per_key(read_csv("sentinel2_wider_registration_gap_weeks"),"week_ending")
    nasa = one_per_key(read_csv("cushing_nasa_satellite_weeks"),"week_ending")
    laser = one_per_key(read_csv("cushing_icesat2_holdout_weeks"),"week_ending")
    native_laser = one_per_key(read_csv("cushing_icesat2_native_holdout_weeks"),"week_ending")
    weather = indexed("cushing_hrrr_holdout_features")
    expected_radar = {(w,) for w,o in optics.items() if o["b04_status"]!="measured"}
    expected_wider = {(w,) for w,o in optics.items() if o["b04_status"]=="no_independently_checked_registration"}
    core = group((r for r in read_csv("cushing_holdout_operator_inputs") if r["instrument"] in CORE),"release_date")
    measured,public = (one_per_key(read_csv(f"cushing_holdout_operator_replay_{s}"),"release_date","instrument")
                       for s in ("manifest","residual_manifest"))
    if (not releases.keys()==optics.keys()==views.keys()==wti.keys()==eia.keys()==cftc.keys()==weather.keys()
        or not set(nasa)==set(laser)==set(native_laser)==set(recovery)==set(rtc)=={(w,) for w in releases}
        or not set(ceda)|set(ceda_regional)<=set(rtc)
        or set(modern)!={(w,) for w in releases} or set(wider)!=expected_wider
        or any(table.keys()!=expected_radar for table in (radar_slc,radar_grd,opera,landsat,landsat_qa,
                                                         landsat_footprint,landsat_pairs,landsat_context,landsat_esri,esri_registration,
                                                         esri_refits,b8_timing,usgs_b8_metadata))
        or set(core)!={(r["release_date"],) for r in releases.values()}):
        raise ValueError("Readiness inputs disagree on the closed release calendar")
    days,coverage,gaps = [],[],[]
    seen_measured,seen_public = set(),set()
    for week,r in sorted(releases.items()):
        issue = r["release_date"]
        o,v,w = optics[week],views[week],wti[week]
        audit,cot = eia[week],cftc[week]
        t = recovery[week,]
        slc,grd,cslc,b8,b8_qa = (table.get((week,),{}) for table in (radar_slc,radar_grd,opera,landsat,landsat_qa))
        rtc_week = rtc[week,]
        ceda_full = ceda.get((week,),())
        ceda_windows = ceda_regional.get((week,),())
        lidar = modern[week,]
        nasa_week,laser_week = nasa[week,],laser[week,]
        atl03 = native_laser[week,]
        hrrr = weather[week]
        z, b8_full = wider.get((week,),{}), landsat_footprint.get((week,),{})
        b8_pairs = landsat_pairs.get((week,),{})
        b8_context = landsat_context.get((week,),{})
        b8_esri = landsat_esri.get((week,),{})
        b8_esri_registration = esri_registration.get((week,),{})
        b8_esri_refit = esri_refits.get((week,),{})
        timing = b8_timing.get((week,),{})
        usgs_meta = usgs_b8_metadata.get((week,),{})
        if any(row["release_date"]!=issue for row in (o,v,w,t,audit,cot,lidar,nasa_week,laser_week,rtc_week,hrrr,atl03)) or any(
            datetime.fromisoformat(stamp)!=datetime.fromisoformat(r["forecast_cutoff_utc"])
            for stamp in (o["cutoff_utc"],v["cutoff_utc"],w["forecast_cutoff_utc"],t["cutoff_utc"],
                          audit["forecast_cutoff_utc"],cot["forecast_cutoff_utc"],lidar["cutoff_utc"],
                          nasa_week["cutoff_utc"],laser_week["cutoff_utc"],rtc_week["forecast_cutoff_utc"],
                          hrrr["forecast_cutoff_utc"],atl03["cutoff_utc"])):
            raise ValueError(f"Different release or cutoff on {issue}")
        if (rtc_week["b04_status"]!=o["b04_status"] or any(
                datetime.fromisoformat(a["forecast_cutoff_utc"])!=datetime.fromisoformat(r["forecast_cutoff_utc"])
                for a in (*ceda_full,*ceda_windows))):
            raise ValueError(f"Radar samples and release calendar differ on {issue}")
        if (audit["source_url"]!=r["eia_target_reference"] or audit["prior_issue"]!=r["prior_stock_vintage"]
            or audit["target_available_at_utc"]!=r["target_available_at_utc"]):
            raise ValueError(f"EIA audit and release index disagree on {issue}")
        if (cot["report_date"]!=r["cftc_report_date"]
            or cot["saved_cftc_available_at_utc"]!=r["cftc_available_at_utc"]):
            raise ValueError(f"CFTC audit and release index disagree on {issue}")
        if r["wti_replay_source"]!=w["execution_source"] or r["b04_tanks"]!=o["baseline_b04_tanks"]:
            raise ValueError(f"Different optical or market coverage on {issue}")
        if (any(x["release_date"]!=issue for x in (slc,grd,cslc,b8,b8_qa) if x)
            or any(x["forecast_cutoff_utc"]!=r["forecast_cutoff_utc"] for x in (b8,b8_qa,b8_full) if x)
            or (b8_full and b8_full["release_date"]!=issue)
            or (z and (z["release_date"],z["cutoff_utc"],z["baseline_status"]) != (
                issue,o["cutoff_utc"],o["b04_status"]))):
            raise ValueError(f"Different alternative satellite release or cutoff on {issue}")
        if (lidar["baseline_b04_status"],lidar["baseline_b04_tanks"])!=(o["b04_status"],o["baseline_b04_tanks"]):
            raise ValueError(f"LiDAR expansion and base optical calendar disagree on {issue}")
        if any(s["b04_status"]!=o["b04_status"] for s in (nasa_week,laser_week,atl03)):
            raise ValueError(f"NASA satellite inventory and base optical calendar disagree on {issue}")
        if b8 and not set(json.loads(b8_full["clear_candidate_ids_json"])) <= set(
            json.loads(b8_qa["clear_candidate_ids_json"])) <= (
                set(json.loads(b8["paired_candidate_ids_json"])) | set(json.loads(b8["mixed_b8_and_sentinel2_candidate_ids_json"]))):
            raise ValueError(f"Native B8 QA has tank identities outside its candidate list on {issue}")
        if b8_pairs and (b8_pairs["release_date"] != issue
            or b8_pairs["forecast_cutoff_utc"] != r["forecast_cutoff_utc"]
            or b8_pairs["candidate_tanks"] != b8_full["clear_any_b8_pair_candidates"]
            or not 0 <= int(b8_pairs["tanks_with_both_fits_resolved"]) <= int(b8_pairs["tanks_with_all_required_b8_staged"])
                       <= int(b8_pairs["candidate_tanks"])):
            raise ValueError(f"Exact B8 pair coverage differs on {issue}")
        if b8_context and (b8_context["release_date"] != issue
            or b8_context["forecast_cutoff_utc"] != r["forecast_cutoff_utc"]
            or b8_context["exact_pair_candidates"] != b8_pairs["exact_pair_candidates"]
            or not 0 <= int(b8_context["checked_context_candidate_tanks"]) <= int(b8_pairs["candidate_tanks"])
            or not int(b8_context["checked_context_candidate_tanks"]) <= int(b8_context["checked_context_pairs"])
                   <= int(b8_pairs["exact_pair_candidates"])):
            raise ValueError(f"Red-band registration context differs on {issue}")
        if b8_esri and (b8_esri["release_date"]!=issue or b8_esri["forecast_cutoff_utc"]!=r["forecast_cutoff_utc"]
            or b8_esri["b04_status"]!=o["b04_status"] or b8_esri["exact_pair_candidates"]!=b8_pairs["exact_pair_candidates"]
            or b8_esri["candidate_tanks"]!=b8_pairs["candidate_tanks"]
            or not 0<=int(b8_esri["tanks_resolved_with_checked_red_context_before_cutoff"])<=min(
                int(b8_esri["tanks_with_both_fits_resolved"]),int(b8_context["checked_context_candidate_tanks"]))
            or not int(b8_esri["tanks_with_both_fits_resolved"])<=int(b8_esri["tanks_with_usable_esri_inputs"])
                   <=int(b8_esri["tanks_with_all_required_windows"])<=int(b8_pairs["candidate_tanks"])
            or int(b8_esri["historical_pairs_verified"])!=0
            or any(b8_esri_registration[k]!=b8_esri[k] for k in ("release_date","forecast_cutoff_utc","b04_status"))
            or b8_esri_registration["candidate_pairs"]!=b8_esri["pairs_with_usable_esri_inputs"]
            or not 0<=int(b8_esri_registration["checked_tanks_with_both_roof_fits_and_source_clock"])
                   <=int(b8_esri_registration["checked_tanks_with_both_roof_fits"])<=int(b8_esri["tanks_with_both_fits_resolved"])):
            raise ValueError(f"Esri B8 retrospective coverage differs on {issue}")
        esri_stable = sum(int(b8_esri_refit.get(f"tanks_with_{sign}_stable_direction",0)) for sign in ("positive","negative"))
        if b8_esri_refit and (any(b8_esri_refit[k]!=b8_esri[k] for k in ("release_date","forecast_cutoff_utc","b04_status"))
            or int(b8_esri_refit["historically_admitted_tanks"])!=0
            or not 0<=esri_stable<=int(b8_esri_refit["tanks_with_jointly_resolved_pair"])
                   <=int(b8_esri_refit["tanks_with_checked_exterior_offsets"])<=int(b8_esri["tanks_with_usable_esri_inputs"])):
            raise ValueError(f"Esri B8 registered change coverage differs on {issue}")
        if timing and (timing["release_date"]!=issue or timing["forecast_cutoff_utc"]!=r["forecast_cutoff_utc"]
            or timing["candidate_pairs"]!=b8_pairs["exact_pair_candidates"]
            or not 0 <= int(timing["either_catalog_tanks_before_cutoff"]) <= int(b8_pairs["candidate_tanks"])
            or not 0 <= int(timing["either_catalog_and_red_context_tanks_before_cutoff"]) <= min(
                int(timing["either_catalog_tanks_before_cutoff"]),int(b8_context["checked_context_candidate_tanks"]))
            or int(timing["confirmed_historical_b8_file_pairs"]) != 0):
            raise ValueError(f"B8 archive timing evidence differs on {issue}")
        if usgs_meta and (usgs_meta["release_date"]!=issue or usgs_meta["forecast_cutoff_utc"]!=r["forecast_cutoff_utc"]
            or usgs_meta["exact_pair_candidates"]!=b8_pairs["exact_pair_candidates"]
            or not 0<=int(usgs_meta["tanks_with_stac_last_updated_before_cutoff"])
                <=int(usgs_meta["tanks_with_stac_created_before_cutoff"])<=int(b8_pairs["candidate_tanks"])
            or int(usgs_meta["original_b8_file_clocks_verified"])!=0):
            raise ValueError(f"USGS STAC metadata timing evidence differs on {issue}")
        instruments = core[issue,]
        if len(instruments)!=len(CORE) or {a["instrument"] for a in instruments}!=set(CORE):
            raise ValueError(f"Missing or repeated operator on {issue}")
        measured_symbols,public_symbols,paired_flags = [],[],set()
        for a in sorted(instruments,key=lambda a:a["instrument"]):
            s = a["instrument"]
            key = issue,s
            m,p = measured.get(key),public.get(key)
            if a["week_ending"]!=week or (a["replay_input_status"]=="ready_for_replay")!=bool(m):
                raise ValueError(f"Operator readiness disagrees on {issue}/{s}")
            if p:
                if yes(p,"optical_residual_eligible")!=bool(m):
                    raise ValueError(f"Equity models disagree on the optical cohort on {issue}/{s}")
                public_symbols.append(s)
                seen_public.add(key)
            if m:
                measured_symbols.append(s)
                seen_measured.add(key)
                paired_flags.add(yes(m,"common_nonoverlapping_issue"))
            tanks = float(a["asset_baseline_b04_tanks"])
            if not tanks.is_integer() or tanks<0:
                raise ValueError(f"Invalid physical observation count on {issue}/{s}")
            coverage.append(dict(release_date=issue,week_ending=week,instrument=s,benchmark=a["benchmark"],
                asset_terminal_ids=a["asset_terminal_ids"],measured_terminal_ids=a["asset_measured_terminal_ids"],
                measured_tanks=int(tanks),input_status=a["replay_input_status"],
                public_model_training_eligible=bool(p),paired_optical_models_eligible=bool(m),
                accounting_input_gaps=a["accounting_input_gaps"],accounting_quote_bridge_keys=a["accounting_quote_bridge_keys"],
                entry_session=a["entry_session"],fifth_session=a["fifth_session"],
                beta_60d_source_window_present=bool(p and p["amlp_beta_60d_first_close"]),
                beta_120d_source_window_present=bool(p and p["amlp_beta_120d_first_close"])))
            if a["accounting_input_gaps"]:
                gaps.append(dict(release_date=issue,instrument=s,entry_session=a["entry_session"],fifth_session=a["fifth_session"],
                    accounting_input_gaps=a["accounting_input_gaps"],measured_tanks=int(tanks)))
        if len(paired_flags)>1:
            raise ValueError(f"Operators disagree on the common issue calendar on {issue}")
        days.append(dict(week_ending=week,release_date=issue,measurement_at_utc=o["current_measurement_at_utc"],
            forecast_cutoff_utc=r["forecast_cutoff_utc"],target_available_at_utc=r["target_available_at_utc"],
            eia_first_release_source=r["eia_target_reference"],eia_prior_stock_vintage=r["prior_stock_vintage"],
            eia_first_release_file_sha256=audit["source_sha256"],eia_first_release_source_row=audit["source_row"],
            eia_release_timing_basis=audit["timing_basis"],
            eia_weekly_workbook_change_matches=yes(audit,"latest_workbook_change_matches"),
            eia_control_metrics=int(r["control_metrics_present"]),cftc_report_date=r["cftc_report_date"],
            cftc_weekly_source_status=cot["source_status"],cftc_weekly_source_url=cot["source_url"],
            cftc_weekly_footer_date=cot["report_footer_updated_date"],
            cftc_weekly_footer_date_before_cutoff=yes(cot,"report_footer_day_ended_by_cutoff") if cot["report_footer_updated_date"] else None,
            cftc_weekly_file_modified_by_cutoff=cot["http_file_modified_by_cutoff"],
            hrrr_native_both_endpoints_before_cutoff=yes(hrrr,"both_endpoints_all_four_before_cutoff"),
            hrrr_native_prior_cutoff_hour_complete=yes(hrrr,"prior_cutoff_hour_all_four_before_cutoff"),
            **{k:o[k] for k in OPTICAL},additional_optical_views=int(r["optical_views_available"]),
            dated_timing_interpolated_tanks=int(t["dated_interpolated_tanks"]),
            dated_timing_candidate_in_b04_gap=int(t["dated_interpolated_tanks_with_no_b04"])>0,
            modern_lidar_additional_resolved_tanks=int(lidar["modern_additional_resolved_tanks"]),
            modern_lidar_additional_area_m2=float(lidar["modern_additional_area_m2"]),
            wider_registration_checked_pairs=int(z.get("checked_candidates",0)),
            wider_registration_resolved_tanks=int(z.get("resolved_tanks",0)),
            wider_registration_resolved_area_m2=float(z.get("resolved_area_m2",0)),
            landsat_c2_gap_tanks_with_two_b8_candidates=b8.get("paired_clear_possible_b8"),
            landsat_c2_gap_tanks_with_b8_and_sentinel2_candidates=b8.get("mixed_b8_and_sentinel2_candidate_tanks"),
            landsat_c2_gap_tanks_with_any_b8_pair_candidate=b8.get("any_b8_pair_candidate_tanks"),
            landsat_c2_gap_pair_candidates_with_native_qa=b8_qa.get("clear_any_b8_pair_candidates"),
            landsat_c2_gap_pair_candidates_with_full_footprint_qa=b8_full.get("clear_any_b8_pair_candidates"),
            landsat_c2_gap_exact_scene_pairs=b8_pairs.get("exact_pair_candidates"),
            landsat_c2_gap_tanks_with_all_required_b8_staged=b8_pairs.get("tanks_with_all_required_b8_staged"),
            landsat_c2_gap_tanks_with_both_fits_resolved=b8_pairs.get("tanks_with_both_fits_resolved"),
            landsat_c2_gap_pairs_with_checked_red_context=b8_context.get("checked_context_pairs"),
            landsat_c2_gap_tanks_with_checked_red_context=b8_context.get("checked_context_candidate_tanks"),
            esri_b8_gap_tanks_with_all_windows_retrospective=b8_esri.get("tanks_with_all_required_windows"),
            esri_b8_gap_tanks_with_usable_esri_inputs=b8_esri.get("tanks_with_usable_esri_inputs"),
            esri_b8_gap_tanks_with_two_resolved_fits_retrospective=b8_esri.get("tanks_with_both_fits_resolved"),
            esri_b8_gap_tanks_resolved_with_checked_red_context=b8_esri.get("tanks_resolved_with_checked_red_context_before_cutoff"),
            esri_b8_gap_tanks_resolved_with_checked_b8_exterior=b8_esri_registration.get("checked_tanks_with_both_roof_fits"),
            esri_b8_gap_tanks_resolved_with_checked_b8_exterior_and_source_clock=b8_esri_registration.get(
                "checked_tanks_with_both_roof_fits_and_source_clock"),
            esri_b8_gap_tanks_with_jointly_resolved_change=b8_esri_refit.get("tanks_with_jointly_resolved_pair"),
            esri_b8_gap_tanks_with_stable_change_direction=esri_stable if b8_esri_refit else None,
            landsat_c2_gap_tanks_with_cdse_bundle_clock=timing.get("cdse_catalog_tanks_before_cutoff"),
            landsat_c2_gap_tanks_with_hls_lineage_clock=timing.get("hls_tanks_before_cutoff"),
            landsat_c2_gap_tanks_with_either_archive_clock=timing.get("either_catalog_tanks_before_cutoff"),
            landsat_c2_gap_tanks_with_archive_clock_and_checked_context=timing.get("either_catalog_and_red_context_tanks_before_cutoff"),
            landsat_c2_gap_tanks_with_usgs_stac_metadata_before_cutoff=usgs_meta.get("tanks_with_stac_created_before_cutoff"),
            slc_gap_dated_roofs_with_full_burst_context=slc.get("roofs_with_full_context"),
            slc_gap_dated_roofs_with_full_context_and_dual_catalog=slc.get("roofs_with_full_context_and_dual_catalog_verification"),
            slc_ceda_full_burst_tanks_with_both_archive_clocks=len({a["candidate_id"] for a in ceda_full
                if yes(a,"both_archive_object_clocks_before_cutoff") and yes(a,"both_parent_catalog_clocks_before_cutoff")}),
            slc_ceda_dual_pol_window_tanks_with_both_archive_clocks=len({a["candidate_id"] for a in ceda_windows
                if yes(a,"all_versions_before_cutoff")}),
            rtc_same_track_tanks_with_complete_local_pixels=int(rtc_week["tank_weeks_with_complete_context"]),
            rtc_same_track_tanks_with_both_archive_versions_pre_cutoff=int(rtc_week["tank_weeks_with_both_versions_before_cutoff"]),
            opera_gap_dated_roofs_with_full_context=cslc.get("opera_full_context_roofs"),
            opera_gap_dated_roofs_with_full_context_before_cutoff=cslc.get("opera_full_context_before_cutoff_roofs"),
            nisar_same_stack_catalog_pairs_before_cutoff=int(nasa_week["nisar_same_stack_pairs_before_cutoff"]),
            nisar_current_catalog_scenes_before_cutoff=len(json.loads(nasa_week["current_nisar_scene_ids_before_cutoff_json"])),
            aster_current_vnir_catalog_scenes_before_cutoff=len(json.loads(nasa_week["current_aster_scene_ids_before_cutoff_json"])),
            icesat2_current_retrospective_roofs=int(laser_week["current_retrospective_roofs"]),
            icesat2_current_roofs_with_cmr_insert_before_cutoff=int(laser_week["current_roofs_with_cmr_insert_before_cutoff"]),
            icesat2_same_roofs_with_both_endpoints=int(laser_week["same_roofs_with_both_endpoints"]),
            atl03_native_current_roofs=int(atl03["current_retrospective_roofs"]),
            atl03_native_current_roofs_with_cmr_insert_before_cutoff=int(atl03["current_roofs_with_cmr_insert_before_cutoff"]),
            atl03_native_same_roofs_at_both_endpoints=int(atl03["same_roofs_with_both_endpoints"]),
            grd_gap_dated_roofs_with_exact_versions_pre_cutoff=grd.get("dated_roofs_with_exact_versions_pre_cutoff"),
            grd_gap_dated_roofs_with_maximum_layover_above_nominal_resolution=grd.get("dated_roofs_with_maximum_layover_above_nominal_resolution"),
            wti_predictor_source=w["predictor_source"],wti_feature_components=int(w["forecast_features_present"]),
            wti_execution_source=w["execution_source"],wti_near_month=r["near_month"],wti_far_month=r["far_month"],
            wti_entry_session=r["entry_session"],wti_exit_session=r["exit_session"],
            wti_nonoverlapping=yes(r,"nonoverlapping_window_all"),wti_b04_nonoverlapping=yes(r,"nonoverlapping_window_b04"),
            equity_public_model_symbols="|".join(public_symbols),equity_optical_symbols="|".join(measured_symbols),
            equity_nonoverlapping=True in paired_flags))
    if seen_measured!=measured.keys() or seen_public!=public.keys():
        raise ValueError("Unused equity replay provenance")
    return days,coverage,gaps


def intervals(dates):
    span = [(b-a).days for a,b in zip(dates,dates[1:])]
    return dict(observations=len(dates),minimum_days=min(span),median_days=median(span),maximum_days=max(span),
                frequencies=dict(sorted(Counter(span).items())))


def main():
    days,equity,gaps = build()
    measured = [r for r in days if int(r["baseline_b04_tanks"])]
    b8_downloads = read_csv("cushing_b8_download_manifest")
    b8_download_status = Counter(r["b8_status"] for r in b8_downloads)
    if (len({r["product_id"] for r in b8_downloads}) != len(b8_downloads)
        or b8_download_status.keys() - {"awaiting_original_b8", "verified_native_crop"}):
        raise ValueError("Invalid original B8 status in the download manifest")
    report = dict(release_cadence=intervals([date.fromisoformat(r["release_date"]) for r in days]),
        b04_observation_cadence=intervals([date.fromisoformat(r["week_ending"]) for r in measured]),
        first_release=days[0]["release_date"],last_release=days[-1]["release_date"],
        eia_release_timing_basis=dict(sorted(Counter(r["eia_release_timing_basis"] for r in days).items())),
        eia_weekly_workbook_change_disagreements=sum(not r["eia_weekly_workbook_change_matches"] for r in days),
        cftc_weekly_source_status=dict(sorted(Counter(r["cftc_weekly_source_status"] for r in days).items())),
        cftc_weekly_footer_dates_before_cutoff=sum(r["cftc_weekly_footer_date_before_cutoff"] is True for r in days),
        cftc_weekly_files_modified_by_cutoff=sum(r["cftc_weekly_file_modified_by_cutoff"]=="True" for r in days),
        hrrr_native_both_endpoints_before_cutoff=sum(r["hrrr_native_both_endpoints_before_cutoff"] for r in days),
        hrrr_native_prior_cutoff_hour_complete=sum(r["hrrr_native_prior_cutoff_hour_complete"] for r in days),
        b04_status=dict(Counter(r["b04_status"] for r in days)),
        b04_tank_weeks=sum(int(r["baseline_b04_tanks"]) for r in days),
        b04_median_visible_area_fraction=median(float(r["baseline_b04_area_fraction"]) for r in measured),
        b04_median_space_time_fraction=median(float(r["baseline_b04_space_time_fraction"]) for r in measured),
        b04_median_area_weighted_capture_interval_hours=median(float(r["area_weighted_capture_interval_hours"]) for r in measured),
        b04_and_all_alternate_views=sum(int(r["baseline_b04_tanks"])>0 and r["additional_optical_views"]==3 for r in days),
        b04_gap_weeks_with_dated_timing_candidate=sum(r["dated_timing_candidate_in_b04_gap"] for r in days),
        b04_gap_weeks_with_modern_lidar_measurement=[r["week_ending"] for r in days if
            r["b04_status"]!="measured" and r["modern_lidar_additional_resolved_tanks"]],
        b04_gap_weeks_with_wider_registration_measurement=[r["week_ending"] for r in days if r["wider_registration_resolved_tanks"]],
        b04_gap_weeks_with_either_auxiliary_measurement=sum(r["b04_status"]!="measured" and (
            r["modern_lidar_additional_resolved_tanks"]>0 or r["wider_registration_resolved_tanks"]>0) for r in days),
        b04_gap_weeks_with_landsat_c2_b8_pair_candidate=sum(int(r["landsat_c2_gap_tanks_with_any_b8_pair_candidate"] or 0)>0 for r in days),
        b04_gap_weeks_with_landsat_c2_native_qa_pair_candidate=sum(
            int(r["landsat_c2_gap_pair_candidates_with_native_qa"] or 0)>0 for r in days),
        b04_gap_weeks_with_landsat_c2_full_footprint_qa_candidate=sum(
            int(r["landsat_c2_gap_pair_candidates_with_full_footprint_qa"] or 0)>0 for r in days),
        b04_gap_tank_weeks_with_landsat_c2_full_footprint_qa_candidate=sum(
            int(r["landsat_c2_gap_pair_candidates_with_full_footprint_qa"] or 0) for r in days),
        landsat_c2_gap_exact_scene_pairs=sum(int(r["landsat_c2_gap_exact_scene_pairs"] or 0) for r in days),
        landsat_c2_gap_tank_weeks_with_all_required_b8_staged=sum(
            int(r["landsat_c2_gap_tanks_with_all_required_b8_staged"] or 0) for r in days),
        landsat_c2_gap_tank_weeks_with_both_fits_resolved=sum(
            int(r["landsat_c2_gap_tanks_with_both_fits_resolved"] or 0) for r in days),
        landsat_c2_gap_pairs_with_checked_red_context=sum(int(r["landsat_c2_gap_pairs_with_checked_red_context"] or 0) for r in days),
        landsat_c2_gap_tank_weeks_with_checked_red_context=sum(
            int(r["landsat_c2_gap_tanks_with_checked_red_context"] or 0) for r in days),
        landsat_c2_gap_weeks_with_checked_red_context=sum(
            int(r["landsat_c2_gap_tanks_with_checked_red_context"] or 0)>0 for r in days),
        esri_b8_retrospective_coverage={field:dict(tank_weeks=sum(int(r[f"esri_b8_gap_tanks_{field}"] or 0) for r in days),
            weeks=sum(int(r[f"esri_b8_gap_tanks_{field}"] or 0)>0 for r in days)) for field in (
                "with_all_windows_retrospective","with_usable_esri_inputs","with_two_resolved_fits_retrospective",
                "resolved_with_checked_red_context","resolved_with_checked_b8_exterior",
                "resolved_with_checked_b8_exterior_and_source_clock","with_jointly_resolved_change","with_stable_change_direction")},
        landsat_c2_archive_timing={field:dict(
            tank_weeks=sum(int(r[f"landsat_c2_gap_tanks_with_{field}"] or 0) for r in days),
            weeks=sum(int(r[f"landsat_c2_gap_tanks_with_{field}"] or 0)>0 for r in days))
            for field in ("cdse_bundle_clock","hls_lineage_clock","either_archive_clock","archive_clock_and_checked_context")},
        landsat_c2_gap_usgs_stac_metadata_timing=dict(tank_weeks=sum(int(r["landsat_c2_gap_tanks_with_usgs_stac_metadata_before_cutoff"] or 0)
            for r in days),weeks=sum(int(r["landsat_c2_gap_tanks_with_usgs_stac_metadata_before_cutoff"] or 0)>0 for r in days)),
        landsat_c2_b8_downloads=dict(total_exact_originals=len(b8_downloads),
            awaiting_original_b8=b8_download_status["awaiting_original_b8"],
            verified_native_crop=b8_download_status["verified_native_crop"],
            independent_icesat2_control_scenes=sum(r["purpose"]=="icesat2_laser_reference" for r in b8_downloads),
            manifest="data/derived/cushing_b8_download_manifest.csv"),
        b04_gap_weeks_with_slc_full_burst_context=sum(int(r["slc_gap_dated_roofs_with_full_burst_context"] or 0)>0 for r in days),
        radar_local_pre_cutoff={key:dict(weeks=sum(bool(r[key]) for r in days),
            tank_weeks=sum(r[key] for r in days), optical_gap_weeks=sum(
                r["b04_status"]!="measured" and bool(r[key]) for r in days)) for key in (
                "slc_ceda_full_burst_tanks_with_both_archive_clocks",
                "slc_ceda_dual_pol_window_tanks_with_both_archive_clocks",
                "rtc_same_track_tanks_with_both_archive_versions_pre_cutoff")},
        b04_gap_weeks_with_opera_full_context_before_cutoff=sum(
            int(r["opera_gap_dated_roofs_with_full_context_before_cutoff"] or 0)>0 for r in days),
        nisar_same_stack_catalog_weeks_before_cutoff=[r["week_ending"] for r in days if r["nisar_same_stack_catalog_pairs_before_cutoff"]],
        b04_gap_weeks_with_nisar_same_stack_catalog_before_cutoff=[r["week_ending"] for r in days if
            r["b04_status"]!="measured" and r["nisar_same_stack_catalog_pairs_before_cutoff"]],
        b04_gap_weeks_with_aster_current_vnir_catalog_before_cutoff=[r["week_ending"] for r in days if
            r["b04_status"]!="measured" and r["aster_current_vnir_catalog_scenes_before_cutoff"]],
        icesat2_current_height_control_weeks=[r["week_ending"] for r in days if r["icesat2_current_retrospective_roofs"]],
        icesat2_same_roofs_both_endpoints_weeks=[r["week_ending"] for r in days if r["icesat2_same_roofs_with_both_endpoints"]],
        atl03_native_current_height_weeks=[r["week_ending"] for r in days if r["atl03_native_current_roofs"]],
        atl03_native_same_roofs_at_both_endpoints_weeks=[r["week_ending"] for r in days if r["atl03_native_same_roofs_at_both_endpoints"]],
        b04_gap_weeks_with_landsat_c2_or_opera_candidate=sum(
            int(r["landsat_c2_gap_tanks_with_any_b8_pair_candidate"] or 0)>0
            or int(r["opera_gap_dated_roofs_with_full_context_before_cutoff"] or 0)>0 for r in days),
        b04_gap_weeks_with_grd_native_pairs_pre_cutoff=sum(int(r["grd_gap_dated_roofs_with_exact_versions_pre_cutoff"] or 0)>0 for r in days),
        wti_paired_windows={k:sum(r[k] for r in days) for k in ("wti_nonoverlapping","wti_b04_nonoverlapping")},
        operator_paired_windows=sum(r["equity_nonoverlapping"] for r in days),
        same_release_in_both_measured_trade_calendars=sum(r["equity_nonoverlapping"] and r["wti_b04_nonoverlapping"] for r in days),
        equity_by_instrument={s:dict(public_model_training_windows=sum(r["public_model_training_eligible"] for r in equity if r["instrument"]==s),
            paired_optical_windows=sum(r["paired_optical_models_eligible"] for r in equity if r["instrument"]==s)) for s in sorted(CORE)},
        equity_accounting_gaps=len(gaps),equity_windows_with_price_basis_bridges=sum(bool(r["accounting_quote_bridge_keys"]) for r in equity),
        interpretation="Data coverage and source provenance. No numerical EIA targets, equity or WTI closing prices, returns or backtest scores are exported.")
    for suffix,rows in (("releases",days),("equities",equity)):
        export(f"{STEM}_{suffix}",rows)
    write_csv(DERIVED/f"{STEM}_equity_quote_gaps.csv",GAP_FIELDS,([r[k] for k in GAP_FIELDS] for r in gaps))
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="")


if __name__=="__main__":
    main()
