"""Write a coverage report from the normalized database and completed downloads."""
import collections
import csv
import datetime as dt
import json
import sqlite3

from collect_imagery import BANDS, candidate_footprint, canonical_items, earliest_alternate_items
from collect_public import RAW, ROOT, write
from collect_s2_l1c import CHANNELS, OUT as L1C

DERIVED = ROOT / "data" / "derived"
LOGS = ROOT / "data" / "logs"


def query(db, statement):
    return [dict(row) for row in db.execute(statement)]


def download_log(pattern):
    latest = {}
    successful = set()
    for path in sorted(LOGS.glob(pattern)):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row["id"]
            latest[key] = row
            if row.get("dimensions") or row.get("bytes"):
                successful.add(key)
    return {
        "unique_attempted": len(latest),
        "completed_in_logs": len(successful),
        "failed_without_later_success": dict(collections.Counter(
            row["error"] for key, row in latest.items() if key not in successful and "error" in row)),
    }


def l1c_source_readback_coverage(db):
    path = DERIVED / "sentinel2_l1c_source_readback.csv"
    if not path.is_file():
        return None
    expected = dict(db.execute("SELECT canonical_l2a_id,version_id FROM sentinel2_l1c_input_inventory"))
    folders = {row[0] for row in db.execute("SELECT source_directory FROM sentinel2_l1c_local_file_check")}
    checked, verified, mismatched, unchecked_folders = set(), set(), set(), set()
    counts = collections.Counter()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            identity = row["canonical_l2a_id"]
            checked.add(identity)
            counts[row["status"]] += 1
            if identity not in expected or row["version_id"] != (expected[identity] or ""):
                mismatched.add(identity)
            if row["status"] == "verified":
                verified.add(identity)
                if row["local_directory"] not in folders:
                    unchecked_folders.add(row["local_directory"])
    return {
        "expected_acquisitions": len(expected), "acquisitions_in_readback_report": len(checked),
        "acquisitions_with_verified_files": len(verified), "status_counts": dict(counts),
        "catalog_acquisitions_absent_from_report": len(expected.keys() - checked),
        "catalog_version_mismatches": len(mismatched), "verified_folders_absent_from_quality_index": len(unchecked_folders),
    }


def sentinel2_coverage():
    by_year = collections.defaultdict(collections.Counter)
    for item in canonical_items():
        year = item["properties"]["datetime"][:4]
        identity = item["id"]
        base = RAW / "imagery_core" / "sentinel2" / year / identity
        meta = RAW / "sentinel2_metadata" / identity
        by_year[year]["selected"] += 1
        by_year[year]["bands_and_item_on_disk"] += int(
            (base / "item.json").is_file() and all((base / f"{band}.tif").is_file() for band in BANDS))
        by_year[year]["granule_xml_on_disk"] += int((meta / "granule_metadata.xml").is_file())
        by_year[year]["product_xml_on_disk"] += int((meta / "product_metadata.xml").is_file())
    return dict(sorted(by_year.items()))


def supplemental_imagery_coverage():
    catalog = RAW / "stac_supplemental"
    layer = DERIVED / "supplemental_tank_candidates.geojson"
    if not catalog.is_dir() or not layer.is_file():
        return {}
    _, area = candidate_footprint(layer, 120)
    canonical = canonical_items(catalog, area)
    alternate = earliest_alternate_items(catalog, area)
    collections_to_check = (
        ("supplemental_l2a", canonical, RAW / "sentinel2_supplemental" / "sentinel2", BANDS),
        ("supplemental_earliest_alternate_l2a", alternate, RAW / "sentinel2_supplemental_original" / "sentinel2", BANDS),
        ("supplemental_l1c_2017_2018", [i for i in canonical if i["properties"]["datetime"][:4] in {"2017", "2018"}],
         L1C / "supplemental" / "granules", CHANNELS),
        ("combined_l1c_2019_2023", [i for i in canonical if "2019" <= i["properties"]["datetime"][:4] <= "2023"],
         L1C / "combined_2019_2023" / "granules", CHANNELS),
        ("combined_l1c_2024_2026", [i for i in canonical if "2024" <= i["properties"]["datetime"][:4] <= "2026"],
         L1C / "combined_2024_2026" / "granules", CHANNELS),
    )
    summary = {}
    for label, items, storage, bands in collections_to_check:
        by_year = collections.defaultdict(collections.Counter)
        for item in items:
            year, identity = item["properties"]["datetime"][:4], item["id"]
            directory = storage / year / identity
            by_year[year]["selected"] += 1
            by_year[year]["bands_and_item_on_disk"] += int(
                (directory / "item.json").is_file() and all((directory / f"{band}.tif").is_file() for band in bands))
        summary[label] = dict(sorted(by_year.items()))
    return summary


def main():
    path = DERIVED / "cushing.sqlite"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        report = {
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "database_integrity": db.execute("PRAGMA integrity_check").fetchone()[0],
            "table_counts": {
                name: db.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
                for name, in db.execute("SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name")
            },
            "eia_series_coverage": query(db, """
                SELECT series, source_key, count(*) AS n, min(date) AS first, max(date) AS last
                FROM eia_series GROUP BY series, source_key ORDER BY series
            """),
            "eia_release": query(db, """
                SELECT count(*) AS releases, count(DISTINCT week_ending) AS current_weeks,
                    min(week_ending) AS first_week, max(week_ending) AS last_week
                FROM eia_release
            """)[0],
            "eia_as_published": query(db, """
                SELECT count(DISTINCT week_ending) AS observed_weeks, count(*) AS published_values,
                    sum(precision_kbbl>1) AS rounded_pdf_values FROM eia_cushing_as_published
            """)[0],
            "eia_control_vintage_coverage": query(db, """
                SELECT metric,unit,count(DISTINCT release_date) AS releases,count(value) AS numeric_values,
                    min(report_date) AS first_observation,max(report_date) AS last_observation,
                    sum(source_kind='recovered_issue_pdf') AS recovered_pdf_values
                FROM eia_control_vintage GROUP BY metric,unit ORDER BY metric
            """),
            "eia_market_context_timing": query(db, """
                SELECT c.phase,count(*) AS metric_rows,count(DISTINCT c.release_date) AS issues,
                    sum(c.value IS NULL) AS missing_values,
                    count(DISTINCT CASE WHEN c.phase='pre_release' AND c.last_wpsr_issue>=c.release_date
                        OR c.phase='post_release' AND c.last_wpsr_issue<>c.release_date THEN c.release_date END) AS issue_timing_conflicts,
                    count(DISTINCT CASE WHEN o.last_complete_wti_curve_session>=substr(c.cutoff_utc,1,10)
                        OR julianday(o.cftc_conservative_from_utc)>julianday(c.cutoff_utc) THEN c.release_date END) AS market_timing_conflicts
                FROM eia_market_context c JOIN oil_market_context o USING (release_date,phase) GROUP BY c.phase
            """),
            "eia_core_equity_horizons": query(db, """
                SELECT instrument,own_quote_status,benchmark_alignment,count(*) AS release_windows
                FROM eia_equity_horizon_input WHERE study_role='core_operator'
                GROUP BY instrument,own_quote_status,benchmark_alignment ORDER BY instrument
            """),
            "eia_cad_equity_reference_inputs": query(db, """
                SELECT instrument,cross_venue_clock_status,count(*) AS release_windows,
                    count(CASE WHEN entry_fx_date IS NOT NULL AND fifth_fx_date IS NOT NULL THEN 1 END)
                        AS windows_with_both_fx_references,
                    count(CASE WHEN julianday(entry_fx_available_at_utc)>julianday(entry_open_utc)
                        OR julianday(fifth_fx_available_at_utc)>julianday(fifth_close_utc) THEN 1 END)
                        AS fx_availability_violations
                FROM eia_cad_equity_reference_inputs GROUP BY instrument,cross_venue_clock_status
                ORDER BY instrument,cross_venue_clock_status
            """),
            "equity_tail_quote_sources": query(db, """
                SELECT session_date,provider,count(*) AS quote_bars FROM market_study_quote
                WHERE provider IN ('ibkr_connector_tail','yahoo_finance_tail') GROUP BY session_date,provider
            """),
            "eia_weekly_satellite_readiness": query(db, """
                SELECT study_phase, count(*) AS target_weeks,
                    sum(current_l1c_archive_products>0 AND previous_l1c_archive_products>0) AS l1c_archive_both_weeks,
                    sum(current_l1c_screened_products>0 AND previous_l1c_screened_products>0) AS l1c_local_both_weeks,
                    sum(paired_l1c_candidates>0) AS weeks_with_l1c_pair,
                    sum(paired_l2a_candidates>0) AS weeks_with_l2a_pair,
                    sum(paired_candidates_same_level>0) AS weeks_with_same_level_pair,
                    sum(paired_with_prior_aerial_tank_seen>0) AS weeks_with_pair_and_prior_visible_tank
                FROM eia_weekly_satellite_readiness GROUP BY study_phase
            """),
            "l1c_local_file_validation": query(db, """
                SELECT capture_year, status, count(*) AS folders, sum(tank_rows) AS tank_rows
                FROM sentinel2_l1c_local_file_check GROUP BY capture_year, status ORDER BY capture_year, status
            """),
            "l1c_exact_input_inventory": query(db, """
                SELECT capture_year,archive,input_status,count(*) AS canonical_granules,
                    sum(local_folders_failed) AS failed_local_folders,
                    sum(clear_geometry_contours>0) AS granules_with_clear_geometry,
                    sum(eligible_week_endpoints>0) AS granules_in_eia_windows
                FROM sentinel2_l1c_input_inventory
                GROUP BY capture_year,archive,input_status ORDER BY capture_year,archive,input_status
            """),
            "l1c_source_readback": l1c_source_readback_coverage(db),
            "l1c_alternate_catalog_evidence": query(db, """
                SELECT a.status,count(*) AS products,min(a.cdse_published_at_utc) AS first_catalog_publication,
                    max(a.cdse_published_at_utc) AS last_catalog_publication,
                    count(CASE WHEN f.status='screened' THEN 1 END) AS locally_screened_products,
                    sum(EXISTS(SELECT 1 FROM eia_satellite_window_input w
                        WHERE w.level='L1C' AND w.canonical_l2a_id=a.canonical_l2a_id
                            AND w.version_id=a.l1c_product_name)) AS products_used_in_study_windows
                FROM sentinel2_l1c_alternate_evidence a
                LEFT JOIN sentinel2_l1c_local_file_check f ON f.source_directory=a.local_directory
                GROUP BY a.status ORDER BY a.status
            """),
            "l1c_tandem_pairs": query(db, """
                SELECT capture_day,count(*) AS paired_tiles,
                    count(CASE WHEN s2a_source_directory IS NOT NULL AND s2c_source_directory IS NOT NULL THEN 1 END)
                        AS paired_local_screened_tiles,
                    min(s2c_minus_s2a_seconds) AS min_capture_gap_seconds,
                    max(s2c_minus_s2a_seconds) AS max_capture_gap_seconds
                FROM sentinel2_l1c_tandem_pair GROUP BY capture_day ORDER BY capture_day
            """),
            "eia_current_column_gaps": query(db, """
                SELECT date, value, (SELECT min(release_date) FROM eia_cushing_vintages v WHERE v.week_ending=e.date)
                    AS earliest_archived_observation
                FROM eia_series e WHERE series='cushing_weekly'
                  AND date BETWEEN (SELECT min(week_ending) FROM eia_release)
                               AND (SELECT max(week_ending) FROM eia_release)
                  AND date NOT IN (SELECT week_ending FROM eia_release)
            """),
            "wpsr_control_tables": query(db, """
                SELECT table_number, count(DISTINCT release_date) AS issues,
                    count(*) AS published_cells, min(release_date) AS first, max(release_date) AS last
                FROM wpsr_vintage_value GROUP BY table_number ORDER BY table_number
            """),
            "equity_coverage": query(db, """
                SELECT instrument, conid, count(*) AS closed_session_rows, min(session_date) AS first,
                    max(session_date) AS last, broker_source
                FROM equity_closed_sessions GROUP BY instrument, conid, broker_source
                ORDER BY instrument
            """),
            "oke_vendor_adjustment_check": query(db, """
                SELECT substr(session_date,1,4) AS year, count(*) AS sessions,
                    sum(issuer_cash>0) AS issuer_payments,
                    sum(abs(vendor_implied_cash-issuer_cash)>0.01) AS vendor_cash_mismatches
                FROM oke_equity_total_return GROUP BY year ORDER BY year
            """),
            "market_identity_evidence": query(db, "SELECT * FROM market_identity_evidence ORDER BY session_date"),
            "equity_research_coverage": query(db, """
                SELECT instrument,provider,count(*) AS sessions,min(session_date) AS first,max(session_date) AS last
                FROM equity_research_trade_bar GROUP BY instrument,provider ORDER BY instrument,provider
            """),
            "equity_reference_price_checks": query(db, """
                SELECT instrument,comparison,count(*) AS checked_sessions,
                    sum(abs(tws_close-independent_close)>0.005) AS differences,
                    max(abs(tws_close-independent_close)) AS max_abs_difference
                FROM equity_reference_price_check GROUP BY instrument,comparison ORDER BY instrument,comparison
            """),
            "dolt_historical_equity_quality": query(db, """
                SELECT listing_id,symbol,quality_status,count(*) AS rows
                FROM dolt_historical_equity_quality GROUP BY listing_id,symbol,quality_status
                ORDER BY listing_id,quality_status
            """),
            "historical_equity_source_coverage": query(db, """
                SELECT listing_id, symbol, count(*) AS xnys_sessions, sum(valid_dolt_bar) AS valid_dolt_bars,
                    sum(adjusted_close IS NOT NULL) AS adjusted_daily_bars,
                    sum(tickertech_close IS NOT NULL) AS tickertech_daily_bars,
                    sum(issuer_reported_price IS NOT NULL) AS issuer_price_anchors,
                    sum(reference_close IS NOT NULL) AS reference_daily_bars,
                    sum(valid_dolt_bar=0 AND adjusted_close IS NULL AND tickertech_close IS NULL
                        AND reference_close IS NULL) AS sessions_without_daily_bar
                FROM historical_equity_source_coverage GROUP BY listing_id, symbol ORDER BY listing_id
            """),
            "historical_equity_monthly_reference": query(db, """
                SELECT listing_id, count(*) AS months, min(month) AS first, max(month) AS last,
                    count(matched_dolt_session) AS independently_matched
                FROM historical_equity_monthly_reference GROUP BY listing_id
            """),
            "historical_distribution_ex_date_estimate": query(db, """
                SELECT instrument,estimate_basis,count(*) AS distributions,count(estimated_ex_date) AS dates_estimated
                FROM historical_distribution_ex_date_estimate GROUP BY instrument,estimate_basis
            """),
            "dolt_independent_closing_price_check": query(db, """
                SELECT d.symbol,count(*) AS overlapping_sessions,
                    sum(abs(d.close-r.close)>0.005) AS close_differences_over_half_cent,
                    max(abs(d.close-r.close)) AS max_absolute_difference
                FROM dolt_historical_equity_bar d
                JOIN equity_research_trade_bar r ON r.instrument=d.symbol AND r.session_date=d.session_date
                WHERE d.symbol IN ('OKE','MMP') GROUP BY d.symbol ORDER BY d.symbol
            """),
            "market_identity_excluded_tws_bars": query(db, """
                SELECT t.instrument,t.series,count(*) AS excluded_rows,
                    min(t.session_date) AS first,max(t.session_date) AS last
                FROM tws_daily_bar t JOIN market_identity_limit i USING (instrument)
                WHERE t.session_date<i.first_research_session GROUP BY t.instrument,t.series
            """),
            "futures_coverage": query(db, """
                SELECT instrument, conid, count(*) AS bars, sum(CASE WHEN volume > 0 THEN 1 ELSE 0 END) AS positive_volume,
                    min(bar_start_utc) AS first, max(bar_start_utc) AS last
                FROM market_bar WHERE security_type='FUT' GROUP BY instrument, conid ORDER BY instrument
            """),
            "ibkr_unavailable": query(db, """
                SELECT instrument, conid, status FROM ibkr_coverage WHERE status!='downloaded' ORDER BY instrument
            """),
            "weather_stations": query(db, """
                SELECT station, count(*) AS n, min(observed_at_utc) AS first, max(observed_at_utc) AS last,
                    sum(report_kind='reported_metar_text') AS reported_metar_text,
                    sum(report_kind='iem_ghcnh_derived') AS iem_ghcnh_derived,
                    count(altimeter_inhg) AS with_altimeter,count(sea_level_pressure_hpa) AS with_sea_level_pressure,
                    sum(precip_trace=1) AS trace_precipitation
                FROM weather_metar GROUP BY station ORDER BY station
            """),
            "satellite_weather_references": query(db, """
                SELECT collection,count(*) AS granules,count(DISTINCT weather_reference_utc) AS times
                FROM satellite_weather_clock GROUP BY collection ORDER BY collection
            """),
            "scene_weather_timing": query(db, """
                SELECT station,report_kind,time_relation,count(*) AS reference_times,count(observed_at_utc) AS matched,
                    sum(abs(observed_minus_reference_seconds)<=3600) AS matched_within_one_hour,
                    sum(report_variants>1) AS ambiguous_report_times,
                    sum(CASE WHEN time_relation='at_or_before' AND observed_minus_reference_seconds>0
                        OR time_relation='at_or_after' AND observed_minus_reference_seconds<0 THEN 1 ELSE 0 END) AS timing_conflicts
                FROM scene_weather_match GROUP BY station,report_kind,time_relation ORDER BY station,report_kind,time_relation
            """),
            "nearest_weather_station_by_contour": query(db, """
                SELECT station,count(*) AS contours,min(distance_km) AS min_km,max(distance_km) AS max_km
                FROM tank_weather_station_distance WHERE distance_rank=1 GROUP BY station ORDER BY station
            """),
            "scene_catalogs": query(db, """
                SELECT collection, count(*) AS items, sum(canonical_sentinel2) AS selected_sentinel2,
                    min(captured_at_utc) AS first, max(captured_at_utc) AS last
                FROM satellite_scene GROUP BY collection ORDER BY collection
            """),
            "sentinel2_local_quality": query(db, """
                SELECT substr(captured_at_utc, 1, 4) AS year, count(DISTINCT scene_item_id) AS granules,
                    count(DISTINCT candidate_id) AS candidates, count(*) AS rows,
                    sum(optical_cells_valid_in_all_bands=optical_touched_cells_expected) AS all_optical_cells_valid,
                    sum(optical_cells_valid_in_all_bands=0) AS no_valid_optical_cells
                FROM sentinel2_tank_scene_quality JOIN tank_candidate USING (candidate_id)
                GROUP BY year ORDER BY year
            """),
            "sentinel2_supplemental_quality": query(db, """
                SELECT substr(captured_at_utc, 1, 4) AS year, count(DISTINCT scene_item_id) AS granules,
                    count(DISTINCT candidate_id) AS candidates, count(*) AS rows,
                    sum(CASE WHEN optical_cells_valid_in_all_bands=optical_touched_cells_expected
                        AND scl_touched_cells_expected=scl_class_0+scl_class_1+scl_class_2+scl_class_3+scl_class_4+
                            scl_class_5+scl_class_6+scl_class_7+scl_class_8+scl_class_9+scl_class_10+scl_class_11
                        AND scl_touched_cells_expected>0
                        AND scl_class_0+scl_class_1+scl_class_3+scl_class_8+scl_class_9+scl_class_10+scl_class_11=0
                        THEN 1 ELSE 0 END) AS rows_with_full_optical_without_excluded_roof_scl_classes
                FROM sentinel2_tank_scene_quality JOIN supplemental_tank_candidate USING (candidate_id)
                GROUP BY year ORDER BY year
            """),
            "eia_sentinel2_clock_coverage": query(db, """
                SELECT substr(latest_week_in_issue, 1, 4) AS year, count(*) AS issues,
                    sum(core_l2a_granules>0) AS core_l2a_clock_eligible_issues,
                    sum(core_l1c_granules>0) AS core_l1c_clock_eligible_issues
                FROM eia_sentinel2_timing_coverage GROUP BY year ORDER BY year
            """),
            "sentinel2_product_timing": query(db, """
                SELECT substr(captured_at_utc, 1, 4) AS year, count(*) AS granules,
                    sum(canonical_generation_lag_hours BETWEEN 0 AND 24) AS canonical_generated_within_24h,
                    sum(earliest_l2a_generation_lag_hours BETWEEN 0 AND 24) AS earliest_l2a_generated_within_24h,
                    count(earliest_l2a_local_directory) AS earliest_l2a_local,
                    count(l1c_local_directory) AS l1c_local,
                    sum(l1c_generation_before_tile_sensing) AS l1c_metadata_clock_inconsistencies
                FROM sentinel2_product_timing GROUP BY year ORDER BY year
            """),
            "tank_aerial_visual_observations": query(db, """
                SELECT substr(observed_date, 1, 4) AS year, image_source, structure_observation,
                    top_observation, o.candidate_id LIKE 'osm-way-%' AS in_core_osm, count(*) AS observations
                FROM tank_aerial_visual_observation o GROUP BY 1, 2, 3, 4, 5 ORDER BY 1, 2, 3, 4, 5
            """),
            "aerial_structural_changes": query(db, """
                SELECT event, count(*) AS intervals, count(DISTINCT candidate_id) AS contours
                FROM tank_aerial_change_interval GROUP BY event ORDER BY event
            """),
            "eia_storage_capacity": query(db, """
                SELECT area, count(DISTINCT period_month) AS months, count(*) AS values_in_source,
                    min(period_month) AS first, max(period_month) AS last
                FROM eia_crude_storage_capacity GROUP BY area ORDER BY area
            """),
            "eia_803_osm": query(db, """
                SELECT county, in_core, count(*) AS candidates, max(equivalent_diameter_m) AS largest_diameter_m
                FROM eia_803_osm_candidate GROUP BY county, in_core ORDER BY county, in_core
            """),
            "eia_803_epa": query(db, """
                SELECT reported_fips, count(*) AS records, count(petroleum_screen) AS petroleum_screened,
                    sum(point_county_fips IS NULL) AS point_outside_three_counties_or_missing,
                    sum(point_county_fips != reported_fips) AS fips_coordinate_mismatches
                FROM eia_803_epa_facility GROUP BY reported_fips ORDER BY reported_fips
            """),
            "eia_803_aerial_screen": query(db, """
                SELECT observation, count(*) AS windows FROM eia_803_facility_aerial_observation
                GROUP BY observation ORDER BY observation
            """),
            "supplemental_tanks": query(db, """
                SELECT source_site, validation_status, count(*) AS candidates,
                    sum(native_tile_ids IS NOT NULL) AS local_original_tile,
                    max(core_overlap_fraction) AS max_overlap_fraction
                FROM supplemental_tank_candidate GROUP BY source_site, validation_status ORDER BY source_site
            """),
            "tws_coverage": query(db, """
                SELECT instrument, conid, series, count(*) AS n, min(session_date) AS first,
                    max(session_date) AS last
                FROM tws_daily_bar GROUP BY instrument, conid, series ORDER BY instrument, series
            """),
            "tws_futures_coverage": query(db, """
                SELECT contract_month, conid, count(*) AS n, min(session_date) AS first, max(session_date) AS last,
                    sum(CASE WHEN volume > 0 THEN 1 ELSE 0 END) AS positive_volume
                FROM tws_futures_closed_sessions GROUP BY contract_month, conid ORDER BY contract_month
            """),
            "wti_dated_archive_coverage": query(db, """
                SELECT count(*) AS bars, count(DISTINCT delivery_month) AS contracts,
                    min(session_date) AS first, max(session_date) AS last FROM wti_tradingcharts_bar
            """),
            "wti_nearby_curve_coverage": query(db, """
                SELECT coverage_status, count(*) AS dates, min(session_date) AS first,
                    max(session_date) AS last FROM wti_curve_daily_coverage GROUP BY coverage_status
            """),
            "wti_incomplete_curve_dates": query(db, """
                SELECT session_date, priced_maturities, coverage_status, cme_no_settlement_notice_url
                FROM wti_curve_daily_coverage WHERE priced_maturities<4 ORDER BY session_date
            """),
            "wti_dated_close_crosscheck": query(db, """
                SELECT comparison, count(*) AS overlaps, sum(abs(difference)>0.005) AS differences_over_half_cent,
                    max(abs(difference)) AS max_absolute_difference
                FROM wti_close_crosscheck GROUP BY comparison
            """),
            "lidar_tank_coverage": query(db, """
                SELECT count(*) AS candidates, sum(returns_inside_polygon > 0) AS interior_covered,
                    sum(ground_class_2_returns_10_to_35m > 0) AS nearby_ground_covered
                FROM lidar_tank_coverage JOIN tank_candidate USING (candidate_id)
            """),
            "tank_engineering_registry_coverage": query(db, """
                SELECT count(DISTINCT candidate_id) AS mapped_candidates,
                    count(footprint_area_m2) AS projected_footprints,
                    count(lidar_upper_outline_diameter_m) AS fitted_lidar_outlines,
                    sum(lidar_fit_status='well_supported_circular_fit') AS well_supported_circular_fits,
                    sum(cohort_id IS NOT NULL) AS documented_cohort_rows,
                    sum(documented_roof_type='EFR') AS open_efr_in_documented_cohorts,
                    count(reported_capacity_min_bbl) AS documented_capacity_rows
                FROM tank_engineering_registry
            """)[0],
            "tank_engineering_cohorts": query(db, """
                SELECT terminal_id,cohort_id,count(*) AS contours,min(permit_diameter_m) AS permit_diameter_m,
                    sum(reported_capacity_min_bbl) AS cohort_capacity_min_bbl,
                    sum(reported_capacity_max_bbl) AS cohort_capacity_max_bbl
                FROM tank_engineering_cohort GROUP BY terminal_id,cohort_id ORDER BY terminal_id,cohort_id
            """),
            "plains_equipment_roster_comparison": query(db, """
                SELECT comparison,count(*) AS equipment_ids FROM plains_permit_equipment_history GROUP BY comparison
            """),
            "enbridge_main_equipment_roster_comparison": query(db, """
                SELECT comparison,count(*) AS equipment_ids FROM enbridge_main_permit_equipment_history GROUP BY comparison
            """),
            "enbridge_main_contour_reconciliation": query(db, """
                SELECT match_status,documented_roof_type,count(*) AS contours,
                    count(coalesce(lidar_upper_outline_diameter_m,lidar_covered_surface_outline_diameter_m)) AS measured_outlines,
                    count(lidar_2011_rim_above_ground_m) AS legacy_rim_rows
                FROM enbridge_main_contour_reconciliation GROUP BY match_status,documented_roof_type
            """),
            "enbridge_main_cohort_capacity": query(db, """
                SELECT * FROM enbridge_main_cohort_capacity ORDER BY cohort_id
            """),
            "ngl_cushing_contour_reconciliation": query(db, """
                SELECT parcel_group,match_status,count(*) AS contours,count(nominal_capacity_bbl) AS capacities,
                    sum(nominal_capacity_bbl) AS nominal_bbl,count(source_document_date) AS dated_specifications,
                    round(min(lidar_2011_rim_above_ground_m),3) AS minimum_2011_rim_m,
                    round(max(lidar_2011_rim_above_ground_m),3) AS maximum_2011_rim_m
                FROM ngl_cushing_contour_reconciliation GROUP BY parcel_group,match_status
            """),
            "enterprise_cushing_contour_reconciliation": query(db, """
                SELECT asset_scope_id,match_status,count(*) AS contours,count(nominal_capacity_bbl) AS capacities,
                    sum(nominal_capacity_bbl) AS nominal_bbl,
                    sum(aerial_top_2021='inset_wall_shadow_visible' OR aerial_top_2023='inset_wall_shadow_visible')
                        AS observed_inset_shadow_contours
                FROM enterprise_cushing_contour_reconciliation GROUP BY asset_scope_id,match_status
            """),
            "terminal_disclosed_capacity_coverage": query(db, """
                SELECT terminal_id,scope_key,capacity_basis,count(*) AS disclosures,
                    count(publication_date) AS dated_publications
                FROM terminal_disclosed_capacity GROUP BY terminal_id,scope_key,capacity_basis
            """),
            "north_farm_contour_reconciliation": query(db, """
                SELECT terminal_id,farm_block,match_status,cohort_id,count(*) AS contours,
                    count(lidar_upper_outline_diameter_m) AS lidar_outlines,sum(reported_capacity_min_bbl) AS permit_capacity_bbl
                FROM north_farm_contour_reconciliation GROUP BY terminal_id,farm_block,match_status,cohort_id
                ORDER BY terminal_id,farm_block,match_status,cohort_id
            """),
            "plains_cushing_contour_reconciliation": query(db, """
                SELECT match_status,cohort_id,count(*) AS contours,sum(capacity_min_bbl) AS cohort_capacity_bbl,
                    min(corporate_capacity_bbl_each) AS as_built_corporate_capacity_bbl_each,
                    min(earlier_corporate_capacity_bbl_each) AS earlier_corporate_capacity_bbl_each,
                    count(CASE WHEN aerial_structure_observation='aboveground_circular_shell_seen' THEN 1 END)
                        AS shells_seen_in_2008_mosaic,
                    min(lidar_upper_outline_diameter_m) AS lidar_diameter_min_m,
                    max(lidar_upper_outline_diameter_m) AS lidar_diameter_max_m,
                    min(lidar_minus_design_diameter_m) AS lidar_minus_design_diameter_min_m,
                    max(lidar_minus_design_diameter_m) AS lidar_minus_design_diameter_max_m
                FROM plains_cushing_contour_reconciliation GROUP BY match_status,cohort_id ORDER BY match_status,cohort_id
            """),
            "permit_epa_review_timeline": query(db, """
                SELECT count(*) AS review_windows,count(DISTINCT document_id) AS documents,
                    count(CASE WHEN epa_review_end_date<deq_memo_date THEN 1 END) AS review_end_before_local_memo
                FROM permit_epa_review_timeline
            """)[0],
            "terminal_data_coverage": query(db, """
                SELECT terminal_id,historical_tickers,permit_documents,associated_contours,parcel_and_permit_contours,
                    contours_with_multiple_terminal_associations,documented_open_roof_contours,
                    documented_covered_roof_contours,contours_with_reported_capacity,contours_with_reported_height,
                    contours_with_independent_design_dimensions
                FROM terminal_data_coverage ORDER BY terminal_id
            """),
            "lidar_supplemental_tank_coverage": query(db, """
                SELECT count(*) AS candidates, sum(returns_inside_polygon > 0) AS interior_covered,
                    sum(ground_class_2_returns_10_to_35m > 0) AS nearby_ground_covered
                FROM lidar_tank_coverage JOIN supplemental_tank_candidate USING (candidate_id)
            """),
            "lidar_flight_observations": query(db, """
                SELECT dataset,acquisition_date_utc, count(DISTINCT candidate_id) AS candidates,
                    count(DISTINCT point_source_id) AS flight_lines, count(*) AS observations,
                    min(first_interior_return_at_utc) AS first, max(last_interior_return_at_utc) AS last
                FROM lidar_tank_observation GROUP BY dataset,acquisition_date_utc ORDER BY acquisition_date_utc
            """),
            "lidar_2011_tank_coverage": query(db, """
                SELECT count(*) AS candidates,sum(a.returns_inside_polygon>0) AS interior_covered,
                    sum(a.ground_class_2_returns_10_to_35m>0) AS nearby_ground_covered,
                    sum(a.returns_inside_polygon>0 AND b.returns_inside_polygon>0) AS interior_in_both_surveys
                FROM lidar_2011_tank_coverage a JOIN lidar_tank_coverage b USING (candidate_id)
            """),
            "lidar_local_ept_nodes": query(db, """
                SELECT dataset,clip_scope,count(*) AS indexed_nodes,count(local_file) AS local_laz,
                    sum(selected_points) AS selected_points
                FROM lidar_ept_node GROUP BY dataset,clip_scope ORDER BY dataset,clip_scope
            """),
            "lidar_stroud_acquisition_evidence": query(db, """
                SELECT candidate_id,source_tile,fgdc_flight_start,fgdc_flight_end,interior_first_returns,
                    gps_week_time_min_s,gps_week_time_max_s,date_status,original_interior_first_returns,
                    ept_returns_matching_original_gps FROM lidar_stroud_legacy_observation
                    ORDER BY candidate_id
            """),
            "stroud_tank_specification_check": query(db, """
                SELECT candidate_id,documented_roof_type,earliest_group_construction,nominal_capacity_bbl,
                    application_shell_height_m,application_diameter_m,lidar_upper_outline_diameter_m,
                    lidar_minus_application_diameter_m,roof_capacity_memo_date,application_receipt_date,legacy_date_status
                FROM stroud_tank_specification_check ORDER BY candidate_id
            """),
            "lidar_optical_calibration_pairs": query(db, """
                SELECT level,substr(optical_captured_at_utc,1,10) AS capture_day,count(*) AS pairs,
                    count(DISTINCT scene_item_id) AS granules,count(DISTINCT candidate_id) AS contours,
                    count(DISTINCT CASE WHEN optical_cells=valid_cells AND roof_clear_by_source_mask THEN candidate_id END)
                        AS contours_complete_and_clear_by_source_mask,
                    CASE WHEN level='L1C' THEN count(DISTINCT CASE WHEN optical_cells=valid_cells
                        AND roof_clear_by_source_mask AND resolved_b04_detector THEN candidate_id END) END
                        AS contours_with_clear_optical_and_resolved_detector,
                    min(optical_minus_lidar_end_hours) AS min_signed_hours,max(optical_minus_lidar_start_hours) AS max_signed_hours,
                    sum(red_scale IS NULL OR red_offset IS NULL) AS missing_red_radiometry_pairs
                FROM lidar_optical_calibration_pair GROUP BY level,capture_day ORDER BY level,capture_day
            """),
            "sentinel2_band_radiometry": query(db, """
                SELECT level,metadata_status,count(*) AS band_rows,count(DISTINCT version_id) AS products
                FROM sentinel2_band_radiometry GROUP BY level,metadata_status ORDER BY level,metadata_status
            """),
            "lidar_calibration_b04_angle_nodes": query(db, """
                SELECT d.capture_day,n.node_role,count(*) AS rows,count(n.source_zenith_deg) AS with_zenith,
                    count(n.source_azimuth_deg) AS with_azimuth,count(DISTINCT n.candidate_id) AS contours
                FROM lidar_calibration_b04_angle_node n
                JOIN sentinel2_l1c_tank_quality q ON q.scene_item_id=n.scene_item_id
                    AND q.candidate_id=n.candidate_id AND q.l1c_product_name=n.version_id
                JOIN lidar_calibration_capture_day d ON substr(q.captured_at_utc,1,10)=d.capture_day
                GROUP BY d.capture_day,n.node_role ORDER BY d.capture_day,n.node_role
            """),
            "lidar_radial_profiles": query(db, """
                SELECT dataset,geometry_basis,count(*) AS profile_rows,count(DISTINCT candidate_id) AS contours,
                    count(CASE WHEN sector=-1 THEN 1 END) AS whole_ring_rows,
                    sum(CASE WHEN sector=-1 THEN first_returns ELSE 0 END) AS first_returns_by_geometry
                FROM lidar_radial_profile GROUP BY dataset,geometry_basis ORDER BY dataset,geometry_basis
            """),
            "lidar_radial_boundary_comparison": query(db, """
                SELECT dataset,geometry_basis,count(*) AS flight_line_pairs,
                    sum(mapped_polygon_rim_minus_interior_native_z<0) AS original_negative_with_profile,
                    sum(mapped_polygon_rim_minus_interior_native_z<0 AND boundary_minus_center_native_z<0)
                        AS original_negative_still_negative,
                    sum(mapped_polygon_rim_minus_interior_native_z<0 AND boundary_minus_center_native_z>=0)
                        AS original_negative_now_nonnegative
                FROM lidar_radial_boundary_comparison GROUP BY dataset,geometry_basis ORDER BY dataset,geometry_basis
            """),
            "roof_physics_input_coverage": query(db, """
                SELECT documented_roof_type,count(*) AS contours,
                    sum(dated_naip_years_through_2022>0) AS with_dated_naip_through_2022,
                    count(CASE WHEN aerial_top_2021 IS NOT NULL AND aerial_top_2023 IS NOT NULL THEN 1 END)
                        AS with_visual_roof_review_both_years,
                    sum(lidar_flight_lines>0) AS with_lidar_flight_lines,
                    sum(reference_flight_lines) AS reference_flight_lines,
                    sum(numeric_references) AS numeric_references,
                    sum(legacy_reference_flight_lines) AS legacy_reference_flight_lines,
                    sum(legacy_numeric_references) AS legacy_numeric_references,
                    sum(clear_l1c_versions_2022>0 AND clear_l1c_versions_2023>0) AS clear_l1c_both_days
                FROM roof_physics_input_coverage GROUP BY documented_roof_type ORDER BY documented_roof_type
            """),
            "lidar_roof_elevation_reference": query(db, """
                SELECT r.dataset,r.geometry_basis,t.documented_roof_type,count(*) AS flight_references,
                    count(DISTINCT r.candidate_id) AS contours,
                    sum(r.center_sectors=8 AND r.rim_sectors=8) AS full_sector_support,
                    min(r.rim_minus_center_plane_native_z) AS min_rim_minus_center_native_z,
                    max(r.rim_minus_center_plane_native_z) AS max_rim_minus_center_native_z
                FROM lidar_roof_elevation_reference r JOIN tank_engineering_registry t USING (candidate_id)
                WHERE t.documented_roof_type IS NOT NULL GROUP BY r.dataset,r.geometry_basis,t.documented_roof_type
                ORDER BY r.dataset,r.geometry_basis,t.documented_roof_type
            """),
            "eia_roof_cohort_weekly_coverage": query(db, """
                SELECT cohort_id,ticker,count(*) AS target_weeks,
                    count(CASE WHEN source_document_dated_before_cutoff THEN 1 END) AS weeks_after_document_date,
                    count(CASE WHEN source_document_dated_before_cutoff AND paired_l1c_prior_visible_contours>0 THEN 1 END)
                        AS weeks_after_document_date_with_l1c_pair,
                    max(paired_l1c_prior_visible_contours) AS max_l1c_paired_contours
                FROM eia_roof_cohort_weekly_coverage GROUP BY cohort_id,ticker ORDER BY cohort_id,ticker
            """),
            "tank_emissions_geometry_comparison": query(db, """
                SELECT cohort_id,application_received_date,count(*) AS contours,min(matched_equipment_ids) AS matched_equipment_ids,
                    min(emissions_diameter_m) AS emissions_diameter_m,min(lidar_upper_outline_diameter_m) AS min_lidar_diameter_m,
                    max(lidar_upper_outline_diameter_m) AS max_lidar_diameter_m,
                    min(lidar_circle_area_above_emissions_circle_pct) AS min_circle_area_difference_pct,
                    max(lidar_circle_area_above_emissions_circle_pct) AS max_circle_area_difference_pct
                FROM tank_emissions_geometry_comparison GROUP BY cohort_id,application_received_date ORDER BY cohort_id
            """),
            "magellan_central_fixed_roof_check": query(db, """
                SELECT tank_id,match_status,count(*) AS source_rows,count(DISTINCT document_id) AS documents,
                    min(capacity_gal) AS min_document_capacity_gal,max(capacity_gal) AS max_document_capacity_gal,
                    count(DISTINCT unassigned_parcel_contour_id) AS unassigned_contours,
                    min(equivalent_fill_depth_at_contour_area_m) AS min_equivalent_fill_depth_m,
                    max(equivalent_fill_depth_at_contour_area_m) AS max_equivalent_fill_depth_m
                FROM magellan_central_fixed_roof_check GROUP BY tank_id,match_status
            """),
            "lidar_negative_height_aerial_evidence": query(db, """
                SELECT latest_aerial_date,latest_aerial_top_observation,count(DISTINCT candidate_id) AS contours
                FROM lidar_radial_boundary_comparison WHERE dataset='OK_Statewide_2_D22'
                    AND geometry_basis='mapped_footprint_circle' AND mapped_polygon_rim_minus_interior_native_z<0
                GROUP BY latest_aerial_date,latest_aerial_top_observation ORDER BY latest_aerial_date,latest_aerial_top_observation
            """),
            "naip_tank_year_coverage": query(db, """
                SELECT substr(acquisition_date, 1, 4) AS year, candidate_id LIKE 'osm-way-%' AS in_core_osm,
                    count(DISTINCT candidate_id) AS mapped_candidates
                FROM naip_tank_image_coverage GROUP BY 1, 2 ORDER BY 1, 2
            """),
            "naip_source_clock_coverage": query(db, """
                SELECT acquisition_year,count(*) AS polygon_matches,count(DISTINCT candidate_id) AS contours,
                    count(DISTINCT CASE WHEN intersecting_source_polygons>1 THEN candidate_id END) AS multiple_polygon_contours,
                    count(DISTINCT CASE WHEN nominal_date_matches=0 THEN candidate_id END) AS contours_with_nominal_date_conflict,
                    count(DISTINCT CASE WHEN json_array_length(asset_paths_json)>0 THEN candidate_id END) AS contours_with_local_assets,
                    count(DISTINCT CASE WHEN json_array_length(dated_asset_paths_json)>0 THEN candidate_id END) AS contours_with_dated_assets
                FROM naip_tank_source_clock GROUP BY acquisition_year ORDER BY acquisition_year
            """),
            "naip_2025_rgb_coverage": query(db, """
                SELECT count(*) AS contours,count(DISTINCT clip_id) AS clips,
                    sum(candidate_pixels=valid_pixels AND candidate_pixels>0) AS fully_covered_contours
                FROM naip_2025_wcs_coverage
            """)[0],
            "aerial_optical_pair_coverage": query(db, """
                SELECT aerial_source,substr(aerial_date,1,4) AS year,level,count(*) AS pairs,
                    count(DISTINCT candidate_id) AS contours,min(minimum_separation_seconds) AS nearest_seconds,
                    count(DISTINCT CASE WHEN aerial_date=substr(satellite_captured_at_utc,1,10) THEN candidate_id END)
                        AS same_day_contours,
                    count(DISTINCT CASE WHEN maximum_separation_seconds<=3600 THEN candidate_id END) AS within_one_hour_contours,
                    count(DISTINCT CASE WHEN maximum_separation_seconds<=3600 AND roof_clear_by_source_mask
                        AND optical_cells=valid_cells THEN candidate_id END) AS within_one_hour_complete_clear_contours
                FROM aerial_optical_pair GROUP BY aerial_source,year,level ORDER BY aerial_source,year,level
            """),
            "landsat_catalog_coverage": query(db, """
                SELECT substr(acquired_date,1,4) AS year,platform,count(*) AS acquisitions,
                    sum(usgs_level1_catalog) AS usgs_level1_catalog,sum(pc_level2_catalog) AS pc_level2_catalog,
                    sum(gcs_pan_local) AS gcs_pan_local,sum(pc_level2_local) AS pc_level2_local,sum(hls_l30_local) AS hls_l30_local
                FROM landsat_route_coverage GROUP BY year,platform ORDER BY year,platform
            """),
            "landsat_archive_clocks": query(db, """
                SELECT provider,count(*) AS products,sum(assets) AS assets,min(latest_archive_object_time_utc) AS earliest,
                    max(latest_archive_object_time_utc) AS latest
                FROM landsat_product_archive_clock GROUP BY provider ORDER BY provider
            """),
            "landsat_hls_recovered_gaps": query(db, """
                SELECT gap_reason,count(*) AS target_acquisitions,sum(exact_hls_catalog_matches>0) AS exact_catalog_matches,
                    sum(local_hls_granules>0) AS local_matches,sum(contours_with_local_pixels>0) AS matches_with_valid_contour_pixels
                FROM landsat_hls_gap_coverage GROUP BY gap_reason ORDER BY gap_reason
            """),
            "landsat_local_roof_qa": query(db, """
                SELECT l.source_kind,substr(l.captured_at_utc,1,4) AS year,count(DISTINCT l.product_id) AS local_products,
                    count(DISTINCT q.candidate_id) AS contours,count(q.candidate_id) AS observations,
                    count(CASE WHEN q.signal_cells=q.valid_signal_cells AND q.qa_roof_cells=q.clear_qa_roof_cells THEN 1 END)
                        AS fully_sampled_with_clear_qa
                FROM landsat_local_product l LEFT JOIN landsat_tank_scene_quality q USING (product_id)
                GROUP BY l.source_kind,year ORDER BY l.source_kind,year
            """),
            "landsat_native_resolution_vs_contours": query(db, """
                SELECT count(*) AS contours,min(footprint_diameter_m) AS minimum_diameter_m,
                    max(footprint_diameter_m) AS maximum_diameter_m,sum(footprint_diameter_m>=4*15) AS diameters_at_least_four_pan_cells,
                    sum(footprint_diameter_m>=6*15) AS diameters_at_least_six_pan_cells
                FROM tank_engineering_registry WHERE footprint_diameter_m>0
            """)[0],
            "ok_aerial_tank_coverage": query(db, """
                SELECT count(DISTINCT candidate_id) AS candidates_with_pixels,
                    count(DISTINCT CASE WHEN valid_pixels=candidate_pixels THEN candidate_id END) AS single_tile_full,
                    (SELECT count(*) FROM ok_aerial_tank_union_coverage WHERE union_valid_pixels=candidate_pixels) AS union_full,
                    count(DISTINCT tile_id) AS tiles_with_tank_pixels
                FROM ok_aerial_tank_image_coverage
            """),
            "assessor_parcels": query(db, """
                SELECT county, snapshot_kind, snapshot_year, count(DISTINCT candidate_id) AS candidate_contours,
                    count(DISTINCT parcel_id) AS parcels,
                    count(DISTINCT CASE WHEN tank_area_fraction >= 0.99 THEN candidate_id END) AS almost_entirely_in_one_parcel
                FROM tank_parcel_intersection GROUP BY county, snapshot_kind, snapshot_year ORDER BY county, snapshot_year
            """),
            "parcel_and_permit_evidence": query(db, """
                SELECT facility_key, count(DISTINCT candidate_id) AS candidates, count(*) AS source_snapshots
                FROM tank_facility_evidence GROUP BY facility_key ORDER BY facility_key
            """),
            "parcel_and_corporate_evidence": query(db, """
                SELECT terminal_id, match_kind, count(DISTINCT candidate_id) AS candidates,
                    count(*) AS source_snapshots
                FROM tank_terminal_parcel_evidence GROUP BY terminal_id, match_kind ORDER BY terminal_id, match_kind
            """),
            "permit_cohort_evidence": query(db, """
                SELECT facility_key, cohort_id, diameter_evidence, count(*) AS contours
                FROM tank_permit_cohort_evidence GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
            """),
            "tank_data_readiness": query(db, """
                SELECT count(*) AS candidates, sum(full_state_aerial_2025) AS full_state_aerial_2025,
                    min(naip_years) AS minimum_naip_years,
                    sum(lidar_interior_returns>0) AS candidates_with_lidar,
                    sum(parcel_snapshot_years>0) AS candidates_with_parcel,
                    sum(corroborated_facility_candidates>0) AS candidates_with_parcel_and_permit_evidence
                FROM tank_data_readiness
            """)[0],
            "lidar_ept": query(db, """
                SELECT dataset, count(*) AS completed_nodes, sum(selected_points) AS selected_points,
                    sum(CASE WHEN local_file IS NOT NULL THEN 1 ELSE 0 END) AS point_files
                FROM lidar_ept_node GROUP BY dataset
            """),
            "downloads": {
                "sentinel2_by_year": sentinel2_coverage(),
                "sentinel2_expanded_by_year": supplemental_imagery_coverage(),
                "naip": download_log("naip_*.jsonl"),
                "lidar_original_laz": download_log("lidar_download.jsonl"),
                "ok_aerial_2025": download_log("ok_aerial_2025.jsonl"),
                "ok_aerial_2025_tank_clips": download_log("ok_aerial_2025_tank_clips.jsonl"),
            },
        }
    candidates = DERIVED / "osm_tank_candidates.geojson"
    if candidates.is_file():
        report["osm_candidate_count"] = len(json.loads(candidates.read_bytes())["features"])
    write(DERIVED / "coverage.json", json.dumps(report, indent=2, ensure_ascii=False).encode())
    print(json.dumps({k: report[k] for k in ("database_integrity", "table_counts", "downloads", "osm_candidate_count")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
