"""Build the local research database from downloaded source files."""
import csv
import datetime as dt
import gzip
import io
import json
import re
import sqlite3
import zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

import xlrd
from shapely.geometry import Point, shape

from collect_eia_803_area import screening_reason
from collect_issuer_distributions import replace_cash_records
from collect_fx_reference import SCHEMA as FX_REFERENCE_SCHEMA, add_fx_reference
from collect_permit_review_timeline import OUTPUT as EPA_REVIEW_CSV, SCHEMA as EPA_REVIEW_SCHEMA
from collect_public import RAW, ROOT
from collect_imagery import canonical_items
from collect_wpsr_factors import TABLES
from publication_clocks import cftc_rows, eia_rows, export
from market_sessions import release_alignment_rows, session_rows
from build_wti_curve import SCHEMA as WTI_SCHEMA, add_wti_curve
from build_eia_market_inputs import SCHEMA as EIA_MARKET_SCHEMA, add_eia_market_inputs
from build_lidar_optical_calibration import SCHEMA as LIDAR_OPTICAL_SCHEMA, add_lidar_optical_calibration
from build_lidar_radial_profiles import SCHEMA as LIDAR_RADIAL_SCHEMA
from build_lidar_roof_reference import SCHEMA as LIDAR_ROOF_REFERENCE_SCHEMA, add_lidar_roof_reference
from build_permit_tank_emissions_geometry import SCHEMA as PERMIT_EMISSIONS_GEOMETRY_SCHEMA, add_permit_tank_emissions_geometry
from build_market_event_panel import SCHEMA as MARKET_EVENT_SCHEMA, add_market_event_panel
from build_market_transaction_timeline import add_market_transaction_timeline
from build_scene_weather import SCHEMA as SCENE_WEATHER_SCHEMA, add_scene_weather
from build_aerial_optical_pairs import SCHEMA as AERIAL_OPTICAL_SCHEMA, add_aerial_optical_pairs
from build_landsat_coverage import SCHEMA as LANDSAT_SCHEMA, add_landsat
from build_weekly_satellite_panel import SCHEMA as WEEKLY_SATELLITE_SCHEMA, build_weekly_satellite_panel
from build_tank_geometry_registry import SCHEMA as TANK_ENGINEERING_SCHEMA, add_tank_geometry_registry
from build_terminal_scope_reconciliation import install_terminal_scopes
from terminal_aerial_roof_evidence import TABLE as EARLY_ROOF_TABLE, install_early_roof_review
from build_tlp_market_history import add_tlp_history
from build_cvrr_market_history import add_cvrr_history
from build_semg_market_history import add_semg_history
from build_hep_parent_exposure import EXPOSURE_SCHEMA as HEP_PARENT_SCHEMA, add_hep_parent_data
from build_deeprock_exposure import add_deeprock_data, export_deeprock_data
from build_execution_cost_inputs import add_execution_cost_inputs, export_execution_cost_inputs
from collect_early_equity_history import (
    SCHEMA as HISTORICAL_EQUITY_SCHEMA,
    add_adjusted_equity,
    add_amid_monthly_reference,
    add_tickertech_jpep,
)

DERIVED = ROOT / "data" / "derived"
DB = DERIVED / "cushing.sqlite"
SCHEMA = """
CREATE TABLE eia_series (series TEXT, source_key TEXT, description TEXT, date TEXT, value REAL,
    PRIMARY KEY (series, date));
CREATE TABLE eia_crude_storage_capacity (area TEXT, period_month TEXT, workbook_header_date TEXT,
    metric_label TEXT, unit TEXT, raw_value TEXT, numeric_value REAL, source_cell TEXT, source_url TEXT,
    PRIMARY KEY (area, source_cell));
CREATE TABLE eia_release (release_date TEXT, week_ending TEXT, previous_week_ending TEXT,
    cushing_kbbl INTEGER, previous_cushing_kbbl INTEGER, change_kbbl INTEGER, file TEXT,
    source_url TEXT, PRIMARY KEY (release_date, week_ending));
CREATE TABLE eia_release_clock (release_date TEXT PRIMARY KEY, oldest_week_in_issue TEXT, latest_week_in_issue TEXT,
    latest_week_observed_at_utc TEXT, scheduled_table_release_at_utc TEXT, timing_basis TEXT,
    conservative_prerelease_before_utc TEXT, conservative_postrelease_from_utc TEXT, notes TEXT, source_url TEXT);
CREATE TABLE equity_session_calendar (venue TEXT, session_date TEXT, open_utc TEXT, close_utc TEXT,
    early_close INTEGER, observed_instruments INTEGER, calendar_source_url TEXT, venue_source_url TEXT,
    PRIMARY KEY (venue, session_date));
CREATE TABLE eia_equity_session_alignment (release_date TEXT, venue TEXT, same_day_close_utc TEXT,
    scheduled_release_before_same_day_close INTEGER, next_session_date TEXT, next_session_open_utc TEXT,
    next_session_close_utc TEXT, next_session_observed_instruments INTEGER, PRIMARY KEY (release_date, venue));
CREATE TABLE wpsr_vintage_value (release_date TEXT, table_number INTEGER, source_row INTEGER,
    source_column INTEGER, report_date TEXT, group_label TEXT, measure_label TEXT,
    raw_value TEXT, numeric_value REAL, source_url TEXT,
    PRIMARY KEY (release_date, table_number, source_row, source_column));
CREATE TABLE wpsr_recovered_value (release_date TEXT, table_number INTEGER, pdf_page INTEGER,
    source_line INTEGER, column_role TEXT, report_date TEXT, raw_label TEXT, raw_value TEXT,
    numeric_value REAL, source_url TEXT,
    PRIMARY KEY (release_date, table_number, pdf_page, source_line, column_role));
CREATE TABLE market_bar (instrument TEXT, conid INTEGER, security_type TEXT, bar_start_utc TEXT,
    session_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, broker_source TEXT,
    downloaded_at_utc TEXT, same_day_snapshot INTEGER, PRIMARY KEY (instrument, bar_start_utc));
CREATE TABLE corporate_action (instrument TEXT, conid INTEGER, effective_date TEXT,
    announce_date TEXT, payment_date TEXT, currency TEXT, action_type TEXT, amount TEXT,
    source_json TEXT);
CREATE TABLE ibkr_coverage (instrument TEXT PRIMARY KEY, conid INTEGER, status TEXT, detail TEXT);
CREATE TABLE cftc_wti (report_date TEXT PRIMARY KEY, contract_market_code TEXT, market TEXT,
    open_interest INTEGER, money_manager_long INTEGER, money_manager_short INTEGER,
    producer_long INTEGER, producer_short INTEGER, source_file TEXT);
CREATE TABLE issuer_cash_distribution (instrument TEXT, ex_date TEXT, record_date TEXT, declared_date TEXT,
    payable_date TEXT, reported_amount TEXT, amount_on_post_2020_amlp_share_basis TEXT, currency TEXT,
    source_basis TEXT, source_url TEXT, PRIMARY KEY (instrument, record_date));
CREATE TABLE market_identity_check (requested_instrument TEXT, session_date TEXT, requested_security_close REAL,
    other_security TEXT, other_security_close REAL, source_url TEXT, PRIMARY KEY (requested_instrument, session_date));
CREATE TABLE market_identity_limit (instrument TEXT PRIMARY KEY, first_research_session TEXT,
    excluded_history_candidate TEXT, source_url TEXT);
CREATE TABLE equity_reference_bar (instrument TEXT, session_date TEXT, open REAL, high REAL, low REAL,
    close REAL, volume REAL, currency TEXT, provider TEXT, requested_symbol TEXT, source_file TEXT,
    verification_source_url TEXT, PRIMARY KEY (instrument, session_date));
CREATE TABLE equity_reference_price_check (instrument TEXT, session_date TEXT, comparison TEXT,
    tws_close REAL, independent_close REAL, independent_source_url TEXT,
    PRIMARY KEY (instrument, session_date, comparison));
CREATE TABLE equity_merger_consideration (retiring_instrument TEXT, acquirer_instrument TEXT,
    last_trading_session TEXT, closing_date TEXT, cash_per_retiring_unit REAL,
    acquirer_shares_per_retiring_unit REAL, currency TEXT, source_url TEXT,
    PRIMARY KEY (retiring_instrument, closing_date));
CREATE TABLE historical_equity_listing_window (listing_id TEXT PRIMARY KEY, symbol TEXT,
    first_study_session TEXT, last_study_session TEXT, exchange TEXT, issuer_identity_source TEXT);
CREATE TABLE dolt_historical_equity_bar (listing_id TEXT, symbol TEXT, session_date TEXT,
    open REAL, high REAL, low REAL, close REAL, volume REAL, exchange TEXT, currency TEXT,
    source_revision TEXT, source_file TEXT, issuer_identity_source TEXT,
    PRIMARY KEY (listing_id, session_date));
CREATE TABLE cftc_release_clock (report_date TEXT PRIMARY KEY, standard_or_original_date TEXT,
    documented_or_inferred_release_date TEXT, conservative_research_from_utc TEXT, timing_basis TEXT, source_url TEXT);
CREATE TABLE weather_metar (station TEXT, observed_at_utc TEXT, lon REAL, lat REAL, temperature_f REAL,
    dew_point_f REAL, relative_humidity_pct REAL, wind_direction_deg REAL, wind_speed_kt REAL,
    precip_1h_in REAL, visibility_mi REAL, gust_kt REAL, sky_cover_1 TEXT, sky_cover_2 TEXT,
    sky_cover_3 TEXT, sky_base_1_ft REAL, sky_base_2_ft REAL, sky_base_3_ft REAL,
    weather_codes TEXT, metar TEXT, altimeter_inhg REAL, sea_level_pressure_hpa REAL, sky_cover_4 TEXT,
    sky_base_4_ft REAL, precip_trace INTEGER, report_kind TEXT, source_file TEXT,
    PRIMARY KEY (station, observed_at_utc, metar));
CREATE TABLE macro_series (series TEXT, date TEXT, value REAL, source_file TEXT,
    PRIMARY KEY (series, date));
CREATE TABLE satellite_scene (collection TEXT, item_id TEXT, captured_at_utc TEXT,
    catalog_created_at TEXT, catalog_updated_at TEXT, platform TEXT, tile TEXT,
    baseline TEXT, cloud_cover_pct REAL, sun_azimuth_deg REAL, sun_elevation_deg REAL,
    canonical_sentinel2 INTEGER, source_manifest TEXT, PRIMARY KEY (collection, item_id));
CREATE TABLE sentinel2_archive_availability (item_id TEXT PRIMARY KEY, canonical_item_id TEXT, collection TEXT,
    captured_at_utc TEXT, catalog_created_at_utc TEXT, raster_objects_last_modified_by_utc TEXT,
    raster_and_granule_objects_last_modified_by_utc TEXT, raster_and_both_xml_objects_last_modified_by_utc TEXT,
    catalog_route_conservative_from_utc TEXT, raster_and_granule_lag_hours REAL, required_object_clocks INTEGER);
CREATE TABLE sentinel2_l1c_archive_availability (canonical_l2a_id TEXT PRIMARY KEY, captured_at_utc TEXT,
    status TEXT, l1c_product_name TEXT, gcs_product_url TEXT, all_required_objects_created_by_utc TEXT,
    with_red_detector_mask_objects_created_by_utc TEXT, gcs_input_lag_hours REAL, cloud_mask_filenames TEXT,
    red_detector_mask_filenames TEXT, required_objects INTEGER);
CREATE TABLE sentinel2_l1c_early_detector_footprint (scene_item_id TEXT PRIMARY KEY, captured_at_utc TEXT,
    l1c_product_name TEXT, polygon_count INTEGER, mask_status TEXT, bytes INTEGER, source_crc32c TEXT,
    source_created_at_utc TEXT, local_file TEXT, source_url TEXT);
CREATE TABLE sentinel2_tank_scene_quality (candidate_id TEXT, scene_item_id TEXT, captured_at_utc TEXT,
    optical_touched_cells_expected INTEGER, optical_cells_valid_in_all_bands INTEGER,
    scl_touched_cells_expected INTEGER,
    scl_class_0 INTEGER, scl_class_1 INTEGER, scl_class_2 INTEGER, scl_class_3 INTEGER,
    scl_class_4 INTEGER, scl_class_5 INTEGER, scl_class_6 INTEGER, scl_class_7 INTEGER,
    scl_class_8 INTEGER, scl_class_9 INTEGER, scl_class_10 INTEGER, scl_class_11 INTEGER,
    context_scl_cells INTEGER, context_nodata_defect_cloud_shadow_snow_cells INTEGER,
    PRIMARY KEY (candidate_id, scene_item_id));
CREATE TABLE lidar_product (id TEXT PRIMARY KEY, title TEXT, date_created TEXT,
    publication_date TEXT, size_bytes INTEGER, url TEXT, bounding_box_json TEXT);
CREATE TABLE tank_candidate (candidate_id TEXT PRIMARY KEY, longitude REAL, latitude REAL,
    area_m2 REAL, equivalent_diameter_m REAL, product_tag TEXT, operator_tag TEXT,
    tank_type_tag TEXT, osm_edit_timestamp TEXT, geometry_geojson TEXT, source_url TEXT,
    validation_status TEXT);
CREATE TABLE eia_803_osm_candidate (candidate_id TEXT PRIMARY KEY, county TEXT, county_fips TEXT,
    in_core INTEGER, longitude REAL, latitude REAL, equivalent_diameter_m REAL, product_tag TEXT,
    geometry_geojson TEXT, source_url TEXT);
CREATE TABLE eia_803_epa_facility (object_id INTEGER PRIMARY KEY, registry_id TEXT, name TEXT, city TEXT,
    reported_county TEXT, reported_fips TEXT, point_county_fips TEXT, active_flag TEXT,
    longitude REAL, latitude REAL, air_ids TEXT, caa_naics TEXT, facility_naics TEXT,
    petroleum_screen TEXT, source_url TEXT);
CREATE TABLE eia_803_facility_aerial_observation (review_number INTEGER PRIMARY KEY, registry_id TEXT,
    observation TEXT, visible_large_round_structures INTEGER, notes TEXT);
CREATE TABLE hifld_pol_eia_county (object_id INTEGER PRIMARY KEY, terminal_id TEXT, name TEXT, city TEXT,
    reported_county_fips TEXT, coordinate_county_fips TEXT, longitude REAL, latitude REAL,
    reported_type TEXT, commodity TEXT, crude_oil TEXT, reported_capacity REAL, status TEXT,
    reported_owner TEXT, reported_operator TEXT, source_date TEXT, validation_date TEXT,
    nearest_tank_candidate TEXT, nearest_tank_distance_m REAL, source_url TEXT);
CREATE TABLE supplemental_tank_candidate (candidate_id TEXT PRIMARY KEY, source_site TEXT, appearance TEXT,
    longitude REAL, latitude REAL, approx_diameter_m REAL, core_overlap_fraction REAL, image_path TEXT,
    review_path TEXT, native_tile_ids TEXT, native_tile_majority_dates TEXT, source_url TEXT,
    validation_status TEXT, notes TEXT);
CREATE TABLE tank_aerial_visual_observation (candidate_id TEXT, review_number INTEGER, observed_date TEXT,
    image_source TEXT, structure_observation TEXT, top_observation TEXT, detail TEXT,
    review_sheet TEXT, review_row INTEGER, review_column INTEGER, source_asset_paths TEXT,
    PRIMARY KEY (candidate_id, observed_date, image_source));
CREATE TABLE tank_aerial_change_interval (candidate_id TEXT, review_number INTEGER, event TEXT,
    earlier_observed_date TEXT, later_observed_date TEXT, earlier_observation TEXT, later_observation TEXT,
    earlier_review_sheet TEXT, later_review_sheet TEXT,
    PRIMARY KEY (candidate_id, earlier_observed_date, later_observed_date));
CREATE TABLE sentinel2_product_timing (canonical_id TEXT PRIMARY KEY, captured_at_utc TEXT,
    canonical_generation_at_utc TEXT, canonical_catalog_created_at_utc TEXT, canonical_generation_lag_hours REAL,
    earliest_l2a_id TEXT, earliest_l2a_generation_at_utc TEXT, earliest_l2a_generation_lag_hours REAL,
    earliest_l2a_local_directory TEXT, l1c_product_name TEXT, l1c_generation_reported_at_utc TEXT,
    l1c_generation_before_tile_sensing INTEGER, l1c_archive_band_and_granule_metadata_created_by_utc TEXT,
    l1c_local_directory TEXT);
CREATE TABLE terminal (terminal_id TEXT PRIMARY KEY, name TEXT, asset_type TEXT,
    capacity_mmbbl REAL, capacity_basis TEXT, capacity_as_of TEXT, source_url TEXT, notes TEXT);
CREATE TABLE terminal_company_history (terminal_id TEXT, issuer TEXT, ticker TEXT,
    exchange_currency TEXT, valid_from TEXT, valid_to_exclusive TEXT, date_precision TEXT,
    role TEXT, interest_pct REAL, source_url TEXT, notes TEXT);
CREATE TABLE terminal_observation (terminal_id TEXT, as_of_date TEXT, observation TEXT,
    capacity_mmbbl REAL, capacity_comparator TEXT, capacity_basis TEXT, source_url TEXT);
CREATE TABLE permit_document (document_id TEXT PRIMARY KEY, facility_key TEXT, terminal_id TEXT,
    facility_id TEXT, permit_number TEXT, memo_date TEXT, document_status TEXT,
    latitude REAL, longitude REAL, position_description TEXT, source_url TEXT);
CREATE TABLE permit_tank (facility_key TEXT, facility_id TEXT, permit_number TEXT, memo_date TEXT,
    document_id TEXT, pdf_page INTEGER, unit_group TEXT, tank_id TEXT, emissions_unit_id TEXT,
    roof_type TEXT, status_in_draft TEXT, capacity_value REAL, capacity_unit TEXT, capacity_bbl REAL,
    construction_or_modification_raw TEXT, height_ft REAL, diameter_ft REAL, source_url TEXT,
    stored_product_raw TEXT, capacity_basis TEXT, height_basis TEXT, bottom_design_raw TEXT,
    PRIMARY KEY (document_id, facility_key, tank_id));
CREATE TABLE tank_permit_section_candidate (candidate_id TEXT, facility_key TEXT,
    plss_section_id TEXT, centroid_distance_from_reference_m REAL, equivalent_diameter_m REAL,
    evidence_type TEXT, permit_source_url TEXT, PRIMARY KEY (candidate_id, facility_key, plss_section_id));
CREATE TABLE tank_parcel_intersection (candidate_id TEXT, snapshot_year INTEGER, object_id INTEGER,
    parcel_id TEXT, property_id TEXT, source_year_id INTEGER, owner_name_recorded TEXT,
    business_name_recorded TEXT, land_use_code TEXT, sale_date_raw TEXT, legal_reference TEXT,
    tank_area_fraction REAL, source_url TEXT, county TEXT, snapshot_kind TEXT,
    PRIMARY KEY (candidate_id, county, snapshot_year, object_id));
CREATE TABLE facility_parcel_alias (facility_key TEXT, owner_name_recorded TEXT,
    identity_source_url TEXT, notes TEXT, PRIMARY KEY (facility_key, owner_name_recorded));
CREATE TABLE terminal_parcel_alias (terminal_id TEXT, recorded_name TEXT, match_kind TEXT,
    identity_source_url TEXT, asset_source_url TEXT, notes TEXT,
    PRIMARY KEY (terminal_id, recorded_name, match_kind));
CREATE TABLE tank_permit_cohort_evidence (candidate_id TEXT, facility_key TEXT, cohort_id TEXT,
    possible_permit_tank_ids TEXT, permit_roof_type TEXT, permit_height_ft REAL, permit_diameter_ft REAL,
    diameter_evidence TEXT, match_basis TEXT, permit_source_url TEXT,
    PRIMARY KEY (candidate_id, facility_key, cohort_id));
CREATE TABLE tank_asset_cohort (candidate_id TEXT, terminal_id TEXT, cohort_id TEXT, capacity_bbl REAL,
    capacity_basis TEXT, documented_by TEXT, documented_build_year INTEGER, map_pdf_page INTEGER,
    source_url TEXT, local_pdf TEXT, notes TEXT, PRIMARY KEY (candidate_id, terminal_id, cohort_id));
CREATE TABLE tws_daily_bar (instrument TEXT, conid INTEGER, series TEXT, session_date TEXT,
    open REAL, high REAL, low REAL, close REAL, volume REAL, wap REAL, trades INTEGER,
    primary_exchange TEXT, currency TEXT, downloaded_at_utc TEXT, source_file TEXT,
    PRIMARY KEY (instrument, series, session_date));
CREATE TABLE tws_futures_bar (instrument TEXT, contract_month TEXT, conid INTEGER, local_symbol TEXT,
    last_trading_date TEXT, session_date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
    wap REAL, trades INTEGER, downloaded_at_utc TEXT, source_file TEXT, download_date_chicago TEXT,
    PRIMARY KEY (conid, session_date));
CREATE TABLE lidar_tank_coverage (candidate_id TEXT PRIMARY KEY, returns_inside_polygon INTEGER,
    returns_10_to_35m_from_polygon INTEGER, ground_class_2_returns_10_to_35m INTEGER);
CREATE TABLE lidar_2011_tank_coverage (candidate_id TEXT PRIMARY KEY, returns_inside_polygon INTEGER,
    returns_10_to_35m_from_polygon INTEGER, ground_class_2_returns_10_to_35m INTEGER);
CREATE TABLE lidar_stroud_legacy_observation (candidate_id TEXT PRIMARY KEY, dataset TEXT, source_tile TEXT,
    fgdc_flight_start TEXT, fgdc_flight_end TEXT, interior_first_returns INTEGER, interior_z_p10_native REAL,
    interior_z_p50_native REAL, interior_z_p90_native REAL, gps_week_time_min_s REAL, gps_week_time_max_s REAL,
    utc_options_for_min_week_time_json TEXT, utc_options_for_max_week_time_json TEXT, date_status TEXT, ept_origin_paths_json TEXT,
    local_laz_files_json TEXT, fgdc_source_url TEXT, ept_source_url TEXT, original_interior_first_returns INTEGER,
    ept_returns_matching_original_gps INTEGER, local_original_laz TEXT, original_laz_sha256 TEXT);
CREATE TABLE lidar_tank_observation (candidate_id TEXT, dataset TEXT, acquisition_date_utc TEXT,
    point_source_id INTEGER, first_interior_return_at_utc TEXT, last_interior_return_at_utc TEXT,
    interior_points INTEGER, rim_points INTEGER, ground_points INTEGER,
    interior_z_p10 REAL, interior_z_p50 REAL, interior_z_p90 REAL, rim_z_p50 REAL, rim_z_p95 REAL,
    ground_z_p10 REAL, ground_z_p50 REAL, ground_z_p90 REAL,
    PRIMARY KEY (candidate_id, dataset, acquisition_date_utc, point_source_id));
CREATE TABLE naip_tank_image_coverage (candidate_id TEXT, acquisition_date TEXT, item_id TEXT,
    asset_path TEXT, pixel_width_m REAL, coverage TEXT, PRIMARY KEY (candidate_id, item_id));
CREATE TABLE ok_aerial_tank_image_coverage (candidate_id TEXT, tile_id TEXT, tile_majority_date TEXT,
    tile_time_raw TEXT, asset_path TEXT, candidate_pixels INTEGER, valid_pixels INTEGER,
    valid_fraction REAL, PRIMARY KEY (candidate_id, tile_id));
CREATE TABLE ok_aerial_tank_union_coverage (candidate_id TEXT PRIMARY KEY, candidate_pixels INTEGER,
    union_valid_pixels INTEGER, tile_majority_dates TEXT);
CREATE TABLE eia_archive_exception (week_ending TEXT PRIMARY KEY, earliest_archive_date TEXT,
    value_at_earliest_archive_kbbl INTEGER, earliest_value_precision_kbbl INTEGER,
    exact_archive_date TEXT, exact_value_kbbl INTEGER, source_url TEXT, notes TEXT);
CREATE TABLE lidar_ept_node (dataset TEXT, node_id TEXT, source_url TEXT, source_points INTEGER,
    selected_points INTEGER, horizontal_crs TEXT, buffer_m REAL, local_file TEXT,
    source_sha256 TEXT, clip_scope TEXT, PRIMARY KEY (dataset, node_id, clip_scope));
CREATE INDEX market_session_idx ON market_bar (instrument, session_date);
CREATE INDEX satellite_date_idx ON satellite_scene (captured_at_utc, collection);
CREATE INDEX satellite_item_idx ON satellite_scene (item_id);
CREATE UNIQUE INDEX issuer_cash_ex_date_idx ON issuer_cash_distribution (instrument, ex_date) WHERE ex_date IS NOT NULL;
CREATE INDEX sentinel2_tank_scene_date_idx ON sentinel2_tank_scene_quality (captured_at_utc);
CREATE VIEW equity_closed_sessions AS
    SELECT m.* FROM market_bar m LEFT JOIN market_identity_limit i USING (instrument)
    WHERE security_type = 'STK' AND same_day_snapshot = 0
        AND (i.first_research_session IS NULL OR m.session_date>=i.first_research_session);
CREATE VIEW tws_equity_research_bar AS
    SELECT t.* FROM tws_daily_bar t LEFT JOIN market_identity_limit i USING (instrument)
    WHERE i.first_research_session IS NULL OR t.session_date>=i.first_research_session;
CREATE VIEW equity_research_trade_bar AS
    SELECT r.instrument,r.session_date,r.open,r.high,r.low,r.close,r.volume,r.currency,
        r.provider,r.requested_symbol,r.source_file,r.verification_source_url,NULL AS conid
    FROM equity_reference_bar r
    UNION ALL
    SELECT t.instrument,t.session_date,t.open,t.high,t.low,t.close,t.volume,t.currency,
        'ibkr_tws',t.instrument,t.source_file,NULL,t.conid
    FROM tws_equity_research_bar t WHERE t.series='TRADES'
      AND NOT EXISTS (SELECT 1 FROM equity_reference_bar r
                      WHERE r.instrument=t.instrument AND r.session_date=t.session_date);
CREATE VIEW market_identity_evidence AS
    SELECT i.*, t.close AS tws_trades_close, m.close AS connector_close
    FROM market_identity_check i
    LEFT JOIN tws_daily_bar t ON t.instrument=i.requested_instrument AND t.session_date=i.session_date AND t.series='TRADES'
    LEFT JOIN market_bar m ON m.instrument=i.requested_instrument AND m.session_date=i.session_date;
CREATE VIEW dolt_historical_equity_quality AS
    SELECT *, CASE
        WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL THEN 'missing_field'
        WHEN low<=0 OR low>min(open,close) OR high<max(open,close) OR volume<0 THEN 'invalid_ohlcv'
        ELSE 'ohlcv_range_valid' END AS quality_status
    FROM dolt_historical_equity_bar;
CREATE VIEW dolt_historical_equity_missing_session AS
    SELECT l.listing_id,l.symbol,c.session_date
    FROM historical_equity_listing_window l
    JOIN equity_session_calendar c ON c.venue='XNYS' AND c.session_date BETWEEN l.first_study_session AND l.last_study_session
    LEFT JOIN dolt_historical_equity_bar d ON d.listing_id=l.listing_id AND d.session_date=c.session_date
    WHERE d.session_date IS NULL;
CREATE VIEW issuer_cash_adjusted_equity_return AS
    WITH aligned AS (
        SELECT t.instrument, t.session_date, t.conid, t.open, t.close, t.currency, t.provider,
            coalesce(CAST(i.reported_amount AS REAL), 0) AS issuer_cash,
            i.source_url AS issuer_cash_source,
            a.close AS vendor_adjusted_close,
            a.close / nullif(t.close, 0) AS vendor_adjustment_ratio
        FROM equity_research_trade_bar t
        LEFT JOIN tws_daily_bar a ON a.instrument=t.instrument AND a.conid=t.conid
            AND a.session_date=t.session_date AND a.series='ADJUSTED_LAST'
        LEFT JOIN issuer_cash_distribution i ON i.instrument=t.instrument AND i.ex_date=t.session_date
        WHERE t.instrument IN ('OKE','MMP')
            AND EXISTS (SELECT 1 FROM issuer_cash_distribution i WHERE i.instrument=t.instrument)
    ), daily AS (
        SELECT *, lag(session_date) OVER chronological AS previous_session_date,
            lag(close) OVER chronological AS previous_close,
            lag(vendor_adjustment_ratio) OVER chronological AS previous_vendor_adjustment_ratio
        FROM aligned WINDOW chronological AS (PARTITION BY instrument ORDER BY session_date)
    )
    SELECT session_date, conid, currency, previous_session_date, previous_close, open, close,
        issuer_cash, issuer_cash_source,
        (close + issuer_cash) / nullif(previous_close, 0) - 1 AS issuer_based_close_to_close_total_return,
        vendor_adjusted_close,
        previous_close * (1 - previous_vendor_adjustment_ratio / nullif(vendor_adjustment_ratio, 0))
            AS vendor_implied_cash,
        instrument, provider
    FROM daily;
CREATE VIEW oke_equity_total_return AS
    SELECT * FROM issuer_cash_adjusted_equity_return WHERE instrument='OKE';
CREATE VIEW mmp_equity_total_return AS
    SELECT * FROM issuer_cash_adjusted_equity_return WHERE instrument='MMP';
CREATE VIEW tws_futures_closed_sessions AS
    SELECT * FROM tws_futures_bar WHERE session_date < download_date_chicago;
CREATE VIEW tank_facility_evidence AS
    SELECT p.candidate_id, p.county, p.snapshot_year, p.snapshot_kind, p.parcel_id,
        p.owner_name_recorded, p.tank_area_fraction, s.facility_key, s.plss_section_id,
        p.source_url AS parcel_source_url, s.permit_source_url, a.identity_source_url, a.notes
    FROM tank_parcel_intersection p
    JOIN tank_permit_section_candidate s USING (candidate_id)
    JOIN facility_parcel_alias a ON a.facility_key=s.facility_key
        AND lower(a.owner_name_recorded)=lower(p.owner_name_recorded)
    WHERE p.tank_area_fraction >= 0.99;
CREATE VIEW tank_terminal_parcel_evidence AS
    SELECT p.candidate_id, p.county, p.snapshot_year, p.snapshot_kind, p.parcel_id,
        p.tank_area_fraction, a.terminal_id, a.match_kind, p.source_url AS parcel_source_url,
        a.identity_source_url, a.asset_source_url, a.notes
    FROM tank_parcel_intersection p
    JOIN terminal_parcel_alias a ON
        (a.match_kind='exact' AND upper(trim(p.owner_name_recorded))=upper(a.recorded_name))
        OR (a.match_kind='co_recorded' AND substr(upper(trim(p.owner_name_recorded)), 1, length(a.recorded_name)+1)
            =upper(a.recorded_name)||' ')
    WHERE p.tank_area_fraction >= 0.99;
CREATE VIEW tank_terminal_research_attribution AS
    SELECT candidate_id, terminal_id, 'documented_asset_cohort' AS basis FROM tank_asset_cohort
    UNION
    SELECT candidate_id, terminal_id, 'recorded_parcel_association' AS basis FROM tank_terminal_parcel_evidence p
    WHERE NOT EXISTS (SELECT 1 FROM tank_asset_cohort c WHERE c.candidate_id=p.candidate_id);
CREATE VIEW eia_cushing_vintages AS
    SELECT release_date, week_ending, cushing_kbbl AS cushing_kbbl, 'current_week' AS reported_as, source_url FROM eia_release
    UNION ALL
    SELECT release_date, previous_week_ending, previous_cushing_kbbl, 'previous_week', source_url FROM eia_release;
CREATE VIEW eia_cushing_as_published AS
    SELECT v.release_date, v.week_ending, v.cushing_kbbl, 1 AS precision_kbbl,
        v.reported_as, v.source_url
    FROM eia_cushing_vintages v
    WHERE NOT EXISTS (SELECT 1 FROM eia_archive_exception e
        WHERE e.earliest_archive_date=v.release_date AND e.earliest_archive_date<e.exact_archive_date)
    UNION ALL
    SELECT earliest_archive_date, week_ending, value_at_earliest_archive_kbbl,
        earliest_value_precision_kbbl, 'current_week_recovered_from_pdf', source_url
    FROM eia_archive_exception WHERE earliest_archive_date<exact_archive_date;
CREATE VIEW eia_sentinel2_timing_coverage AS
    WITH archive_versions AS (
        SELECT canonical_item_id, julianday(captured_at_utc) AS sensed,
            julianday(catalog_route_conservative_from_utc) AS available, 'L2A' AS level
        FROM sentinel2_archive_availability
        UNION ALL
        SELECT canonical_l2a_id, julianday(captured_at_utc),
            julianday(all_required_objects_created_by_utc), 'L1C'
        FROM sentinel2_l1c_archive_availability a
        WHERE NOT EXISTS (
            SELECT 1 FROM sentinel2_l1c_early_detector_footprint m
            WHERE m.scene_item_id=a.canonical_l2a_id AND m.mask_status='empty_source_mask')
    ), core AS (
        SELECT item_id FROM satellite_scene WHERE canonical_sentinel2=1
    )
    SELECT e.release_date, e.latest_week_in_issue, e.latest_week_observed_at_utc,
        e.conservative_prerelease_before_utc,
        count(DISTINCT CASE WHEN a.level='L2A' THEN a.canonical_item_id END) AS l2a_granules,
        count(DISTINCT CASE WHEN a.level='L1C' THEN a.canonical_item_id END) AS l1c_granules,
        count(DISTINCT CASE WHEN a.level='L2A' THEN c.item_id END) AS core_l2a_granules,
        count(DISTINCT CASE WHEN a.level='L1C' THEN c.item_id END) AS core_l1c_granules
    FROM eia_release_clock e
    LEFT JOIN archive_versions a
        ON a.sensed > julianday(e.latest_week_observed_at_utc)-7
        AND a.sensed <= julianday(e.latest_week_observed_at_utc)
        AND a.available < julianday(e.conservative_prerelease_before_utc)
    LEFT JOIN core c ON c.item_id=a.canonical_item_id
    GROUP BY e.release_date;
CREATE VIEW tank_data_readiness AS
    WITH optical AS (
        SELECT candidate_id, count(*) AS catalog_granules,
            sum(optical_cells_valid_in_all_bands=optical_touched_cells_expected) AS complete_optical_granules
        FROM sentinel2_tank_scene_quality GROUP BY candidate_id
    ), historical_photo AS (
        SELECT candidate_id, count(DISTINCT substr(acquisition_date, 1, 4)) AS naip_years
        FROM naip_tank_image_coverage GROUP BY candidate_id
    ), land AS (
        SELECT candidate_id, count(DISTINCT snapshot_year) AS parcel_snapshot_years
        FROM tank_parcel_intersection GROUP BY candidate_id
    ), site AS (
        SELECT candidate_id, count(DISTINCT facility_key) AS corroborated_facility_candidates
        FROM tank_facility_evidence GROUP BY candidate_id
    )
    SELECT t.candidate_id, t.longitude, t.latitude, t.equivalent_diameter_m, t.validation_status,
        coalesce(o.catalog_granules, 0) AS sentinel2_granules,
        coalesce(o.complete_optical_granules, 0) AS complete_optical_granules,
        coalesce(h.naip_years, 0) AS naip_years,
        coalesce(a.union_valid_pixels = a.candidate_pixels, 0) AS full_state_aerial_2025,
        coalesce(l.returns_inside_polygon, 0) AS lidar_interior_returns,
        coalesce(land.parcel_snapshot_years, 0) AS parcel_snapshot_years,
        coalesce(site.corroborated_facility_candidates, 0) AS corroborated_facility_candidates
    FROM tank_candidate t
    LEFT JOIN optical o USING (candidate_id)
    LEFT JOIN historical_photo h USING (candidate_id)
    LEFT JOIN ok_aerial_tank_union_coverage a USING (candidate_id)
    LEFT JOIN lidar_tank_coverage l USING (candidate_id)
    LEFT JOIN land USING (candidate_id)
    LEFT JOIN site USING (candidate_id);
"""


def add_eia(db):
    for path in sorted((RAW / "eia").glob("*.xls")):
        workbook = xlrd.open_workbook(path)
        sheet = workbook.sheet_by_name("Data 1")
        key, title = sheet.cell_value(1, 1), sheet.cell_value(2, 1)
        rows = []
        for row in sheet.get_rows():
            if len(row) < 2 or row[0].ctype != xlrd.XL_CELL_DATE or row[1].ctype != xlrd.XL_CELL_NUMBER:
                continue
            date = xlrd.xldate_as_datetime(row[0].value, workbook.datemode).date().isoformat()
            rows.append((path.stem, key, title, date, row[1].value))
        if not rows:
            raise ValueError(f"No dated numeric observations in {path}")
        db.executemany("INSERT INTO eia_series VALUES (?,?,?,?,?)", rows)
    for path in sorted((RAW / "eia" / "wpsr").glob("*_table4.csv")):
        rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
        header = next(row for row in rows if row and row[0] == "STUB_1")
        cushing = next(row for row in rows if row and row[0].strip().lower() in {"cushing", "cushing, oklahoma"})
        dates = [dt.datetime.strptime(value, "%m/%d/%y").date().isoformat() for value in header[1:3]]
        values = [round(float(value.replace(",", "")) * 1000) for value in cushing[1:4]]
        release = path.name[:10].replace("_", "-")
        source = path.with_suffix(".source").read_text().strip()
        if not dates[1] < dates[0] < release:
            raise ValueError(f"Unexpected reporting dates in {path}")
        db.execute("INSERT INTO eia_release VALUES (?,?,?,?,?,?,?,?)",
                   (release, *dates, *values, str(path.relative_to(ROOT)), source))


def add_wpsr_factors(db):
    directory = RAW / "eia" / "wpsr"
    releases = {date: (week, previous) for date, week, previous in db.execute(
        "SELECT release_date, week_ending, previous_week_ending FROM eia_release")}
    for table in TABLES:
        for path in sorted(directory.glob(f"*_table{table}.csv")):
            source_file = path.with_suffix(".source")
            if not source_file.is_file():
                continue
            raw = path.read_bytes()
            try:
                content = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                content = raw.decode("cp1252")
            rows = csv.reader(io.StringIO(content))
            header = next(rows)
            stubs = sum(cell.startswith("STUB_") for cell in header)
            report_dates = tuple(dt.datetime.strptime(day, "%m/%d/%Y" if len(day.rsplit("/", 1)[-1]) == 4
                                                        else "%m/%d/%y").date().isoformat()
                                 for day in header[stubs:stubs + 2])
            release = path.name[:10].replace("_", "-")
            if report_dates != releases.get(release):
                raise ValueError(f"Reporting dates do not match archived Table 4 in {path}")
            source = source_file.read_text().strip()
            for number, row in enumerate(rows, start=2):
                if len(row) < stubs + 2 or not any(cell.strip() for cell in row[:stubs]):
                    continue
                group = row[0].strip() if stubs == 2 else None
                label = row[stubs - 1].strip()
                for offset, report_date in enumerate(report_dates):
                    value = row[stubs + offset].strip()
                    try:
                        numeric = float(value.replace(",", ""))
                    except ValueError:
                        numeric = None
                    db.execute("INSERT INTO wpsr_vintage_value VALUES (?,?,?,?,?,?,?,?,?,?)",
                               (release, table, number, stubs + offset + 1, report_date,
                                group, label, value, numeric, source))


def add_ibkr(db):
    directory = RAW / "ibkr"
    ny = ZoneInfo("America/New_York")
    successful = set()
    files = {path.stem.removesuffix("_daily"): path for path in directory.glob("*_daily.json")}
    for path in directory.glob("cl_*_retry.json"):
        if json.loads(path.read_bytes())["response"].get("time"):
            instrument = path.stem.removesuffix("_retry")
            first = files.get(instrument)
            if not first or not json.loads(first.read_bytes())["response"].get("time"):
                files[instrument] = path
    for key, path in sorted(files.items()):
        instrument = key.upper()
        payload = json.loads(path.read_bytes())
        request, response = payload["request"], payload["response"]
        conid = request.get("con_id", request.get("contract_id"))
        security_type = request.get("security_type", "STK")
        times = response.get("time")
        if not times:
            db.execute("INSERT INTO ibkr_coverage VALUES (?,?,?,?)",
                       (instrument, conid, "no_bars", json.dumps(response)))
            successful.add(instrument)
            continue
        columns = [response.get(key, []) for key in ("open", "high", "low", "close", "volume")]
        if any(len(column) != len(times) for column in columns):
            raise ValueError(f"Unequal bar array lengths in {path}")
        stamp = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
        rows = []
        for timestamp, prices in zip(times, zip(*columns), strict=True):
            ts = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            date = ts.astimezone(ny).date() if security_type == "STK" else None
            same_day = int(ts.astimezone(ny).date() == stamp.astimezone(ny).date())
            rows.append((instrument, conid, security_type, timestamp, date.isoformat() if date else None,
                         *prices, response.get("source"), stamp.isoformat(), same_day))
        db.executemany("INSERT INTO market_bar VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        for event in response.get("corp_actions", []):
            db.execute("INSERT INTO corporate_action VALUES (?,?,?,?,?,?,?,?,?)", (
                instrument, conid, event.get("date"), event.get("announce_date"),
                event.get("payment_date"), event.get("currency"), event.get("type"),
                event.get("value"), json.dumps(event, ensure_ascii=False),
            ))
        db.execute("INSERT INTO ibkr_coverage VALUES (?,?,?,?)", (instrument, conid, "downloaded", None))
        successful.add(instrument)
    for path in sorted(directory.glob("*_resolution.json")):
        instrument = path.stem.removesuffix("_resolution").upper()
        if instrument not in successful:
            payload = json.loads(path.read_bytes())
            chosen = payload.get("contract") or {}
            status = "contract_ladder_only" if payload.get("ladder") else "contract_unresolved"
            db.execute("INSERT INTO ibkr_coverage VALUES (?,?,?,?)", (
                instrument, chosen.get("underlying_contract_id"), status,
                payload.get("error") or json.dumps(payload.get("exact_symbols", [])),
            ))


def add_catalogs(db):
    preferred = {item["id"] for item in canonical_items()}
    core_paths = sorted((RAW / "stac").glob("*/*.geojson"))
    supplemental_paths = sorted((RAW / "stac_supplemental").glob("*/*.geojson"))
    for path in core_paths + supplemental_paths:
        collection = path.parent.name
        for item in json.loads(path.read_bytes())["features"]:
            props = item["properties"]
            db.execute("INSERT OR IGNORE INTO satellite_scene VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                collection, item["id"], props.get("datetime"), props.get("created"),
                props.get("updated"), props.get("platform"), props.get("grid:code"),
                props.get("s2:processing_baseline"), props.get("eo:cloud_cover"),
                props.get("view:sun_azimuth"), props.get("view:sun_elevation"),
                int(item["id"] in preferred and collection in {"sentinel-2-c1-l2a", "sentinel-2-l2a"}),
                str(path.relative_to(ROOT)),
            ))
    path = RAW / "usgs" / "lidar_products.json"
    if path.is_file():
        for item in json.loads(path.read_bytes())["items"]:
            db.execute("INSERT OR IGNORE INTO lidar_product VALUES (?,?,?,?,?,?,?)", (
                item["sourceId"], item["title"], item.get("dateCreated"), item.get("publicationDate"),
                item.get("sizeInBytes"), item.get("downloadLazURL") or item.get("downloadURL"),
                json.dumps(item.get("boundingBox")),
            ))


def add_factors(db):
    root = RAW / "factors"
    for path in sorted(root.glob("cftc_disaggregated_*.zip")):
        with zipfile.ZipFile(path) as archive:
            filename = next(name for name in archive.namelist() if name.lower().endswith((".txt", ".csv")))
            with archive.open(filename) as file:
                for row in csv.DictReader(io.TextIOWrapper(file, encoding="utf-8-sig", errors="replace")):
                    row = {key.strip(): value.strip() for key, value in row.items() if key and value is not None}
                    if row.get("CFTC_Contract_Market_Code") != "067651":
                        continue
                    db.execute("INSERT OR REPLACE INTO cftc_wti VALUES (?,?,?,?,?,?,?,?,?)", (
                        row["Report_Date_as_YYYY-MM-DD"], "067651", row["Market_and_Exchange_Names"],
                        int(row["Open_Interest_All"]), int(row["M_Money_Positions_Long_All"]),
                        int(row["M_Money_Positions_Short_All"]), int(row["Prod_Merc_Positions_Long_All"]),
                        int(row["Prod_Merc_Positions_Short_All"]), path.name,
                    ))
    for path in sorted(root.glob("*.csv")):
        with path.open() as file:
            rows = csv.DictReader(file)
            if path.stem not in (rows.fieldnames or []):
                continue
            for row in rows:
                value = row.get(path.stem, "")
                date = row.get("observation_date") or row.get("DATE")
                if value and value != "." and date:
                    db.execute("INSERT INTO macro_series VALUES (?,?,?,?)", (path.stem, date, float(value), path.name))
    for index, value_column in (("OVX", "OVX"), ("VIX", "CLOSE")):
        cboe = root / f"{index}_cboe.csv"
        if cboe.is_file():
            with cboe.open() as file:
                for row in csv.DictReader(file):
                    if row.get(value_column):
                        date = dt.datetime.strptime(row["DATE"], "%m/%d/%Y").date().isoformat()
                        db.execute("INSERT INTO macro_series VALUES (?,?,?,?)",
                                   (f"{index}_CBOE", date, float(row[value_column]), cboe.name))
    french = root / "fama_french_daily.zip"
    if french.is_file():
        with zipfile.ZipFile(french) as archive:
            lines = archive.read(archive.namelist()[0]).decode("ascii", errors="replace").splitlines()
        names = ("MKT_RF", "SMB", "HML", "RMW", "CMA", "RF")
        for row in csv.reader(lines):
            if len(row) != 7 or not re.fullmatch(r"\d{8}", row[0].strip()):
                continue
            date = dt.datetime.strptime(row[0], "%Y%m%d").date().isoformat()
            db.executemany("INSERT INTO macro_series VALUES (?,?,?,?)",
                           [(name, date, float(value) / 100, french.name) for name, value in zip(names, row[1:])])


def add_weather(db):
    columns = ("lon", "lat", "tmpf", "dwpf", "relh", "drct", "sknt", "p01i", "vsby", "gust",
               "skyc1", "skyc2", "skyc3", "skyl1", "skyl2", "skyl3", "wxcodes", "metar")
    root = RAW / "weather"
    paths = sorted(root.glob("asos_*.csv")) + sorted(root.glob("asos_expanded_*.csv.gz"))
    for path in paths:
        expanded = path.suffix == ".gz"
        source = str(path.relative_to(ROOT))
        with (gzip.open(path, "rt") if expanded else path.open()) as file:
            def reports():
                for row in csv.DictReader(file):
                    if not row.get("station") or not row.get("valid"):
                        continue
                    timestamp = row["valid"].replace(" ", "T") + ":00Z"
                    values = [row.get(column) or None for column in columns]
                    values[-1] = values[-1] or ""
                    metar = values[-1]
                    kind = ("iem_ghcnh_derived" if "IEM_GHCNH" in metar else
                            "reported_metar_text" if metar else "archive_without_metar_text")
                    precip = row.get("p01i")
                    trace = int(precip == "T") if precip else None
                    if trace:
                        values[7] = None
                    extra = [row.get(column) or None for column in ("alti", "mslp", "skyc4", "skyl4")]
                    yield (row["station"], timestamp, *values, *extra, trace, kind, source)

            conflict = "REPLACE" if expanded else "IGNORE"
            db.executemany(f"INSERT OR {conflict} INTO weather_metar VALUES (" +
                           ",".join("?" for _ in range(27)) + ")", reports())


def load_csv(db, table, path, fixed_fields=()):
    if not path.is_file():
        return
    with path.open() as file:
        rows = csv.reader(file)
        columns = next(rows)
        placeholders = ",".join("?" for _ in range(len(columns) + len(fixed_fields)))
        db.executemany(f"INSERT INTO {table} VALUES ({placeholders})",
                       ([value or None for value in row] + list(fixed_fields) for row in rows))


def load_cash_override(db, path):
    if path.is_file():
        with path.open(newline="") as file:
            rows = [tuple(value or None for value in row.values()) for row in csv.DictReader(file)]
        if rows:
            replace_cash_records(db, rows, {row[0] for row in rows})


def add_eia_803_area(db):
    osm = DERIVED / "eia_803_osm_tank_candidates.geojson"
    if osm.is_file():
        for feature in json.loads(osm.read_bytes())["features"]:
            attributes = feature["properties"]
            point = shape(feature["geometry"]).centroid
            db.execute("INSERT INTO eia_803_osm_candidate VALUES (?,?,?,?,?,?,?,?,?,?)", (
                attributes["candidate_id"], attributes["county"], attributes["county_fips"],
                attributes["inside_existing_core_inventory"], point.x, point.y,
                attributes["equivalent_diameter_m"], attributes["product_tag"],
                json.dumps(feature["geometry"], separators=(",", ":")), attributes["source"],
            ))
    county_file = RAW / "documents" / "eia_803_counties.geojson"
    epa_file = RAW / "documents" / "epa_echo_eia_803_counties.geojson"
    if not epa_file.is_file():
        return
    counties = [(f["properties"]["GEOID"], shape(f["geometry"]))
                for f in json.loads(county_file.read_bytes())["features"]]
    for feature in json.loads(epa_file.read_bytes())["features"]:
        p = feature["properties"]
        lon, lat = p["FAC_LONG"], p["FAC_LAT"]
        point = Point(lon, lat) if lon is not None and lat is not None else None
        point_fips = next((code for code, area in counties if point is not None and area.covers(point)), None)
        db.execute("INSERT INTO eia_803_epa_facility VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            p["OBJECTID"], p["REGISTRY_ID"], p["FAC_NAME"], p["FAC_CITY"], p["FAC_COUNTY"],
            p["FAC_FIPS_CODE"], point_fips, p["FAC_ACTIVE_FLAG"], lon, lat, p["AIR_IDS"],
            p["CAA_NAICS"], p["FAC_NAICS_CODES"], screening_reason(p) or None, p["DFR_URL"],
        ))


def add_registry(db):
    reference = ROOT / "data" / "reference"
    candidates = DERIVED / "osm_tank_candidates.geojson"
    if candidates.is_file():
        for feature in json.loads(candidates.read_bytes())["features"]:
            attributes = feature["properties"]
            point = shape(feature["geometry"]).centroid
            db.execute("INSERT INTO tank_candidate VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                attributes["candidate_id"], point.x, point.y, attributes["area_m2"], attributes["equivalent_diameter_m"],
                attributes.get("product_tag"), attributes.get("operator_tag"), attributes.get("tank_type_tag"),
                attributes.get("osm_edit_timestamp"), json.dumps(feature["geometry"], separators=(",", ":")),
                attributes["source"], attributes["validation_status"],
            ))
    add_eia_803_area(db)
    load_csv(db, "hifld_pol_eia_county", DERIVED / "hifld_pol_eia_counties.csv")
    load_csv(db, "eia_803_facility_aerial_observation", reference / "eia_803_facility_aerial_observations.csv")
    load_csv(db, "supplemental_tank_candidate", DERIVED / "supplemental_tank_candidates.csv")
    for table, filename in [("terminal", "terminals.csv"), ("terminal_company_history", "terminal_company_history.csv"),
                            ("terminal_observation", "terminal_observations.csv"), ("permit_document", "permit_documents.csv")]:
        load_csv(db, table, reference / filename)
    load_csv(db, "permit_tank", DERIVED / "permit_tanks.csv")
    load_csv(db, "permit_epa_review_timeline", EPA_REVIEW_CSV)
    load_csv(db, "permit_tank_capacity_crosscheck", DERIVED / "permit_tank_capacity_crosscheck.csv")
    load_csv(db, "permit_planned_tank_cohort", reference / "permit_planned_tank_cohorts.csv")
    load_csv(db, "plains_cushing_2008_aerial_observation", reference / "plains_cushing_2008_aerial_observations.csv")
    load_csv(db, "enbridge_main_contour_group", reference / "enbridge_main_contour_groups.csv")
    load_csv(db, "enbridge_main_group_source", reference / "enbridge_main_group_sources.csv")
    load_csv(db, "terminal_disclosed_capacity", reference / "cushing_terminal_disclosed_capacity.csv")
    load_csv(db, "tank_footprint_geometry", DERIVED / "tank_footprint_geometry.csv")
    load_csv(db, "tank_lidar_outline", DERIVED / "tank_lidar_outlines.csv")
    load_csv(db, "tank_permit_section_candidate", DERIVED / "tank_permit_section_candidates.csv")
    load_csv(db, "tank_parcel_intersection", DERIVED / "tank_parcel_intersections.csv", ("payne", "annual_assessment"))
    load_csv(db, "tank_parcel_intersection", DERIVED / "tank_lincoln_parcel_intersections.csv", ("lincoln", "retrieved_current"))
    load_csv(db, "facility_parcel_alias", reference / "facility_parcel_aliases.csv")
    load_csv(db, "terminal_parcel_alias", reference / "terminal_parcel_aliases.csv")
    load_csv(db, "tank_permit_cohort_evidence", reference / "tank_permit_cohort_evidence.csv")
    load_csv(db, "tank_asset_cohort", reference / "tank_asset_cohorts.csv")
    install_terminal_scopes(db)
    load_csv(db, "tank_aerial_visual_observation", reference / "tank_aerial_visual_observations.csv")
    load_csv(db, "tank_aerial_visual_observation", DERIVED / "tank_aerial_supplemental_observations.csv")
    install_early_roof_review(db)
    load_csv(db, "tank_aerial_change_interval", DERIVED / "tank_aerial_change_intervals.csv")
    load_csv(db, "eia_crude_storage_capacity", DERIVED / "eia_crude_storage_capacity.csv")
    load_csv(db, "issuer_cash_distribution", DERIVED / "issuer_cash_distributions.csv")
    load_cash_override(db, DERIVED / "operator_issuer_cash_distributions.csv")
    load_csv(db, "issuer_cash_distribution", reference / "mmp_issuer_distributions.csv")
    load_csv(db, "issuer_cash_distribution", reference / "jpep_amid_issuer_distributions.csv")
    load_cash_override(db, DERIVED / "legacy_operator_issuer_cash_distributions.csv")
    load_cash_override(db, DERIVED / "canadian_operator_issuer_cash_distributions.csv")
    load_cash_override(db, DERIVED / "additional_us_issuer_cash_distributions.csv")
    load_cash_override(db, DERIVED / "refiner_issuer_cash_distributions.csv")
    load_cash_override(db, DERIVED / "spdr_issuer_cash_distributions.csv")
    load_cash_override(db, DERIVED / "historical_midstream_issuer_cash.csv")
    load_csv(db, "equity_cash_quote_basis", DERIVED / "spdr_cash_quote_basis.csv")
    load_csv(db, "equity_share_split_event", reference / "market_control_share_splits.csv")
    load_csv(db, "oil_fund_cash_distribution_coverage", DERIVED / "oil_fund_cash_distribution_coverage.csv")
    load_csv(db, "market_study_symbol_history", reference / "market_study_symbol_history.csv")
    load_csv(db, "equity_in_kind_distribution", reference / "equity_in_kind_distributions.csv")
    load_csv(db, "market_identity_limit", reference / "market_identity_limits.csv")
    load_csv(db, "market_identity_check", reference / "market_identity_checks.csv")
    load_csv(db, "equity_reference_bar", DERIVED / "equity_reference_bar.csv")
    load_csv(db, "equity_reference_bar", DERIVED / "equity_tail_reference.csv")
    load_csv(db, "equity_reference_bar", DERIVED / "equity_gap_reference.csv")
    load_csv(db, "equity_reference_bar", DERIVED / "legacy_equity_reference_bar.csv")
    load_csv(db, "equity_reference_price_check", DERIVED / "equity_reference_price_checks.csv")
    load_csv(db, "equity_merger_consideration", reference / "equity_merger_consideration.csv")
    load_csv(db, "historical_equity_price_anchor", reference / "historical_equity_price_anchors.csv")
    load_csv(db, "historical_equity_listing_window", reference / "historical_equity_listing_windows.csv")
    load_csv(db, "dolt_historical_equity_bar", DERIVED / "dolt_historical_equity_bars.csv")
    load_csv(db, "sentinel2_product_timing", DERIVED / "sentinel2_product_timing.csv")
    load_csv(db, "sentinel2_archive_availability", DERIVED / "sentinel2_archive_availability.csv")
    load_csv(db, "sentinel2_l1c_archive_availability", DERIVED / "sentinel2_l1c_archive_availability.csv")
    load_csv(db, "sentinel2_l1c_early_detector_footprint", DERIVED / "sentinel2_l1c_early_detector_footprints.csv")
    load_csv(db, "sentinel2_tank_scene_quality", DERIVED / "sentinel2_tank_scene_quality.csv")
    load_csv(db, "sentinel2_tank_scene_quality", DERIVED / "sentinel2_supplemental_tank_scene_quality.csv")
    load_csv(db, "sentinel2_l1c_tank_quality", DERIVED / "sentinel2_l1c_tank_quality.csv")
    load_csv(db, "sentinel2_l1c_local_file_check", DERIVED / "sentinel2_l1c_local_file_checks.csv")
    load_csv(db, "sentinel2_l1c_alternate_evidence", DERIVED / "sentinel2_l1c_alternate_evidence.csv")
    load_csv(db, "lidar_tank_coverage", DERIVED / "lidar_tank_coverage.csv")
    load_csv(db, "lidar_tank_observation", DERIVED / "lidar_tank_observations.csv")
    load_csv(db, "naip_tank_image_coverage", DERIVED / "naip_tank_image_coverage.csv")
    load_csv(db, "lidar_tank_coverage", DERIVED / "lidar_supplemental_tank_coverage.csv")
    load_csv(db, "lidar_tank_observation", DERIVED / "lidar_supplemental_tank_observations.csv")
    for layer in ("tank", "supplemental_tank"):
        load_csv(db, "lidar_2011_tank_coverage", DERIVED / f"lidar_2011_{layer}_coverage.csv")
        load_csv(db, "lidar_tank_observation", DERIVED / f"lidar_2011_{layer}_observations.csv")
    load_csv(db, "lidar_stroud_legacy_observation", DERIVED / "lidar_stroud_legacy_observations.csv")
    load_csv(db, "naip_tank_image_coverage", DERIVED / "naip_supplemental_tank_image_coverage.csv")
    load_csv(db, "ok_aerial_tank_image_coverage", DERIVED / "ok_aerial_tank_image_coverage.csv")
    load_csv(db, "ok_aerial_tank_union_coverage", DERIVED / "ok_aerial_tank_union_coverage.csv")
    load_csv(db, "eia_archive_exception", reference / "eia_archive_exceptions.csv")
    load_csv(db, "wpsr_recovered_value", DERIVED / "wpsr_2019_07_03_pdf_values.csv")


def add_tws(db):
    root = RAW / "tws"
    for path in sorted(root.glob("*/*.json"), key=lambda file: (file.stat().st_mtime, file.name)):
        if path.parent.name == "TGE":
            continue  # This raw response contains the successor use of the ticker, unrelated to Tallgrass.
        payload = json.loads(path.read_bytes())
        contract = payload["contract"]
        for row in payload["bars"]:
            date = dt.date.fromisoformat(row["date"])
            if not dt.date(2016, 1, 1) <= date <= dt.date(2026, 11, 10):
                continue
            db.execute("INSERT OR REPLACE INTO tws_daily_bar VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                path.parent.name, contract["conId"], payload["what_to_show"], date.isoformat(),
                row["open"], row["high"], row["low"], row["close"], row["volume"], row.get("average"),
                row.get("barCount"), contract.get("primaryExchange"), contract.get("currency"),
                payload["downloaded_at_utc"], str(path.relative_to(ROOT)),
            ))


def add_tws_futures(db):
    paths = sorted((RAW / "tws_futures").glob("CL_*.json"))
    paths.extend(sorted((RAW / "tws_futures_snapshots").glob("*/CL_*.json")))
    for path in paths:
        payload = json.loads(path.read_bytes())
        contract, catalog = payload["contract"], payload["catalog_contract"]
        expiry = dt.datetime.strptime(catalog["last_trading_date"], "%Y%m%d").date().isoformat()
        downloaded = payload["downloaded_at_utc"]
        chicago_date = dt.datetime.fromisoformat(downloaded).astimezone(ZoneInfo("America/Chicago")).date().isoformat()
        for row in payload["bars"]:
            date = dt.date.fromisoformat(row["date"]).isoformat()
            db.execute("""INSERT INTO tws_futures_bar VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(conid,session_date) DO UPDATE SET open=excluded.open,high=excluded.high,low=excluded.low,
                    close=excluded.close,volume=excluded.volume,wap=excluded.wap,trades=excluded.trades,
                    downloaded_at_utc=excluded.downloaded_at_utc,source_file=excluded.source_file,
                    download_date_chicago=excluded.download_date_chicago
                WHERE tws_futures_bar.session_date>=tws_futures_bar.download_date_chicago
                    AND excluded.session_date<excluded.download_date_chicago""", (
                "CL", catalog["contract_month"], contract["conId"], contract["localSymbol"],
                expiry, date, row["open"], row["high"], row["low"], row["close"], row["volume"],
                row.get("average"), row.get("barCount"), downloaded,
                str(path.relative_to(ROOT)), chicago_date,
            ))


def add_lidar_ept(db):
    for scope in ("ept", "ept_supplemental"):
        for path in sorted((RAW / "usgs" / scope).glob("*/nodes/*.json")):
            item = json.loads(path.read_bytes())
            laz = path.with_suffix(".laz")
            if item["selected_points"] > 0 and not laz.is_file():
                raise ValueError(f"Missing LiDAR file for {path}")
            db.execute("INSERT INTO lidar_ept_node VALUES (?,?,?,?,?,?,?,?,?,?)", (
                item["dataset"], item["id"], item["source_url"], item["source_points"],
                item["selected_points"], item["horizontal_crs"], item["buffer_m"],
                str(laz.relative_to(ROOT)) if laz.is_file() else None, item["source_sha256"], scope,
            ))


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    temporary = DB.with_suffix(".part")
    temporary.unlink(missing_ok=True)
    with sqlite3.connect(temporary) as db:
        db.executescript(SCHEMA)
        db.executescript(WTI_SCHEMA)
        db.executescript(EIA_MARKET_SCHEMA)
        db.executescript(MARKET_EVENT_SCHEMA)
        db.executescript(HEP_PARENT_SCHEMA)
        db.executescript(FX_REFERENCE_SCHEMA)
        db.executescript(HISTORICAL_EQUITY_SCHEMA)
        db.executescript(WEEKLY_SATELLITE_SCHEMA)
        db.executescript(TANK_ENGINEERING_SCHEMA)
        db.executescript(EPA_REVIEW_SCHEMA)
        db.executescript(PERMIT_EMISSIONS_GEOMETRY_SCHEMA)
        db.executescript(LIDAR_OPTICAL_SCHEMA)
        db.executescript(LIDAR_RADIAL_SCHEMA)
        db.executescript(LIDAR_ROOF_REFERENCE_SCHEMA)
        db.executescript(SCENE_WEATHER_SCHEMA)
        db.executescript(AERIAL_OPTICAL_SCHEMA)
        db.executescript(LANDSAT_SCHEMA)
        for loader in (add_eia, add_wpsr_factors, add_ibkr, add_catalogs, add_factors, add_weather,
                       add_registry, add_tank_geometry_registry, add_adjusted_equity, add_amid_monthly_reference, add_tickertech_jpep,
                       add_tws, add_tws_futures,
                       add_wti_curve, add_lidar_ept):
            loader(db)
        load_csv(db, "sentinel2_band_radiometry", DERIVED / "sentinel2_band_radiometry.csv")
        load_csv(db, "lidar_radial_profile", DERIVED / "lidar_radial_profile.csv")
        add_lidar_roof_reference(db)
        add_permit_tank_emissions_geometry(db)
        load_csv(db, "naip_2025_wcs_coverage", DERIVED / "naip_2025_wcs_coverage.csv")
        add_lidar_optical_calibration(db)
        add_scene_weather(db)
        add_aerial_optical_pairs(db)
        add_landsat(db)
        db.executemany("INSERT INTO eia_release_clock VALUES (?,?,?,?,?,?,?,?,?,?)", eia_rows(db))
        build_weekly_satellite_panel(db)
        db.executemany("INSERT INTO cftc_release_clock VALUES (?,?,?,?,?,?)", cftc_rows(db))
        db.executemany("INSERT INTO equity_session_calendar VALUES (?,?,?,?,?,?,?,?)", session_rows(db))
        db.executemany("INSERT INTO eia_equity_session_alignment VALUES (?,?,?,?,?,?,?,?)", release_alignment_rows(db))
        add_tlp_history(db)
        add_cvrr_history(db)
        add_semg_history(db)
        add_eia_market_inputs(db)
        add_market_event_panel(db)
        add_market_transaction_timeline(db)
        add_hep_parent_data(db)
        add_deeprock_data(db)
        add_fx_reference(db)
        add_execution_cost_inputs(db)
        if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("Database integrity check failed")
        for table, in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            print(table, db.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        for table in ("eia_release_clock", "cftc_release_clock", "equity_session_calendar", "eia_equity_session_alignment",
                      "eia_sentinel2_timing_coverage", "oke_equity_total_return", "mmp_equity_total_return",
                      "market_identity_evidence", "dolt_historical_equity_missing_session",
                      "wti_delivery_calendar", "wti_nearby_curve", "wti_curve_daily_coverage", "wti_close_crosscheck",
                      "historical_equity_source_coverage", "historical_equity_monthly_reference",
                      "historical_equity_daily_reference", "historical_distribution_ex_date_estimate", "eia_weekly_satellite_readiness",
                      "eia_satellite_window", "eia_satellite_window_input", "eia_satellite_tank_screen",
                      "tank_engineering_registry", "tank_engineering_cohort", "plains_permit_equipment_history", "lidar_tank_observation",
                      "lidar_optical_calibration_pair", "lidar_calibration_b04_angle_node", "lidar_radial_boundary_comparison",
                      "satellite_weather_clock", "scene_weather_observation", "tank_weather_station_distance",
                      "eia_inventory_release_panel", "eia_control_vintage", "eia_market_context", "oil_market_context",
                      "market_study_quote", "eia_event_market_session", "eia_equity_horizon_input", "eia_event_operator_exposure",
                      "tlp_market_session_coverage", "tlp_equity_total_return", "cvrr_market_session_coverage"):
            export(db, table)
        export(db, "lidar_roof_optical_reference", order_by="candidate_id, acquisition_date_utc, point_source_id, optical_captured_at_utc, level, version_id")
        export(db, "lidar_roof_elevation_reference", order_by="candidate_id, dataset, acquisition_date_utc, point_source_id")
        export(db, "roof_physics_input_coverage", order_by="candidate_id")
        export(db, "eia_roof_cohort_weekly_coverage", order_by="week_ending, cohort_id, ticker")
        export(db, "hep_parent_unit_observation", order_by="report_period_end")
        export(db, "cushing_connect_parent_equity_regime", order_by="legal_effective_date")
        export(db, "cushing_connect_pipeline_contract_term", order_by="agreement_version")
        export(db, "eia_cushing_connect_parent_equity_exposure", order_by="release_date")
        export_deeprock_data(db)
        export(db, "permit_tank_emissions_geometry", order_by="cohort_id, application_received_date, calculation_kind, tank_id")
        export(db, "tank_emissions_geometry_comparison", order_by="cohort_id, candidate_id, application_received_date")
        export(db, "magellan_central_fixed_roof_check", order_by="document_id, pdf_page, unassigned_parcel_contour_id")
        export(db, "stroud_tank_specification_check", order_by="candidate_id")
        export(db, "plains_original_design_reference", order_by="nominal_capacity_bbl")
        export(db, "plains_expansion_capacity_reference", order_by="cohort_id, statement_date")
        export(db, "plains_cushing_contour_reconciliation", order_by="match_status, cohort_id, parcel_ids, candidate_id")
        export(db, "north_farm_contour_reconciliation", order_by="terminal_id, farm_block, match_status, cohort_id, candidate_id")
        export(db, "enbridge_main_permit_equipment_history", order_by="tank_id")
        export(db, "enbridge_main_contour_reconciliation", order_by="match_status, cohort_id, candidate_id")
        export(db, "enbridge_main_cohort_capacity", order_by="cohort_id")
        export(db, "enbridge_main_covered_roof_outline", order_by="candidate_id")
        export(db, "terminal_disclosed_capacity", order_by="terminal_id, scope_key, reference_date")
        export(db, "nei_roof_equipment_evidence", order_by="inventory_year, terminal_id, cohort_id, equipment_id")
        export(db, "ngl_cushing_contour_reconciliation", order_by="parcel_group, cohort_id, candidate_id")
        export(db, "enterprise_cushing_contour_reconciliation", order_by="asset_scope_id, candidate_id")
        export(db, "terminal_data_coverage", order_by="terminal_id")
        export(db, EARLY_ROOF_TABLE, order_by="scope_id,candidate_id,observed_date")
        export(db, "market_transaction_event", order_by="disclosed_from_utc,event_id")
        export(db, "market_transaction_session_alignment", order_by="disclosed_from_utc,event_id,instrument")
        export(db, "eia_market_transaction_window", order_by="release_date,instrument,transaction_id")
        export(db, "eia_terminal_transfer_disclosure", order_by="release_date,terminal_id")
        export(db, "sentinel2_l1c_tandem_pair", order_by="capture_day, tile")
        export(db, "sentinel2_l1c_input_inventory", order_by="captured_at_utc, canonical_l2a_id")
        export(db, "cad_usd_reference_rate", order_by="observation_date")
        export(db, "eia_cad_equity_reference_inputs", order_by="release_date, instrument")
        export_execution_cost_inputs(db)
        export(db, "naip_tank_source_clock", order_by="acquisition_year, candidate_id, start_utc, source_polygon_id")
        export(db, "aerial_optical_pair", order_by="aerial_date, candidate_id, minimum_separation_seconds, level, scene_item_id, aerial_id")
        export(db, "landsat_route_coverage", order_by="acquired_date, platform, wrs_path_row")
        export(db, "landsat_local_product", order_by="captured_at_utc, source_kind, product_id")
        export(db, "landsat_tank_scene_quality", order_by="product_id, candidate_id")
        export(db, "landsat_reflectance_coefficients", order_by="product_id, band")
        export(db, "landsat_archive_asset_clock", order_by="provider, product_id, asset_label")
        export(db, "landsat_product_archive_clock", order_by="provider, product_id")
        export(db, "landsat_hls_source_scene_match", order_by="acquisition_key, hls_id")
        export(db, "landsat_hls_gap_coverage", order_by="acquisition_key")
        export(db, "landsat_pc_asset_exception", order_by="product_id, asset_label")
    temporary.replace(DB)


if __name__ == "__main__":
    main()
