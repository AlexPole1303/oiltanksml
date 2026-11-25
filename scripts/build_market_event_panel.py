"""Prepare dated equity quote and calendar inputs for weekly event research."""

CORE = ("PAA", "ENB", "ET", "OKE", "EPD", "NGL")
CONTROLS = ("AMLP", "ENFR", "SPY", "XLE", "XOP", "USO", "BNO")
COMPARISONS = ("PAGP", "SOBO", "TRP", "DINO", "CVI", "PSX", "KEY", "KMI", "WMB")
HISTORICAL = ("MMP", "JPEP", "AMID", "BKEP", "HEP", "PSXP", "SEMG", "TEP", "TEGP", "TGE", "ETE", "ETP", "SXL", "TLP", "CVRR")

SYMBOL_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_study_symbol_history (
    instrument TEXT, exchange_symbol TEXT, issuer_cik TEXT, valid_from TEXT,
    valid_to_exclusive TEXT, listing_venue TEXT, source_url TEXT,
    PRIMARY KEY (instrument, valid_from));
CREATE TABLE IF NOT EXISTS equity_in_kind_distribution (
    issuer_symbol TEXT, distributed_symbol TEXT, ex_date TEXT, record_date TEXT, payable_date TEXT,
    due_bill_from TEXT, due_bill_through TEXT, cash_usd_per_issuer_share TEXT,
    distributed_shares_total INTEGER, issuer_shares_on_record_date INTEGER, fractional_share_policy TEXT,
    entitlement_source_url TEXT, final_amount_source_url TEXT,
    PRIMARY KEY (issuer_symbol, distributed_symbol, ex_date));
CREATE VIEW IF NOT EXISTS market_study_issuer_cash_distribution AS
    SELECT coalesce(h.instrument,d.instrument) AS instrument,d.instrument AS issuer_symbol,
        d.ex_date,d.record_date,d.declared_date,d.payable_date,d.reported_amount,
        d.amount_on_post_2020_amlp_share_basis,d.currency,d.source_basis,d.source_url
    FROM issuer_cash_distribution d
    LEFT JOIN market_study_symbol_history h ON h.exchange_symbol=d.instrument
        AND coalesce(d.ex_date,d.record_date)>=h.valid_from
        AND (h.valid_to_exclusive IS NULL OR coalesce(d.ex_date,d.record_date)<h.valid_to_exclusive);
CREATE VIEW IF NOT EXISTS market_study_quote_issuer_identity AS
    SELECT q.instrument,q.session_date,coalesce(h.exchange_symbol,q.instrument) AS exchange_symbol,
        h.issuer_cik,coalesce(h.listing_venue,q.venue) AS venue,q.conid,h.source_url AS identity_source_url
    FROM market_study_quote q LEFT JOIN market_study_symbol_history h ON h.instrument=q.instrument
        AND q.session_date>=h.valid_from AND (h.valid_to_exclusive IS NULL OR q.session_date<h.valid_to_exclusive);
"""

CONTROL_CASH_SCHEMA = """
CREATE TABLE IF NOT EXISTS equity_cash_quote_basis (
    instrument TEXT, ex_date TEXT, amount_on_quote_share_basis TEXT NOT NULL,
    PRIMARY KEY (instrument, ex_date));
CREATE TABLE IF NOT EXISTS equity_share_split_event (
    instrument TEXT, first_split_adjusted_session TEXT, new_shares INTEGER, old_shares INTEGER,
    prior_cusip TEXT, current_cusip TEXT, source_file TEXT, source_url TEXT,
    PRIMARY KEY (instrument, first_split_adjusted_session));
CREATE TABLE IF NOT EXISTS oil_fund_cash_distribution_coverage (
    instrument TEXT PRIMARY KEY, issuer_confirmed_from TEXT, issuer_confirmed_through TEXT,
    issuer_distributions INTEGER, annual_source_url TEXT, interim_source_url TEXT,
    annual_source_file TEXT, interim_source_file TEXT,
    first_paired_broker_session TEXT, last_paired_broker_session TEXT, paired_broker_sessions INTEGER,
    max_close_difference REAL);
CREATE VIEW IF NOT EXISTS market_study_cash_on_quote_basis AS
    SELECT d.*,b.amount_on_quote_share_basis FROM market_study_issuer_cash_distribution d
    LEFT JOIN equity_cash_quote_basis b ON b.instrument=d.issuer_symbol AND b.ex_date=d.ex_date;
"""

SCHEMA = SYMBOL_HISTORY_SCHEMA + CONTROL_CASH_SCHEMA + """
CREATE TABLE market_study_instrument (
    instrument TEXT PRIMARY KEY, venue TEXT, currency TEXT, study_role TEXT);
CREATE TABLE market_study_quote (
    instrument TEXT, session_date TEXT, venue TEXT, currency TEXT, conid INTEGER,
    open REAL, high REAL, low REAL, close REAL, volume REAL, provider TEXT, price_basis TEXT, source_file TEXT,
    listing_segment TEXT, adjusted_ibkr_close REAL, adjusted_ibkr_source_file TEXT,
    issuer_cash_on_ex_date TEXT, issuer_cash_post_2020_amlp_basis TEXT, issuer_cash_source_url TEXT,
    PRIMARY KEY (instrument, session_date));
CREATE TABLE eia_event_market_session (
    release_date TEXT, venue TEXT, ordinal INTEGER, session_date TEXT, open_utc TEXT, close_utc TEXT,
    early_close INTEGER, PRIMARY KEY (release_date, venue, ordinal));
CREATE VIEW eia_event_operator_exposure AS
    SELECT c.release_date,c.latest_week_in_issue,t.terminal_id,t.issuer,t.ticker,t.role,t.interest_pct,
        t.valid_from,t.valid_to_exclusive,t.date_precision,
        CASE WHEN t.date_precision='unknown' THEN 'effective_dates_unspecified'
             WHEN t.date_precision='documented_from' THEN 'documented_as_of_lower_bound'
             WHEN t.date_precision='month' AND (substr(t.valid_from,1,7)=substr(c.release_date,1,7)
                    OR substr(t.valid_to_exclusive,1,7)=substr(c.release_date,1,7)) THEN 'boundary_month_uncertain'
             WHEN t.valid_from IS NULL OR t.valid_to_exclusive IS NULL THEN 'partially_bounded'
             ELSE 'within_documented_effective_dates' END AS exposure_date_status,t.source_url
    FROM eia_release_clock c JOIN terminal_company_history t
      ON (t.valid_from IS NULL OR c.release_date>=t.valid_from)
     AND (t.valid_to_exclusive IS NULL OR c.release_date<t.valid_to_exclusive
          OR (t.date_precision='month' AND substr(c.release_date,1,7)=t.valid_to_exclusive))
    WHERE EXISTS (SELECT 1 FROM eia_inventory_release_panel w WHERE w.release_date=c.release_date);
CREATE VIEW eia_equity_horizon_input AS
    WITH event_quotes AS (
        SELECT e.release_date,i.instrument,count(q.session_date) AS observed_first_five_sessions,
            count(b.session_date) AS amlp_observed_first_five_sessions
        FROM eia_event_market_session e JOIN market_study_instrument i USING (venue)
        LEFT JOIN market_study_quote q ON q.instrument=i.instrument AND q.session_date=e.session_date
        LEFT JOIN market_study_quote b ON b.instrument='AMLP' AND b.session_date=e.session_date
        WHERE e.ordinal BETWEEN 1 AND 5 GROUP BY e.release_date,i.instrument
    ), event_targets AS (
        SELECT release_date, min(week_ending) AS oldest_first_released_week,
            max(week_ending) AS latest_first_released_week, count(*) AS weeks_first_released
        FROM eia_inventory_release_panel GROUP BY release_date
    )
    SELECT q.release_date,r.oldest_first_released_week,r.latest_first_released_week,r.weeks_first_released,
        q.instrument,i.study_role,i.venue,i.currency,
        s.session_date AS entry_session,s.open_utc AS entry_open_utc,f.session_date AS fifth_session,
        f.close_utc AS fifth_close_utc,q.observed_first_five_sessions,q.amlp_observed_first_five_sessions,
        first.open AS first_session_open,fifth.close AS fifth_session_close,
        first.provider AS first_quote_provider,fifth.provider AS fifth_quote_provider,
        first.price_basis AS first_price_basis,fifth.price_basis AS fifth_price_basis,
        first.source_file AS first_quote_source,fifth.source_file AS fifth_quote_source,
        first.listing_segment AS first_listing_segment,fifth.listing_segment AS fifth_listing_segment,
        EXISTS (SELECT 1 FROM terminal_company_history t WHERE t.ticker=q.instrument AND t.date_precision='day'
            AND (t.valid_from BETWEEN s.session_date AND f.session_date
                 OR t.valid_to_exclusive BETWEEN s.session_date AND f.session_date)) AS terminal_relation_boundary_in_window,
        hfirst.open AS amlp_same_date_open,hfifth.close AS amlp_same_date_close,
        hfirst.provider AS amlp_first_quote_provider,hfifth.provider AS amlp_fifth_quote_provider,
        CASE WHEN f.session_date IS NULL THEN 'fifth_session_outside_calendar'
             WHEN first.open IS NULL THEN 'missing_first_bar'
             WHEN fifth.close IS NULL THEN 'missing_fifth_bar'
             WHEN q.observed_first_five_sessions<>5 THEN 'missing_intermediate_bar'
             WHEN first.listing_segment<>fifth.listing_segment THEN 'crosses_listing_segment'
             ELSE 'five_observed_quote_sessions' END AS own_quote_status,
        CASE WHEN i.currency<>'USD' THEN 'requires_fx_and_cross_venue_alignment'
             WHEN hfirst.open IS NULL OR hfifth.close IS NULL THEN 'missing_amlp_endpoint'
             WHEN q.amlp_observed_first_five_sessions<>5 THEN 'missing_intermediate_amlp_bar'
             ELSE 'same_date_usd_amlp_endpoints' END AS benchmark_alignment
    FROM event_quotes q JOIN event_targets r USING (release_date) JOIN market_study_instrument i USING (instrument)
    JOIN eia_event_market_session s ON s.release_date=q.release_date AND s.venue=i.venue AND s.ordinal=1
    LEFT JOIN eia_event_market_session f ON f.release_date=q.release_date AND f.venue=i.venue AND f.ordinal=5
    LEFT JOIN market_study_quote first ON first.instrument=i.instrument AND first.session_date=s.session_date
    LEFT JOIN market_study_quote fifth ON fifth.instrument=i.instrument AND fifth.session_date=f.session_date
    LEFT JOIN market_study_quote hfirst ON hfirst.instrument='AMLP' AND hfirst.session_date=s.session_date
    LEFT JOIN market_study_quote hfifth ON hfifth.instrument='AMLP' AND hfifth.session_date=f.session_date;
"""


def add_market_study_quotes(db, instrument=None):
    db.execute("""INSERT INTO market_study_quote
        WITH candidates AS (
            SELECT 1 AS priority,instrument,session_date,currency,conid,open,high,low,close,volume,provider,
                CASE WHEN provider LIKE 'ibkr_tws%' THEN 'ibkr_trades_split_adjusted'
                     WHEN provider='ibkr_connector_tail' THEN 'ibkr_connector_last_regular_session'
                     WHEN provider IN ('tickertech_tlp','tickertech_semg','tickertech_cvrr','tickertech_amid',
                                       'tickertech_tep','tickertech_sxl','tickertech_tegp')
                         THEN 'reference_reported_ohlcv_rounded_volume'
                     ELSE 'reference_reported_ohlcv' END AS price_basis,source_file
            FROM equity_research_trade_bar
            UNION ALL
            SELECT 2,symbol,session_date,'USD',NULL,open,high,low,close,volume,'tickertech',
                'archive_reported_ohlcv',source_file FROM historical_equity_daily_reference
            UNION ALL
            SELECT 3,symbol,session_date,currency,NULL,open,high,low,close,volume,'dolt_archive',
                'archive_reported_ohlcv',source_file FROM dolt_historical_equity_quality WHERE quality_status='ohlcv_range_valid'
        ), usable AS (
            SELECT c.*,i.venue,row_number() OVER (PARTITION BY c.instrument,c.session_date ORDER BY c.priority) AS n
            FROM candidates c JOIN market_study_instrument i USING (instrument)
            JOIN equity_session_calendar e ON e.venue=i.venue AND e.session_date=c.session_date
            WHERE (? IS NULL OR c.instrument=?) AND c.currency=i.currency AND c.volume>0 AND c.low>0 AND c.low<=min(c.open,c.close)
              AND c.high>=max(c.open,c.close)
              AND (i.study_role<>'historical_operator' OR EXISTS (
                   SELECT 1 FROM historical_equity_listing_window l WHERE l.symbol=c.instrument
                   AND c.session_date BETWEEN l.first_study_session AND l.last_study_session))
        )
        SELECT q.instrument,q.session_date,q.venue,q.currency,q.conid,q.open,q.high,q.low,q.close,q.volume,
            q.provider,q.price_basis,q.source_file,CASE WHEN q.instrument='OKE' THEN 'OKE' ELSE coalesce(l.listing_id,q.instrument) END,
            a.close,a.source_file,
            CASE WHEN d.amount_on_quote_share_basis IS NULL THEN d.reported_amount
                 WHEN q.price_basis='ibkr_trades_split_adjusted' THEN d.amount_on_quote_share_basis END,
            d.amount_on_post_2020_amlp_share_basis,d.source_url
        FROM usable q LEFT JOIN historical_equity_listing_window l ON l.symbol=q.instrument
            AND q.session_date BETWEEN l.first_study_session AND l.last_study_session
        LEFT JOIN tws_equity_research_bar a ON a.instrument=q.instrument AND a.conid=q.conid
            AND a.session_date=q.session_date AND a.series='ADJUSTED_LAST'
        LEFT JOIN market_study_cash_on_quote_basis d ON d.instrument=q.instrument AND d.ex_date=q.session_date
            AND d.currency=q.currency
        WHERE q.n=1""", (instrument, instrument))


def add_market_event_panel(db):
    for role, symbols in (("core_operator", CORE), ("market_control", CONTROLS),
                          ("comparison_operator", COMPARISONS), ("historical_operator", HISTORICAL)):
        db.executemany("INSERT INTO market_study_instrument VALUES (?,?,?,?)",
                       ((symbol, "XTSE" if symbol == "KEY" else "XNYS", "CAD" if symbol == "KEY" else "USD", role)
                        for symbol in symbols))
    add_market_study_quotes(db)
    db.execute("""INSERT INTO eia_event_market_session
        WITH ordered AS (
            SELECT venue,session_date,open_utc,close_utc,early_close,
                row_number() OVER (PARTITION BY venue ORDER BY session_date) AS n
            FROM equity_session_calendar
        )
        SELECT a.release_date,a.venue,s.n-e.n+1,s.session_date,s.open_utc,s.close_utc,s.early_close
        FROM eia_equity_session_alignment a JOIN ordered e ON e.venue=a.venue AND e.session_date=a.next_session_date
        JOIN ordered s ON s.venue=e.venue AND s.n BETWEEN e.n AND e.n+9
        WHERE EXISTS (SELECT 1 FROM eia_inventory_release_panel p WHERE p.release_date=a.release_date)""")
