"""Attach issuer cash and security entitlements to the scheduled five-session windows."""
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from decimal import Decimal

from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write


def actions(db):
    estimated_ex = {
        (instrument, record): (day, rule)
        for instrument, record, day, rule in db.execute("""
            SELECT instrument,record_date,estimated_ex_date,rule_source_url FROM historical_distribution_ex_date_estimate
        """)
    }
    for row in db.execute("""
        SELECT instrument,issuer_symbol,ex_date,record_date,declared_date,payable_date,reported_amount,
            amount_on_quote_share_basis,amount_on_post_2020_amlp_share_basis,currency,source_basis,source_url
        FROM market_study_cash_on_quote_basis
    """):
        r = dict(row)
        estimate, rule = estimated_ex.get((r["issuer_symbol"], r["record_date"]), (None, None))
        day = r["ex_date"] or estimate or r["record_date"]
        status = "confirmed_ex_date" if r["ex_date"] else "estimated_ex_date" if estimate else "record_date_only"
        yield (r["instrument"], day, day, "cash_distribution", status, r["source_url"],
               dict(r, estimated_ex_date=estimate, estimation_rule_source_url=rule))

    for row in db.execute("SELECT * FROM equity_share_split_event"):
        r = dict(row)
        day = r["first_split_adjusted_session"]
        yield r["instrument"], day, day, "share_split", "exchange_effective_session", r["source_url"], r

    for row in db.execute("SELECT * FROM equity_merger_consideration"):
        r = dict(row)
        yield (r["retiring_instrument"], r["last_trading_session"], r["closing_date"], "retiring_security_conversion",
               "documented_last_trade_and_legal_close", r["source_url"], r)

    for row in db.execute("SELECT * FROM equity_in_kind_distribution"):
        r = dict(row)
        if r["cash_usd_per_issuer_share"]:
            cash = db.execute("""
                SELECT reported_amount FROM market_study_cash_on_quote_basis WHERE instrument=? AND ex_date=? AND currency='USD'
            """, (r["issuer_symbol"], r["ex_date"])).fetchall()
            r["cash_component_in_issuer_cash_ledger"] = any(
                Decimal(amount) == Decimal(r["cash_usd_per_issuer_share"]) for amount, in cash)
        yield (r["issuer_symbol"], r["due_bill_from"], r["ex_date"], "in_kind_share_distribution",
               "documented_due_bills_and_ex_date", r["entitlement_source_url"], r)

    for row in db.execute("SELECT * FROM market_spin_off_terms"):
        r = dict(row)
        yield (r["parent_instrument"], r["due_bill_first_session"], r["parent_ex_distribution_session"],
               "spin_off", "documented_due_bills_and_ex_date", r["trading_mechanics_source_url"], r)


def window_links(db, by_instrument):
    for w in db.execute("""
        SELECT h.release_date,w.study_phase,w.week_ending,h.instrument,h.entry_session,h.fifth_session,
            h.first_price_basis,h.fifth_price_basis,h.currency
        FROM eia_equity_horizon_input h JOIN eia_release_clock r USING(release_date)
        JOIN eia_satellite_window w ON w.week_ending=r.latest_week_in_issue AND w.endpoint='current'
        WHERE h.fifth_session IS NOT NULL
        ORDER BY h.release_date,h.instrument
    """):
        window = dict(w)
        instrument, entry, exit_day = (w[field] for field in ("instrument", "entry_session", "fifth_session"))
        for event, terms in by_instrument[instrument]:
            start, finish = (event[field] for field in ("first_relevant_date", "last_relevant_date"))
            if finish < entry or start > exit_day:
                continue
            kind, status = event["event_type"], event["date_status"]
            treatment = "review_security_entitlement"
            if kind == "cash_distribution":
                if status != "confirmed_ex_date":
                    treatment = "ex_date_needs_confirmation"
                elif finish == entry:
                    treatment = "entry_session_is_already_ex_distribution"
                elif terms["currency"] != w["currency"]:
                    treatment = "cash_during_holding_requires_fx"
                else:
                    treatment = "cash_ex_date_after_entry_through_exit"
            elif kind == "share_split":
                if start == entry:
                    treatment = "entry_session_already_on_new_share_basis"
                elif w["first_price_basis"] == w["fifth_price_basis"] == "ibkr_trades_split_adjusted":
                    treatment = "quote_endpoints_use_broker_split_adjusted_basis"
                else:
                    treatment = "endpoint_share_basis_needs_review"
            elif kind == "retiring_security_conversion" and terms["closing_date"] > exit_day:
                treatment = "last_trading_session_in_window_legal_close_later"
            yield dict(window, event_id=event["event_id"], event_type=kind, date_status=status,
                       first_relevant_date=start, last_relevant_date=finish, handling_note=treatment)


def main():
    events, by_instrument = [], defaultdict(list)
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        for instrument, start, end, kind, status, source, terms in sorted(
            actions(db), key=lambda row: (row[0], row[1], row[3], row[2], row[4])
        ):
            if not start or not end or start > end:
                raise ValueError(f"Invalid security event interval for {instrument} {kind}")
            event_id = f"{instrument}/{kind}/{start}/{end}"
            event = dict(event_id=event_id, instrument=instrument, first_relevant_date=start, last_relevant_date=end,
                         event_type=kind, date_status=status, source_url=source,
                         issuer_terms_json=json.dumps(terms, ensure_ascii=False, separators=(",", ":")))
            events.append(event)
            by_instrument[instrument].append((event, terms))
        links = list(window_links(db, by_instrument))
    if len(events) != len({r["event_id"] for r in events}):
        raise ValueError("Equity action identifiers are not unique")
    if len(links) != len({(r["release_date"], r["instrument"], r["event_id"]) for r in links}):
        raise ValueError("A security action was linked twice in a holding window")
    export("study_equity_action_events", events)
    export("study_equity_window_actions", links)
    summary = dict(source_actions=len(events), action_window_links=len(links),
                   source_actions_by_type=dict(sorted(Counter(r["event_type"] for r in events).items())),
                   linked_actions_by_handling_note=dict(sorted(Counter(r["handling_note"] for r in links).items())))
    write(DERIVED / "study_equity_window_actions_summary.json", json.dumps(summary, indent=2).encode() + b"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
