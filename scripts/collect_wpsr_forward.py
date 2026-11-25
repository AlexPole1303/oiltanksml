"""Preserve WPSR issues released after the end of the fixed research calendar."""
import argparse
import hashlib
from datetime import date, datetime, time, timedelta
from threading import Event

from audit_cushing_holdout_eia_vintages import archive, stocks
from build_s1_reference_matches import export, read
from collect_public import RAW, get, wpsr_jobs
from collect_wpsr_factors import TABLES, fetch_table
from publication_clocks import EIA_SCHEDULE, table_rows, utc


RELEASES = RAW / "eia" / "wpsr_forward"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    cutoff = date.fromisoformat(max(r["release_date"] for r in read("eia_release_clock")))
    first = cutoff + timedelta(days=1)
    if args.as_of < first:
        parser.error("Choose an as-of day after the fixed research calendar")
    for _, collect, options in wpsr_jobs(args.as_of.isoformat(), first.isoformat(), RELEASES):
        collect(*options)
    schedule_path = RAW / "eia" / "forward_release_schedule.html"
    schedule_raw = get(EIA_SCHEDULE, schedule_path, refresh=True)
    special = {}
    for cells in table_rows(schedule_raw.decode("utf-8")):
        if len(cells) != 5 or not cells[0] or not cells[0][0].isalpha():
            continue
        try:
            period, issue = [datetime.strptime(value, "%B %d, %Y").date() for value in cells[:2]]
            clock = datetime.strptime(cells[3].replace(".", ""), "%I:%M %p").time()
        except ValueError:
            continue
        special[issue.isoformat()] = period.isoformat(), clock
    rows, sources = [], []
    paused = Event()
    for path in sorted(RELEASES.glob("*_table4.csv")):
        slug = path.name[:10]
        issue = slug.replace("_", "-")
        day = date.fromisoformat(issue)
        if not first <= day <= args.as_of:
            continue
        week, prior, inventory, previous, difference, precision, _, row, proof = stocks(issue, directory=RELEASES)
        basis, published = "archived_day_only", None
        if issue in special:
            reference, clock = special[issue]
            if week != reference:
                raise ValueError(f"WPSR archive week disagrees with published schedule for {issue}")
            basis, published = "dated_holiday_schedule", utc(day, clock)
        elif day.weekday() == 2 and (day - date.fromisoformat(week)).days == 5:
            basis, published = "standard_schedule_rule", utc(day, time(10, 30))
        for table in TABLES:
            status = fetch_table(RELEASES, slug, table, paused)
            if status and "error" in status:
                raise ValueError(status["error"])
            _, _, source = archive(issue, table, directory=RELEASES)
            sources.append(dict(release_date=issue, table_number=table, **source))
        rows.append(dict(release_date=issue, latest_week_in_issue=week, previous_week_ending=prior,
            measured_at_utc=utc(date.fromisoformat(week), time(7)), scheduled_release_at_utc=published, timing_basis=basis,
            conservative_prerelease_before_utc=utc(day), conservative_postrelease_from_utc=utc(day + timedelta(days=1)),
            schedule_source_url=EIA_SCHEDULE, schedule_sha256=hashlib.sha256(schedule_raw).hexdigest(), cushing_kbbl=inventory,
            previous_cushing_kbbl=previous, cushing_change_kbbl=difference, printed_precision_kbbl=precision,
            table4_row=row, **proof))
    if not rows:
        raise ValueError("No newer WPSR issues were available in the requested interval")
    export("eia_wpsr_forward_releases", rows)
    export("eia_wpsr_forward_sources", sources)
    print(f"Verified {len(rows)} releases after {cutoff}, {len(sources)} source tables.")


if __name__ == "__main__":
    main()
