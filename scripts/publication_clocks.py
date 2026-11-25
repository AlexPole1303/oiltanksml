"""Attach report clocks with separate archived, scheduled, and conservative timestamps."""
import csv
import datetime as dt
import html
import io
import re
from zoneinfo import ZoneInfo

from collect_public import RAW, ROOT, write

EASTERN = ZoneInfo("America/New_York")
EIA_SCHEDULE = "https://www.eia.gov/petroleum/supply/weekly/schedule.php"
CFTC_SCHEDULE = "https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm"
CFTC_SPECIAL = "https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm"
CFTC_2019 = "https://www.cftc.gov/PressRoom/PressReleases/7864-19"
CFTC_2025 = "https://www.cftc.gov/PressRoom/PressReleases/9147-25"
CFTC_2025_YEAR_END_REPORT = (
    "https://www.cftc.gov/sites/default/files/files/dea/cotarchives/2025/futures/petroleum_lf122325.htm"
)
DERIVED = ROOT / "data" / "derived"


def utc(date, time=dt.time()):
    return dt.datetime.combine(date, time, EASTERN).astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def table_rows(source):
    for raw in re.findall(r"<tr\b[^>]*>(.*?)</tr>", source, re.I | re.S):
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", raw, re.I | re.S)
        yield [" ".join(html.unescape(re.sub(r"<[^>]*>", "", cell)).split()) for cell in cells]


def eia_rows(db):
    schedule = (RAW / "eia" / "release_schedule.html").read_text()
    special = {}
    for cells in table_rows(schedule):
        if len(cells) != 5 or not re.fullmatch(r"[A-Za-z]+ \d{1,2}, \d{4}", cells[0]):
            continue
        period = dt.datetime.strptime(cells[0], "%B %d, %Y").date()
        day = dt.datetime.strptime(cells[1], "%B %d, %Y").date()
        time = dt.datetime.strptime(cells[3].replace(".", ""), "%I:%M %p").time()
        special[str(day)] = (time, cells[4], period)
    if not special:
        raise ValueError("Could not parse the archived WPSR holiday schedule")
    dated_notices = {
        "2022-06-29": ("Hardware failure. Two reference weeks released together.", "press506"),
        "2023-11-15": ("Planned system upgrade. Two reference weeks released together.", "press545"),
    }
    for release_day, latest_week, earliest_week, first_pdf, source in db.execute("""
        SELECT release_date, max(week_ending), min(week_ending),
            max(reported_as='current_week_recovered_from_pdf'), max(source_url)
        FROM eia_cushing_as_published GROUP BY release_date ORDER BY release_date
    """):
        date = dt.date.fromisoformat(release_day)
        last = dt.date.fromisoformat(latest_week)
        timing_source, note = EIA_SCHEDULE, ""
        time = None
        if first_pdf:
            basis, timing_source, note = "archived_pdf_day_only", source, "Archived CSV duplicates the prior issue."
        elif release_day in special:
            time, note, period = special[release_day]
            if period != last:
                raise ValueError(f"WPSR schedule disagrees with archived week on {release_day}")
            basis = "dated_holiday_schedule"
        elif release_day in dated_notices:
            note, notice = dated_notices[release_day]
            time, basis = dt.time(10, 30), "dated_notice_regular_time"
            timing_source = f"https://www.eia.gov/pressroom/releases/{notice}.php"
        elif date.weekday() == 2 and (date - last).days == 5:
            time, basis = dt.time(10, 30), "standard_schedule_rule"
        else:
            basis, timing_source = "archived_day_only", source
        yield (release_day, earliest_week, latest_week, utc(last, dt.time(7)), utc(date, time) if time else None,
               basis, utc(date), utc(date + dt.timedelta(days=1)), note, timing_source)


def cftc_rows(db):
    raw = (RAW / "factors" / "cftc_catchup_2025.html").read_text()
    backlogged = {}
    for cells in table_rows(raw):
        if len(cells) == 3 and all(re.fullmatch(r"\d{2}/\d{2}/\d{4}\+*", cell) for cell in cells):
            report, original, rescheduled = [dt.datetime.strptime(cell.rstrip("+"), "%m/%d/%Y").date() for cell in cells]
            backlogged[report] = (original, rescheduled)
    if not backlogged:
        raise ValueError("Could not parse CFTC December 2025 catch-up schedule")
    observed_delays = {
        "2020-12-21": "2020-12-28", "2021-06-15": "2021-06-21", "2023-01-31": "2023-02-24",
        "2023-02-07": "2023-03-03", "2023-02-14": "2023-03-08", "2023-02-21": "2023-03-10",
        "2023-02-28": "2023-03-14", "2023-03-07": "2023-03-16", "2023-03-14": "2023-03-21",
        "2025-01-07": "2025-01-13",
    }
    current = (RAW / "factors" / "cftc_release_schedule.html").read_text()
    current_dates = []
    for cells in table_rows(current):
        if not cells or cells[0] not in (dt.date(2026, m, 1).strftime("%B") for m in range(1, 13)):
            continue
        for value in cells[1:]:
            if re.fullmatch(r"\d{2}\*?", value):
                current_dates.append(dt.datetime.strptime(f"2026 {cells[0]} {value.rstrip('*')}", "%Y %B %d").date())
    for (observed,) in db.execute("SELECT report_date FROM cftc_wti ORDER BY report_date"):
        date = dt.date.fromisoformat(observed)
        weekday = date.weekday()
        regular_friday = date + dt.timedelta(days=(4 - weekday) % 7)
        release, original = None, regular_friday
        basis, source = "regular_rule_date_unverified", CFTC_SCHEDULE
        if date in backlogged:
            original, release = backlogged[date]
            basis, source = "dated_catchup_schedule", CFTC_2025
        elif observed in observed_delays:
            release = dt.date.fromisoformat(observed_delays[observed])
            basis, source = "dated_special_announcement", CFTC_SPECIAL
        elif dt.date(2018, 12, 24) <= date <= dt.date(2019, 2, 26):
            rank = (date - dt.date(2018, 12, 24)).days // 7
            # The CFTC announced one Friday followed by alternating Tuesday and Friday.
            release = dt.date(2019, 2, 1) + dt.timedelta(days=(rank // 2) * 7 + (4 if rank % 2 else 0))
            basis, source = "catchup_cadence_inferred", CFTC_2019
        elif date.year == 2026 or date == dt.date(2025, 12, 30):
            if dates := [value for value in current_dates if 0 < (value - date).days < 8]:
                release = min(dates)
                basis = "dated_tentative_schedule"
        if observed == "2025-12-23":
            # The WTI report footer is dated December 31. Admit it after that day
            # because the catch-up schedule only gives an intended December 29 release.
            release = dt.date(2025, 12, 31)
            basis, source = "archived_report_updated_after_schedule", CFTC_2025_YEAR_END_REPORT
        # For a regular rule with no archived clock, allow two business days of ordinary holiday delay.
        cutoff_day = (release + dt.timedelta(days=1) if release else regular_friday + dt.timedelta(days=5))
        if release is not None and release < date:
            raise ValueError(f"CFTC release before its reference date: {date}")
        yield (observed, str(original), str(release) if release else None, utc(cutoff_day), basis, source)


def export(db, table, order_by="1"):
    rows = db.execute(f"SELECT * FROM {table} ORDER BY {order_by}")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([column[0] for column in rows.description])
    writer.writerows(rows)
    write(DERIVED / f"{table}.csv", output.getvalue().encode())
