"""Validate the historical WTI/Brent feature against public ALFRED vintages."""

import argparse
import csv
import hashlib
import io
import json
import math
import time
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from build_s1_reference_matches import DERIVED, export, read
from collect_public import RAW, ROOT, TLS, write
from fit_cushing_public_state import basis, spot_reference
from publication_clocks import EASTERN

STEM = "cushing_spot_vintages"
SERIES = ("DCOILWTICO","DCOILBRENTEU")
STORE = RAW/"eia"/"alfred_spot_vintages"
ENDPOINT = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"


def fetch(day, series):
    oldest = (date.fromisoformat(day)-timedelta(days=130)).isoformat()
    url = ENDPOINT+"?"+urlencode(dict(id=series,vintage_date=day,cosd=oldest,coed=day))
    path = STORE/f"{series}_{day}.csv"
    if path.is_file():
        raw = path.read_bytes()
    else:
        for delay in (.25,1.,None):
            try:
                with urlopen(url,context=TLS,timeout=16) as response:
                    raw = response.read()
                break
            except (URLError,TimeoutError) as error:
                if isinstance(error,HTTPError) and error.code<429 or delay is None:
                    raise
                time.sleep(delay)
    column = f"{series}_{day.replace('-','')}"
    table = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if table.fieldnames != ["observation_date",column]:
        raise ValueError(f"Wrong archival headers for {series} at {day}: {table.fieldnames}")
    marks = {}
    chronological = []
    for row in table:
        when = date.fromisoformat(row["observation_date"]).isoformat()
        if not oldest<=when<=day or chronological and chronological[-1]>=when:
            raise ValueError(f"Invalid observation date for {series} at {day}")
        chronological.append(when)
        value = row[column]
        if value not in ("","."):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Invalid archived price for {series} at {when}")
            marks[when] = number
    if len(marks)<63:
        raise ValueError(f"Insufficient archived prices for {series} at {day}")
    if not path.exists():
        write(path,raw)
    return marks,dict(url=url,path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest())


def inspect(row):
    cutoff = datetime.fromisoformat(row["cutoff_utc"])
    vintage = (cutoff.astimezone(EASTERN).date()-timedelta(days=1)).isoformat()
    books,provenance = zip(*(fetch(vintage,s) for s in SERIES),strict=True)
    common = sorted(books[0].keys() & books[1].keys())
    latest = row["spot_reference_day"]
    at = bisect_left(common,latest)
    if at<62:
        raise ValueError(f"Fewer than 63 known pairs at the reference date on {row['release_date']}")
    known = common[:at+int(latest in books[0] and latest in books[1])][-63:]
    if len(known)!=63:
        raise ValueError(f"Insufficient paired observations on {row['release_date']}")
    spread = sorted(books[0][d]-books[1][d] for d in known)
    mark = books[0][known[-1]]-books[1][known[-1]]
    pair = (mark-spread[31], mark-(books[0][known[-6]]-books[1][known[-6]]),
            mark-(books[0][known[-21]]-books[1][known[-21]]))
    return dict(release_date=row["release_date"],study_phase=row["study_phase"],vintage_date=vintage,
        cutoff=row["cutoff_utc"],stored_reference_date=latest,archived_reference_date=known[-1],
        last_joint_vintage_date=common[-1],archived_basis_level=pair[0],archived_basis_move_5=pair[1],
        archived_basis_move_20=pair[2],
        **{f"{label}_{key}":value for label,record in zip(("wti","brent"),provenance,strict=True) for key,value in record.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-only",action="store_true")
    args = parser.parse_args()
    days,marks = spot_reference()
    source = [r | dict(spot_reference_day=basis(datetime.fromisoformat(r["cutoff_utc"]),days,marks)[1])
              for r in read("cushing_outer_capture_nowcast_forecasts")
              if not args.holdout_only or r["study_phase"]=="holdout"]
    with ThreadPoolExecutor(max_workers=12) as pool:
        audited = list(pool.map(inspect,source))
    for r in audited:
        original,_,_ = basis(datetime.fromisoformat(r["cutoff"]),days,marks)
        vintage = [r[f"archived_basis_{k}"] for k in ("level","move_5","move_20")]
        r["maximum_basis_difference_usd_bbl"] = float(max(abs(a-b) for a,b in zip(original,vintage,strict=True)))
    report = dict(releases=len(audited),matched_reference_dates=sum(r["stored_reference_date"]==r["archived_reference_date"] for r in audited),
        exact_basis_features=sum(r["maximum_basis_difference_usd_bbl"]<1e-9 for r in audited),
        maximum_basis_difference_usd_bbl=max(r["maximum_basis_difference_usd_bbl"] for r in audited))
    export(STEM,audited)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
