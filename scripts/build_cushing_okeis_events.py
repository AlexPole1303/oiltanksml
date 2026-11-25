"""Preserve dated operator events independently of any satellite height estimates."""

import hashlib
import json

from build_s1_reference_matches import export, read
from collect_public import ROOT


DOCUMENTS = {
    "-44534761069947207": "cef0618f6f23586e01eb8b0de24fd920e9b764ba4ae1fa9210af87db64516813",
    "-8465169134545152864": "897c56b4c7dfaba4a58e68e5c48a934c1382ab25410ec53459733e3e3375be05",
    "-5258296408881310420": "4b493c9f8ff6c9a946c00e4ba69086d15f75deeba6351e291077cb3351a9dc54",
    "4989303168206992388": "c5b04fed363c1271f2a57f45b1bf5a598027ae0cb70dcef5c6401d6267e83ae2",
}


def main():
    sources = {r["document_id"]: r for r in read("cushing_okeis_documents") if r["document_id"] in DOCUMENTS}
    for key, digest in DOCUMENTS.items():
        if hashlib.sha256((ROOT / sources[key]["local_pdf"]).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Reviewed operator source changed: {key}")
    tanks = read("tank_engineering_registry")
    rows = []

    def add(doc, page, terminal, equipment, kind, reported_date, *, actual=None, earliest=None, hours=None, emissions_unit=None):
        source = sources[doc]
        candidates = sorted(r["candidate_id"] for r in tanks if r["documented_terminal_id"] == terminal and equipment in {
            n.removeprefix("CG-").removeprefix("T-") for n in json.loads(r["possible_equipment_ids_json"] or "[]")})
        rows.append(dict(terminal_id=terminal, equipment_id=equipment, emission_unit_id=emissions_unit, event_kind=kind,
                         operator_document_date=reported_date, observed_event_local_date=actual,
                         planned_event_on_or_after_local_date=earliest, reported_duration_hours=hours,
                         actual_event_local_time=None, measured_liquid_level_m=None, measured_volume_bbl=None,
                         possible_candidate_ids_json=json.dumps(candidates), individual_contour_verified=len(candidates) == 1,
                         source_document_id=doc, source_page=page, source_pdf=source["local_pdf"],
                         source_sha256=DOCUMENTS[doc], source_url=source["document_url"],
                         listed_site_id=source["okeis_site_id"]))

    permit, = [r for r in read("permit_tanks") if r["document_id"] == "8499" and r["tank_id"] == "CG-2227"]
    if (permit["facility_id"], permit["emissions_unit_id"], permit["roof_type"]) != ("1514", "6491", "EFR"):
        raise ValueError("Enbridge roof landing equipment differs from the permit")
    add("-44534761069947207", 40, "enbridge_cushing", "2227", "observed_roof_landing_then_refill", "2026-10-20",
        actual="2026-07-23", hours=2, emissions_unit="6491")
    for n, day in ((2230, "2026-04-07"), (2233, "2026-04-08"), (2235, "2026-04-08"), (2236, "2026-05-05"),
                   (2237, "2026-04-08"), (2239, "2026-05-05"), (2244, "2026-06-24")):
        kind = "observed_primary_seal_inspection" if n == 2244 else "observed_primary_and_secondary_seal_inspection"
        add("-44534761069947207", 3, "enbridge_cushing", str(n), kind, "2026-10-20", actual=day)
    for n in (7300, 7600):
        add("-8465169134545152864", 1, "plains_cushing", str(n), "planned_refill", "2026-08-25", earliest="2026-09-25")
    for n in (1007, 1009, 2525, 2526, 2527, 2533, 2534, 2535, 2539):
        add("-5258296408881310420", 1, "semgroup_cushing", str(n), "planned_seal_inspection", "2026-10-06", earliest="2026-11-11")
    add("4989303168206992388", 1, "deeprock_development", "7706", "out_of_service_as_reported", "2026-09-08", actual="2026-09-08")
    add("4989303168206992388", 1, "deeprock_development", "7706", "planned_refill", "2026-09-08", earliest="2026-11-09")
    export("cushing_okeis_operator_events", rows)
    print(f"Archived {len(rows)} operator events and plans")


if __name__ == "__main__":
    main()
