"""Inventory same-burst Sentinel-1 coverage for all holdout weeks missing B04."""
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from urllib.parse import urlencode
from urllib.error import HTTPError

from pyproj import Transformer
from shapely import make_valid, transform
from shapely.geometry import Point, shape

from build_holdout_optical_inputs import indexed
from build_s1_reference_matches import read
from build_s1_slc_2019_reference_plan import ASF, CDSE, ROOT_STORE
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_public import BBOX, ROOT, get, write, write_csv

STEM = "s1_holdout_gap_burst"
PROJECT = Transformer.from_crs(4326,32614,always_xy=True).transform
CONTEXT_M = 120
SIDES = ("previous","current")


def timestamp(r,k):
    return datetime.fromisoformat(r[k])


def candidates():
    optics = {w:r for w,r in indexed("holdout_optical_weeks").items() if r["b04_status"]!="measured"}
    catalog = {r["product_id"]:r for r in read("s1_cushing_catalog")}
    windows = {r["week_ending"]:r for r in read("s1_cushing_weekly_catalog_coverage") if r["study_phase"]=="holdout"}
    with closing(read_only(SOURCE)) as db:
        clock = {(w,e):(r,*map(datetime.fromisoformat,(a,b,c))) for w,e,r,a,b,c in db.execute("""
            SELECT week_ending,endpoint,release_date,window_starts_after_utc,window_ends_by_utc,inputs_available_before_utc
            FROM eia_satellite_window WHERE study_phase='holdout'""")}
    pairs = {}
    for w,o in sorted(optics.items()):
        row = windows[w]
        sides = [[catalog[i] for i in json.loads(row[f"slc_{side}_product_ids_json"])] for side in SIDES]
        for side,products in zip(SIDES,sides,strict=True):
            issue,lower,upper,cutoff = clock[w,side]
            if issue!=o["release_date"] or cutoff!=timestamp(o,"cutoff_utc"):
                raise ValueError(f"Wrong radar reporting clock for {w}/{side}")
            for p in products:
                if not (p["product_type"]=="IW_SLC__1S" and lower<timestamp(p,"capture_start_utc")<=
                        timestamp(p,"capture_end_utc")<=upper and timestamp(p,"catalog_publication_utc")<cutoff):
                    raise ValueError(f"Ineligible radar acquisition at {w}/{side}")
        pairs[w] = [(a,b) for a in sides[0] for b in sides[1]
                    if (a["relative_orbit"],a["direction"])==(b["relative_orbit"],b["direction"])]
        if bool(pairs[w])!=(row["slc_common_relative_track"]=="True"):
            raise ValueError(f"Same-orbit search differs from the archived result on {w}")
    return optics,pairs


def cdse_records(product):
    identity = product["product_id"]
    west,south,east,north = BBOX.split(",")
    polygon = f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"
    criteria = f"ParentProductId eq '{identity}' and OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')"
    path = ROOT_STORE/f"{identity}.json"
    query = CDSE+"?"+urlencode({"$filter":criteria,"$count":"true","$top":100})
    response = json.loads(get(query,path))
    rows = response["value"]
    if len(rows)!=response["@odata.count"] or response.get("@odata.nextLink"):
        raise ValueError(f"Incomplete Copernicus burst query for {identity}")
    for b in rows:
        if ((b["ParentProductId"],b["ParentProductName"],b["PlatformSerialIdentifier"],str(b["RelativeOrbitNumber"]),
             b["OrbitDirection"].upper())!=(identity,product["product_name"],product["satellite"][-1],
                                            product["relative_orbit"],product["direction"].upper())
            or f"/measurement/{product['satellite'].lower()}-" not in b["S3Path"]):
            raise ValueError(f"Wrong Copernicus burst identity for {identity}")
    return rows,str(path.relative_to(ROOT))


def asf_records(product, cdse):
    root = product["product_name"].removesuffix(".SAFE")
    start,end = (product[k] for k in ("capture_start_utc","capture_end_utc"))
    fields = dict(processingLevel="BURST",start=start,end=end,output="geojson")
    if cdse:
        fields["fullBurstID"] = ",".join(sorted({f"{b['RelativeOrbitNumber']:03d}_{b['BurstId']:06d}_{b['SwathIdentifier']}"
                                                 for b in cdse}))
    else:
        fields["intersectsWith"] = bbox_wkt()
    path = ROOT_STORE/f"asf_holdout_{product['product_id']}.geojson"
    try:
        payload = json.loads(get(ASF+"?"+urlencode(fields),path))
    except HTTPError as exc:
        if exc.code!=404:
            raise
        return [],None
    rows = [r for r in payload["features"] if r["properties"]["url"].split("/")[3]==root]
    return rows,str(path.relative_to(ROOT))


def bbox_wkt():
    w,s,e,n = BBOX.split(",")
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


def geometry(value):
    return transform(make_valid(shape(value)),PROJECT,interleaved=False)


def bursts(product):
    cdse,cdse_file = cdse_records(product)
    asf,asf_file = asf_records(product,cdse)
    matching = {}
    for row in asf:
        p = row["properties"]
        b = p["burst"]
        if p["platform"]!=f"SENTINEL-1{product['satellite'][-1]}":
            raise ValueError(f"Different ASF platform for {product['product_id']}")
        key = int(b["relativeBurstID"]),b["subswath"],p["polarization"]
        if key in matching:
            raise ValueError(f"Duplicate ASF burst {key}")
        matching[key] = row
    used,rows = set(),[]
    for b in cdse:
        key = int(b["BurstId"]),b["SwathIdentifier"],b["PolarisationChannels"]
        item = matching.get(key)
        p = item["properties"] if item else {}
        footprint = geometry(b["GeoFootprint"])
        if item:
            used.add(key)
            footprint = footprint.intersection(geometry(item["geometry"]))
        if item and not math.isclose(float(p["burst"]["azimuthAnxTime"]),float(b["AzimuthAnxTime"]),rel_tol=0,abs_tol=.01):
            raise ValueError(f"Cross-catalog azimuth time differs for {key}")
        rows.append(dict(cdse_absolute_burst_id=b["AbsoluteBurstId"],asf_absolute_burst_id=p.get("burst",{}).get("absoluteBurstID"),
            relative_burst_id=key[0],swath=key[1],polarization=key[2],
            footprint=footprint,cdse_uuid=b["Id"],cdse_catalog_file=cdse_file,
            asf_catalog_file=asf_file,asf_data_url=p.get("url"),asf_annotation_url=(p.get("additionalUrls") or [None])[0],
            identity_evidence="cdse_and_asf" if item else "cdse"))
    for key,item in matching.items():
        if key in used:
            continue
        p = item["properties"]
        rows.append(dict(cdse_absolute_burst_id=None,asf_absolute_burst_id=p["burst"]["absoluteBurstID"],relative_burst_id=key[0],
            swath=key[1],polarization=key[2],footprint=geometry(item["geometry"]),
            cdse_uuid=None,cdse_catalog_file=cdse_file,asf_catalog_file=asf_file,
            asf_data_url=p["url"],asf_annotation_url=(p.get("additionalUrls") or [None])[0],identity_evidence="asf"))
    keys = [(r["relative_burst_id"],r["swath"],r["polarization"]) for r in rows]
    if len(set(keys))!=len(keys):
        raise ValueError(f"Repeated burst in {product['product_id']}")
    return dict(product,bursts=rows,cdse_records=len(cdse),asf_records=len(asf),crosschecked=len(used),
                cdse_catalog_file=cdse_file,asf_catalog_file=asf_file)


def roofs():
    records = defaultdict(list)
    with closing(read_only(SOURCE)) as db:
        centres = {r[0]:(r[1],r[2]) for r in db.execute("SELECT candidate_id,longitude,latitude FROM tank_engineering_registry")}
    registry = {r["candidate_id"]:r for r in read("sentinel2_space_time_coverage_roof_registry")}
    for r in read("holdout_optical_tank_weeks"):
        key = r["candidate_id"]
        evidence = registry[key]
        if any(not r[k] or r[k]>r["previous_measurement_at_utc"][:10] for k in ("first_seen","lidar_source_available_date","roof_evidence_date")):
            continue
        lon,lat = centres[key]
        radius = math.sqrt(float(r["circular_area_m2"])/math.pi)
        point = Point(*PROJECT(lon,lat))
        records[r["week_ending"]].append((r,point,point.buffer(radius),point.buffer(radius+CONTEXT_M),
                                           float(evidence["shell_above_ground_m"])))
    return records


def pair_matches(week, eligible, pair, outputs):
    first,last = (outputs[p["product_id"]] for p in pair)
    earlier,later = ({(r["relative_burst_id"],r["swath"],r["polarization"]):r for r in p["bursts"]} for p in (first,last))
    overlap = {k:(earlier[k],later[k],earlier[k]["footprint"].intersection(later[k]["footprint"]))
               for k in earlier.keys() & later.keys() if k[2]=="VV"}
    result = []
    for tank,point,outline,context,shell in eligible:
        covering = [(shape_.covers(context),shape_.boundary.distance(point),key,a,b) for key,(a,b,shape_) in overlap.items()
                    if shape_.covers(outline)]
        if not covering:
            continue
        full,_,key,a,b = max(covering,key=lambda c:(c[0],c[1],c[2]))
        result.append(dict(week_ending=week,release_date=tank["release_date"],candidate_id=tank["candidate_id"],
            terminal_id=tank["terminal_id"],first_seen=tank["first_seen"],lidar_source_available_date=tank["lidar_source_available_date"],
            roof_evidence_date=tank["roof_evidence_date"],diameter_m=2*math.sqrt(float(tank["circular_area_m2"])/math.pi),
            rim_to_ground_m=shell,relative_orbit=first["relative_orbit"],direction=first["direction"],
            common_relative_burst_id=key[0],swath=key[1],polarization=key[2],full_context_in_both_bursts=full,
            satellite_pair=first["satellite"]+"/"+last["satellite"],
            capture_separation_days=(timestamp(last,"capture_start_utc")-timestamp(first,"capture_start_utc")).total_seconds()/86400,
            **{f"{side}_{k}":p[k] for side,p in zip(SIDES,(first,last),strict=True)
               for k in ("product_id","product_name","capture_start_utc","catalog_publication_utc")},
            **{f"{side}_{k}":b[k] for side,b in zip(SIDES,(a,b),strict=True) for k in
               ("cdse_absolute_burst_id","asf_absolute_burst_id","cdse_uuid","cdse_catalog_file","asf_catalog_file",
                "asf_data_url","asf_annotation_url","identity_evidence")}))
    return result


def save(name,rows):
    fields = tuple(rows[0])
    write_csv(DERIVED/f"{STEM}_{name}.csv",fields,([r[k] for k in fields] for r in rows))


def grd_manifest(gaps):
    archive = {r["product_id"]:r for r in read("s1_cushing_catalog")}
    reasons = defaultdict(set)
    for r in read("s1_cushing_weekly_catalog_coverage"):
        week = r["week_ending"]
        if r["study_phase"]!="holdout" or week not in gaps:
            continue
        sides = [[archive[i] for i in json.loads(r[f"grd_{s}_product_ids_json"])] for s in SIDES]
        for a in sides[0]:
            for b in sides[1]:
                if (a["relative_orbit"],a["direction"])==(b["relative_orbit"],b["direction"]):
                    for side,p in zip(SIDES,(a,b),strict=True):
                        if not timestamp(p,"catalog_publication_utc")<timestamp(gaps[week],"cutoff_utc"):
                            raise ValueError(f"Late GRD product on {week}")
                        reasons[p["product_id"]].add(f"{week}/{side}")
    rows = [dict(archive[i],selection_reasons="|".join(sorted(rs))) for i,rs in sorted(reasons.items())]
    fields = tuple(rows[0])
    write_csv(DERIVED/"s1_grd_holdout_gap_manifest.csv",fields,([r[k] for k in fields] for r in rows))
    return len(rows)


def main():
    weeks,pairs = candidates()
    products = {p["product_id"]:p for sample in pairs.values() for pair in sample for p in pair}
    with ThreadPoolExecutor(max_workers=6) as pool:
        data = {}
        for r in pool.map(bursts,products.values()):
            data[r["product_id"]] = r
            print(r["product_id"],r["cdse_records"],r["asf_records"],flush=True)
    records = roofs()
    tank_rows,calendar,downloads = [],[],{}
    for week,o in weeks.items():
        sample = [r for p in pairs[week] for r in pair_matches(week,records[week],p,data)]
        tank_rows.extend(sample)
        full = [r for r in sample if r["full_context_in_both_bursts"]]
        cross = [r for r in full if all(r[f"{side}_identity_evidence"]=="cdse_and_asf" for side in SIDES)]
        for r in full:
            for side in SIDES:
                identity = r[f"{side}_product_id"],r[f"{side}_cdse_uuid"] or r[f"{side}_asf_data_url"]
                item = downloads.setdefault(identity,dict(**{k:r[f"{side}_{k}"] for k in (
                    "product_id","product_name","cdse_uuid","asf_data_url","asf_annotation_url","identity_evidence")},
                    swath=r["swath"],polarization=r["polarization"],relative_burst_id=r["common_relative_burst_id"],weeks=set()))
                item["weeks"].add(week)
        calendar.append(dict(week_ending=week,release_date=o["release_date"],b04_status=o["b04_status"],
            original_slc_same_track_pairs=len(pairs[week]),dated_open_roofs=len(records[week]),
            roofs_in_shared_vv_burst=len({r["candidate_id"] for r in sample}),
            roofs_with_full_context=len({r["candidate_id"] for r in full}),
            roofs_with_full_context_and_dual_catalog_verification=len({r["candidate_id"] for r in cross})))
    save("weeks",calendar)
    if tank_rows:
        save("tank_pairs",tank_rows)
    if downloads:
        save("download_manifest",[dict(r,weeks="|".join(sorted(r["weeks"]))) for _,r in sorted(downloads.items())])
    save("parents",[{k:r[k] for k in ("product_id","product_name","satellite","relative_orbit","direction","catalog_publication_utc",
        "source_file","cdse_records","asf_records","crosschecked","cdse_catalog_file","asf_catalog_file")} for r in data.values()])
    report = dict(optical_gap_weeks=len(calendar),same_track_weeks=sum(bool(r["original_slc_same_track_pairs"]) for r in calendar),
        same_track_parent_pairs=sum(r["original_slc_same_track_pairs"] for r in calendar),unique_parent_products=len(data),
        original_grd_parent_products_for_gap_weeks=grd_manifest(weeks),
        parents_with_copernicus_bursts=sum(bool(r["cdse_records"]) for r in data.values()),
        parents_with_asf_bursts=sum(bool(r["asf_records"]) for r in data.values()),
        parent_burst_records_crosschecked=sum(r["crosschecked"] for r in data.values()),
        required_vv_burst_assets=len(downloads),
        required_vv_burst_assets_with_asf_url=sum(bool(r["asf_data_url"]) for r in downloads.values()),
        weeks_with_shared_vv_burst=sum(bool(r["roofs_in_shared_vv_burst"]) for r in calendar),
        weeks_with_full_context=sum(bool(r["roofs_with_full_context"]) for r in calendar),
        weeks_with_full_context_and_dual_catalog_verification=sum(bool(r["roofs_with_full_context_and_dual_catalog_verification"]) for r in calendar),
        tank_weeks_with_full_context=sum(r["roofs_with_full_context"] for r in calendar),
        full_context_m_beyond_tank_radius=CONTEXT_M,
        interpretation="A common relative orbit and common VV burst footprint establish only coverage. Original parent availability "
        "is checked against the issue cutoff. Burst extraction may be a later archive service. No radar samples, EIA targets or market data are read.")
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="")


if __name__=="__main__":
    main()
