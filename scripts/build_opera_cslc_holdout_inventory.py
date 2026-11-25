"""Match NASA OPERA CSLC granules to Sentinel-1 burst identities and EIA clocks."""
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from shapely import make_valid, transform
from shapely.geometry import Polygon

from build_holdout_optical_inputs import indexed
from build_s1_holdout_gap_burst_plan import PROJECT, roofs
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import BBOX, RAW, ROOT, TLS, get, write

STEM = "opera_cslc_holdout"
STORE = RAW/"opera_cslc_cushing"
COLLECTION = "C2777443834-ASF"
STATIC_COLLECTION = "C2795135668-ASF"
API = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
NAME = re.compile(r"OPERA_L2_CSLC-S1_T(\d{3})-(\d{6})-(IW[123])_(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)_(S1[A-D])_(VV|VH)_v(\d+\.\d+)")
SIDES = ("previous","current")


def utc(value):
    return datetime.fromisoformat(value)


def day_records(day):
    later = (datetime.fromisoformat(day)+timedelta(days=1)).date().isoformat()
    query = dict(collection_concept_id=COLLECTION,bounding_box=BBOX,
                 temporal=f"{day}T00:00:00Z,{later}T00:00:00Z",page_size=100)
    file = STORE/f"{day}.umm.json"
    answer = json.loads(get(API+"?"+urlencode(query),file))
    if answer["hits"]!=len(answer["items"]):
        raise ValueError(f"Incomplete NASA daily inventory for {day}")
    return [parse(r,file) for r in answer["items"]]


def polygon(granule):
    polygons = granule["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"]
    if len(polygons)!=1:
        raise ValueError("Expected one OPERA burst footprint")
    points = polygons[0]["Boundary"]["Points"]
    return transform(make_valid(Polygon([(p["Longitude"],p["Latitude"]) for p in points])),PROJECT,interleaved=False)


def file_fields(granule):
    name = granule["GranuleUR"]
    measurement, = [r for r in granule["DataGranule"]["ArchiveAndDistributionInformation"]
                    if r["Format"]=="HDF5" and r["Name"]==f"{name}.h5"]
    links = [r["URL"] for r in granule["RelatedUrls"] if r.get("Format")=="HDF5" and r["Type"]=="GET DATA"
             and urlparse(r["URL"]).hostname in ("cumulus.asf.earthdatacloud.nasa.gov","datapool.asf.alaska.edu")]
    digest = measurement["Checksum"]
    if (digest["Algorithm"]!="MD5" or not re.fullmatch("[0-9a-f]{32}",digest["Value"]) or not links
        or int(measurement["SizeInBytes"])<=0 or any(not u.endswith(f"/{name}.h5") for u in links)):
        raise ValueError(f"Wrong HDF5 file metadata for {name}")
    return dict(size_bytes=int(measurement["SizeInBytes"]),md5=digest["Value"],download_url=links[0])


def parse(item,file):
    m,u = item["meta"],item["umm"]
    granule = u["GranuleUR"]
    match = NAME.fullmatch(granule)
    if not match or m["native-id"]!=granule or m["collection-concept-id"]!=COLLECTION:
        raise ValueError(f"Unexpected OPERA granule identity {granule}")
    orbit,relative,swath,capture,_,sat,pol,version = match.groups()
    begin = utc(u["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"])
    end = utc(u["TemporalExtent"]["RangeDateTime"]["EndingDateTime"])
    attrs = {x["Name"]:x["Values"] for x in u["AdditionalAttributes"]}
    (group_id,) = attrs["GROUP_ID"]
    if (attrs["OPERA_BURST_ID"]!=[f"T{orbit}_{relative}_{swath}"] or int(attrs["PATH_NUMBER"][0])!=int(orbit)
        or group_id.split("_")[0]!=sat or group_id.split("_")[-1]!=orbit
        or datetime.strptime(capture,"%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)!=begin or end<=begin):
        raise ValueError(f"Different burst properties for {granule}")
    archive = u["ProviderDates"]
    inserted = min(utc(r["Date"]) for r in archive if r["Type"]=="Insert")
    generated = utc(u["DataGranule"]["ProductionDateTime"])
    revision = utc(m["revision-date"])
    if not end<=generated<=inserted or inserted>revision:
        raise ValueError(f"Wrong production/availability clock for {granule}")
    return dict(granule=granule,concept_id=m["concept-id"],revision=m["revision-id"],version=version,group_id=group_id,
        orbit=int(orbit),relative_burst_id=int(relative),swath=swath,satellite=sat,polarization=pol,
        absolute_orbit=int(group_id.split("_")[-2]),capture_start=begin,capture_end=end,
        direction=attrs["ASCENDING_DESCENDING"][0],production_at=generated,inserted_at=inserted,
        revision_at=revision,footprint=polygon(u),catalog_file=str(file.relative_to(ROOT)),**file_fields(u))


def static_layers(matched):
    keys = {(g["orbit"],g["relative_burst_id"],g["swath"]) for sample in matched.values() for g in sample}
    file = STORE/"static.umm.json"
    q = dict(collection_concept_id=STATIC_COLLECTION,bounding_box=BBOX,page_size=100)
    result = json.loads(get(API+"?"+urlencode(q),file))
    if result["hits"]!=len(result["items"]):
        raise ValueError("Incomplete NASA static geometry inventory")
    out = {}
    for item in result["items"]:
        m,u = item["meta"],item["umm"]
        attrs = {x["Name"]:x["Values"] for x in u["AdditionalAttributes"]}
        orbit,relative,swath = attrs["OPERA_BURST_ID"][0].split("_")
        key = int(orbit.removeprefix("T")),int(relative),swath
        if key not in keys:
            continue
        name = u["GranuleUR"]
        if key in out or m["collection-concept-id"]!=STATIC_COLLECTION or not name.startswith(
                f"OPERA_L2_CSLC-S1-STATIC_{orbit}-{relative}-{swath}_"):
            raise ValueError(f"Duplicate or inconsistent geometry layer for {key}")
        inserted = min(utc(d["Date"]) for d in u["ProviderDates"] if d["Type"]=="Insert")
        out[key] = dict(granule=name,concept_id=m["concept-id"],inserted_at=inserted,footprint=polygon(u),
                        catalog_file=str(file.relative_to(ROOT)),**file_fields(u))
    if keys!=out.keys():
        raise ValueError("No static radar geometry for one or more measured burst keys")
    return out


def needed():
    requests = {}
    for r in read("s1_holdout_gap_burst_tank_pairs"):
        if r["full_context_in_both_bursts"]!="True":
            continue
        for side in SIDES:
            parent = r[f"{side}_product_name"]
            key = parent,int(r["common_relative_burst_id"]),r["swath"]
            requests[key] = dict(parent_product_name=parent,relative_burst_id=key[1],swath=key[2],
                orbit=int(r["relative_orbit"]),direction=r["direction"],satellite=parent.split("_")[0],
                absolute_orbit=int(parent.split("_")[-3]),start=utc(r[f"{side}_capture_start_utc"]),scope="holdout")
    for r in read("s1_slc_2019_reference_burst_plan"):
        if r["qualified_aerial_shadow_reference"]!="True":
            continue
        parent = r["original_slc_parent_product"]
        filename = r["asf_vv_file"].split("_")
        key = parent,int(filename[1]),r["selected_swath"]
        if filename[2]!=key[2]:
            raise ValueError("2019 control has inconsistent burst identifiers")
        requests[key] = dict(parent_product_name=parent,relative_burst_id=key[1],swath=key[2],orbit=34,
            direction="ascending",satellite="S1B",absolute_orbit=int(parent.split("_")[-3]),
            start=datetime.strptime(parent.split("_")[-5],"%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc),
            scope="aerial_control")
    return requests


def match_granules(requests, by_day):
    result = {}
    for key,r in requests.items():
        matched = []
        end = datetime.strptime(r["parent_product_name"].split("_")[-4],"%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        for g in by_day[r["start"].date().isoformat()]:
            fields = ("orbit","relative_burst_id","swath","satellite","absolute_orbit")
            if (all(r[k]==g[k] for k in fields) and r["direction"].upper()==g["direction"].upper()
                and g["polarization"]=="VV" and r["start"]<=g["capture_start"]<g["capture_end"]<=end):
                matched.append(g)
        result[key] = sorted(matched,key=lambda g:(g["inserted_at"],g["concept_id"]))
    return result


def weekly(matched,static):
    clock = indexed("holdout_optical_weeks")
    geometry = {(week,r["candidate_id"]):context for week,rows in roofs().items() for r,_,_,context,_ in rows}
    weeks = {week:dict(release_date=r["release_date"],eligible=set(),covered=set(),available=set()) for week,r in clock.items()
             if r["b04_status"]!="measured"}
    output = []
    for r in read("s1_holdout_gap_burst_tank_pairs"):
        if r["full_context_in_both_bursts"]!="True":
            continue
        week,tank = r["week_ending"],r["candidate_id"]
        group = weeks[week]
        group["eligible"].add(tank)
        area = geometry[week,tank]
        cutoff = utc(clock[week]["cutoff_utc"])
        chosen = []
        for side in SIDES:
            key = r[f"{side}_product_name"],int(r["common_relative_burst_id"]),r["swath"]
            selected = next((g for g in matched[key] if g["footprint"].covers(area)),None)
            chosen.append(selected)
        if not all(chosen):
            continue
        g = chosen[0]
        layer = static[g["orbit"],g["relative_burst_id"],g["swath"]]
        if not layer["footprint"].covers(area):
            continue
        group["covered"].add(tank)
        causal = all(g["inserted_at"]<cutoff for g in (*chosen,layer))
        if causal:
            group["available"].add(tank)
        output.append({k:r[k] for k in ("week_ending","release_date","candidate_id","terminal_id","common_relative_burst_id","swath")} |
            dict(forecast_cutoff_utc=cutoff.isoformat(),both_granules_and_geometry_inserted_before_cutoff=causal,
                 static_geometry_concept_id=layer["concept_id"],static_geometry_inserted_at=layer["inserted_at"].isoformat(),
                 **{f"{side}_{k}":(g[k].isoformat() if isinstance(g[k],datetime) else g[k])
                    for side,g in zip(SIDES,chosen,strict=True) for k in ("concept_id","granule","inserted_at","download_url")}))
    return [dict(week_ending=w,release_date=r["release_date"],native_slc_full_context_roofs=len(r["eligible"]),
                opera_full_context_roofs=len(r["covered"]),opera_full_context_before_cutoff_roofs=len(r["available"]))
            for w,r in sorted(weeks.items())],output


def access_probe(url):
    try:
        with urlopen(Request(url,headers={"Range":"bytes=0-7"}),timeout=15,context=TLS) as result:
            return dict(http_status=result.status,host=urlparse(result.geturl()).hostname,
                        hdf5_signature=result.read(8)==b"\x89HDF\r\n\x1a\n")
    except HTTPError as exc:
        return dict(http_status=exc.code,host=urlparse(exc.geturl()).hostname,hdf5_signature=False)
    except OSError as exc:
        return dict(error=type(exc).__name__,hdf5_signature=False)


def controls(tanks, assets):
    optical = defaultdict(list)
    for r in read("shadow_timing_holdout_observations"):
        if r["geometry_mode"]=="dated_2011_lidar" and r["fit_status"]=="resolved_in_geometry_diagnostic":
            optical[r["candidate_id"]].append(r)
    endpoints = {(r["candidate_id"],r[f"{side}_concept_id"],r["terminal_id"]) for r in tanks for side in SIDES}
    result = []
    for tank,concept,terminal in sorted(endpoints):
        asset = assets[concept]
        begin,end = (utc(asset[k]) for k in ("capture_start","capture_end"))
        choices = []
        for r in optical[tank]:
            captured = utc(r["captured_at_utc"])
            separation = max(abs((captured-begin).total_seconds()),abs((captured-end).total_seconds()))/3600
            if separation<=24:
                choices.append((separation,r["version_id"],r["canonical_l2a_id"],r))
        if not choices:
            continue
        gap,_,_,r = min(choices,key=lambda q:q[:3])
        if (r["terminal_id"]!=terminal or not utc(r["captured_at_utc"])<=utc(r["archive_available_at_utc"])
            <=utc(r["geometry_available_at_utc"]) or not (ROOT/r["tile_metadata_path"]).is_file()):
            raise ValueError(f"Optical control has inconsistent identity or clock for {tank}")
        result.append(dict(candidate_id=tank,terminal_id=terminal,opera_concept_id=concept,opera_granule=asset["granule"],
            radar_begin_utc=asset["capture_start"],radar_end_utc=asset["capture_end"],maximum_separation_hours=gap,
            **{f"optical_{k}":r[k] for k in ("version_id","captured_at_utc","geometry_available_at_utc","estimated_gap_m",
                "near_optimal_gap_min_m","near_optimal_gap_max_m","tile_metadata_path","measurement_source","observation_id")}))
    return result


def main():
    requests = needed()
    days = sorted({r["start"].date().isoformat() for r in requests.values()})
    with ThreadPoolExecutor(max_workers=6) as pool:
        by_day = dict(zip(days,pool.map(day_records,days),strict=True))
    matched = match_granules(requests,by_day)
    static = static_layers(matched)
    weeks,tanks = weekly(matched,static)
    inventory = [{k:v for k,v in requests[key].items() if k!="start"} |
        {k:(v.isoformat() if isinstance(v,datetime) else v) for k,v in g.items() if k not in ("footprint",*requests[key])}
        for key,group in matched.items() for g in group]
    assets = {r["concept_id"]:r for r in inventory}
    static_inventory = [dict(orbit=k[0],relative_burst_id=k[1],swath=k[2],**{
        n:(v.isoformat() if isinstance(v,datetime) else v) for n,v in r.items() if n!="footprint"}) for k,r in sorted(static.items())]
    aerial = [r for r in inventory if r["scope"]=="aerial_control"]
    probes = {key:access_probe(r["download_url"]) for key,r in (("aerial_control",aerial[0]),("static",static_inventory[0]),
              ("holdout",next(r for r in inventory if r["scope"]=="holdout")))}
    optical = controls(tanks,assets)
    by_role = defaultdict(set)
    for r in tanks:
        if r["both_granules_and_geometry_inserted_before_cutoff"]:
            for side in SIDES:
                by_role[r[f"{side}_concept_id"]].add(r["week_ending"])
    control_counts = Counter(r["opera_concept_id"] for r in optical)
    download = []
    for r in (*static_inventory,*assets.values()):
        identity = r["concept_id"]
        role = ("static_geometry" if r in static_inventory else "aerial_control" if r["scope"]=="aerial_control"
                else "optical_control" if control_counts[identity] else
                "holdout_pre_cutoff" if by_role[identity] else "holdout_retrospective")
        download.append({k:r[k] for k in ("concept_id","granule","download_url","size_bytes","md5","catalog_file")} |
            dict(role=role,optical_control_count=control_counts[identity],pre_cutoff_weeks="|".join(sorted(by_role[identity]))))
    priorities = ("static_geometry","aerial_control","optical_control","holdout_pre_cutoff","holdout_retrospective")
    download.sort(key=lambda r:(priorities.index(r["role"]),-r["optical_control_count"],r["granule"]))
    primer = [r for r in download if r["role"] in priorities[:2]]
    report = dict(collection=COLLECTION,query_days=len(days),cmr_granules=sum(map(len,by_day.values())),
        native_bursts_requested_by_scope=dict(Counter(r["scope"] for r in requests.values())),
        native_bursts_with_opera_match_by_scope=dict(Counter(requests[k]["scope"] for k,v in matched.items() if v)),
        exact_granules=len(assets),exact_granule_bytes=sum(r["size_bytes"] for r in assets.values()),
        static_geometry_granules=len(static),static_geometry_bytes=sum(r["size_bytes"] for r in static.values()),
        anonymous_download_probes=probes,
        gap_weeks_with_opera_full_context=sum(r["opera_full_context_roofs"]>0 for r in weeks),
        gap_weeks_with_opera_full_context_before_cutoff=sum(r["opera_full_context_before_cutoff_roofs"]>0 for r in weeks),
        gap_tank_weeks_with_opera_full_context=sum(r["opera_full_context_roofs"] for r in weeks),
        gap_tank_weeks_with_opera_full_context_before_cutoff=sum(r["opera_full_context_before_cutoff_roofs"] for r in weeks),
        optical_controls_within_24h=len(optical),optical_control_granules=len({r["opera_concept_id"] for r in optical}),
        optical_controls_within_12h=sum(r["maximum_separation_hours"]<=12 for r in optical),
        optical_controls_within_1h=sum(r["maximum_separation_hours"]<=1 for r in optical),
        download_queue_bytes=sum(r["size_bytes"] for r in download),
        initial_aerial_check_files=len(primer),initial_aerial_check_bytes=sum(r["size_bytes"] for r in primer),
        interpretation="The CMR insert timestamp is the recorded catalog availability of the processed granule. "
        "Both acquisition endpoints and the static geometry must cover the roof and 120 m context. OPERA uses geocoded complex data; "
        "catalog evidence alone does not establish usable roof height or successful pixel access.")
    export(f"{STEM}_granules",inventory)
    export(f"{STEM}_static_geometry",static_inventory)
    export(f"{STEM}_download_manifest",download)
    export(f"{STEM}_weeks",weeks)
    if tanks:
        export(f"{STEM}_tank_pairs",tanks)
    if optical:
        export(f"{STEM}_optical_controls",optical)
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="")


if __name__=="__main__":
    main()
