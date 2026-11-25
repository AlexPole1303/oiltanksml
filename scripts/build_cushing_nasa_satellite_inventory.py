"""Inventory ASTER VNIR and NISAR GSLC against dated roofs and reporting clocks."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta
from itertools import product
from struct import unpack
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from pyproj import Transformer
from shapely import make_valid, transform, union_all
from shapely.geometry import Point, Polygon

from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from build_weekly_shadow_validation import legacy_geometry
from collect_public import RAW, ROOT, TLS, get, write, write_csv

STEM = "cushing_nasa_satellite"
STORE = RAW / STEM
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
BBOX = "-96.81,35.73,-96.54,36.10"
ASTER = "C3306888411-LPCLOUD"
NISAR = {"beta":"C2850259510-ASF", "provisional":"C2854332392-ASF", "validated":"C3622244601-ASF"}
NISAR_NAME = re.compile(r"NISAR_L2_PR_GSLC_(\d{3})_(\d{3})_([AD])_(\d{3})_(\d{4})_(\w{4})_([AL])_(\d{8}T\d{6})_.*")
PROJECT = Transformer.from_crs(4326,32614,always_xy=True).transform
END = "2026-10-31T23:59:59Z"
RESOLVED = "resolved_in_geometry_diagnostic"


def catalog(label, filters, first, last, refresh):
    path = STORE / f"{label}.umm.json"
    url = CMR+"?"+urlencode(dict(filters,bounding_box=BBOX,temporal=f"{first},{last}",page_size=200))
    raw = get(url,path,refresh=refresh)
    data = json.loads(raw)
    if data["hits"] != len(data["items"]):
        raise ValueError(f"NASA search results are incomplete for {label}")
    return data["items"], str(path.relative_to(ROOT)), hashlib.sha256(raw).hexdigest()


def footprint(u):
    polygons = u["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"]
    result = union_all([make_valid(Polygon([(p["Longitude"],p["Latitude"]) for p in polygon["Boundary"]["Points"]]))
                        for polygon in polygons])
    if result.is_empty or result.geom_type not in ("Polygon","MultiPolygon"):
        raise ValueError(f"Invalid footprint in {u['GranuleUR']}")
    return transform(result,PROJECT,interleaved=False)


def file_asset(u, suffix, algorithm, *, required=True):
    name = u["GranuleUR"]+suffix
    records = [r for r in u["DataGranule"]["ArchiveAndDistributionInformation"] if r["Name"]==name]
    urls = {x["URL"] for x in u["RelatedUrls"] if x["URL"].startswith("https://") and x["URL"].endswith("/"+name)
            and (not suffix.endswith(".png") or "/BROWSE/" in x["URL"])}
    if not required and not records and not urls:
        return {}
    if len(records)!=1 or len(urls)!=1:
        raise ValueError(f"Incomplete original asset identity for {name}")
    digest = records[0]["Checksum"]
    hex_digits = hashlib.new(algorithm.replace("-","").lower()).digest_size*2
    size = int(records[0]["SizeInBytes"])
    if digest["Algorithm"]!=algorithm or not re.fullmatch(rf"[0-9a-fA-F]{{{hex_digits}}}",digest["Value"]) or size<=0:
        raise ValueError(f"Unexpected checksum for {name}")
    return dict(url=next(iter(urls)),size_bytes=size,checksum=digest["Value"].lower())


def scene_rows(label, result, roofs):
    items, path, digest = result
    for r in items:
        m,u = r["meta"],r["umm"]
        name = u["GranuleUR"]
        production = datetime.fromisoformat(u["DataGranule"]["ProductionDateTime"])
        revision = datetime.fromisoformat(m["revision-date"])
        dates = {r["Type"]:datetime.fromisoformat(r["Date"]) for r in u["ProviderDates"]}
        temporal = u["TemporalExtent"]
        start = datetime.fromisoformat(temporal.get("SingleDateTime") or temporal["RangeDateTime"]["BeginningDateTime"])
        end = start if "SingleDateTime" in temporal else datetime.fromisoformat(temporal["RangeDateTime"]["EndingDateTime"])
        area = footprint(u)
        contained = {tank for tank,context in roofs.items() if area.covers(context)}
        base = dict(sensor="ASTER" if label.startswith("aster") else "NISAR", granule_id=name,
            cmr_id=m["concept-id"],cmr_collection=m["collection-concept-id"],catalog_snapshot=path,catalog_sha256=digest,
            capture_start_utc=start,capture_end_utc=end,product_generated_at_utc=production,cmr_revision_at_utc=revision,
            candidate_roofs_with_full_catalog_context=len(contained),roof_ids_json=json.dumps(sorted(contained)),
            footprint_utm_wkt=area.wkt)
        if base["sensor"]=="ASTER":
            attrs = {x["Name"]:x["Values"][0] for x in u.get("AdditionalAttributes",[])}
            image = file_asset(u,"_VNIR_B02.tif","SHA-512",required=False)
            if name!=m["native-id"] or m["collection-concept-id"]!=ASTER or not start<=production<=revision:
                raise ValueError(f"Unexpected ASTER capture or product identity for {name}")
            row = dict(base, product_kind="AST_L1T.004",availability_clock_utc=max(production,revision),
                availability_basis="current_catalog_revision_and_product_generation",day_night=u["DataGranule"]["DayNightFlag"],
                scene_cloud_percent=u["CloudCover"],solar_elevation_deg=attrs.get("SolarElevationAngle"),
                solar_azimuth_deg=attrs.get("SolarAzimuthAngle"),vnir_pointing_deg=attrs.get("VNIRPointingAngle"),
                vnir_b02_present=bool(image),vnir_b02_pixel_m=float(attrs["XPixelSize"]) if image else None)
            if image and (row["vnir_b02_pixel_m"]!=15 or not row["vnir_pointing_deg"]):
                raise ValueError(f"VNIR resolution or pointing differs for {name}")
            row.update({f"b02_{k}":image.get(k) for k in ("url","size_bytes","checksum")})
            qa = file_asset(u,"_QA.txt","SHA-512",required=False)
            row.update({f"qa_{k}":qa.get(k) for k in ("url","size_bytes","checksum")})
        else:
            cycle,track,direction,frame,bands,pol,coverage,stamp = NISAR_NAME.fullmatch(name).groups()
            inserted = dates["Insert"]
            if (m["collection-concept-id"]!=NISAR[label] or m["native-id"]!=name or not end<=production<=inserted<=revision
                    or stamp!=start.strftime("%Y%m%dT%H%M%S")):
                raise ValueError(f"Unexpected NISAR clocks or frame identity for {name}")
            first = int(bands[:2])
            row = dict(base,product_kind=f"GSLC_{label}",availability_clock_utc=inserted,
                availability_basis="provider_insert",repeat_cycle=int(cycle),track=int(track),direction=direction,
                frame=int(frame),frequency_a_bandwidth_mhz=first,frequency_b_bandwidth_mhz=int(bands[2:]),
                frequency_polarization_code=pol,frequency_a_north_pixel_m=5,
                frequency_a_east_pixel_m={5:40,20:10,40:5,77:2.5}[first],coverage_code=coverage)
            row.update({f"hdf5_{k}":v for k,v in file_asset(u,".h5","MD5").items()})
            row.update({f"metadata_{k}":v for k,v in file_asset(u,".h5.iso.xml","MD5").items()})
            row.update({f"native_hh_browse_{k}":v for k,v in file_asset(u,"_NATIVE_A_HH.png","MD5",required=False).items()})
        row["browse_url"] = next((x["URL"] for x in u["RelatedUrls"] if x["URL"].startswith("https://")
                                  and x["Type"]=="GET RELATED VISUALIZATION" and x["URL"].endswith((".jpg",".png"))),None)
        yield row,contained


def windows_and_roofs():
    g = legacy_geometry()
    roofs = {tank:Point(float(r["center_east_m"]),float(r["center_north_m"])).buffer(float(r["diameter_m"])/2+120)
             for tank,r in g.items()}
    base = {r["week_ending"]:r for r in read_csv("holdout_optical_weeks")}
    eligible = defaultdict(set)
    for r in read_csv("holdout_optical_tank_weeks"):
        eligible[r["week_ending"]].add(r["candidate_id"])
    with closing(read_only(SOURCE)) as db:
        rows = db.execute("""SELECT week_ending,endpoint,window_starts_after_utc,window_ends_by_utc
            FROM eia_satellite_window WHERE study_phase='holdout' ORDER BY week_ending,endpoint""")
        windows = defaultdict(dict)
        for week,side,start,end in rows:
            windows[week][side] = datetime.fromisoformat(start),datetime.fromisoformat(end)
    if windows.keys()!=base.keys() or eligible.keys()!=base.keys():
        raise ValueError("Different closed-period measurement calendars")
    return base,windows,eligible,roofs


def radar_stack(r):
    return tuple(r[k] for k in ("track","direction","frame","frequency_a_bandwidth_mhz",
                                "frequency_b_bandwidth_mhz","frequency_polarization_code","coverage_code"))


def match_weeks(base,windows,eligible,optical,radar):
    for week, edges in windows.items():
        cutoff = datetime.fromisoformat(base[week]["cutoff_utc"])
        groups = {}
        for side,(first,last) in edges.items():
            groups[side] = {}
            for sensor,scenes in (("aster",optical),("nisar",radar)):
                groups[side][sensor] = [(r,tanks&eligible[week]) for r,tanks in scenes if
                    first < r["capture_start_utc"] <= r["capture_end_utc"] <= last and tanks&eligible[week]]
        row = dict(week_ending=week,release_date=base[week]["release_date"],cutoff_utc=cutoff,b04_status=base[week]["b04_status"])
        for side,scenes in groups.items():
            for sensor,matches in scenes.items():
                ids = [r["cmr_id"] for r,_ in matches]
                early = [r["cmr_id"] for r,_ in matches if r["availability_clock_utc"]<cutoff]
                row.update({f"{side}_{sensor}_scene_ids_json":json.dumps(ids),
                            f"{side}_{sensor}_scene_ids_before_cutoff_json":json.dumps(early)})
        pairs = []
        for (a,ra),(b,rb) in product(groups["previous"]["nisar"],groups["current"]["nisar"]):
            common = ra&rb
            if radar_stack(a)!=radar_stack(b) or not common:
                continue
            pairs.append(dict(week_ending=week,release_date=row["release_date"],cutoff_utc=cutoff,b04_status=row["b04_status"],
                previous_cmr_id=a["cmr_id"],current_cmr_id=b["cmr_id"],previous_capture_utc=a["capture_start_utc"],
                current_capture_utc=b["capture_start_utc"],capture_gap_hours=(b["capture_start_utc"]-a["capture_start_utc"]).total_seconds()/3600,
                both_before_cutoff=max(a["availability_clock_utc"],b["availability_clock_utc"])<cutoff,
                matching_roofs=len(common),matching_roof_ids_json=json.dumps(sorted(common)),
                track=a["track"],direction=a["direction"],frame=a["frame"],frequency_a_pixel_east_m=a["frequency_a_east_pixel_m"],
                frequency_a_pixel_north_m=a["frequency_a_north_pixel_m"],polarization_code=a["frequency_polarization_code"]))
        row["nisar_same_stack_pairs"] = len(pairs)
        row["nisar_same_stack_pairs_before_cutoff"] = sum(p["both_before_cutoff"] for p in pairs)
        yield row,pairs


def nearby_optical_controls(aster):
    candidate_dates = defaultdict(list)
    for r,tanks in aster:
        if r["day_night"] not in ("Day","Both"):
            continue
        day = r["capture_start_utc"].date()
        candidate_dates[day].append((r,tanks))
    for phase in ("development","validation","holdout"):
        with (DERIVED/f"shadow_timing_{phase}_observations.csv").open(newline="") as stream:
            for s in csv.DictReader(stream):
                if s["geometry_mode"]!="dated_2011_lidar" or s["fit_status"]!=RESOLVED:
                    continue
                capture = datetime.fromisoformat(s["captured_at_utc"])
                for day in (capture.date()-timedelta(days=1),capture.date(),capture.date()+timedelta(days=1)):
                    for a,tanks in candidate_dates[day]:
                        lag = (capture-a["capture_start_utc"]).total_seconds()/60
                        if abs(lag)<=120 and s["candidate_id"] in tanks:
                            yield dict(study_phase=phase,candidate_id=s["candidate_id"],terminal_id=s["terminal_id"],
                                aster_cmr_id=a["cmr_id"],aster_capture_utc=a["capture_start_utc"],aster_scene_cloud_percent=a["scene_cloud_percent"],
                                sentinel2_version=s["version_id"],sentinel2_capture_utc=capture,sentinel2_archive_at_utc=s["archive_available_at_utc"],
                                sentinel2_geometry_at_utc=s["geometry_available_at_utc"],
                                sentinel2_gap_m=s["estimated_gap_m"],sentinel2_minus_aster_minutes=lag)


def radar_optical_controls(pairs,scenes):
    optical = defaultdict(list)
    for row in read_csv("shadow_timing_holdout_observations"):
        if row["geometry_mode"]=="dated_2011_lidar" and row["fit_status"]==RESOLVED:
            optical[row["candidate_id"]].append((datetime.fromisoformat(row["captured_at_utc"]),row))
    for pair in pairs:
        for side in ("previous","current"):
            radar = scenes[pair[f"{side}_cmr_id"]]
            start,end = radar["capture_start_utc"],radar["capture_end_utc"]
            for tank in json.loads(pair["matching_roof_ids_json"]):
                choices = []
                for capture,s in optical[tank]:
                    hours = max(abs(capture-start),abs(capture-end)).total_seconds()/3600
                    if hours<=24:
                        choices.append((hours,s["version_id"],s["observation_id"],s))
                if not choices:
                    continue
                hours,_,_,s = min(choices,key=lambda choice:choice[:3])
                capture,archive,geometry = (datetime.fromisoformat(s[k]) for k in
                    ("captured_at_utc","archive_available_at_utc","geometry_available_at_utc"))
                if not capture<=archive<=geometry or not (ROOT/s["tile_metadata_path"]).is_file():
                    raise ValueError(f"Inconsistent optical control provenance for {tank}")
                yield dict(week_ending=pair["week_ending"],endpoint=side,nisar_cmr_id=radar["cmr_id"],
                    radar_capture_start_utc=start,radar_capture_end_utc=end,max_separation_hours=hours,
                    both_inputs_before_cutoff=max(geometry,radar["availability_clock_utc"])<pair["cutoff_utc"],
                    **{k:s[k] for k in ("candidate_id","terminal_id","version_id","captured_at_utc",
                        "archive_available_at_utc","geometry_available_at_utc","estimated_gap_m",
                        "near_optimal_gap_min_m","near_optimal_gap_max_m","tile_metadata_path",
                        "measurement_source","observation_id")})


def stereo_controls(controls,aster,refresh):
    needed = {r["aster_cmr_id"] for r in controls}
    scenes = {r["cmr_id"]:r for r,_ in aster if r["cmr_id"] in needed}
    rows = []
    for day in sorted({r["capture_start_utc"].date().isoformat() for r in scenes.values()}):
        targets = [r for r in scenes.values() if r["capture_start_utc"].date().isoformat()==day]
        for product_type, extension in (("AST14DEM",".tif"),("AST_L1B",".hdf")):
            result,path,digest = catalog(f"{product_type}_{day}",{"short_name":product_type,"version":"004"},
                                         day+"T00:00:00Z",day+"T23:59:59Z",refresh)
            for a in targets:
                capture_key = a["granule_id"].rsplit("_",2)[1]
                for item in result:
                    m,u = item["meta"],item["umm"]
                    if u["GranuleUR"].rsplit("_",2)[1]!=capture_key:
                        continue
                    capture = datetime.fromisoformat(u["TemporalExtent"]["SingleDateTime"])
                    separation = (capture-a["capture_start_utc"]).total_seconds()
                    if abs(separation)>1:
                        raise ValueError(f"ASTER stereo source time differs for {u['GranuleUR']}")
                    rows.append(dict(aster_vnir_cmr_id=a["cmr_id"],product_type=product_type,granule_id=u["GranuleUR"],
                         cmr_id=m["concept-id"],capture_utc=capture,catalog_capture_difference_s=separation,
                         generated_at_utc=u["DataGranule"]["ProductionDateTime"],
                         catalog_snapshot=path,catalog_sha256=digest,**file_asset(u,extension,"SHA-512")))
    return rows


def radar_download_manifest(pairs,scenes,roofs):
    for pair in pairs:
        context = union_all([roofs[t] for t in json.loads(pair["matching_roof_ids_json"])])
        west,south,east,north = context.bounds
        for side in ("previous","current"):
            scene = scenes[pair[f"{side}_cmr_id"]]
            yield dict(week_ending=pair["week_ending"],endpoint=side,
                priority="optical_control" if pair["b04_status"]=="measured" else "b04_gap",
                both_before_cutoff=pair["both_before_cutoff"],cutoff_utc=pair["cutoff_utc"],
                matching_roofs=pair["matching_roofs"],subset_crs="EPSG:32614",
                subset_west_m=west,subset_south_m=south,subset_east_m=east,subset_north_m=north,
                complex_raster="/science/LSAR/GSLC/grids/frequencyA/HH",
                geometry_group="/science/LSAR/GSLC/metadata/radarGrid",
                geometry_variables="incidenceAngle|losUnitVectorX|losUnitVectorY",
                **{k:scene[k] for k in ("cmr_id","granule_id","capture_start_utc","capture_end_utc",
                    "availability_clock_utc","product_kind","hdf5_url","hdf5_size_bytes","hdf5_checksum",
                    "metadata_url","metadata_size_bytes","metadata_checksum","native_hh_browse_url",
                    "native_hh_browse_size_bytes","native_hh_browse_checksum","catalog_snapshot","catalog_sha256")})


def save_radar_browse(scene):
    url = scene["native_hh_browse_url"]
    path = STORE/"browse"/urlsplit(url).path.rsplit("/",1)[1]
    raw = path.read_bytes() if path.is_file() else get(url)
    if (len(raw)!=scene["native_hh_browse_size_bytes"] or hashlib.md5(raw).hexdigest()!=scene["native_hh_browse_checksum"]
        or raw[:16]!=b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR"):
        raise ValueError(f"ASF browse image identity differs for {scene['cmr_id']}")
    if not path.is_file():
        write(path,raw)
    width,height = unpack(">II",raw[16:24])
    return dict(cmr_id=scene["cmr_id"],week_ending=scene["week_ending"],endpoint=scene["endpoint"],
        browse_url=url,browse_path=str(path.relative_to(ROOT)),browse_bytes=len(raw),browse_md5=hashlib.md5(raw).hexdigest(),
        browse_width=width,browse_height=height)


def probe(sensor,kind,url):
    try:
        with urlopen(Request(url,headers={"Range":"bytes=0-31","User-Agent":"Mozilla/5.0"}),context=TLS,timeout=25) as r:
            status,host,content,head = r.status,urlsplit(r.url).hostname,r.headers.get_content_type(),r.read(32)
    except HTTPError as error:
        status,host,content,head = error.code,urlsplit(error.url).hostname,error.headers.get_content_type(),b""
    except OSError as error:
        return dict(sensor=sensor,asset=kind,url=url,probe_error=type(error).__name__)
    signature = ("hdf5" if head.startswith(b"\x89HDF\r\n\x1a\n") else "tiff" if head[:4] in (b"II*\0",b"MM\0*")
                 else "jpg" if head.startswith(b"\xff\xd8\xff") else "png" if head.startswith(b"\x89PNG\r\n\x1a\n") else None)
    return dict(sensor=sensor,asset=kind,url=url,http_status=status,response_host=host,content_type=content,
                file_signature=signature,earthdata_signin=host=="urs.earthdata.nasa.gov")


def export(suffix,rows,fields=None):
    columns = fields or tuple(dict.fromkeys(k for row in rows for k in row))
    write_csv(DERIVED/f"{STEM}_{suffix}.csv",columns,([r.get(k) for k in columns] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    jobs = [("aster_development",{"collection_concept_id":ASTER},"2017-01-01T00:00:00Z","2023-12-31T23:59:59Z"),
            ("aster_holdout",{"collection_concept_id":ASTER},"2024-01-01T00:00:00Z",END),
            ("aster_v3",{"short_name":"AST_L1T","version":"003"},"2017-01-01T00:00:00Z",END)]
    jobs.extend((label,{"collection_concept_id":cid},"2025-10-01T00:00:00Z",END) for label,cid in NISAR.items())
    jobs.extend((label.lower(),{"short_name":label},"2017-01-01T00:00:00Z",END) for label in ("UAVSAR_POLSAR","UAVSAR_RPI"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(zip((j[0] for j in jobs),pool.map(lambda j:catalog(*j,args.refresh),jobs),strict=True))
    base,windows,eligible,roofs = windows_and_roofs()
    records = [pair for label,result in results.items() if label in ("aster_development","aster_holdout",*NISAR)
               for pair in scene_rows(label,result,roofs)]
    if len({r["cmr_id"] for r,_ in records})!=len(records):
        raise ValueError("NASA inventory includes duplicate granule concepts")
    aster = [(r,t) for r,t in records if r["sensor"]=="ASTER" and r["vnir_b02_present"]]
    radar = [(r,t) for r,t in records if r["sensor"]=="NISAR"]
    weeks,radar_pairs = [],[]
    for w,p in match_weeks(base,windows,eligible,aster,radar):
        weeks.append(w)
        radar_pairs.extend(p)
    controls = list(nearby_optical_controls(aster))
    stereo = stereo_controls(controls,aster,args.refresh)
    scenes = {r["cmr_id"]:r for r,_ in records}
    radar_controls = list(radar_optical_controls(radar_pairs,scenes))
    control_roofs = defaultdict(set)
    for control in radar_controls:
        control_roofs[control["week_ending"],control["endpoint"]].add(control["candidate_id"])
    control_roof_pairs = {week:len(control_roofs[week,"previous"]&control_roofs[week,"current"])
                         for week in sorted({p["week_ending"] for p in radar_pairs})}
    radar_manifest = list(radar_download_manifest(radar_pairs,scenes,roofs))
    with ThreadPoolExecutor(max_workers=3) as pool:
        browse = list(pool.map(save_radar_browse,radar_manifest))
    a = scenes[controls[0]["aster_cmr_id"]]
    main_pair = next(p for p in radar_pairs if p["both_before_cutoff"] and p["b04_status"]!="measured")
    n = scenes[main_pair["current_cmr_id"]]
    requests = [(sensor,k,r[k]) for sensor,r,keys in (("ASTER",a,("b02_url","qa_url","browse_url")),
                ("NISAR",n,("hdf5_url","metadata_url","browse_url"))) for k in keys if r[k]]
    requests.extend(("ASTER",r["product_type"],r["url"]) for r in stereo)
    with ThreadPoolExecutor(max_workers=4) as pool:
        access = list(pool.map(lambda request:probe(*request),requests))
    gaps = [w for w in weeks if w["b04_status"]!="measured"]
    summary = dict(aster_cmr_records=sum(r["sensor"]=="ASTER" for r,_ in records),
        aster_vnir_b02_records=len(aster),aster_vnir_b02_distinct_days=len({r["capture_start_utc"].date() for r,t in aster if t}),
        aster_vnir_b02_holdout_records=sum(r["capture_start_utc"].year>=2024 for r,t in aster),
        aster_v3_cmr_records=len(results["aster_v3"][0]),
        aster_sentinel2_controls=len(controls),aster_sentinel2_control_roof_days=len({(r["candidate_id"],r["aster_capture_utc"].date()) for r in controls}),
        aster_stereo_products=len(stereo),
        uavsar_records={k:len(results[k][0]) for k in ("uavsar_polsar","uavsar_rpi")},
        nisar_cmr_records=len(radar),nisar_records_by_collection=dict(Counter(r["product_kind"] for r,t in radar)),
        nisar_records_with_roof_context=sum(bool(t) for _,t in radar),
        nisar_same_stack_pairs=len(radar_pairs),nisar_same_stack_pairs_before_cutoff=sum(r["both_before_cutoff"] for r in radar_pairs),
        nisar_optical_controls_within_24h=len(radar_controls),
        nisar_optical_controls_before_cutoff=sum(r["both_inputs_before_cutoff"] for r in radar_controls),
        nisar_optical_control_endpoints=dict(Counter(r["nisar_cmr_id"] for r in radar_controls)),
        nisar_same_roof_optical_controls_at_both_endpoints_by_week=control_roof_pairs,
        nisar_native_hh_browse_files=len(browse),nisar_native_hh_browse_bytes=sum(r["browse_bytes"] for r in browse),
        nisar_primary_pair_hdf5_bytes=sum(r["hdf5_size_bytes"] for r in radar_manifest if r["priority"]=="b04_gap"),
        gap_weeks_with_nisar_current_catalog_before_cutoff=[w["week_ending"] for w in gaps if json.loads(w["current_nisar_scene_ids_before_cutoff_json"])],
        gap_weeks_with_nisar_pair_before_cutoff=[w["week_ending"] for w in gaps if w["nisar_same_stack_pairs_before_cutoff"]],
        gap_weeks_with_aster_current_vnir_before_cutoff=[w["week_ending"] for w in gaps if json.loads(w["current_aster_scene_ids_before_cutoff_json"])],
        access=access,scope="Catalog footprints and availability of exact versions. These records do not establish local clear pixels, "
                           "radar visibility of roofs, or a calibrated change estimate. ASTER cloud percentage is for the entire scene.")
    for suffix,rows in (("scenes",[r for r,_ in records]),("weeks",weeks),("nisar_pairs",radar_pairs),
                        ("nisar_download_manifest",radar_manifest),("nisar_browse",browse),("nisar_optical_controls",radar_controls),
                        ("aster_controls",controls),("aster_stereo_controls",stereo),("access",access)):
        export(suffix,rows)
    payload = json.dumps(summary,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
