"""Collect public ICESat-2 ATL03 extracts and index independent Cushing roof controls."""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import unquote, urlencode, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window, from_bounds
from scipy.spatial import cKDTree
from shapely import transform, union_all
from shapely.geometry import Point

from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED
from build_weekly_shadow_validation import legacy_geometry
from build_cushing_nasa_satellite_inventory import windows_and_roofs
from collect_public import RAW, ROOT, TLS, TNM, get, write, write_csv
from collect_umbra_sicd_windows import with_ellipsoid_heights

STEM = "cushing_icesat2"
STORE = RAW / STEM
OA = "https://openaltimetry.earthdatacloud.nasa.gov/data/api"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
COLLECTION = "C3326974349-NSIDC_CPRD"
FIRST, LAST = "2018-10-13", "2026-10-31"
GRANULE = re.compile(r"ATL03_(\d{14})_(\d{4})\d{4}_007_\d{2}\.h5")
BEAMS = {f"gt{i}{side}" for i in (1,2,3) for side in ("l","r")}
TO_GEO = Transformer.from_crs(32614,4326,always_xy=True)
TO_UTM = Transformer.from_crs(4326,32614,always_xy=True)
TO_DEM = Transformer.from_crs(32614,26914,always_xy=True)
INSET_M = 12
CONTEXT_M = 120


def query(url, fields):
    return url+"?"+urlencode(fields)


def archive(url,path,refresh=False):
    return json.loads(get(url,path,refresh=refresh))


def identities():
    dated = {r["candidate_id"]:r for r in read_csv("tank_engineering_registry")}
    legacy = legacy_geometry()
    modern = {r["candidate_id"]:r for r in read_csv("cushing_modern_lidar_optical_registry")
              if r["modern_input_status"]=="available"}
    tanks = {}
    for key in sorted(legacy.keys() | modern.keys()):
        older,newer = legacy.get(key),modern.get(key)
        row = older or newer
        x,y,dia = (float(row[k]) for k in ("center_east_m","center_north_m","diameter_m"))
        wall,ground = (float(row[k]) for k in ("prior_rim_z_p95_native","prior_ground_z_p50_native"))
        if not 0<wall-ground<35 or dated[key]["documented_roof_type"]!="EFR" or not dated[key]["first_seen"]:
            raise ValueError(f"Unverified external roof identity {key}")
        lon,lat = TO_GEO.transform(x,y)
        tanks[key] = dict(candidate_id=key,terminal_id=dated[key]["documented_terminal_id"],
            first_seen=dated[key]["first_seen"],east_m=x,north_m=y,latitude=lat,longitude=lon,radius_m=dia/2,
            geometry_dataset=row["dataset"],lidar_ground_navd88_m=ground,lidar_top_navd88_m=wall,
            geometry_source_directory=row["source_directory"],
            additional_geometry_available_date=newer["latest_engineering_date"] if not older else None)
    circles = [(t["east_m"],t["north_m"],t["radius_m"]) for t in tanks.values()]
    circles.extend((float(t["center_east_m"]),float(t["center_north_m"]),float(t["diameter_m"])/2)
                   for t in modern.values())
    coverage = union_all([Point(x,y).buffer(r+CONTEXT_M) for x,y,r in circles])
    box = transform(coverage,TO_GEO.transform,interleaved=False).bounds
    box = tuple((math.floor if i<2 else math.ceil)(x*1e5)/1e5 for i,x in enumerate(box))
    return with_ellipsoid_heights(tanks),coverage.bounds,box


def source_catalog(box,refresh):
    # Outward rounding keeps all original circular contexts within the service query.
    path = STORE/"atl03_v7.umm.json"
    url = query(CMR,dict(collection_concept_id=COLLECTION,bounding_box=",".join(map(str,box)),
        temporal=f"{FIRST}T00:00:00Z,{LAST}T23:59:59Z",page_size=200))
    data = archive(url,path,refresh)
    if data["hits"]!=len(data["items"]):
        raise ValueError("Incomplete ATL03 CMR snapshot")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {}
    for item in data["items"]:
        m,u = item["meta"],item["umm"]
        files = [r["URL"] for r in u["RelatedUrls"] if r["Type"]=="GET DATA" and r["URL"].startswith("https://")
                 and GRANULE.fullmatch(r["URL"].rsplit("/",1)[-1])]
        filename, = files
        name = filename.rsplit("/",1)[-1]
        stamp,track = GRANULE.fullmatch(name).groups()
        day = datetime.strptime(stamp,"%Y%m%d%H%M%S").date().isoformat()
        key = day,int(track)
        times = u["TemporalExtent"]["RangeDateTime"]
        if key in result or m["collection-concept-id"]!=COLLECTION or not u["GranuleUR"].startswith(name):
            raise ValueError(f"Unexpected ATL03 identity {name}")
        result[key] = dict(cmr_id=m["concept-id"],filename=name,hdf5_url=filename,
            granule_start_utc=times["BeginningDateTime"],granule_end_utc=times["EndingDateTime"],
            granule_produced_utc=u["DataGranule"]["ProductionDateTime"],
            granule_catalog_inserted_utc=next(r["Date"] for r in u["ProviderDates"] if r["Type"]=="Insert"),
            cmr_snapshot_path=str(path.relative_to(ROOT)),cmr_snapshot_sha256=digest)
    return result


def oa_track_dates(track,refresh):
    url = f"{OA}/v2/icesat2/collections/atl03/tracks/{track}/dates"
    data = archive(url,STORE/f"openaltimetry_dates_{track:04d}.json",refresh)
    return {(d,track) for d in data["dates"] if FIRST<=d<=LAST}


def rows_from_csv(raw):
    lines = raw.decode().splitlines()
    return list(csv.DictReader(line for line in lines if not line.startswith("#")))


def collect_photons(key,box,*,store=STORE):
    day,track = key
    stem = f"{day}_{track:04d}"
    fields = dict(zip(("minx","miny","maxx","maxy"),box,strict=True))
    fields.update(date=day,trackId=track,photonConfidence="high,medium,low",sampling="false",outputFormat="csv")
    csv_url = query(f"{OA}/icesat2/atl03",fields)
    path = store/"photons"/f"{stem}.csv"
    raw = get(csv_url,path)
    header = b"beam,latitude,longitude,photon height,confidence code"
    if header not in raw[:500]:
        raise ValueError(f"OpenAltimetry returned an invalid extract for {stem}")
    source = dict(requested_date=day,track_id=track,openaltimetry_url=csv_url,
        csv_path=str(path.relative_to(ROOT)),csv_sha256=hashlib.sha256(raw).hexdigest(),
        csv_bytes=len(raw),packet_path="",packet_sha256="",packet_granules_json="[]")
    photons = rows_from_csv(raw)
    if not photons and not raw.startswith(b"# subsample_stride="):
        source.update(photon_count=0,maximum_subsample_stride=1)
        return source,[]
    fields["outputFormat"] = "zip"
    packet_path = store/"packets"/f"{stem}.zip"
    packet = get(query(f"{OA}/icesat2/atl03",fields),packet_path)
    with zipfile.ZipFile(io.BytesIO(packet)) as z:
        names = z.namelist()
        csv_name, = [n for n in names if n.endswith(".csv")]
        xml_name, = [n for n in names if n.endswith(".xml")]
        packaged = rows_from_csv(z.read(csv_name))
        xml = ElementTree.fromstring(z.read(xml_name))
    expected_counts = Counter(row["beam"] for row in packaged)
    if (Counter((row["beam"],row["latitude"],row["longitude"],row["photon height"],row["confidence code"]) for row in photons)
            != Counter((row["beam"],row["latitude"],row["longitude"],row["photon height"],row["confidence code"]) for row in packaged)):
        raise ValueError(f"OpenAltimetry packet differs from CSV for {stem}")
    extent = tuple(float(xml.findtext(f"input/extent/{k}")) for k in ("minx","miny","maxx","maxy"))
    if extent!=box or xml.findtext("input/date")!=day or int(xml.findtext("input/trackid"))!=track:
        raise ValueError(f"OpenAltimetry packet belongs to a different pass for {stem}")
    packet_granules,strides,counts = set(),[],Counter()
    for file in xml.findall("output/files/file"):
        link = unquote(file.findtext("filename"))
        match = re.search(GRANULE.pattern,link)
        if match is None or int(match[2])!=track or f"/collections/{COLLECTION}/" not in link:
            raise ValueError(f"OpenAltimetry did not identify ATL03 v7 for {stem}")
        packet_granules.add(match[0])
        for beam in file.findall("beam"):
            counts[beam.attrib["name"]] += int(beam.findtext("photon_count"))
            strides.append(int(beam.findtext("subsample_stride")))
    if +counts!=expected_counts or not set(counts)<=BEAMS or not strides or min(strides)<1:
        raise ValueError(f"OpenAltimetry photon counts are inconsistent for {stem}")
    source.update(photon_count=len(photons),maximum_subsample_stride=max(strides),
        packet_path=str(packet_path.relative_to(ROOT)),packet_sha256=hashlib.sha256(packet).hexdigest(),
        packet_granules_json=json.dumps(sorted(packet_granules)))
    return source,photons


def dem_assets(box,utm_bounds):
    query_url = query(TNM,dict(datasets="Digital Elevation Model (DEM) 1 meter",
        bbox=",".join(map(str,box)),prodFormats="GeoTIFF",outputFormat="JSON",max=100))
    data = archive(query_url,STORE/"3dep_tiles.json")
    if not data["total"] or data["total"]!=len(data["items"]):
        raise ValueError("Incomplete USGS ground elevation catalog")
    bounds = TO_DEM.transform_bounds(*utm_bounds,densify_pts=21)
    for item in data["items"]:
        url = item["downloadURL"]
        path = STORE/"ground"/url.rsplit("/",1)[-1]
        if not path.is_file():
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",CPL_VSIL_CURL_USE_HEAD="NO"):
                with rasterio.open(url) as src:
                    if src.crs.to_epsg()!=26914 or src.res!=(1,1):
                        raise ValueError(f"Unexpected source DEM grid for {url}")
                    raw = from_bounds(*bounds,src.transform)
                    window = Window(math.floor(raw.col_off),math.floor(raw.row_off),
                        math.ceil(raw.col_off+raw.width)-math.floor(raw.col_off),
                        math.ceil(raw.row_off+raw.height)-math.floor(raw.row_off))
                    window = window.intersection(Window(0,0,src.width,src.height))
                    profile = src.profile.copy()
                    profile.update(width=int(window.width),height=int(window.height),
                        transform=src.window_transform(window),compress="deflate")
                    path.parent.mkdir(parents=True,exist_ok=True)
                    tmp = path.with_suffix(".part")
                    with rasterio.open(tmp,"w",**profile) as dst:
                        dst.write(src.read(window=window))
                        dst.update_tags(source_url=url,vertical_datum="NAVD88",source_id=item["sourceId"])
                    tmp.replace(path)
        with rasterio.open(path) as src:
            if src.tags().get("source_id")!=item["sourceId"]:
                raise ValueError(f"Unexpected cached DEM identity {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream,"sha256").hexdigest()
        yield dict(dem_file=str(path.relative_to(ROOT)),dem_source_url=url,dem_source_id=item["sourceId"],
            dem_publication_date=item["publicationDate"],dem_file_sha256=digest)


def ground_heights(xy,dems):
    result = np.full(len(xy),np.nan)
    x,y = TO_DEM.transform(xy[:,0],xy[:,1])
    for dem in dems:
        with rasterio.open(ROOT/dem["dem_file"]) as src:
            use = ((x>=src.bounds.left)&(x<src.bounds.right)&(y>src.bounds.bottom)&(y<=src.bounds.top))
            if use.any():
                z = np.array([p[0] for p in src.sample(zip(x[use],y[use],strict=True))],dtype=float)
                z[z==src.nodata] = np.nan
                missing = ~np.isfinite(result[use])
                result[np.flatnonzero(use)[missing]] = z[missing]
    return result


def other_structure_context():
    records = read_csv("tank_engineering_registry")
    centers = np.array([(float(r["footprint_east_m"]),float(r["footprint_north_m"])) for r in records])
    radii = np.array([float(r["footprint_diameter_m"])/2+INSET_M for r in records])
    return centers,radii


def measurements(source,photons,tanks,dems,structures):
    if not photons:
        return
    coords = np.array([(float(r["longitude"]),float(r["latitude"]),float(r["photon height"])) for r in photons])
    x,y = TO_UTM.transform(coords[:,0],coords[:,1])
    xy = np.column_stack((x,y))
    if not np.isfinite(coords).all():
        raise ValueError(f"Invalid photon position in {source['csv_path']}")
    height = coords[:,2]
    beam = np.array([r["beam"] for r in photons])
    conf = np.array([int(r["confidence code"]) for r in photons])
    if not set(conf)<={2,3,4} or not set(beam)<=BEAMS:
        raise ValueError(f"Invalid OpenAltimetry classification in {source['csv_path']}")
    photon_tree = cKDTree(xy)
    centers,radii = structures
    nearby_tanks = cKDTree(centers)
    contexts = {t["candidate_id"]:photon_tree.query_ball_point((t["east_m"],t["north_m"]),t["radius_m"]+CONTEXT_M)
                for t in tanks.values()}
    nearby = np.array(sorted({i for indices in contexts.values() for i in indices if conf[i]>=3}),dtype=int)
    ground_z = np.full(len(xy),np.nan)
    ground_z[nearby] = ground_heights(xy[nearby],dems)
    for t in tanks.values():
        x0,y0,r = (t[k] for k in ("east_m","north_m","radius_m"))
        center = np.array([x0,y0])
        idx = np.array(photon_tree.query_ball_point(center,max(r-INSET_M,0)),dtype=int)
        if not len(idx):
            continue
        exterior = np.array(contexts[t["candidate_id"]],dtype=int)
        for obj in nearby_tanks.query_ball_point(center,r+CONTEXT_M+max(radii)):
            exterior = exterior[np.linalg.norm(xy[exterior]-centers[obj],axis=1)>radii[obj]]
        for b in sorted(set(beam[idx])):
            local = idx[beam[idx]==b]
            high = local[conf[local]>=3]
            ref = exterior[(beam[exterior]==b)&(conf[exterior]>=3)&np.isfinite(ground_z[exterior])]
            datum = t["geoid18_interpolated_m"]
            offsets = height[ref]-ground_z[ref]-datum
            near = abs(offsets)<2
            ref = ref[near]
            offsets = offsets[near]
            _,cell_ids = np.unique(np.floor(xy[ref]/2),axis=0,return_inverse=True)
            cell_medians = [np.median(offsets[cell_ids==i]) for i in np.unique(cell_ids)]
            independent_cells = len(cell_medians)
            ground_bias = float(np.median(cell_medians)) if independent_cells>=10 else None
            p10,p50,p90 = np.percentile(height[high],[10,50,90]) if len(high) else (None,None,None)
            extent = float(np.linalg.norm(np.ptp(xy[high],axis=0))) if len(high) else 0
            deep = high[np.linalg.norm(xy[high]-center,axis=1)<r-15]
            stricter = float(np.median(height[deep])) if len(deep)>=10 else None
            depth = t["top_hae_approx_m"]+ground_bias-p50 if ground_bias is not None and p50 is not None else None
            first_seen = t["first_seen"]<=source["utc_granule_day"]
            stable = len(high)>=10 and extent>=8 and p90-p10<=1
            usable = bool(first_seen and stable and depth is not None and 0<=depth<=t["lidar_top_navd88_m"]-t["lidar_ground_navd88_m"])
            yield dict(requested_date=source["requested_date"],utc_granule_day=source["utc_granule_day"],
                filename=source["filename"],cmr_id=source["cmr_id"],
                granule_start_utc=source["granule_start_utc"],granule_end_utc=source["granule_end_utc"],
                granule_catalog_inserted_utc=source["granule_catalog_inserted_utc"],track_id=source["track_id"],beam=b,
                candidate_id=t["candidate_id"],terminal_id=t["terminal_id"],geometry_dataset=t["geometry_dataset"],
                first_seen=t["first_seen"],existed_by_capture=first_seen,roof_inset_m=INSET_M,
                geometry_center_east_m=x0,geometry_center_north_m=y0,geometry_radius_m=r,
                geometry_source_directory=t["geometry_source_directory"],
                additional_geometry_available_date=t["additional_geometry_available_date"],
                interior_photons=len(local),interior_medium_or_high_photons=len(high),interior_photon_extent_m=extent,
                roof_hae_p10_m=p10,roof_hae_p50_m=p50,roof_hae_p90_m=p90,
                roof_hae_p50_inset_15m=stricter,roof_hae_shift_inset_15m=stricter-p50 if stricter is not None else None,
                stable_interior_profile=stable,ground_control_photons=len(ref),ground_control_2m_cells=independent_cells,
                geoid18_offset_m=datum,local_ground_bias_m=ground_bias,wall_navd88_m=t["lidar_top_navd88_m"],
                estimated_wall_to_roof_m=depth,passes_roof_profile_checks=usable,
                maximum_subsample_stride=source["maximum_subsample_stride"],photon_csv=source["csv_path"],
                photon_packet=source["packet_path"],packet_granules_json=source["packet_granules_json"])


def export(name, rows):
    fields = tuple(dict.fromkeys(k for r in rows for k in r))
    write_csv(DERIVED/f"{STEM}_{name}.csv",fields,([r.get(k) for k in fields] for r in rows))


def unique_profiles(profiles):
    groups = defaultdict(list)
    for p in profiles:
        groups[p["filename"],p["candidate_id"],p["beam"]].append(p)
    for rows in groups.values():
        best = min(rows,key=lambda r:(not r["passes_roof_profile_checks"],r["maximum_subsample_stride"],
            -r["interior_medium_or_high_photons"],r["requested_date"]))
        yield dict(best,request_count=len(rows),all_requested_dates_json=json.dumps(sorted(r["requested_date"] for r in rows)))


def optical_matches(profiles):
    eligible = {p["candidate_id"] for p in profiles}
    optical = defaultdict(list)
    for phase in ("development","validation","holdout"):
        with (DERIVED/f"shadow_timing_{phase}_observations.csv").open(newline="") as stream:
            for r in csv.DictReader(stream):
                if (r["candidate_id"] in eligible and r["geometry_mode"]=="dated_2011_lidar"
                    and r["fit_status"]=="resolved_in_geometry_diagnostic"):
                    optical[r["candidate_id"]].append((datetime.fromisoformat(r["captured_at_utc"]),phase,r))
    for p in profiles:
        start,end = (datetime.fromisoformat(p[k]) for k in ("granule_start_utc","granule_end_utc"))
        choices = []
        for t,phase,s in optical[p["candidate_id"]]:
            hours = max(abs((t-edge).total_seconds()) for edge in (start,end))/3600
            if hours<=48:
                choices.append((hours,s["version_id"],s["observation_id"],phase,s))
        if not choices:
            continue
        hours,_,_,phase,s = min(choices,key=lambda r:r[:3])
        if p["terminal_id"]!=s["terminal_id"] or not (ROOT/s["tile_metadata_path"]).is_file():
            raise ValueError(f"Optical source identity differs for {p['candidate_id']}")
        difference = float(s["estimated_gap_m"])-p["estimated_wall_to_roof_m"]
        yield dict(candidate_id=p["candidate_id"],terminal_id=p["terminal_id"],beam=p["beam"],
            icesat_granule=p["filename"],icesat_cmr_id=p["cmr_id"],icesat_granule_start_utc=start,
            icesat_granule_end_utc=end,icesat_wall_to_roof_m=p["estimated_wall_to_roof_m"],
            icesat_source_csv=p["photon_csv"],max_time_separation_hours=hours,study_phase=phase,
            optical_minus_icesat_m=difference,
            **{f"sentinel2_{k}":s[k] for k in ("version_id","captured_at_utc","geometry_available_at_utc",
                "estimated_gap_m","near_optimal_gap_min_m","near_optimal_gap_max_m","tile_metadata_path",
                "measurement_source","observation_id")})


def airborne_matches(profiles):
    references = defaultdict(list)
    for r in read_csv("lidar_roof_elevation_reference"):
        if (r["rim_minus_center_plane_native_z"] and int(r["center_sectors"])>=6 and int(r["rim_sectors"])>=6
            and 0<=float(r["rim_minus_center_plane_native_z"])<=35):
            references[r["candidate_id"]].append(r)
    for p in profiles:
        start,end = (datetime.fromisoformat(p[k]) for k in ("granule_start_utc","granule_end_utc"))
        for s in references[p["candidate_id"]]:
            bounds = (datetime.fromisoformat(s[k]) for k in ("first_return_at_utc","last_return_at_utc"))
            hours = max(abs((t-edge).total_seconds()) for edge in (start,end) for t in bounds)/3600
            if hours<=14*24:
                yield dict(candidate_id=p["candidate_id"],terminal_id=p["terminal_id"],beam=p["beam"],
                    icesat_granule=p["filename"],icesat_cmr_id=p["cmr_id"],icesat_granule_start_utc=start,
                    icesat_granule_end_utc=end,icesat_wall_to_roof_m=p["estimated_wall_to_roof_m"],
                    max_time_separation_hours=hours,
                    airborne_minus_icesat_m=float(s["rim_minus_center_plane_native_z"])-p["estimated_wall_to_roof_m"],
                    **{f"airborne_{k}":s[k] for k in ("dataset","point_source_id","first_return_at_utc",
                        "last_return_at_utc","rim_minus_center_plane_native_z","source_directory","source_node_ids_json")})


def holdout_calendar(profiles):
    baseline,windows,eligible,_ = windows_and_roofs()
    sources = defaultdict(list)
    for p in profiles:
        if p["utc_granule_day"]>="2024-01-01":
            sources[p["filename"]].append(p)
    for week,edges in windows.items():
        o = baseline[week]
        cutoff = datetime.fromisoformat(o["cutoff_utc"])
        cutoff_day = cutoff.date().isoformat()
        row = dict(week_ending=week,release_date=o["release_date"],cutoff_utc=o["cutoff_utc"],b04_status=o["b04_status"])
        endpoint_roofs = {}
        for side,(first,last) in edges.items():
            matches = []
            for rows in sources.values():
                start,end = (datetime.fromisoformat(rows[0][k]) for k in ("granule_start_utc","granule_end_utc"))
                if first<start<=end<=last:
                    matches.extend(p for p in rows if p["candidate_id"] in eligible[week] or (
                        p["additional_geometry_available_date"] and p["additional_geometry_available_date"]<cutoff_day
                        and p["first_seen"]<=edges["previous"][0].date().isoformat()))
            roofs = {p["candidate_id"] for p in matches}
            endpoint_roofs[side] = roofs
            row.update({f"{side}_catalog_granules_json":json.dumps(sorted({p["filename"] for p in matches})),
                        f"{side}_retrospective_roofs":len(roofs),
                        f"{side}_roofs_with_cmr_insert_before_cutoff":len({p["candidate_id"] for p in matches
                            if datetime.fromisoformat(p["granule_catalog_inserted_utc"])<cutoff})})
        row["same_roofs_with_both_endpoints"] = len(endpoint_roofs["previous"]&endpoint_roofs["current"])
        yield row


def original_access(info):
    url = info["hdf5_url"]
    try:
        request = Request(url,headers={"Range":"bytes=0-15","User-Agent":"Mozilla/5.0"})
        with urlopen(request,context=TLS,timeout=15) as response:
            status,host,kind = response.status,urlsplit(response.url).hostname,response.headers.get_content_type()
            hdf5 = response.read(8)==b"\x89HDF\r\n\x1a\n"
    except HTTPError as error:
        status,host,kind,hdf5 = error.code,urlsplit(error.url).hostname,error.headers.get_content_type(),False
    return dict(cmr_id=info["cmr_id"],hdf5_url=url,http_status=status,response_host=host,
        content_type=kind,hdf5_signature=hdf5,earthdata_signin=host=="urs.earthdata.nasa.gov")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-catalog",action="store_true")
    parser.add_argument("--probe-access",action="store_true")
    args = parser.parse_args()
    tanks,utm_bounds,box = identities()
    cmr = source_catalog(box,args.refresh_catalog)
    with ThreadPoolExecutor(max_workers=5) as pool:
        per_track = pool.map(lambda t:oa_track_dates(t,args.refresh_catalog),sorted({t for _,t in cmr}))
        requests = sorted(set(cmr).union(*per_track))
    dems = list(dem_assets(box,utm_bounds))
    structures = other_structure_context()
    catalog,roof = [],[]
    by_filename = {r["filename"]:r for r in cmr.values()}
    with ThreadPoolExecutor(max_workers=5) as pool:
        for key,(asset,photons) in zip(requests,pool.map(lambda k:collect_photons(k,box),requests),strict=True):
            filenames = json.loads(asset["packet_granules_json"])
            info = cmr.get(key,{})
            if filenames:
                if len(filenames)!=1 or filenames[0] not in by_filename:
                    raise ValueError(f"ATL03 packet needs separate granule attribution for {key}: {filenames}")
                info = by_filename[filenames[0]]
            record = {**asset,**info}
            record["utc_granule_day"] = info.get("granule_start_utc","")[:10]
            catalog.append(record)
            roof.extend(measurements(record,photons,tanks,dems,structures))
    roof = list(unique_profiles(roof))
    good = [r for r in roof if r["passes_roof_profile_checks"]]
    optical,airborne = list(optical_matches(good)),list(airborne_matches(good))
    weeks = list(holdout_calendar(good))
    if args.probe_access:
        info = by_filename[max(p["filename"] for p in good)]
        export("original_access",[original_access(info)])
    for kind,rows in (("extracts",catalog),("ground_assets",dems),("roof_profiles",roof),
                      ("sentinel2_controls",optical),("airborne_controls",airborne),("holdout_weeks",weeks)):
        export(kind,rows)
    report = dict(cmr_granules=len(cmr),date_track_requests=len(catalog),
        photon_extracts_with_returns=sum(bool(r["photon_count"]) for r in catalog),
        unique_granules_with_returns=len({r["filename"] for r in catalog if r["photon_count"]}),
        photon_rows_across_responses=sum(r["photon_count"] for r in catalog),verified_packets=sum(bool(r["packet_path"]) for r in catalog),
        extract_rows_with_subsampling=sum(r["maximum_subsample_stride"]>1 for r in catalog),
        tanks_considered=len(tanks),dem_assets=len(dems),beam_roof_profiles=len(roof),passing_beam_roof_profiles=len(good),
        passing_unique_roofs=len({r["candidate_id"] for r in good}),passing_days=sorted({r["utc_granule_day"] for r in good}),
        sentinel2_matches_within_48h=len(optical),sentinel2_matches_within_24h=sum(r["max_time_separation_hours"]<=24 for r in optical),
        sentinel2_median_absolute_difference_within_48h_m=float(np.median([abs(r["optical_minus_icesat_m"]) for r in optical])),
        airborne_matches_within_14d=len(airborne),
        profiles_with_inset_15m_comparison=sum(r["roof_hae_shift_inset_15m"] is not None for r in good),
        maximum_absolute_inset_sensitivity_m=max(abs(r["roof_hae_shift_inset_15m"]) for r in good if r["roof_hae_shift_inset_15m"] is not None),
        holdout_weeks_with_current_height_control=[r["week_ending"] for r in weeks if r["current_retrospective_roofs"]],
        holdout_weeks_with_current_cmr_insert_before_cutoff=[r["week_ending"] for r in weeks if r["current_roofs_with_cmr_insert_before_cutoff"]],
        holdout_weeks_with_same_roofs_at_both_endpoints=[r["week_ending"] for r in weeks if r["same_roofs_with_both_endpoints"]],
        interpretation="Retrospective independent height controls. OpenAltimetry does not export photon timestamps or individual "
                       "quality flags. Photon time is bounded by the verified granule timestamps. Repeated date requests for the same "
                       "granule, roof and beam are counted once. The roof edge margin and local terrain alignment are engineering filters. "
                       "No oil stock targets or market results are used.")
    payload = json.dumps(report,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
