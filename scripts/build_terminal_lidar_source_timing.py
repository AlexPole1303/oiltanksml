"""Attach USGS publication dates to the source tiles behind the terminal roof geometries."""
import csv
import datetime as dt
import json
import re
import sqlite3
from collections import defaultdict
from contextlib import closing

import numpy as np
from pyproj import Transformer
from shapely import union_all
from shapely.geometry import box

from audit_lidar_tanks import DATASET, LEGACY_DATASET, SURVEYS
from build_lidar_radial_profiles import Circle, ZONES, select_returns
from build_tank_lidar_outlines import DERIVED
from build_tank_geometry_registry import TERMINAL_GEOMETRY_STUDY
from collect_public import RAW, ROOT, write_csv

SOURCE_FILE = DERIVED/"terminal_lidar_source_timing.csv"


def legacy_geometry():
    path = DERIVED/"terminal_legacy_lidar_geometry.csv"
    with path.open(newline="") as stream:
        return {r["candidate_id"]:r for r in csv.DictReader(stream) if r["diameter_m"] and r["fitted_2011_gap_m"]}


def old_origins(geometry):
    origins = json.loads((RAW / "usgs/ept" / LEGACY_DATASET / "ept-sources-list.json").read_text())
    folders = defaultdict(list)
    for candidate, row in geometry.items():
        folders[ROOT / row["source_directory"]].append(Circle(
            candidate, "independent_2011_elevated_circle", row["fit_status"], float(row["center_east_m"]),
            float(row["center_north_m"]), float(row["diameter_m"]) / 2))
    for folder, circles in folders.items():
        selected = select_returns(circles, SURVEYS[LEGACY_DATASET], sorted(folder.glob("*.laz")), include_origin=True)
        for i, circle in enumerate(circles):
            points = np.concatenate(selected[i])
            eligible = ((points[:, 6] == 0) & np.isin(points[:, 2], (ZONES.index("center"), ZONES.index("boundary")))
                        & (points[:, 1] == int(geometry[circle.candidate]["point_source_id"])))
            for origin in sorted(np.unique(points[eligible, -1]).astype(int)):
                path = origins[origin]["path"]
                tile = re.search(r"_(\d{5}[A-Z]\d_[A-Z]{3})_LAS_2016\.laz$", path)
                if tile is None:
                    raise ValueError(f"Unexpected EPT origin path {path}")
                yield circle.candidate, tile.group(1), origin, path


def main():
    legacy = legacy_geometry()
    metadata = {}
    with (DERIVED/"lidar_2011_source_tiles.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            metadata[r["tile_id"]] = r
    uri = (DERIVED/"cushing.sqlite").as_uri()+"?mode=ro"
    with closing(sqlite3.connect(uri,uri=True)) as db:
        db.row_factory = sqlite3.Row
        markers = ",".join("?" for _ in TERMINAL_GEOMETRY_STUDY)
        roofs = {r["candidate_id"]:r for r in db.execute(f"""SELECT t.candidate_id,t.documented_terminal_id,
            o.center_east_m,o.center_north_m,o.diameter_m FROM tank_engineering_registry t JOIN tank_lidar_outline o USING(candidate_id)
            WHERE t.documented_terminal_id IN ({markers}) AND t.documented_roof_type='EFR'""", TERMINAL_GEOMETRY_STUDY)}
        modern_tiles = list(db.execute("""SELECT id,title,publication_date,date_created,url,bounding_box_json
            FROM lidar_product WHERE url LIKE '%/OK_Statewide_2_D22/%'"""))
    tile_boxes = []
    for tile in modern_tiles:
        bounds = json.loads(tile["bounding_box_json"])
        tile_boxes.append((tile, box(bounds["minX"], bounds["minY"], bounds["maxX"], bounds["maxY"])))
    rows = []
    for candidate,tile,origin,path in old_origins(legacy):
        r = metadata[tile]
        date = dt.datetime.strptime(r["fgdc_publication_date_raw"],"%Y%m%d").date().isoformat()
        rows.append((candidate,roofs[candidate]["documented_terminal_id"],LEGACY_DATASET,float(legacy[candidate]["diameter_m"]),
                     r["usgs_source_id"],r["tnm_publication_date"],date,max(date,r["tnm_publication_date"]),
                     "ept_origin_id_and_fgdc_source_tile",origin,path,r["original_laz_url"],r["fgdc_url"]))
    project = Transformer.from_crs(32614,4326,always_xy=True)
    for candidate,r in roofs.items():
        east,north,radius = r["center_east_m"],r["center_north_m"],r["diameter_m"]/2+12
        roof_box = box(*project.transform_bounds(east-radius,north-radius,east+radius,north+radius))
        covered = [tile for tile, footprint in tile_boxes if footprint.covers(roof_box)]
        basis = "single_tnm_tile_covers_fitted_rim_search_bounds"
        if len(covered) != 1:
            intersecting = [(tile, footprint) for tile, footprint in tile_boxes if footprint.intersects(roof_box)]
            if not union_all([footprint for _, footprint in intersecting]).covers(roof_box):
                raise ValueError(f"TNM tile footprints do not cover the fitted rim search bounds for {candidate}")
            covered = [tile for tile, _ in intersecting]
            basis = "union_of_tnm_tiles_covers_fitted_rim_search_bounds"
        for tile in covered:
            rows.append((candidate, r["documented_terminal_id"], DATASET, r["diameter_m"], tile["id"],
                         tile["publication_date"], None, tile["publication_date"], basis, None, None, tile["url"], None))
    if {row[0] for row in rows if row[2]==LEGACY_DATASET}!=legacy.keys():
        raise ValueError("Not all legacy diameters have a verified EPT origin and publication date")
    fields = ("candidate_id","terminal_id","dataset","diameter_m","usgs_source_id","tnm_publication_date",
              "fgdc_publication_date","source_catalog_date","source_match_basis","ept_origin_id","ept_origin_path",
              "original_laz_url","source_metadata_url")
    write_csv(SOURCE_FILE,fields,sorted(rows))
    print(f"{len(rows)} USGS source-tile rows covering {len(roofs)} terminal roofs")


if __name__=="__main__":
    main()
