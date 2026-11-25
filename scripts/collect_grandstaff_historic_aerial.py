"""Archive original USDA county mosaics and recover dated Grandstaff image windows."""

import argparse
import hashlib
import io
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

import rasterio
import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import Point, shape

from build_s1_reference_matches import export, read
from collect_public import RAW, ROOT, TLS

STORE = RAW / "naip" / "historic_payne"
BOX = "https://nrcs.app.box.com"
PUBLIC_SHARE = "kx173sgjkk9bn6rl5s9klmwf19vio963"
# Name, public file ID, byte count, SHA-256 from the verified local archive.
ARCHIVES = {
    2003: ("ortho_1-1_1n_s_ok119_2003_1", "140407175957", 239110479,
           "3c804c4ffcaeb451e53d26a8a8ce7009369a08d7f5a1d972b42d90907fc79e22"),
    2004: ("naip_1-1_2n_s_ok119_2004_1", "140221704613", 38890140,
           "cfbeef2d0f209a4d3b0290aecdad747af38d1ca85c160401a89d01e9df4cc25d"),
    2005: ("naip_1-1_2n_s_ok119_2005_1", "140465838284", 135148853,
           "e02e94192ab11e2176c959a2a88b8134ff82a840652353e1ef5b7c10d30b67f7"),
    2006: ("naip_1-1_2n_s_ok119_2006_1", "140055769214", 135086586,
           "7593bdeba00bcc64a938eee620da1ec59796a15a3eb9395344c0f872bd94bf91"),
}
# Decoding does not resample the source raster. These bounds are in the USDA archive CRS.
CROP = (701725, 3986300, 702950, 3988825)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, length, digest):
    if path.stat().st_size != length or sha256(path) != digest:
        raise ValueError(f"Archive differs from the recorded USDA download: {path}")
    with ZipFile(path) as archive:
        bad = archive.testzip()
    if bad:
        raise ValueError(f"Invalid member in {path}: {bad}")


def archive_file(name, file_id, length, digest):
    path = STORE / "archives" / (name + ".zip")
    if path.is_file():
        verify(path, length, digest)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = BOX + "/index.php?" + urlencode({
        "rm": "box_download_shared_file", "shared_name": PUBLIC_SHARE, "file_id": f"f_{file_id}",
    })
    temporary = path.with_suffix(".part")
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), context=TLS, timeout=40) as response:
        if response.headers.get_content_type() != "application/zip":
            raise ValueError(f"Unexpected USDA file type for {name}")
        with temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    verify(temporary, length, digest)
    temporary.replace(path)
    return path


def acquisition_tiles(archive, name):
    with shapefile.Reader(
        shp=io.BytesIO(archive.read(name + ".shp")),
        shx=io.BytesIO(archive.read(name + ".shx")),
        dbf=io.BytesIO(archive.read(name + ".dbf")),
        encoding="latin1",
    ) as vector:
        return [(record.record.as_dict(), shape(record.shape.__geo_interface__)) for record in vector.iterShapeRecords()]


def decode(archive, name, image, crs, decoder):
    with tempfile.TemporaryDirectory(prefix="grandstaff_naip_") as work:
        directory = Path(work)
        for extension in (".sid", ".sdw", ".prj"):
            member = name + extension
            with archive.open(member) as src, (directory / member).open("wb") as dst:
                shutil.copyfileobj(src, dst)
        west, south, east, north = CROP
        temp_image = image.with_suffix(".part.tif")
        subprocess.run([
            decoder, "-i", str(directory / (name + ".sid")), "-o", str(temp_image),
            "-of", "tifg", "-coord", "geo", "-ulxy", str(west), str(north),
            "-lrxy", str(east), str(south), "-s", "0", "-progress", "none",
        ], check=True, capture_output=True)
        with rasterio.open(temp_image, "r+") as dataset:
            if dataset.crs is None or not dataset.crs.is_projected:
                dataset.crs = crs.to_wkt()
            elif dataset.crs.to_epsg() != crs.to_epsg():
                raise ValueError(f"Decoder CRS disagrees with the source for {name}")
        temp_image.replace(image)


def build_year(year, definition, tanks, decoder):
    name, file_id, size, digest = definition
    path = archive_file(*definition)
    source_url = f"{BOX}/v/naip/file/{file_id}"
    image = STORE / "grandstaff" / f"{year}.tif"
    image.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path) as archive:
        crs = CRS.from_wkt(archive.read(name + ".prj").decode())
        if crs.to_epsg() != 26914:
            raise ValueError(f"Unexpected archive CRS for {name}")
        x_step, _, _, y_step, _, _ = map(float, archive.read(name + ".sdw").split())
        decode(archive, name, image, crs, decoder)
        tiles = acquisition_tiles(archive, name)
    with rasterio.open(image) as ds:
        if ds.crs.to_epsg() != crs.to_epsg() or ds.res != (x_step, -y_step) or ds.count != 3:
            raise ValueError(f"Unexpected native raster dimensions for {name}")
        if not ds.read().any():
            raise ValueError(f"Empty decoded raster {image}")
        bounds = ds.bounds
    projection = Transformer.from_crs(4326, crs, always_xy=True)
    rows = []
    for tank in tanks:
        x, y = projection.transform(float(tank["longitude"]), float(tank["latitude"]))
        if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
            raise ValueError(f"Decoded image excludes {tank['candidate_id']} in {year}")
        tile, = [attrs for attrs, geometry in tiles if geometry.covers(Point(x, y))]
        stamp = tile["IDAT"].replace("-", "")
        if len(stamp) != 8 or int(stamp[:4]) != year:
            raise ValueError(f"Unexpected USDA source date {stamp} for {tank['candidate_id']}")
        rows.append(dict(
            candidate_id=tank["candidate_id"], cohort_id=tank["cohort_id"], year=year,
            source_acquisition_date=f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}",
            quarter_quad=tile["DOQQ"], source_pixel_size_m=x_step,
            pixel_grid_crs="EPSG:26914", easting_m=round(x, 3), northing_m=round(y, 3),
            crop_file=str(image.relative_to(ROOT)), archive_file=str(path.relative_to(ROOT)),
            archive_url=source_url,
        ))
    source = dict(
        year=year, archive_file=str(path.relative_to(ROOT)), archive_url=source_url, archive_bytes=size,
        archive_sha256=digest, archive_member_crc_verified=True, crop_file=str(image.relative_to(ROOT)),
        crop_sha256=sha256(image), source_pixel_size_m=x_step, crop_crs="EPSG:26914",
    )
    return source, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", required=True, help="Path to the LizardTech mrsiddecode binary")
    args = parser.parse_args()
    decoder = str(Path(args.decoder).resolve(strict=True))
    tanks = [r for r in read("tank_engineering_registry") if r["documented_terminal_id"] == "enbridge_blueknight"]
    sources, dates = [], []
    for year, definition in ARCHIVES.items():
        source, observations = build_year(year, definition, tanks, decoder)
        sources.append(source)
        dates.extend(observations)
    export("cushing_grandstaff_historic_aerial_sources", sources)
    export("cushing_grandstaff_historic_aerial_dates", dates)
    print(f"Verified {len(sources)} USDA originals and {len(dates)} dated tank-centre image intersections")


if __name__ == "__main__":
    main()
