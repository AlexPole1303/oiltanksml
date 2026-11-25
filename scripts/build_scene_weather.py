"""Match exact satellite acquisition clocks to nearby surface weather records."""
import datetime as dt
from bisect import bisect_left, bisect_right

from pyproj import Geod

SCHEMA = """
CREATE TABLE satellite_weather_clock (
    collection TEXT,item_id TEXT,captured_at_utc TEXT,weather_reference_utc TEXT,
    PRIMARY KEY (collection,item_id));
CREATE TABLE scene_weather_match (
    weather_reference_utc TEXT,station TEXT,report_kind TEXT,time_relation TEXT,observed_at_utc TEXT,
    observed_minus_reference_seconds REAL,report_variants INTEGER,
    PRIMARY KEY(weather_reference_utc,station,report_kind,time_relation));
CREATE TABLE tank_weather_station_distance (
    candidate_id TEXT,station TEXT,station_longitude REAL,station_latitude REAL,distance_km REAL,distance_rank INTEGER,
    PRIMARY KEY(candidate_id,station));
CREATE VIEW scene_weather_observation AS
    SELECT m.weather_reference_utc,m.station,m.report_kind,m.time_relation,m.observed_at_utc,
        m.observed_minus_reference_seconds,m.report_variants,w.lon AS station_longitude,w.lat AS station_latitude,
        w.temperature_f,w.dew_point_f,w.relative_humidity_pct,
        round((w.temperature_f-w.dew_point_f)*5.0/9,2) AS dew_point_depression_c,
        w.wind_direction_deg,w.wind_speed_kt,w.gust_kt,w.precip_1h_in,w.precip_trace,w.visibility_mi,
        w.altimeter_inhg,w.sea_level_pressure_hpa,w.sky_cover_1,w.sky_cover_2,w.sky_cover_3,w.sky_cover_4,
        w.sky_base_1_ft,w.sky_base_2_ft,w.sky_base_3_ft,w.sky_base_4_ft,w.weather_codes,w.metar,w.source_file
    FROM scene_weather_match m LEFT JOIN weather_metar w
    ON w.station=m.station AND w.observed_at_utc=m.observed_at_utc AND w.report_kind=m.report_kind;
"""


def utc_timestamp(value):
    timestamp = dt.datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"Missing time zone for satellite or weather timestamp: {value}")
    return timestamp.astimezone(dt.timezone.utc)


def weather_matches(db, references):
    groups = db.execute("SELECT DISTINCT station,report_kind FROM weather_metar ORDER BY station,report_kind").fetchall()
    for station, kind in groups:
        records = db.execute("""SELECT observed_at_utc,count(*) FROM weather_metar
            WHERE station=? AND report_kind=? GROUP BY observed_at_utc ORDER BY observed_at_utc""", (station, kind)).fetchall()
        times = [utc_timestamp(record[0]) for record in records]
        for reference, captured in references.items():
            before = bisect_right(times, captured) - 1
            after = bisect_left(times, captured)
            for side, index in (("at_or_before", before), ("at_or_after", after)):
                if 0 <= index < len(records):
                    observed, variants = records[index]
                    offset = round((times[index] - captured).total_seconds(), 3)
                    yield (reference, station, kind, side, observed, offset, variants)
                else:
                    yield (reference, station, kind, side, None, None, 0)


def station_distances(db):
    geod = Geod(ellps="WGS84")
    stations = db.execute("""SELECT station,min(lon),min(lat),count(DISTINCT lon||'/'||lat)
        FROM weather_metar WHERE lon IS NOT NULL AND lat IS NOT NULL GROUP BY station ORDER BY station""").fetchall()
    if any(row[3] != 1 for row in stations):
        raise ValueError("Station coordinates vary within the local weather archive")
    for candidate, lon, lat in db.execute("SELECT candidate_id,longitude,latitude FROM tank_engineering_registry"):
        distances = sorted((geod.inv(lon, lat, station_lon, station_lat)[2] / 1000, station, station_lon, station_lat)
                           for station, station_lon, station_lat, _ in stations)
        for rank, (km, station, station_lon, station_lat) in enumerate(distances, 1):
            yield candidate, station, station_lon, station_lat, round(km, 3), rank


def add_scene_weather(db):
    references, clocks = {}, []
    for collection, item, captured in db.execute("""SELECT collection,item_id,captured_at_utc FROM satellite_scene
        WHERE collection LIKE 'sentinel-%' ORDER BY captured_at_utc,collection,item_id"""):
        timestamp = utc_timestamp(captured)
        reference = timestamp.isoformat(timespec="microseconds")
        references[reference] = timestamp
        clocks.append((collection, item, captured, reference))
    db.executemany("INSERT INTO satellite_weather_clock VALUES (?,?,?,?)", clocks)
    db.executemany("INSERT INTO scene_weather_match VALUES (?,?,?,?,?,?,?)", weather_matches(db, references))
    db.executemany("INSERT INTO tank_weather_station_distance VALUES (?,?,?,?,?,?)", station_distances(db))
