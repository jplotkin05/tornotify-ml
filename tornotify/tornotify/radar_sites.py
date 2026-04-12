"""
Radar site catalog helpers.

Py-ART ships a NEXRAD_LOCATIONS table with radar latitude, longitude, and
elevation metadata. Loading Py-ART itself has noisy import-time side effects, so
this module reads that table from source with ast.literal_eval instead of
importing pyart at config import time.
"""
from __future__ import annotations

import ast
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INCLUDE_TDWR_SITES_ENV = "TORNOTIFY_INCLUDE_TDWR"
MISSING_ELEVATION = -99999

CORE_RADAR_SITES: dict[str, dict[str, Any]] = {
    "KLWX": {"lat": 38.9753, "lon": -77.4778, "elev_ft": 327, "name": "Sterling, VA"},
    "KTLX": {"lat": 35.3331, "lon": -97.2778, "elev_ft": 1213, "name": "Oklahoma City, OK"},
    "KTWX": {"lat": 38.99695, "lon": -96.23255, "elev_ft": 1415, "name": "Topeka, KS"},
    "KILX": {"lat": 40.1506, "lon": -89.3368, "elev_ft": 754, "name": "Lincoln, IL"},
    "KDOX": {"lat": 38.8257, "lon": -75.4401, "elev_ft": 50, "name": "Dover, DE"},
    "KMPX": {"lat": 44.8488, "lon": -93.5654, "elev_ft": 946, "name": "Minneapolis, MN"},
    "KJAX": {"lat": 30.4847, "lon": -81.7019, "elev_ft": 33, "name": "Jacksonville, FL"},
    "KMXX": {"lat": 32.5366, "lon": -85.7897, "elev_ft": 460, "name": "Maxwell AFB, AL"},
    "KMAF": {"lat": 31.943461, "lon": -102.18925, "elev_ft": 2961, "name": "Midland/Odessa, TX"},
}

# Display names for the WSR-88D/NEXRAD sites in Py-ART's coordinate table.
# The base list comes from the NOAA/NWS WSR-88D radar list; newer Py-ART
# entries not present in that older PDF are added from ROC/Level-II metadata.
RADAR_SITE_NAMES: dict[str, str] = {
    "KABR": "Aberdeen, SD",
    "KABX": "Albuquerque, NM",
    "KAKQ": "Wakefield, VA",
    "KAMA": "Amarillo, TX",
    "KAMX": "Miami, FL",
    "KAPX": "Gaylord, MI",
    "KARX": "La Crosse, WI",
    "KATX": "Seattle, WA",
    "KBBX": "Beale AFB, CA",
    "KBGM": "Binghamton, NY",
    "KBHX": "Eureka, CA",
    "KBIS": "Bismarck, ND",
    "KBLX": "Billings, MT",
    "KBMX": "Birmingham, AL",
    "KBOX": "Boston, MA",
    "KBRO": "Brownsville, TX",
    "KBUF": "Buffalo, NY",
    "KBYX": "Key West, FL",
    "KCAE": "Columbia, SC",
    "KCBW": "Caribou, ME",
    "KCBX": "Boise, ID",
    "KCCX": "State College, PA",
    "KCLE": "Cleveland, OH",
    "KCLX": "Charleston, SC",
    "KCRI": "Norman, OK",
    "KCRP": "Corpus Christi, TX",
    "KCXX": "Burlington, VT",
    "KCYS": "Cheyenne, WY",
    "KDAX": "Sacramento, CA",
    "KDDC": "Dodge City, KS",
    "KDFX": "Laughlin AFB, TX",
    "KDGX": "Jackson, MS",
    "KDIX": "Philadelphia, PA",
    "KDLH": "Duluth, MN",
    "KDMX": "Des Moines, IA",
    "KDOX": "Dover AFB, DE",
    "KDTX": "Detroit, MI",
    "KDVN": "Quad Cities/Davenport, IA",
    "KDYX": "Dyess AFB, TX",
    "KEAX": "Kansas City, MO",
    "KEMX": "Tucson, AZ",
    "KENX": "Albany, NY",
    "KEOX": "Ft Rucker, AL",
    "KEPZ": "El Paso, TX",
    "KESX": "Las Vegas, NV",
    "KEVX": "Northwest Florida/Eglin AFB, FL",
    "KEWX": "San Antonio, TX",
    "KEYX": "Edwards AFB, CA",
    "KFCX": "Blacksburg, VA",
    "KFDR": "Frederick/Altus AFB, OK",
    "KFDX": "Cannon AFB, NM",
    "KFFC": "Atlanta, GA",
    "KFSD": "Sioux Falls, SD",
    "KFSX": "Flagstaff, AZ",
    "KFTG": "Denver, CO",
    "KFWS": "Ft Worth, TX",
    "KGGW": "Glasgow, MT",
    "KGJX": "Grand Junction, CO",
    "KGLD": "Goodland, KS",
    "KGRB": "Green Bay, WI",
    "KGRK": "Central Texas/Ft Hood, TX",
    "KGRR": "Grand Rapids, MI",
    "KGSP": "Greer, SC",
    "KGWX": "Columbus AFB, MS",
    "KGYX": "Portland, ME",
    "KHDC": "Hammond/New Orleans, LA",
    "KHDX": "Holloman AFB, NM",
    "KHGX": "Houston, TX",
    "KHNX": "San Joaquin Valley, CA",
    "KHPX": "Ft Campbell, KY",
    "KHTX": "Huntsville/Hytop, AL",
    "KICT": "Wichita, KS",
    "KICX": "Cedar City, UT",
    "KILN": "Cincinnati/Wilmington, OH",
    "KILX": "Lincoln, IL",
    "KIND": "Indianapolis, IN",
    "KINX": "Tulsa, OK",
    "KIWA": "Phoenix, AZ",
    "KIWX": "Northern Indiana/North Webster, IN",
    "KJAX": "Jacksonville, FL",
    "KJGX": "Robins AFB, GA",
    "KJKL": "Jackson, KY",
    "KLBB": "Lubbock, TX",
    "KLCH": "Lake Charles, LA",
    "KLGX": "Langley Hill, WA",
    "KLIX": "New Orleans, LA",
    "KLNX": "North Platte, NE",
    "KLOT": "Chicago, IL",
    "KLRX": "Elko, NV",
    "KLSX": "St Louis, MO",
    "KLTX": "Wilmington, NC",
    "KLVX": "Louisville, KY",
    "KLWX": "Sterling, VA",
    "KLZK": "Little Rock, AR",
    "KMAF": "Midland/Odessa, TX",
    "KMAX": "Medford, OR",
    "KMBX": "Minot AFB, ND",
    "KMHX": "Morehead City, NC",
    "KMKX": "Milwaukee, WI",
    "KMLB": "Melbourne, FL",
    "KMOB": "Mobile, AL",
    "KMPX": "Minneapolis, MN",
    "KMQT": "Marquette, MI",
    "KMRX": "Knoxville/Morristown, TN",
    "KMSX": "Missoula, MT",
    "KMTX": "Salt Lake City, UT",
    "KMUX": "San Francisco, CA",
    "KMVX": "Grand Forks, ND",
    "KMXX": "Maxwell AFB, AL",
    "KNKX": "San Diego, CA",
    "KNQA": "Memphis, TN",
    "KOAX": "Omaha, NE",
    "KOHX": "Nashville, TN",
    "KOKX": "New York City/Upton, NY",
    "KOTX": "Spokane, WA",
    "KPAH": "Paducah, KY",
    "KPBZ": "Pittsburgh, PA",
    "KPDT": "Pendleton, OR",
    "KPOE": "Ft Polk, LA",
    "KPUX": "Pueblo, CO",
    "KRAX": "Raleigh/Durham, NC",
    "KRGX": "Reno, NV",
    "KRIW": "Riverton, WY",
    "KRLX": "Charleston, WV",
    "KRTX": "Portland, OR",
    "KSFX": "Pocatello, ID",
    "KSGF": "Springfield, MO",
    "KSHV": "Shreveport, LA",
    "KSJT": "San Angelo, TX",
    "KSOX": "Santa Ana Mountains, CA",
    "KSRX": "Fort Smith, AR",
    "KTBW": "Tampa Bay, FL",
    "KTFX": "Great Falls, MT",
    "KTLH": "Tallahassee, FL",
    "KTLX": "Oklahoma City, OK",
    "KTWX": "Topeka, KS",
    "KTYX": "Montague/Ft Drum, NY",
    "KUDX": "Rapid City, SD",
    "KUEX": "Hastings, NE",
    "KVAX": "Moody AFB, GA",
    "KVBX": "Vandenberg AFB, CA",
    "KVNX": "Vance AFB, OK",
    "KVTX": "Los Angeles, CA",
    "KVWX": "Evansville, IN",
    "KYUX": "Yuma, AZ",
    "LPLA": "Lajes Field, Azores",
    "PABC": "Bethel, AK",
    "PACG": "Biorka Island/Sitka, AK",
    "PAEC": "Nome, AK",
    "PAHG": "Anchorage/Kenai, AK",
    "PAIH": "Middleton Island, AK",
    "PAKC": "King Salmon, AK",
    "PAPD": "Fairbanks/Pedro Dome, AK",
    "PGUA": "Andersen AFB, Guam",
    "PHKI": "South Kauai, HI",
    "PHKM": "Kamuela/Kohala, HI",
    "PHMO": "Molokai, HI",
    "PHWA": "South Shore, HI",
    "RKJK": "Kunsan, South Korea",
    "RKSG": "Camp Humphreys, South Korea",
    "RODN": "Kadena, Japan",
    "TJUA": "San Juan, PR",
}

STATE_TIMEZONES: dict[str, str] = {
    "AL": "America/Chicago",
    "AR": "America/Chicago",
    "AZ": "America/Phoenix",
    "CA": "America/Los_Angeles",
    "CO": "America/Denver",
    "CT": "America/New_York",
    "DE": "America/New_York",
    "FL": "America/New_York",
    "GA": "America/New_York",
    "HI": "Pacific/Honolulu",
    "IA": "America/Chicago",
    "ID": "America/Denver",
    "IL": "America/Chicago",
    "IN": "America/Indiana/Indianapolis",
    "KS": "America/Chicago",
    "KY": "America/New_York",
    "LA": "America/Chicago",
    "MA": "America/New_York",
    "MD": "America/New_York",
    "ME": "America/New_York",
    "MI": "America/Detroit",
    "MN": "America/Chicago",
    "MO": "America/Chicago",
    "MS": "America/Chicago",
    "MT": "America/Denver",
    "NC": "America/New_York",
    "ND": "America/Chicago",
    "NE": "America/Chicago",
    "NM": "America/Denver",
    "NV": "America/Los_Angeles",
    "NY": "America/New_York",
    "OH": "America/New_York",
    "OK": "America/Chicago",
    "OR": "America/Los_Angeles",
    "PA": "America/New_York",
    "PR": "America/Puerto_Rico",
    "SC": "America/New_York",
    "SD": "America/Chicago",
    "TN": "America/Chicago",
    "TX": "America/Chicago",
    "UT": "America/Denver",
    "VA": "America/New_York",
    "VT": "America/New_York",
    "WA": "America/Los_Angeles",
    "WI": "America/Chicago",
    "WV": "America/New_York",
    "WY": "America/Denver",
}

RADAR_SITE_TIMEZONES: dict[str, str] = {
    "PABC": "America/Anchorage",
    "PACG": "America/Sitka",
    "PAEC": "America/Nome",
    "PAHG": "America/Anchorage",
    "PAIH": "America/Anchorage",
    "PAKC": "America/Anchorage",
    "PAPD": "America/Anchorage",
    "PGUA": "Pacific/Guam",
    "PHKI": "Pacific/Honolulu",
    "PHKM": "Pacific/Honolulu",
    "PHMO": "Pacific/Honolulu",
    "PHWA": "Pacific/Honolulu",
    "RKJK": "Asia/Seoul",
    "RKSG": "Asia/Seoul",
    "RODN": "Asia/Tokyo",
    "LPLA": "Atlantic/Azores",
    "KEPZ": "America/Denver",
    "KFSX": "America/Phoenix",
    "KEVX": "America/Chicago",
    "KHPX": "America/Chicago",
    "KMRX": "America/New_York",
    "KUDX": "America/Denver",
    "KCBX": "America/Boise",
}


def build_radar_site_catalog(
    overrides: dict[str, dict[str, Any]] | None = None,
    *,
    include_tdwr: bool | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Build the site catalog used by scanning and geo conversion.

    The default catalog includes non-TDWR sites from Py-ART's NEXRAD table and
    overlays locally validated metadata for the sites we have already smoke
    tested. If the Py-ART source file is unavailable, the validated core site
    list is used as a fallback so the app can still start.
    """
    if include_tdwr is None:
        include_tdwr = os.getenv(INCLUDE_TDWR_SITES_ENV, "0") == "1"

    overrides = overrides or CORE_RADAR_SITES
    raw_locations = _load_pyart_nexrad_locations()

    if raw_locations:
        catalog = {}
        for site_id, location in sorted(raw_locations.items()):
            if not include_tdwr and _is_tdwr_site(site_id):
                continue
            catalog[site_id] = _normalize_site_record(site_id, location)
    else:
        logger.warning("Falling back to core radar site list; Py-ART NEXRAD_LOCATIONS was not available")
        catalog = {site_id: dict(record) for site_id, record in sorted(CORE_RADAR_SITES.items())}

    for site_id, override in overrides.items():
        base = dict(catalog.get(site_id, {"name": site_id}))
        base.update(override)
        base.setdefault("timezone", _site_timezone(site_id))
        catalog[site_id] = base

    return dict(sorted(catalog.items()))


def _load_pyart_nexrad_locations() -> dict[str, dict[str, Any]]:
    source_path = _pyart_nexrad_common_path()
    if source_path is None:
        return {}

    try:
        tree = ast.parse(source_path.read_text())
    except OSError as exc:
        logger.debug("Could not read Py-ART NEXRAD location source: %s", exc)
        return {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "NEXRAD_LOCATIONS":
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, SyntaxError) as exc:
                    logger.debug("Could not parse Py-ART NEXRAD_LOCATIONS: %s", exc)
                    return {}
                return value if isinstance(value, dict) else {}
    return {}


def _pyart_nexrad_common_path() -> Path | None:
    spec = importlib.util.find_spec("pyart")
    if spec is None or not spec.submodule_search_locations:
        return None

    pyart_root = Path(next(iter(spec.submodule_search_locations)))
    source_path = pyart_root / "io" / "nexrad_common.py"
    return source_path if source_path.exists() else None


def _normalize_site_record(site_id: str, location: dict[str, Any]) -> dict[str, Any]:
    elev = location.get("elev")
    elev_ft = None if elev is None or elev == MISSING_ELEVATION else int(round(float(elev)))
    return {
        "lat": float(location["lat"]),
        "lon": float(location["lon"]),
        "elev_ft": elev_ft,
        "name": RADAR_SITE_NAMES.get(site_id, site_id),
        "timezone": _site_timezone(site_id),
    }


def _site_timezone(site_id: str) -> str:
    if site_id in RADAR_SITE_TIMEZONES:
        return RADAR_SITE_TIMEZONES[site_id]

    name = RADAR_SITE_NAMES.get(site_id, "")
    state = name.rsplit(", ", 1)[-1] if ", " in name else ""
    return STATE_TIMEZONES.get(state, "UTC")


def _is_tdwr_site(site_id: str) -> bool:
    # TDWR airport radars use Txxx IDs; TJUA is the Puerto Rico WSR-88D.
    return site_id.startswith("T") and site_id != "TJUA"
