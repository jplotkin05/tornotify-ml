"""
TorNotify — Radar Analysis GUI

Interactive Streamlit app for uploading/fetching NEXRAD radar data,
running tornado detection inference, and visualizing results.

Usage:
    streamlit run scripts/radar_gui.py
"""
import os
import sys
import time
import re

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

os.environ.setdefault("KERAS_BACKEND", "torch")

import tempfile
import logging
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import streamlit as st
import matplotlib.pyplot as plt

from tornotify.config import (
    RADAR_SITES,
    CONFIDENCE_THRESHOLD,
    CELL_DBZ_THRESHOLD,
    CELL_MIN_GATES,
    DETECTION_MIN_RANGE_KM,
    DETECTION_MIN_GATES,
    DETECTION_MIN_HEATMAP_EDGE_MARGIN,
)
from tornotify.tracking import (
    TrackManager,
    TrackingConfig,
    DetectionCandidate,
)

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TorNotify — Radar Analysis",
    page_icon="🌪️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Lazy imports cached at app level
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading ML model...")
def get_model():
    from tornotify.ml.detector import load_model
    return load_model()


def import_pipeline():
    """Import pipeline modules (heavy deps: pyart, etc.)."""
    from tornotify.ingest.nexrad import (
        get_available_scans, download_scan, parse_scan, get_scan_time,
    )
    from tornotify.preprocess.cells import identify_cells
    from tornotify.preprocess.chips import extract_chip, ALL_VARIABLES
    from tornotify.ml.detector import predict_batch_detailed
    from tornotify.ml.quality import evaluate_detection
    return {
        "get_available_scans": get_available_scans,
        "download_scan": download_scan,
        "parse_scan": parse_scan,
        "get_scan_time": get_scan_time,
        "identify_cells": identify_cells,
        "extract_chip": extract_chip,
        "predict_batch_detailed": predict_batch_detailed,
        "evaluate_detection": evaluate_detection,
        "ALL_VARIABLES": ALL_VARIABLES,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("TorNotify")
st.sidebar.markdown("**Radar Analysis Tool**")

input_mode = st.sidebar.radio("Input mode", ["Fetch from S3", "Upload file"])

site_options = list(RADAR_SITES.keys())
site = st.sidebar.selectbox(
    "NEXRAD site",
    options=site_options,
    format_func=lambda s: f"{s} - {RADAR_SITES[s]['name']}",
    index=site_options.index("KTLX") if "KTLX" in site_options else 0,
)

scan_date = st.sidebar.date_input(
    "Scan date",
    value=date.today(),
    min_value=date(1991, 1, 1),
    max_value=date.today(),
)

st.sidebar.markdown("---")
st.sidebar.subheader("Detection settings")
conf_threshold = st.sidebar.slider(
    "Confidence threshold", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05,
)
dbz_threshold = CELL_DBZ_THRESHOLD
min_gates = CELL_MIN_GATES
shear_threshold = 3.0

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Radar Analysis")

# Initialize session state
if "radar" not in st.session_state:
    st.session_state.radar = None
    st.session_state.scan_time = None
    st.session_state.results = None
    st.session_state.chips = None
    st.session_state.cells = None
    st.session_state.timings = {}
    st.session_state.detected_site = None
    st.session_state.replay_results = None


def _effective_site():
    """Return detected site from uploaded filename if available, else sidebar."""
    return st.session_state.get("detected_site") or site


def _format_scan_filename_for_display(filename: str) -> str:
    """Return a user-readable label for NEXRAD scan names."""
    match = re.match(r"^(?P<site>[A-Z0-9]{4})(?P<date>\d{8})_(?P<time>\d{6})", filename.upper())
    if not match:
        return filename

    site_id = match.group("site")
    utc_dt = datetime.strptime(
        f"{match.group('date')}{match.group('time')}",
        "%Y%m%d%H%M%S",
    ).replace(tzinfo=timezone.utc)

    site_record = RADAR_SITES.get(site_id, {})
    try:
        local_dt = utc_dt.astimezone(ZoneInfo(site_record.get("timezone", "UTC")))
    except ZoneInfoNotFoundError:
        local_dt = utc_dt

    hour = local_dt.hour % 12 or 12
    return f"{local_dt.strftime('%b')} {local_dt.day}, {hour}:{local_dt:%M %p %Z}"


def _result_above_threshold(result: dict) -> bool:
    return float(result["P(tornado)"]) >= conf_threshold


def _result_gate_reasons(result: dict) -> list[str]:
    reasons = []
    if float(result.get("range_km", 0.0)) < DETECTION_MIN_RANGE_KM:
        reasons.append(f"range < {DETECTION_MIN_RANGE_KM:.0f} km")
    if int(result.get("n_gates", 0)) < DETECTION_MIN_GATES:
        reasons.append(f"gates < {DETECTION_MIN_GATES}")
    edge_margin = result.get("edge_margin")
    if edge_margin is not None and float(edge_margin) < DETECTION_MIN_HEATMAP_EDGE_MARGIN:
        reasons.append("edge max")
    return reasons


def _result_gated_at_current_threshold(result: dict) -> bool:
    return _result_above_threshold(result) and bool(_result_gate_reasons(result))


def _label_offset(index: int) -> tuple[int, int]:
    offsets = [
        (5, 5),
        (5, -10),
        (-26, 5),
        (-28, -10),
        (10, 14),
        (-36, 14),
        (12, -18),
        (-40, -18),
    ]
    return offsets[index % len(offsets)]


def _add_detailed_map_underlay(
    ax,
    *,
    extent,
    city_min_pop=10_000,
    city_limit=90,
    label_style="dark",
):
    """Add detailed map context without taking over the storm/track layer."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io.shapereader import Reader, natural_earth

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ece9df", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d8eef7", zorder=0)
    ax.add_feature(
        cfeature.LAKES.with_scale("50m"),
        facecolor="#d8eef7", edgecolor="#7f9eaa", linewidth=0.35, zorder=2,
    )
    ax.add_feature(
        cfeature.RIVERS.with_scale("50m"),
        edgecolor="#6f9db5", linewidth=0.35, zorder=2,
    )
    ax.add_feature(
        cfeature.STATES.with_scale("50m"),
        linewidth=0.75, edgecolor="#222222", facecolor="none", zorder=4,
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.9, edgecolor="#222222", zorder=4,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=0.7, edgecolor="#222222", zorder=4,
    )
    counties = cfeature.NaturalEarthFeature(
        "cultural", "admin_2_counties", "10m",
        facecolor="none", edgecolor="#5c5c5c", linewidth=0.28,
    )
    ax.add_feature(counties, zorder=4)

    try:
        roads = cfeature.NaturalEarthFeature(
            "cultural", "roads", "10m",
            facecolor="none", edgecolor="#8a8175", linewidth=0.35,
        )
        ax.add_feature(roads, zorder=3)
    except Exception:
        pass

    try:
        places_shp = natural_earth(
            resolution="10m", category="cultural", name="populated_places"
        )
        records = []
        for rec in Reader(places_shp).records():
            lon_c, lat_c = rec.geometry.x, rec.geometry.y
            if not (extent[0] <= lon_c <= extent[1] and extent[2] <= lat_c <= extent[3]):
                continue
            pop = rec.attributes.get("POP_MAX", 0) or 0
            if pop >= city_min_pop:
                records.append((pop, lon_c, lat_c, rec.attributes["NAME"]))
        records.sort(reverse=True)

        if label_style == "light":
            text_color = "white"
            dot_color = "white"
            box = dict(boxstyle="round,pad=0.16", fc="black", alpha=0.62, lw=0)
        else:
            text_color = "#111111"
            dot_color = "#111111"
            box = dict(boxstyle="round,pad=0.13", fc="white", ec="#777777", alpha=0.82, lw=0.25)

        lon_min, lon_max, lat_min, lat_max = extent
        lon_span = max(lon_max - lon_min, 0.01)
        lat_span = max(lat_max - lat_min, 0.01)
        edge_margin_lon = lon_span * 0.025
        edge_margin_lat = lat_span * 0.025
        occupied_boxes = []

        def _intersects(box_a, box_b):
            return not (
                box_a[2] < box_b[0]
                or box_b[2] < box_a[0]
                or box_a[3] < box_b[1]
                or box_b[3] < box_a[1]
            )

        labels_drawn = 0
        for pop, lon_c, lat_c, name in records:
            if labels_drawn >= city_limit:
                break
            if (
                lon_c <= lon_min + edge_margin_lon
                or lon_c >= lon_max - edge_margin_lon
                or lat_c <= lat_min + edge_margin_lat
                or lat_c >= lat_max - edge_margin_lat
            ):
                continue

            fontsize = 7.5 if pop >= 50_000 else 6.2
            label_width = lon_span * (0.008 * len(name) + 0.018)
            label_height = lat_span * (0.030 if pop >= 50_000 else 0.024)
            place_left = lon_c > lon_min + lon_span * 0.72
            if place_left:
                text = f"{name} "
                ha = "right"
                label_box = (
                    lon_c - label_width,
                    lat_c,
                    lon_c,
                    lat_c + label_height,
                )
            else:
                text = f" {name}"
                ha = "left"
                label_box = (
                    lon_c,
                    lat_c,
                    lon_c + label_width,
                    lat_c + label_height,
                )
            if (
                label_box[0] < lon_min + edge_margin_lon
                or label_box[2] > lon_max - edge_margin_lon
                or label_box[1] < lat_min + edge_margin_lat
                or label_box[3] > lat_max - edge_margin_lat
            ):
                continue
            if any(_intersects(label_box, existing) for existing in occupied_boxes):
                continue

            ax.text(
                lon_c, lat_c, text,
                transform=ccrs.PlateCarree(),
                fontsize=fontsize, fontweight="bold", color=text_color,
                ha=ha, va="bottom", bbox=box, zorder=5,
            )
            ax.plot(
                lon_c, lat_c, "o", color=dot_color, markersize=2.4,
                transform=ccrs.PlateCarree(), zorder=5,
            )
            occupied_boxes.append(label_box)
            labels_drawn += 1
    except Exception:
        pass


def _parse_uploaded_file(uploaded_file):
    """Save uploaded file to temp dir and parse."""
    pipe = import_pipeline()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, uploaded_file.name)
        with open(local_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        radar = pipe["parse_scan"](local_path)
    return radar


def _fetch_and_parse(site, scan_date, selected_scan_idx=None):
    """Fetch scan from S3 and parse."""
    pipe = import_pipeline()

    dt = datetime(scan_date.year, scan_date.month, scan_date.day, tzinfo=timezone.utc)
    scans = pipe["get_available_scans"](site, dt)
    if not scans:
        st.error(f"No scans found for {site} on {scan_date}")
        return None, scans

    idx = selected_scan_idx if selected_scan_idx is not None else len(scans) - 1
    scan = scans[idx]

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = pipe["download_scan"](scan, tmpdir)
        radar = pipe["parse_scan"](local_path)
    return radar, scans


def _run_tracking_replay(
    site,
    scans,
    start_idx,
    end_idx,
    conf_threshold,
    dbz_threshold,
    min_gates,
    shear_threshold,
    progress_callback=None,
):
    """
    Process a sequence of scans with a shared TrackManager, mirroring the
    live scanner's temporal confirmation logic. Returns a dict with:
        per_scan: list of per-scan summary dicts (ordered)
        tracks:   list of track summary dicts (final state)
        config:   TrackingConfig used
    """
    import math
    import numpy as np
    from pyproj import Geod

    pipe = import_pipeline()
    selected_scans = scans[start_idx:end_idx + 1]

    config = TrackingConfig(threshold=conf_threshold)
    tm = TrackManager(config)

    per_scan = []
    tracks_final: dict[str, dict] = {}

    # We also capture the final successful scan's reflectivity grid so we
    # can draw an overview map with track paths on top.
    backdrop = None
    radar_lat_cached = None
    radar_lon_cached = None
    _geod = Geod(ellps="WGS84")

    for i, scan in enumerate(selected_scans):
        if progress_callback:
            progress_callback(i, len(selected_scans), scan.filename)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = pipe["download_scan"](scan, tmpdir)
                radar = pipe["parse_scan"](local_path)
        except Exception as exc:
            per_scan.append({
                "idx": start_idx + i,
                "filename": scan.filename,
                "scan_time": None,
                "status": f"parse_error: {exc}",
                "n_cells": 0,
                "n_raw": 0,
                "n_pending": 0,
                "n_confirmed": 0,
                "n_newly_confirmed": 0,
                "max_prob": None,
            })
            continue

        scan_time = pipe["get_scan_time"](radar)
        scan_key = f"{site}-{scan_time.strftime('%Y%m%d%H%M%S')}"

        # Radar lat/lon — cache the first valid pair (fall back to site config)
        if radar_lat_cached is None:
            rlat = float(radar.latitude["data"][0])
            rlon = float(radar.longitude["data"][0])
            if abs(rlat) < 1.0 and abs(rlon) < 1.0 and site in RADAR_SITES:
                rlat = RADAR_SITES[site]["lat"]
                rlon = RADAR_SITES[site]["lon"]
            radar_lat_cached = rlat
            radar_lon_cached = rlon

        cells = pipe["identify_cells"](
            radar, scan_key,
            dbz_threshold=dbz_threshold,
            min_gates=min_gates,
            shear_threshold=shear_threshold,
        )

        chips = []
        valid_cells = []
        for cell in cells:
            chip = pipe["extract_chip"](radar, cell)
            if chip is not None:
                chips.append(chip)
                valid_cells.append(cell)

        # Build a backdrop from this scan's 0.5° sweep (use the latest
        # successful scan — the track endpoints will be most meaningful there).
        try:
            from tornotify.ingest.nexrad import find_sweep_for_elevation, get_field
            sweep_idx = find_sweep_for_elevation(radar, 0.5) or 0
            dbz = np.ma.filled(get_field(radar, "reflectivity", sweep_idx), np.nan)
            sweep_start = radar.sweep_start_ray_index["data"][sweep_idx]
            sweep_end = radar.sweep_end_ray_index["data"][sweep_idx] + 1
            azimuths = radar.azimuth["data"][sweep_start:sweep_end]
            ranges_km = radar.range["data"] / 1000.0
            az_grid, rng_grid = np.meshgrid(azimuths, ranges_km, indexing="ij")
            lon0 = np.full_like(az_grid, radar_lon_cached, dtype=float)
            lat0 = np.full_like(az_grid, radar_lat_cached, dtype=float)
            lon_grid, lat_grid, _ = _geod.fwd(
                lon0, lat0, az_grid, rng_grid * 1000.0
            )
            backdrop = {
                "lon_grid": lon_grid,
                "lat_grid": lat_grid,
                "dbz": dbz,
                "scan_time": scan_time,
                "filename": scan.filename,
            }
        except Exception:
            pass

        if not chips:
            # Still age existing tracks with an empty update
            tm.update_scan(site, scan_time, [])
            per_scan.append({
                "idx": start_idx + i,
                "filename": scan.filename,
                "scan_time": scan_time,
                "status": "no_chips" if not cells else "no_valid_chips",
                "n_cells": len(cells),
                "n_raw": 0,
                "n_pending": 0,
                "n_confirmed": 0,
                "n_newly_confirmed": 0,
                "max_prob": None,
            })
            continue

        detection_results = pipe["predict_batch_detailed"](chips)

        candidates = []
        for cell, detection in zip(valid_cells, detection_results):
            decision = pipe["evaluate_detection"](cell, detection, conf_threshold)
            candidates.append(DetectionCandidate(
                site=site,
                scan_time=scan_time,
                cell=cell,
                detection_result=detection,
                quality_decision=decision,
                threshold=conf_threshold,
            ))

        updates = tm.update_scan(site, scan_time, candidates)

        for update in updates:
            if update.track is None:
                continue
            tid = update.track.track_id
            # Snapshot the full observation history each time so pruned
            # tracks keep their last-known state.
            obs_snapshot = []
            for obs in update.track.observations:
                if obs.missed or obs.x_km is None or obs.y_km is None:
                    obs_snapshot.append({
                        "scan_time": obs.scan_time,
                        "probability": obs.probability,
                        "above_threshold": obs.above_threshold,
                        "quality_pass": obs.quality_pass,
                        "quality_reasons": obs.quality_reasons,
                        "lat": None,
                        "lon": None,
                        "missed": True,
                    })
                    continue
                rng_km = math.hypot(obs.x_km, obs.y_km)
                az_deg = (math.degrees(math.atan2(obs.x_km, obs.y_km)) + 360.0) % 360.0
                obs_lon, obs_lat, _ = _geod.fwd(
                    radar_lon_cached, radar_lat_cached, az_deg, rng_km * 1000.0
                )
                obs_snapshot.append({
                    "scan_time": obs.scan_time,
                    "probability": obs.probability,
                    "above_threshold": obs.above_threshold,
                    "quality_pass": obs.quality_pass,
                    "quality_reasons": obs.quality_reasons,
                    "lat": float(obs_lat),
                    "lon": float(obs_lon),
                    "missed": False,
                })
            tracks_final[tid] = {
                "track_id": tid,
                "created_at": update.track.created_at,
                "confirmed_at": update.track.confirmed_at,
                "n_observations": update.track.observations_count,
                "max_prob": update.track.max_probability,
                "observations": obs_snapshot,
            }

        n_raw = sum(1 for c in candidates if c.above_threshold)
        n_pending = sum(1 for u in updates if u.status == "pending")
        n_newly = sum(1 for u in updates if u.newly_confirmed)
        n_confirmed = sum(
            1 for u in updates if u.status in ("confirmed", "newly_confirmed")
        )
        max_prob = max((c.probability for c in candidates), default=0.0)

        per_scan.append({
            "idx": start_idx + i,
            "filename": scan.filename,
            "scan_time": scan_time,
            "status": "processed",
            "n_cells": len(cells),
            "n_valid_cells": len(valid_cells),
            "n_raw": n_raw,
            "n_pending": n_pending,
            "n_confirmed": n_confirmed,
            "n_newly_confirmed": n_newly,
            "max_prob": max_prob,
        })

    tracks_list = sorted(
        tracks_final.values(),
        key=lambda t: (t["confirmed_at"] is None, -t["max_prob"]),
    )
    return {
        "per_scan": per_scan,
        "tracks": tracks_list,
        "config": config,
        "backdrop": backdrop,
        "radar_lat": radar_lat_cached,
        "radar_lon": radar_lon_cached,
        "site": site,
    }


# ---------------------------------------------------------------------------
# Step 1: Load radar data
# ---------------------------------------------------------------------------
st.header("Scan Source")
st.caption("Choose a radar volume that can be used by either workflow below.")
col_load, col_info = st.columns([2, 3])

with col_load:
    if input_mode == "Upload file":
        uploaded = st.file_uploader(
            "Upload NEXRAD Level-II file (.gz / .ar2v / V06)",
            type=None,
        )
        if uploaded and st.button("Parse uploaded file", type="primary"):
            with st.spinner("Parsing radar file..."):
                t0 = time.time()
                radar = _parse_uploaded_file(uploaded)
                parse_time = time.time() - t0

            # Auto-detect site from filename (e.g. "KLWX20240507_123456_V06")
            fname = uploaded.name.upper()
            detected_site = fname[:4] if len(fname) >= 4 and fname[:4].isalpha() else None
            if detected_site and detected_site in RADAR_SITES:
                st.session_state.detected_site = detected_site
            else:
                st.session_state.detected_site = None

            pipe = import_pipeline()
            st.session_state.radar = radar
            st.session_state.scan_time = pipe["get_scan_time"](radar)
            st.session_state.results = None
            st.session_state.timings = {"parse": parse_time}
            st.rerun()

    else:  # Fetch from S3
        if st.button("Fetch available scans", type="primary"):
            with st.spinner(f"Querying S3 for {site} on {scan_date}..."):
                pipe = import_pipeline()
                dt = datetime(scan_date.year, scan_date.month, scan_date.day, tzinfo=timezone.utc)
                scans = pipe["get_available_scans"](site, dt)
            if scans:
                st.session_state.available_scans = scans
                st.session_state.scan_filenames = [s.filename for s in scans]
                st.session_state.scan_labels = [
                    _format_scan_filename_for_display(s.filename) for s in scans
                ]
            else:
                st.warning(f"No scans found for {site} on {scan_date}")

        if "available_scans" in st.session_state and st.session_state.available_scans:
            scan_labels = st.session_state.get(
                "scan_labels",
                [
                    _format_scan_filename_for_display(s.filename)
                    for s in st.session_state.available_scans
                ],
            )
            scan_idx = st.selectbox(
                f"Select scan ({len(st.session_state.available_scans)} available)",
                range(len(st.session_state.available_scans)),
                index=len(st.session_state.available_scans) - 1,
                format_func=lambda i: scan_labels[i],
            )
            if st.button("Download & parse selected scan"):
                with st.spinner("Downloading and parsing..."):
                    try:
                        t0 = time.time()
                        radar, _ = _fetch_and_parse(site, scan_date, scan_idx)
                        parse_time = time.time() - t0
                    except OSError as e:
                        st.error(f"Failed to parse this scan: {e}\n\nTry selecting a different scan from the list — not all files use a compatible compression format.")
                        radar = None
                        parse_time = 0

                if radar is not None:
                    pipe = import_pipeline()
                    st.session_state.radar = radar
                    st.session_state.scan_time = pipe["get_scan_time"](radar)
                    st.session_state.results = None
                    st.session_state.timings = {"download_and_parse": parse_time}
                    st.rerun()

with col_info:
    if st.session_state.radar is not None:
        radar = st.session_state.radar
        st.subheader("Scan info")
        info_cols = st.columns(4)
        info_cols[0].metric("Scan time (UTC)", st.session_state.scan_time.strftime("%H:%M:%S"))
        info_cols[1].metric("Sweeps", radar.nsweeps)
        info_cols[2].metric("Fields", len(radar.fields))
        info_cols[3].metric("Rays", radar.nrays)

        with st.expander("Available fields"):
            st.write(list(radar.fields.keys()))

        if st.session_state.timings:
            timing_str = " | ".join(f"{k}: {v:.1f}s" for k, v in st.session_state.timings.items())
            st.caption(f"Timing: {timing_str}")

# ---------------------------------------------------------------------------
# Step 2: Run analysis
# ---------------------------------------------------------------------------
st.markdown("---")
st.header("Single-Frame Detection Capture")
st.caption(
    "Run TorNet on one selected radar frame. This section shows per-cell model "
    "scores and the exact single-frame detection points."
)
if st.session_state.radar is None:
    st.info("Load or parse a scan above to run single-frame detection.")
if st.session_state.radar is not None:
    if st.button("Run single-frame tornado detection", type="primary", use_container_width=True):
        radar = st.session_state.radar
        pipe = import_pipeline()
        timings = dict(st.session_state.timings)

        status = st.status("Running analysis...", expanded=True)

        # Load model
        status.write("Loading model...")
        t0 = time.time()
        get_model()
        timings["model_load"] = time.time() - t0

        # Use detected site from filename, or sidebar selection
        radar_site = _effective_site()

        # Identify cells
        scan_key = f"{radar_site}-{st.session_state.scan_time.strftime('%Y%m%d%H%M%S')}"
        status.write("Identifying storm cells...")
        t0 = time.time()
        cells = pipe["identify_cells"](
            radar, scan_key,
            dbz_threshold=dbz_threshold,
            min_gates=min_gates,
            shear_threshold=shear_threshold,
        )
        timings["cell_id"] = time.time() - t0
        status.write(f"Found **{len(cells)}** storm cells")

        if cells:
            # Extract chips
            status.write("Extracting radar chips...")
            t0 = time.time()
            chips = []
            valid_cells = []
            for cell in cells:
                chip = pipe["extract_chip"](radar, cell)
                if chip is not None:
                    chips.append(chip)
                    valid_cells.append(cell)
            timings["chip_extraction"] = time.time() - t0
            status.write(f"Extracted **{len(chips)}** valid chips")

            if chips:
                # Batch inference
                status.write("Running inference...")
                t0 = time.time()
                detections = pipe["predict_batch_detailed"](chips)
                timings["inference"] = time.time() - t0

                # Build results — use radar lat/lon for coordinate conversion,
                # falling back to RADAR_SITES config if metadata is zeroed
                from pyproj import Geod
                _geod = Geod(ellps="WGS84")
                _radar_lon = float(radar.longitude["data"][0])
                _radar_lat = float(radar.latitude["data"][0])
                if abs(_radar_lat) < 1.0 and abs(_radar_lon) < 1.0 and radar_site in RADAR_SITES:
                    _radar_lat = RADAR_SITES[radar_site]["lat"]
                    _radar_lon = RADAR_SITES[radar_site]["lon"]

                results = []
                for cell, detection in zip(valid_cells, detections):
                    prob = detection.probability
                    decision = pipe["evaluate_detection"](cell, detection, conf_threshold)
                    _lon, _lat, _ = _geod.fwd(_radar_lon, _radar_lat, cell.az_deg, cell.range_km * 1000)
                    lat, lon = float(_lat), float(_lon)
                    activation_lat = activation_lon = None
                    if detection.activation_az_deg is not None and detection.activation_range_km is not None:
                        _hit_lon, _hit_lat, _ = _geod.fwd(
                            _radar_lon,
                            _radar_lat,
                            detection.activation_az_deg,
                            detection.activation_range_km * 1000,
                        )
                        activation_lat, activation_lon = float(_hit_lat), float(_hit_lon)
                    results.append({
                        "cell_id": cell.cell_id,
                        "P(tornado)": prob,
                        "above_threshold": decision.above_threshold,
                        "quality_pass": decision.actionable,
                        "gated": decision.above_threshold and not decision.actionable,
                        "gate_reasons": "; ".join(decision.reasons),
                        "shear": round(cell.shear, 1),
                        "az_deg": round(cell.az_deg, 1),
                        "range_km": round(cell.range_km, 1),
                        "lat": round(lat, 4),
                        "lon": round(lon, 4),
                        "model_az_deg": None if detection.activation_az_deg is None else round(detection.activation_az_deg, 1),
                        "model_range_km": None if detection.activation_range_km is None else round(detection.activation_range_km, 1),
                        "model_lat": None if activation_lat is None else round(activation_lat, 4),
                        "model_lon": None if activation_lon is None else round(activation_lon, 4),
                        "edge_margin": None if detection.heatmap_edge_margin is None else round(detection.heatmap_edge_margin, 3),
                        "edge_max": detection.heatmap_edge_margin is not None and detection.heatmap_edge_margin < 0.10,
                        "dBZ_max": round(cell.dbz_max, 1),
                        "n_gates": cell.n_gates,
                    })
                results.sort(key=lambda r: r["P(tornado)"], reverse=True)

                st.session_state.results = results
                st.session_state.chips = chips
                st.session_state.cells = valid_cells
                st.session_state.timings = timings
                st.session_state.radar_site = radar_site
            else:
                st.session_state.results = []
        else:
            st.session_state.results = []
            st.session_state.cells = []
            st.session_state.chips = []

        timings["total"] = sum(v for v in timings.values())
        st.session_state.timings = timings
        status.update(label="Analysis complete", state="complete")

# ---------------------------------------------------------------------------
# Step 3: Results display
# ---------------------------------------------------------------------------
if st.session_state.results is not None:
    results = st.session_state.results
    timings = st.session_state.timings

    st.markdown("---")

    if not results:
        st.info("No storm cells detected. Try lowering the dBZ threshold or min gates.")
    else:
        # Summary metrics
        n_detections = sum(1 for r in results if _result_above_threshold(r))
        n_gated = sum(1 for r in results if _result_gated_at_current_threshold(r))
        max_prob = max(r["P(tornado)"] for r in results)
        st.subheader(f"Results: {len(results)} cells, {n_detections} raw over threshold")
        st.caption("The GUI shows single-scan scores; live actionable detections require temporal confirmation.")

        metric_cols = st.columns(5)
        metric_cols[0].metric("Total cells", len(results))
        metric_cols[1].metric("Raw over threshold", n_detections)
        metric_cols[2].metric("Max P(tornado)", f"{max_prob:.4f}")
        metric_cols[3].metric("Gated", n_gated)
        metric_cols[4].metric("Threshold", f"{conf_threshold:.2f}")

        # ---------------------------------------------------------------------------
        # Visualization: radar PPI + results table side by side
        # ---------------------------------------------------------------------------
        viz_col = st.container()
        table_col = st.container()

        with viz_col:
            st.subheader("Radar PPI — Reflectivity")
            radar = st.session_state.radar

            radar_site = _effective_site()

            try:
                import cartopy.crs as ccrs
                from pyproj import Geod
                from tornotify.ingest.nexrad import find_sweep_for_elevation, get_field
                import numpy as np

                _geod_plot = Geod(ellps="WGS84")
                radar_lat = float(radar.latitude["data"][0])
                radar_lon = float(radar.longitude["data"][0])

                # Fall back to RADAR_SITES config if metadata coords are
                # missing / zeroed (common in older NEXRAD files)
                if abs(radar_lat) < 1.0 and abs(radar_lon) < 1.0 and radar_site in RADAR_SITES:
                    radar_lat = RADAR_SITES[radar_site]["lat"]
                    radar_lon = RADAR_SITES[radar_site]["lon"]

                sweep_idx = find_sweep_for_elevation(radar, 0.5) or 0
                dbz = np.ma.filled(get_field(radar, "reflectivity", sweep_idx), np.nan)
                start_ray = radar.sweep_start_ray_index["data"][sweep_idx]
                end_ray = radar.sweep_end_ray_index["data"][sweep_idx] + 1
                azimuths = radar.azimuth["data"][start_ray:end_ray]
                ranges_km = radar.range["data"] / 1000.0

                # Build lon/lat grids for each gate
                az_grid, rng_grid = np.meshgrid(azimuths, ranges_km, indexing="ij")
                lon0 = np.full_like(az_grid, radar_lon, dtype=float)
                lat0 = np.full_like(az_grid, radar_lat, dtype=float)
                lon_grid, lat_grid, _ = _geod_plot.fwd(lon0, lat0, az_grid, rng_grid * 1000.0)

                projection = ccrs.LambertConformal(
                    central_latitude=radar_lat,
                    central_longitude=radar_lon,
                )
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection=projection)

                # Radar data overlay — semi-transparent so map shows through
                mesh = ax.pcolormesh(
                    lon_grid, lat_grid, dbz,
                    cmap="NWSRef", vmin=-10, vmax=75,
                    shading="auto",
                    transform=ccrs.PlateCarree(),
                    alpha=0.72, zorder=1,
                )
                cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
                cbar.set_label("Reflectivity (dBZ)")

                # Set map extent to ~300 km around radar
                from math import cos, radians
                extent_deg = 300.0 / 111.0
                lon_extent = extent_deg / cos(radians(radar_lat))
                ax.set_extent([
                    radar_lon - lon_extent, radar_lon + lon_extent,
                    radar_lat - extent_deg, radar_lat + extent_deg,
                ], crs=ccrs.PlateCarree())

                extent = ax.get_extent(ccrs.PlateCarree())
                _add_detailed_map_underlay(
                    ax,
                    extent=extent,
                    city_min_pop=8_000,
                    city_limit=110,
                    label_style="light",
                )

                # Radar site marker
                ax.plot(
                    radar_lon, radar_lat, "+", markersize=11, color="black",
                    transform=ccrs.PlateCarree(), zorder=5,
                )

                # Gridlines
                gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False

                ax.set_title(f"{radar_site} — {st.session_state.scan_time.strftime('%Y-%m-%d %H:%M UTC')}")

                # Overlay only cells at/above the confidence threshold so the
                # map stays readable. Gated candidates (over-threshold but
                # failed quality gates) are kept and drawn in orange.
                above_thresh_results = [
                    r for r in results if _result_above_threshold(r)
                ]
                detection_label_limit = 12
                if not above_thresh_results:
                    st.caption(
                        f"No cells at/above threshold {conf_threshold:.2f} — "
                        "nothing to plot on the map."
                    )
                elif len(above_thresh_results) > detection_label_limit:
                    st.caption(
                        f"Showing markers for all {len(above_thresh_results)} "
                        f"over-threshold cells; labeling the top {detection_label_limit} "
                        "by probability to keep the map readable."
                    )
                for i, r in enumerate(above_thresh_results):
                    prob = r["P(tornado)"]
                    hit_lon = r.get("model_lon") or r["lon"]
                    hit_lat = r.get("model_lat") or r["lat"]
                    edge_max = bool(r.get("edge_max"))
                    gated = _result_gated_at_current_threshold(r)

                    color = "#111111"
                    size = 58 if not gated else 50

                    if hit_lon != r["lon"] or hit_lat != r["lat"]:
                        ax.plot(
                            [r["lon"], hit_lon], [r["lat"], hit_lat],
                            color=color, linewidth=0.85, linestyle=":",
                            transform=ccrs.PlateCarree(), zorder=10,
                        )
                    ax.scatter(
                        hit_lon, hit_lat,
                        facecolors="none", marker="o", s=size + 18,
                        edgecolors="white", linewidths=2.2, zorder=11,
                        transform=ccrs.PlateCarree(),
                    )
                    ax.scatter(
                        hit_lon, hit_lat,
                        facecolors="none", marker="o", s=size,
                        edgecolors=color, linewidths=1.6, zorder=12,
                        transform=ccrs.PlateCarree(),
                    )
                    ax.scatter(
                        hit_lon, hit_lat,
                        c=color, marker=".", s=10,
                        linewidths=0, zorder=13,
                        transform=ccrs.PlateCarree(),
                    )
                    if i < detection_label_limit:
                        xytext = _label_offset(i)
                        ax.annotate(
                            f"{prob:.2f}" + (" G" if gated else " E" if edge_max else ""),
                            xy=(hit_lon, hit_lat),
                            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                            textcoords="offset points", xytext=xytext,
                            fontsize=6.5, fontweight="bold", color=color,
                            ha="left" if xytext[0] >= 0 else "right",
                            va="bottom" if xytext[1] >= 0 else "top",
                            bbox=dict(
                                boxstyle="round,pad=0.12",
                                fc="white", ec=color, alpha=0.88, lw=0.4,
                            ),
                            arrowprops=dict(
                                arrowstyle="-",
                                color=color,
                                lw=0.55,
                                alpha=0.9,
                                shrinkA=1,
                                shrinkB=2,
                            ),
                            zorder=13,
                        )

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.warning(f"PPI plot failed: {e}")

        with table_col:
            st.subheader("Cell detections")
            import pandas as pd

            df = pd.DataFrame(results)
            df["raw_above_threshold"] = df.apply(
                lambda row: float(row["P(tornado)"]) >= conf_threshold,
                axis=1,
            )
            df["gated"] = df.apply(
                lambda row: bool(_result_gate_reasons(row.to_dict()))
                and float(row["P(tornado)"]) >= conf_threshold,
                axis=1,
            )
            df["gate_reasons"] = df.apply(
                lambda row: "; ".join(_result_gate_reasons(row.to_dict()))
                if float(row["P(tornado)"]) >= conf_threshold else "",
                axis=1,
            )
            df["P(tornado)"] = df["P(tornado)"].map(lambda p: f"{p:.4f}")
            df = df.drop(columns=["above_threshold"], errors="ignore")

            def highlight_row(row):
                if row["raw_above_threshold"] and not row.get("gated"):
                    return ["background-color: #ff000030"] * len(row)
                if row.get("gated"):
                    return ["background-color: #ffa50030"] * len(row)
                return [""] * len(row)

            st.dataframe(
                df.style.apply(highlight_row, axis=1),
                use_container_width=True,
                height=min(400, 50 + 35 * len(df)),
            )

        # ---------------------------------------------------------------------------
        # Chip visualization for selected cell
        # ---------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("Chip inspector")

        if st.session_state.chips:
            pipe = import_pipeline()
            cell_options = [
                f"{r['cell_id']} — P={r['P(tornado)']} dBZ={r['dBZ_max']}"
                for r in results
            ]
            selected_idx = st.selectbox("Select cell to inspect", range(len(cell_options)),
                                        format_func=lambda i: cell_options[i])

            # Map results order back to chips order (results are sorted by prob)
            result = results[selected_idx]
            chip_idx = next(
                i for i, c in enumerate(st.session_state.cells)
                if c.cell_id == result["cell_id"]
            )
            chip = st.session_state.chips[chip_idx]

            var_names = pipe["ALL_VARIABLES"]
            chip_cols = st.columns(3)
            cmaps = {
                "DBZ": "NWSRef",
                "VEL": "NWSVel",
                "KDP": "RdBu_r",
                "RHOHV": "viridis",
                "ZDR": "RdYlBu_r",
                "WIDTH": "plasma",
            }

            for i, var in enumerate(var_names):
                with chip_cols[i % 3]:
                    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
                    for tilt in range(2):
                        data = chip[var][:, :, tilt]
                        cmap = cmaps.get(var, "viridis")
                        try:
                            axes[tilt].imshow(
                                data, aspect="auto", cmap=cmap,
                                origin="lower", interpolation="nearest",
                            )
                        except ValueError:
                            axes[tilt].imshow(
                                data, aspect="auto", cmap="viridis",
                                origin="lower", interpolation="nearest",
                            )
                        axes[tilt].set_title(f"Tilt {tilt}", fontsize=8)
                        axes[tilt].tick_params(labelsize=6)
                    fig.suptitle(var, fontsize=10, fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

    # ---------------------------------------------------------------------------
    # Timing breakdown
    # ---------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Performance")

    timing_cols = st.columns(len(timings))
    for i, (stage, secs) in enumerate(timings.items()):
        timing_cols[i].metric(stage.replace("_", " ").title(), f"{secs:.2f}s")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 2.5))
    stages = [k for k in timings if k != "total"]
    times = [timings[k] for k in stages]
    colors = ["#ff6b6b" if t == max(times) else "#4dabf7" for t in times]
    ax.barh([s.replace("_", " ").title() for s in stages], times, color=colors)
    ax.set_xlabel("Seconds")
    ax.set_title("Pipeline stage timing")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ---------------------------------------------------------------------------
    # Download results as CSV
    # ---------------------------------------------------------------------------
    if results:
        import pandas as pd
        csv_df = pd.DataFrame(results)
        csv_data = csv_df.to_csv(index=False)
        st.download_button(
            "Download results CSV",
            csv_data,
            file_name=f"tornotify_{st.session_state.get('radar_site', site)}_{scan_date}.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Forward tracking replay controls
# ---------------------------------------------------------------------------
st.markdown("---")
st.header("Forward Track Analysis")
st.caption(
    "Process a sequence of scans through the temporal tracker. This section "
    "separates valid tornado tracks anywhere in the replay from tracks still "
    "active on the latest scan."
)

if input_mode != "Fetch from S3":
    st.info("Forward track analysis uses fetched S3 scan sequences. Switch input mode to Fetch from S3 to run it.")
elif "available_scans" not in st.session_state or not st.session_state.available_scans:
    st.info("Fetch available scans first to run forward track analysis.")
else:
    scan_labels = st.session_state.get(
        "scan_labels",
        [
            _format_scan_filename_for_display(s.filename)
            for s in st.session_state.available_scans
        ],
    )
    n_available = len(st.session_state.available_scans)
    col_a, col_b = st.columns(2)
    default_start = max(0, n_available - 12)
    replay_start = col_a.number_input(
        "Start scan",
        min_value=0, max_value=n_available - 1,
        value=default_start, step=1,
        key="replay_start_idx",
        format="%d",
    )
    replay_end = col_b.number_input(
        "End scan",
        min_value=0, max_value=n_available - 1,
        value=n_available - 1, step=1,
        key="replay_end_idx",
        format="%d",
    )

    start_i = int(replay_start)
    end_i = int(replay_end)
    if end_i >= start_i:
        st.caption(
            f"Replay window: {scan_labels[start_i]} to "
            f"{scan_labels[end_i]} ({end_i - start_i + 1} scans)"
        )
    else:
        st.warning("End scan must be at or after start scan.")

    if st.button(
        "Run forward track analysis",
        type="primary",
        key="replay_run_btn",
        disabled=end_i < start_i,
        use_container_width=True,
    ):
        progress = st.progress(0.0, text="Starting replay...")
        status_line = st.empty()

        def _progress_cb(i, total, filename):
            scan_label = _format_scan_filename_for_display(filename)
            progress.progress(
                i / total if total else 1.0,
                text=f"Scan {i + 1}/{total}: {scan_label}",
            )
            status_line.caption(f"Processing {scan_label}...")

        get_model()

        replay = _run_tracking_replay(
            _effective_site(),
            st.session_state.available_scans,
            start_i,
            end_i,
            conf_threshold,
            dbz_threshold,
            min_gates,
            shear_threshold,
            progress_callback=_progress_cb,
        )
        progress.progress(1.0, text="Replay complete")
        status_line.empty()
        st.session_state.replay_results = replay
        st.rerun()

# ---------------------------------------------------------------------------
# Forward tracking replay results
# ---------------------------------------------------------------------------
def _track_last_observation(track):
    for obs in reversed(track.get("observations", [])):
        if not obs.get("missed") and obs.get("lat") is not None and obs.get("lon") is not None:
            return obs
    return None


def _track_observation_is_valid_hit(obs) -> bool:
    return (
        not obs.get("missed")
        and obs.get("above_threshold")
        and obs.get("quality_pass", True)
        and obs.get("lat") is not None
        and obs.get("lon") is not None
    )


def _track_has_valid_confirmation(track, config) -> bool:
    observations = track.get("observations", [])
    if len(observations) < config.confirm_scans:
        return False

    for start in range(0, len(observations) - config.confirm_window_scans + 1):
        window = observations[start:start + config.confirm_window_scans]
        hits = sum(1 for obs in window if _track_observation_is_valid_hit(obs))
        if hits >= config.confirm_scans:
            return True
    return False


def _track_has_recent_valid_confirmation(track, config) -> bool:
    recent = track.get("observations", [])[-config.confirm_window_scans:]
    return sum(1 for obs in recent if _track_observation_is_valid_hit(obs)) >= config.confirm_scans


def _track_is_likely_active(track, latest_scan_time, config) -> bool:
    if track.get("confirmed_at") is None or latest_scan_time is None:
        return False
    if not _track_has_valid_confirmation(track, config):
        return False
    last_obs = _track_last_observation(track)
    return (
        last_obs is not None
        and last_obs.get("scan_time") == latest_scan_time
        and _track_observation_is_valid_hit(last_obs)
    )


if st.session_state.get("replay_results"):
    replay = st.session_state.replay_results
    per_scan = replay["per_scan"]
    tracks = replay["tracks"]
    config = replay["config"]

    st.markdown("---")
    st.subheader("Track Analysis Results")
    st.caption(
        f"confirm {config.confirm_scans}-of-{config.confirm_window_scans} scans · "
        f"match_base {config.match_base_km:.0f} km · "
        f"max_speed {config.max_speed_kmh:.0f} km/h · "
        f"max_missed {config.max_missed_scans}"
    )
    if abs(config.threshold - conf_threshold) > 1e-9:
        st.warning(
            f"This replay was run at threshold {config.threshold:.2f}. "
            f"The current slider is {conf_threshold:.2f}; run the replay again "
            "to rebuild tracks at the new threshold."
        )

    n_confirmed_tracks = sum(1 for t in tracks if t["confirmed_at"] is not None)
    latest_scan_time = replay.get("backdrop", {}).get("scan_time") if replay.get("backdrop") else None
    valid_tornado_tracks = [
        t for t in tracks
        if t.get("confirmed_at") is not None and _track_has_valid_confirmation(t, config)
    ]
    likely_active_tracks = [
        t for t in tracks
        if _track_is_likely_active(t, latest_scan_time, config)
    ]
    stale_valid_tracks = [
        t for t in valid_tornado_tracks
        if not _track_is_likely_active(t, latest_scan_time, config)
    ]
    n_processed = sum(1 for s in per_scan if s["status"] == "processed")
    total_raw_hits = sum(s.get("n_raw", 0) or 0 for s in per_scan)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Scans processed", n_processed, f"of {len(per_scan)}")
    metric_cols[1].metric("Raw model candidates", total_raw_hits)
    metric_cols[2].metric("Tracks created", len(tracks))
    metric_cols[3].metric("Valid tornado tracks", len(valid_tornado_tracks))
    metric_cols[4].metric("Active on latest scan", len(likely_active_tracks))

    if total_raw_hits and not valid_tornado_tracks:
        st.info(
            f"{total_raw_hits} raw model candidates were seen, but none became "
            "quality-passing tornado tracks during this replay."
        )
    elif valid_tornado_tracks:
        st.success(
            f"{len(valid_tornado_tracks)} valid tornado track(s) likely occurred "
            "during this replay."
        )
    if stale_valid_tracks:
        st.caption(
            f"{len(stale_valid_tracks)} valid track(s) are not active on the latest "
            "scan, but remain shown because they likely occurred earlier in the replay."
        )
    if n_confirmed_tracks > len(valid_tornado_tracks):
        st.caption(
            f"{n_confirmed_tracks - len(valid_tornado_tracks)} persistent model "
            "track(s) are not counted as valid tornado tracks because they failed "
            "quality gates."
        )

    # -----------------------------------------------------------------------
    # Track-path overlay map: valid active tracks at the latest backdrop scan
    # -----------------------------------------------------------------------
    backdrop = replay.get("backdrop")
    radar_lat = replay.get("radar_lat")
    radar_lon = replay.get("radar_lon")

    st.markdown("**Valid tornado track paths**")
    if backdrop is None or radar_lat is None:
        st.caption("No radar backdrop available (all scans failed to parse).")
    else:
        try:
            import cartopy.crs as ccrs
            from math import cos, radians

            projection = ccrs.LambertConformal(
                central_latitude=radar_lat,
                central_longitude=radar_lon,
            )
            fig_tr = plt.figure(figsize=(12, 10))
            ax_tr = fig_tr.add_subplot(111, projection=projection)

            extent_deg = 220.0 / 111.0
            lon_extent = extent_deg / cos(radians(radar_lat))
            ax_tr.set_extent([
                radar_lon - lon_extent, radar_lon + lon_extent,
                radar_lat - extent_deg, radar_lat + extent_deg,
            ], crs=ccrs.PlateCarree())

            extent = ax_tr.get_extent(ccrs.PlateCarree())
            _add_detailed_map_underlay(
                ax_tr,
                extent=extent,
                city_min_pop=5_000,
                city_limit=130,
                label_style="dark",
            )

            # Radar site marker
            ax_tr.plot(
                radar_lon, radar_lat, "+", markersize=8, color="#333333",
                transform=ccrs.PlateCarree(), zorder=5,
            )

            # Draw each valid track as a line connecting its observations
            # through time. Keep the map radar-free and labels sparse so
            # overlapping tracks remain readable.
            import matplotlib.patheffects as patheffects
            track_line_stroke = [
                patheffects.Stroke(linewidth=3.4, foreground="white"),
                patheffects.Normal(),
            ]
            track_color = "#111111"
            track_label_limit = 12

            if len(valid_tornado_tracks) > track_label_limit:
                st.caption(
                    f"Showing all {len(valid_tornado_tracks)} valid tracks; "
                    f"labeling the top {track_label_limit} by probability to reduce overlap."
                )

            for track_idx, t in enumerate(valid_tornado_tracks):
                obs_pts = [
                    (o["lon"], o["lat"], o["probability"])
                    for o in t.get("observations", [])
                    if not o.get("missed") and o.get("lat") is not None
                ]
                if not obs_pts:
                    continue

                lons = [p[0] for p in obs_pts]
                lats = [p[1] for p in obs_pts]

                # Trail line — white with black outline, visible on any backdrop
                line_objs = ax_tr.plot(
                    lons, lats,
                    color=track_color, linewidth=1.8,
                    transform=ccrs.PlateCarree(), zorder=6,
                    solid_capstyle="round",
                )
                for line_obj in line_objs:
                    line_obj.set_path_effects(track_line_stroke)

                # Intermediate observation dots — small white with black edge
                if len(lons) > 2:
                    ax_tr.scatter(
                        lons[1:-1], lats[1:-1],
                        c=track_color, s=7, edgecolors="white",
                        linewidths=0.6, zorder=7,
                        transform=ccrs.PlateCarree(),
                    )

                # START marker: small gold circle
                ax_tr.scatter(
                    lons[0], lats[0],
                    facecolors="white", marker="o", s=24,
                    edgecolors=track_color, linewidths=0.75, zorder=9,
                    transform=ccrs.PlateCarree(),
                )

                # Direction arrow between the last two observations — black
                if len(lons) >= 2:
                    ax_tr.annotate(
                        "",
                        xy=(lons[-1], lats[-1]),
                        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax_tr),
                        xytext=(lons[-2], lats[-2]),
                        textcoords=ccrs.PlateCarree()._as_mpl_transform(ax_tr),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=track_color,
                            lw=1.2,
                            mutation_scale=10,
                        ),
                        zorder=9,
                    )

                # END marker: small red X
                ax_tr.scatter(
                    lons[-1], lats[-1],
                    c=track_color, marker="X", s=32,
                    edgecolors="#333333", linewidths=0.55,
                    zorder=10,
                    transform=ccrs.PlateCarree(),
                )

                # Label only the latest track head to avoid overcrowding.
                if track_idx < track_label_limit:
                    xytext = _label_offset(track_idx)
                    ax_tr.annotate(
                        f"P={t['max_prob']:.2f}",
                        xy=(lons[-1], lats[-1]),
                        xycoords=ccrs.PlateCarree()._as_mpl_transform(ax_tr),
                        textcoords="offset points", xytext=xytext,
                        fontsize=6.2, fontweight="bold", color="#111111",
                        ha="left" if xytext[0] >= 0 else "right",
                        va="bottom" if xytext[1] >= 0 else "top",
                        bbox=dict(
                            boxstyle="round,pad=0.10",
                            fc="white", ec="#555555",
                            alpha=0.72, lw=0.3,
                        ),
                        arrowprops=dict(
                            arrowstyle="-",
                            color="#555555",
                            lw=0.45,
                            alpha=0.7,
                            shrinkA=1,
                            shrinkB=2,
                        ),
                        zorder=11,
                    )

            # Legend explaining start/end markers
            if valid_tornado_tracks:
                from matplotlib.lines import Line2D
                legend_handles = [
                    Line2D(
                        [0], [0], marker="o", color="none",
                        markerfacecolor="white", markeredgecolor=track_color,
                        markersize=4.5, markeredgewidth=0.6,
                        label="track start",
                    ),
                    Line2D(
                        [0], [0], marker="X", color="none",
                        markerfacecolor=track_color, markeredgecolor=track_color,
                        markersize=5, markeredgewidth=0.6,
                        label="track end (latest)",
                    ),
                    Line2D(
                        [0], [0], color=track_color, linewidth=1.8,
                        label="track path",
                    ),
                ]
                ax_tr.legend(
                    handles=legend_handles, loc="upper left",
                    framealpha=0.9, fontsize=8,
                )

            gl_tr = ax_tr.gridlines(
                draw_labels=True, linewidth=0.4, color="gray", alpha=0.5,
            )
            gl_tr.top_labels = False
            gl_tr.right_labels = False

            backdrop_time = backdrop["scan_time"].strftime("%H:%M UTC") if backdrop.get("scan_time") else ""
            ax_tr.set_title(
                f"{replay.get('site', '')} — Forward tracking replay\n"
                f"backdrop: {backdrop_time} · "
                f"{len(valid_tornado_tracks)} valid tornado tracks · "
                f"{len(likely_active_tracks)} active on latest scan"
            )
            st.pyplot(fig_tr, use_container_width=True)
            plt.close(fig_tr)

            if not valid_tornado_tracks:
                st.caption(
                    "No valid tornado tracks to draw — only the backdrop is shown."
                )
        except Exception as exc:
            st.warning(f"Track path plot failed: {exc}")

    import pandas as pd

    st.markdown("**Per-scan summary**")
    scan_rows = []
    for s in per_scan:
        filename = s.get("filename", "")
        scan_rows.append({
            "idx": s.get("idx"),
            "scan_time": (
                s["scan_time"].strftime("%H:%M:%S")
                if s.get("scan_time") is not None else ""
            ),
            "scan": _format_scan_filename_for_display(filename),
            "status": s.get("status", ""),
            "cells": s.get("n_cells", 0),
            "raw_hits": s.get("n_raw", 0),
            "max_prob": (
                f"{s.get('max_prob'):.3f}"
                if s.get("max_prob") is not None else ""
            ),
            "pending": s.get("n_pending", 0),
            "confirmed": s.get("n_confirmed", 0),
            "newly_confirmed": s.get("n_newly_confirmed", 0),
        })
    scan_df = pd.DataFrame(scan_rows)

    def _scan_row_highlight(row):
        if row["newly_confirmed"]:
            return ["background-color: #ff000055"] * len(row)
        if row["confirmed"]:
            return ["background-color: #ff000025"] * len(row)
        if row["raw_hits"]:
            return ["background-color: #ffa50025"] * len(row)
        return [""] * len(row)

    st.dataframe(
        scan_df.style.apply(_scan_row_highlight, axis=1),
        use_container_width=True,
        height=min(420, 50 + 35 * len(scan_df)),
    )

    st.markdown("**Track summary**")
    if tracks:
        track_rows = []
        for t in tracks:
            if _track_is_likely_active(t, latest_scan_time, config):
                status = "valid - active latest scan"
            elif t["confirmed_at"] and _track_has_valid_confirmation(t, config):
                status = "valid - earlier in replay"
            elif t["confirmed_at"]:
                status = "persistent model track"
            else:
                status = "pending/stale"
            track_rows.append({
                "track_id": t["track_id"],
                "status": status,
                "created_at": (
                    t["created_at"].strftime("%H:%M:%S")
                    if t["created_at"] else ""
                ),
                "confirmed_at": (
                    t["confirmed_at"].strftime("%H:%M:%S")
                    if t["confirmed_at"] else "—"
                ),
                "n_observations": t["n_observations"],
                "max_prob": f"{t['max_prob']:.4f}",
            })
        tracks_df = pd.DataFrame(track_rows)

        def _track_row_highlight(row):
            if row["status"] == "valid - active latest scan":
                return ["background-color: #ff000040"] * len(row)
            if row["status"] == "valid - earlier in replay":
                return ["background-color: #ff000025"] * len(row)
            if row["status"] == "persistent model track":
                return ["background-color: #ffa50025"] * len(row)
            return [""] * len(row)

        st.dataframe(
            tracks_df.style.apply(_track_row_highlight, axis=1),
            use_container_width=True,
            height=min(420, 50 + 35 * len(tracks_df)),
        )

        csv_tracks = tracks_df.to_csv(index=False)
        st.download_button(
            "Download track summary CSV",
            csv_tracks,
            file_name=f"tornotify_tracks_{_effective_site()}_{scan_date}.csv",
            mime="text/csv",
            key="download_tracks_csv",
        )
    else:
        st.info("No tracks were created during replay (no over-threshold candidates).")

    if st.button("Clear replay results", key="clear_replay"):
        st.session_state.replay_results = None
        st.rerun()
