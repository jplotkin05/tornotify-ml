"""
Radar image artifacts for model analysis.
"""
import logging
import threading
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyart
from pyproj import Geod

from tornotify.ingest.nexrad import find_sweep_for_elevation, get_field
from tornotify.preprocess.cells import StormCell

logger = logging.getLogger(__name__)

_GEOD = Geod(ellps="WGS84")
_plot_lock = threading.Lock()


def save_marked_radar_image(
    radar: pyart.core.Radar,
    site: str,
    scan_time,
    cells: list[StormCell],
    probabilities: list[float],
    detection_results: list[Any] | None = None,
    detection_decisions: list[Any] | None = None,
    output_dir: str | Path = "data/radar_images",
    threshold: float = 0.6,
    elevation_deg: float = 0.5,
    max_range_km: float = 300.0,
) -> str:
    """
    Save a reflectivity image with detected cell centroids and probabilities.

    The primary artifact is georeferenced with a Cartopy map underlay. If map
    rendering fails for any reason, the function falls back to a radar-relative
    kilometer plot so inference logging can continue. If detector heatmap
    results are provided, the model's strongest sub-cell activation point is
    drawn separately from the reflectivity centroid.
    """
    if not cells or not probabilities:
        raise ValueError("At least one cell/probability pair is required")

    sweep_idx = find_sweep_for_elevation(radar, elevation_deg)
    if sweep_idx is None:
        raise ValueError(f"No sweep found near {elevation_deg:.1f} degrees")
    if "reflectivity" not in radar.fields:
        raise ValueError("Radar scan does not include reflectivity")

    scan_time_utc = scan_time.astimezone(timezone.utc) if scan_time.tzinfo else scan_time
    output_path = _radar_image_path(output_dir, site, scan_time_utc)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_data = _radar_plot_data(radar, sweep_idx, cells, max_range_km)

    with _plot_lock:
        try:
            _save_map_image(
                output_path=output_path,
                radar=radar,
                site=site,
                scan_time_utc=scan_time_utc,
                cells=cells,
                probabilities=probabilities,
                detection_results=detection_results,
                detection_decisions=detection_decisions,
                threshold=threshold,
                elevation_deg=elevation_deg,
                plot_data=plot_data,
            )
        except Exception as exc:
            logger.warning("Map radar image failed; falling back to radar-relative image: %s", exc)
            _save_relative_image(
                output_path=output_path,
                site=site,
                scan_time_utc=scan_time_utc,
                cells=cells,
                probabilities=probabilities,
                detection_results=detection_results,
                detection_decisions=detection_decisions,
                threshold=threshold,
                plot_data=plot_data,
            )

    logger.info("Saved marked radar image: %s", output_path)
    return str(output_path)


def _radar_plot_data(
    radar: pyart.core.Radar,
    sweep_idx: int,
    cells: list[StormCell],
    max_range_km: float,
) -> dict[str, np.ndarray | float]:
    dbz = np.ma.filled(get_field(radar, "reflectivity", sweep_idx), np.nan)
    start_ray = radar.sweep_start_ray_index["data"][sweep_idx]
    end_ray = radar.sweep_end_ray_index["data"][sweep_idx] + 1
    azimuths = radar.azimuth["data"][start_ray:end_ray]
    ranges_km = radar.range["data"] / 1000.0

    farthest_cell_km = max((cell.range_km for cell in cells), default=0.0)
    plot_range_km = min(float(ranges_km[-1]), max(max_range_km, farthest_cell_km + 40.0))
    range_mask = ranges_km <= plot_range_km
    ranges_plot = ranges_km[range_mask]
    dbz_plot = dbz[:, range_mask]

    az_rad = np.deg2rad(azimuths)
    range_grid_km, az_grid_rad = np.meshgrid(ranges_plot, az_rad)
    x_grid = range_grid_km * np.sin(az_grid_rad)
    y_grid = range_grid_km * np.cos(az_grid_rad)

    return {
        "dbz": dbz_plot,
        "azimuths": azimuths,
        "ranges_km": ranges_plot,
        "range_grid_km": range_grid_km,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "plot_range_km": plot_range_km,
    }


def _save_map_image(
    output_path: Path,
    radar: pyart.core.Radar,
    site: str,
    scan_time_utc,
    cells: list[StormCell],
    probabilities: list[float],
    detection_results: list[Any] | None,
    detection_decisions: list[Any] | None,
    threshold: float,
    elevation_deg: float,
    plot_data: dict[str, np.ndarray | float],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    radar_lat = float(radar.latitude["data"][0])
    radar_lon = float(radar.longitude["data"][0])
    lon_grid, lat_grid = _gate_lonlat(
        radar_lat=radar_lat,
        radar_lon=radar_lon,
        azimuths=plot_data["azimuths"],
        ranges_km=plot_data["ranges_km"],
    )

    projection = ccrs.LambertConformal(
        central_latitude=radar_lat,
        central_longitude=radar_lon,
    )
    fig = plt.figure(figsize=(10, 8), dpi=140)
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f4f0e8", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d8eef7", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#d8eef7", edgecolor="#8aa6b1", linewidth=0.4, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#555555", linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#555555", linewidth=0.6, zorder=2)
    _add_county_boundaries(ax, ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lon_grid,
        lat_grid,
        plot_data["dbz"],
        cmap="turbo",
        vmin=-10,
        vmax=75,
        shading="auto",
        transform=ccrs.PlateCarree(),
        alpha=0.72,
        zorder=3,
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label("Reflectivity (dBZ)")

    detection_results = _normalize_detection_results(detection_results, len(cells))
    detection_decisions = _normalize_detection_results(detection_decisions, len(cells))
    for cell, prob, result, decision in zip(cells, probabilities, detection_results, detection_decisions):
        lon, lat, _ = _GEOD.fwd(radar_lon, radar_lat, cell.az_deg, cell.range_km * 1000.0)
        _draw_map_cell(ax, lon, lat, cell, prob, threshold, ccrs.PlateCarree(), decision)
        _draw_map_activation(
            ax,
            radar_lon=radar_lon,
            radar_lat=radar_lat,
            centroid_lon=lon,
            centroid_lat=lat,
            detection_result=result,
            transform=ccrs.PlateCarree(),
        )

    ax.plot(
        radar_lon,
        radar_lat,
        marker="+",
        markersize=11,
        color="black",
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    _set_map_extent(ax, lon_grid, lat_grid, ccrs.PlateCarree())
    _add_city_labels(ax, ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.55)
    gl.top_labels = False
    gl.right_labels = False

    title_time = scan_time_utc.strftime("%Y-%m-%d %H:%M:%SZ")
    ax.set_title(f"{site} Reflectivity with TorNotify Cells\n{title_time} | tilt={elevation_deg:.1f} deg")
    ax.text(
        0.01,
        0.01,
        f"threshold={threshold:.2f} | cells={len(cells)}",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "linewidth": 0},
        zorder=6,
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_relative_image(
    output_path: Path,
    site: str,
    scan_time_utc,
    cells: list[StormCell],
    probabilities: list[float],
    detection_results: list[Any] | None,
    detection_decisions: list[Any] | None,
    threshold: float,
    plot_data: dict[str, np.ndarray | float],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8), dpi=140)
    mesh = ax.pcolormesh(
        plot_data["x_grid"],
        plot_data["y_grid"],
        plot_data["dbz"],
        cmap="turbo",
        vmin=-10,
        vmax=75,
        shading="auto",
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
    cbar.set_label("Reflectivity (dBZ)")

    detection_results = _normalize_detection_results(detection_results, len(cells))
    detection_decisions = _normalize_detection_results(detection_decisions, len(cells))
    for cell, prob, result, decision in zip(cells, probabilities, detection_results, detection_decisions):
        az = np.deg2rad(cell.az_deg)
        x = cell.range_km * np.sin(az)
        y = cell.range_km * np.cos(az)
        _draw_relative_cell(ax, x, y, cell, prob, threshold, decision)
        _draw_relative_activation(ax, x, y, result)

    title_time = scan_time_utc.strftime("%Y-%m-%d %H:%M:%SZ")
    plot_range_km = plot_data["plot_range_km"]
    ax.set_title(f"{site} Reflectivity with TorNotify Cells\n{title_time}")
    ax.set_xlabel("East of radar (km)")
    ax.set_ylabel("North of radar (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-plot_range_km, plot_range_km)
    ax.set_ylim(-plot_range_km, plot_range_km)
    ax.grid(True, color="black", alpha=0.16, linewidth=0.5)
    ax.text(
        0.01,
        0.01,
        f"threshold={threshold:.2f} | cells={len(cells)}",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "linewidth": 0},
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _gate_lonlat(
    radar_lat: float,
    radar_lon: float,
    azimuths: np.ndarray,
    ranges_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    az_grid, range_grid_km = np.meshgrid(azimuths, ranges_km, indexing="ij")
    lon0 = np.full_like(az_grid, radar_lon, dtype=float)
    lat0 = np.full_like(az_grid, radar_lat, dtype=float)
    lon_grid, lat_grid, _ = _GEOD.fwd(lon0, lat0, az_grid, range_grid_km * 1000.0)
    return lon_grid, lat_grid


def _draw_map_cell(ax, lon, lat, cell, prob, threshold, transform, decision=None) -> None:
    above = _decision_above_threshold(decision, prob, threshold)
    actionable = _decision_actionable(decision, prob, threshold)
    gated = above and not actionable
    color = "#d62728" if actionable else "#ff9f1c" if gated else "#ffd23f"
    marker = "*" if actionable else "X" if gated else "o"
    size = 180 if actionable else 115 if gated else 90
    label = cell.cell_id.rsplit("-", 1)[-1]

    ax.scatter(
        lon,
        lat,
        s=size,
        marker=marker,
        c=color,
        edgecolors="black",
        linewidths=0.9,
        transform=transform,
        zorder=5,
    )
    ax.text(
        lon,
        lat,
        f" {label}\n P={_format_probability(prob)}" + ("\n gated" if gated else ""),
        fontsize=8,
        color="black",
        transform=transform,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.84, "linewidth": 0},
        zorder=6,
    )


def _draw_map_activation(
    ax,
    radar_lon: float,
    radar_lat: float,
    centroid_lon: float,
    centroid_lat: float,
    detection_result,
    transform,
) -> None:
    az_deg = getattr(detection_result, "activation_az_deg", None)
    range_km = getattr(detection_result, "activation_range_km", None)
    if az_deg is None or range_km is None:
        return

    lon, lat, _ = _GEOD.fwd(radar_lon, radar_lat, az_deg, range_km * 1000.0)
    edge_margin = getattr(detection_result, "heatmap_edge_margin", None)
    near_edge = edge_margin is not None and edge_margin < 0.10
    color = "#b000b5" if near_edge else "#0057d8"

    ax.plot(
        [centroid_lon, lon],
        [centroid_lat, lat],
        color=color,
        linewidth=0.9,
        linestyle=":",
        transform=transform,
        zorder=6,
    )
    ax.scatter(
        lon,
        lat,
        s=70,
        marker="D",
        c=color,
        edgecolors="white",
        linewidths=0.8,
        transform=transform,
        zorder=7,
    )
    if near_edge:
        ax.text(
            lon,
            lat,
            " edge max",
            fontsize=7,
            color=color,
            transform=transform,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.75, "linewidth": 0},
            zorder=8,
        )


def _draw_relative_cell(ax, x, y, cell, prob, threshold, decision=None) -> None:
    above = _decision_above_threshold(decision, prob, threshold)
    actionable = _decision_actionable(decision, prob, threshold)
    gated = above and not actionable
    color = "#d62728" if actionable else "#ff9f1c" if gated else "#ffd23f"
    marker = "*" if actionable else "X" if gated else "o"
    size = 180 if actionable else 115 if gated else 90
    label = cell.cell_id.rsplit("-", 1)[-1]

    ax.scatter(
        x,
        y,
        s=size,
        marker=marker,
        c=color,
        edgecolors="black",
        linewidths=0.9,
        zorder=4,
    )
    ax.text(
        x + 4,
        y + 4,
        f"{label}\nP={_format_probability(prob)}" + ("\ngated" if gated else ""),
        fontsize=8,
        color="black",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.82, "linewidth": 0},
        zorder=5,
    )


def _draw_relative_activation(ax, centroid_x, centroid_y, detection_result) -> None:
    az_deg = getattr(detection_result, "activation_az_deg", None)
    range_km = getattr(detection_result, "activation_range_km", None)
    if az_deg is None or range_km is None:
        return

    az = np.deg2rad(az_deg)
    x = range_km * np.sin(az)
    y = range_km * np.cos(az)
    edge_margin = getattr(detection_result, "heatmap_edge_margin", None)
    near_edge = edge_margin is not None and edge_margin < 0.10
    color = "#b000b5" if near_edge else "#0057d8"
    ax.plot([centroid_x, x], [centroid_y, y], color=color, linewidth=0.9, linestyle=":", zorder=5)
    ax.scatter(x, y, s=70, marker="D", c=color, edgecolors="white", linewidths=0.8, zorder=6)
    if near_edge:
        ax.text(
            x + 4,
            y + 4,
            "edge max",
            fontsize=7,
            color=color,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.75, "linewidth": 0},
            zorder=7,
        )


def _set_map_extent(ax, lon_grid, lat_grid, transform) -> None:
    lon_min, lon_max = np.nanpercentile(lon_grid, [0.1, 99.9])
    lat_min, lat_max = np.nanpercentile(lat_grid, [0.1, 99.9])
    lon_pad = max((lon_max - lon_min) * 0.05, 0.1)
    lat_pad = max((lat_max - lat_min) * 0.05, 0.1)
    ax.set_extent(
        [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad],
        crs=transform,
    )


def _normalize_detection_results(detection_results: list[Any] | None, count: int) -> list[Any]:
    if detection_results is None:
        return [None] * count
    if len(detection_results) < count:
        return list(detection_results) + [None] * (count - len(detection_results))
    return list(detection_results[:count])


def _decision_above_threshold(decision, probability: float, threshold: float) -> bool:
    if decision is None:
        return probability >= threshold
    return bool(getattr(decision, "above_threshold", probability >= threshold))


def _decision_actionable(decision, probability: float, threshold: float) -> bool:
    if decision is None:
        return probability >= threshold
    return bool(getattr(decision, "actionable", probability >= threshold))


def _add_county_boundaries(ax, transform) -> None:
    try:
        import cartopy.feature as cfeature

        counties = cfeature.NaturalEarthFeature(
            "cultural",
            "admin_2_counties",
            "10m",
            facecolor="none",
            edgecolor="#777777",
            linewidth=0.25,
        )
        ax.add_feature(counties, zorder=2)
    except Exception as exc:
        logger.debug("County boundary overlay skipped: %s", exc)


def _add_city_labels(ax, transform, min_population: int = 50_000, max_labels: int = 30) -> None:
    try:
        from cartopy.io.shapereader import Reader, natural_earth

        lon_min, lon_max, lat_min, lat_max = ax.get_extent(transform)
        places_shp = natural_earth(
            resolution="10m",
            category="cultural",
            name="populated_places",
        )

        candidates = []
        for rec in Reader(places_shp).records():
            lon, lat = rec.geometry.x, rec.geometry.y
            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                continue
            population = rec.attributes.get("POP_MAX") or 0
            if population < min_population:
                continue
            candidates.append((population, lon, lat, rec.attributes.get("NAME", "")))

        for _population, lon, lat, name in sorted(candidates, reverse=True)[:max_labels]:
            ax.plot(
                lon,
                lat,
                marker="o",
                markersize=2.5,
                color="#202020",
                transform=transform,
                zorder=6,
            )
            ax.text(
                lon,
                lat,
                f" {name}",
                fontsize=7,
                color="#202020",
                transform=transform,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.68, "linewidth": 0},
                zorder=6,
            )
    except Exception as exc:
        logger.debug("City label overlay skipped: %s", exc)


def _radar_image_path(output_dir: str | Path, site: str, scan_time) -> Path:
    stamp = scan_time.strftime("%Y%m%dT%H%M%SZ")
    return Path(output_dir) / f"{site}_{stamp}_cells.png"


def _format_probability(probability: float) -> str:
    return f"{probability:.4f}" if 0 < probability < 0.001 else f"{probability:.3f}"
