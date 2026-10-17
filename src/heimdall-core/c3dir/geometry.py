"""GOES-19 line-of-sight geometry, from the ABI fixed grid navigation equations in the GOES-R
Product User Guide (section 4.2.8). numpy only, so the model side can use it without the
download libraries.

positions are in the satellite frame used there: sx toward earth's center, sy east-west, sz
north-south, all in meters. a point at height h is handled by inflating the ellipsoid by h,
which is far more accurate than a 2 km pixel at cloud heights.

p is always the attributes of goes_imager_projection.
"""
import numpy as np
from pyproj import Proj


def goes_proj(ds):
    """pyproj projection for the goes grid (x/y in meters = scan angle * satellite height)."""
    p = ds["goes_imager_projection"].attrs
    return Proj(proj="geos", h=p["perspective_point_height"],
                lon_0=p["longitude_of_projection_origin"], sweep=p["sweep_angle_axis"],
                a=p["semi_major_axis"], b=p["semi_minor_axis"])


def view_circle(lon0, radius_deg, n=72):
    """lat / lon of n points on the circle radius_deg (earth-center angle) around the point
    under a geostationary satellite at lon0, counter-clockwise."""
    azimuth = np.deg2rad(np.linspace(360, 0, n, endpoint=False))
    d = np.deg2rad(radius_deg)
    lat = np.rad2deg(np.arcsin(np.sin(d) * np.cos(azimuth)))
    lon = lon0 + np.rad2deg(np.arctan2(np.sin(azimuth) * np.sin(d), np.cos(d)))
    return lat, lon


def within_view(lat, lon, lon0, radius_deg):
    """which points are within radius_deg (earth-center angle) of the point under a geostationary
    satellite at lon0."""
    cos_d = np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon - lon0))
    return np.isfinite(cos_d) & (cos_d >= np.cos(np.deg2rad(radius_deg)))


def _ellipsoid(p, height_m=0.0):
    """(a, b, H, lon0): semi-axes inflated by height_m, satellite distance from earth's center, origin (rad)."""
    return (p["semi_major_axis"] + height_m, p["semi_minor_axis"] + height_m,
            p["perspective_point_height"] + p["semi_major_axis"],
            np.deg2rad(p["longitude_of_projection_origin"]))


def satellite_frame(lat, lon, height_m, p):
    """satellite-frame position (sx, sy, sz) of points at geodetic lat/lon (degrees) and height_m."""
    a, b, H, lon0 = _ellipsoid(p)
    lat, dlon = np.deg2rad(lat), np.deg2rad(lon) - lon0
    e2 = 1 - (b / a) ** 2
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)  # prime vertical radius of curvature
    horizontal = (n + height_m) * np.cos(lat)
    return (H - horizontal * np.cos(dlon), -horizontal * np.sin(dlon),
            (n * (1 - e2) + height_m) * np.sin(lat))


def scan_angles(lat, lon, height_m, p):
    """goes scan angles (x, y) in radians at which a point at lat/lon/height_m is seen.

    nan where the earth blocks the view. at height_m = 0 this matches goes_proj.
    """
    sx, sy, sz = satellite_frame(lat, lon, height_m, p)
    a, b, H, _ = _ellipsoid(p, height_m)
    hidden = H * (H - sx) < sy ** 2 + (a / b) ** 2 * sz ** 2  # behind the limb
    x = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
    y = np.arctan(sz / sx)
    return np.where(hidden, np.nan, x), np.where(hidden, np.nan, y)


def ray_point(x, y, height_m, p):
    """satellite-frame position (sx, sy, sz) where the line of sight at scan angles x/y crosses height_m.

    nan where the line of sight misses the earth.
    """
    a, b, H, _ = _ellipsoid(p, height_m)
    qa = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (a / b) ** 2 * np.sin(y) ** 2)
    qb = -2 * H * np.cos(x) * np.cos(y)
    qc = H ** 2 - a ** 2
    disc = qb ** 2 - 4 * qa * qc
    rs = np.where(disc >= 0, (-qb - np.sqrt(np.maximum(disc, 0))) / (2 * qa), np.nan)  # near side
    return rs * np.cos(x) * np.cos(y), -rs * np.sin(x), rs * np.cos(x) * np.sin(y)


def ray_latlon(x, y, height_m, p):
    """lat/lon (degrees) where the line of sight at scan angles x/y crosses height_m."""
    sx, sy, sz = ray_point(x, y, height_m, p)
    a, b, H, lon0 = _ellipsoid(p, height_m)
    lat = np.arctan((a / b) ** 2 * sz / np.sqrt((H - sx) ** 2 + sy ** 2))
    return np.rad2deg(lat), np.rad2deg(lon0 - np.arctan(sy / (H - sx)))


def closest_ray(voxel, z, row, col, x, y, p, reach=2, chunk=50_000):
    """among the pixels within reach of (row, col), the one whose line of sight passes
    closest to each voxel at height z; returns its row, col and the distance in km.

    voxel: [3, n] satellite-frame positions. done in chunks to bound memory.
    """
    offsets = np.arange(-reach, reach + 1)
    dr, dc = (o.ravel() for o in np.meshgrid(offsets, offsets, indexing="ij"))
    best_row, best_col = np.empty_like(row), np.empty_like(col)
    best_km = np.empty(len(row))
    for s in range(0, len(row), chunk):
        part = slice(s, s + chunk)
        rr = np.clip(row[part, None] + dr, 0, len(y) - 1)
        cc = np.clip(col[part, None] + dc, 0, len(x) - 1)
        ray = ray_point(x[cc], y[rr], z[part, None], p)
        d = np.sqrt(sum((voxel[i, part, None] - ray[i]) ** 2 for i in range(3)))
        d = np.where(np.isfinite(d), d, np.inf)  # lines of sight that miss the earth
        k = np.argmin(d, axis=1)
        n = np.arange(len(k))
        best_row[part], best_col[part], best_km[part] = rr[n, k], cc[n, k], d[n, k] / 1000.0
    return best_row, best_col, best_km
