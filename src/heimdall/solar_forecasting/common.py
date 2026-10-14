"""shared SURFRAD loading and clear-sky index analysis for the solar irradiance study.

same per-minute analysis as explore_stations.py (pvlib Ineichen clear sky, k = GHI / GHI_cs,
closure, detect_clearsky, PV power for the reference system, 10-min averages), but importable
and reading only the already-downloaded daily files under <DATA_DIR>/stations/surfrad/.
each station's analyzed year is pickled to <DATA_DIR>/stations/cache/ on first use.

typical use:
    meta, df = load_station_1min("gwn")       # 1-min, tz-aware UTC index
    k10 = to_10min(df)                        # GOES cadence
    k10["sky"] = classify_intervals(k10, df.attrs["clearsky_alpha"])
    days = classify_days(df)                  # one row per local solar day
"""
import warnings

import numpy as np
import pandas as pd
import pvlib

from heimdall.paths import DATA_DIR, FIGURES_DIR

# settings, identical to explore_stations.py
STATIONS = ["bon", "dra", "fpk", "gwn", "psu", "sxf", "tbl"]  # all 7 SURFRAD sites
STATION_NAMES = {"bon": "Bondville", "dra": "Desert Rock", "fpk": "Fort Peck", "gwn": "Goodwin Creek",
                 "psu": "Penn State", "sxf": "Sioux Falls", "tbl": "Table Mountain"}
YEAR = 2025
START = "2025-01-01"  # UTC days, inclusive
END = "2025-12-31"
AVERAGE = "10min"  # GOES-19 full-disk cadence
MAX_ZENITH = 80  # k gets unreliable near sunrise/sunset (small GHI, cosine errors)
MIN_GHI_CS = 50  # W m^-2; k only where the clear sky is at least this bright
PV_TILT, PV_AZIMUTH = 30, 180  # degrees; reference fixed-tilt, south-facing system
PV_PDC0, PV_GAMMA = 5000, -0.004  # W at 1000 W m^-2 and 25 °C; power temperature coefficient (1/°C)

STATION_DIR = DATA_DIR / "stations"
SURFRAD_DIR = STATION_DIR / "surfrad"
CACHE_DIR = STATION_DIR / "cache"

# 10-min sky classes (k corrected by the station's clearsky_alpha)
OVERCAST_K = 0.3  # corrected k below this: overcast
CLEAR_K = 0.9  # corrected k at least this ...
CLEAR_K_STD = 0.05  # ... and std of 1-min k inside the interval at most this: clear

# day types (daily corrected k and Stein variability index)
CLEAR_DAY_K = 0.85  # clear: daily k >= this and VI < CLEAR_DAY_VI
# ~95th percentile (1.44) of daily VI on days detect_clearsky marks >= 90% clear, all stations 2025;
# by eye, VI 1.5-2 days already show cloud passages (station_selection/calibrate_day_types.py)
CLEAR_DAY_VI = 1.5
OVERCAST_DAY_K = 0.4  # overcast: daily k < this and VI < OVERCAST_DAY_VI
# by eye, k < 0.4 days stay unbroken overcast up to VI ~3; above it sun breaks through
OVERCAST_DAY_VI = 3.0
MIN_DAY_MINUTES = 60  # valid daylight minutes needed for a daily VI / day type


def figure_dir(*parts):
    """FIGURES_DIR / "solar_forecasting" / *parts, created if needed."""
    path = FIGURES_DIR.joinpath("solar_forecasting", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def table_dir(*parts):
    """STATION_DIR / *parts, created if needed."""
    path = STATION_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_surfrad_year(code, year=YEAR):
    """1-min SURFRAD data for every day of year from the local cache; flagged values become nan.
    returns (data, meta). raises FileNotFoundError if any daily file is missing (never downloads)."""
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    paths = [SURFRAD_DIR / code / f"{code}{day:%y}{day.dayofyear:03d}.dat" for day in days]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{code}: {len(missing)} SURFRAD files missing under {SURFRAD_DIR / code} "
                                f"(e.g. {missing[:3]}); run explore_stations.py to download them")
    frames, meta = [], None
    for path in paths:
        data, meta = pvlib.iotools.read_surfrad(str(path))
        for v in ["ghi", "dni", "dhi", "temp_air", "wind_speed"]:
            data.loc[data[f"{v}_flag"] != 0, v] = np.nan  # SURFRAD flag 0 = good
        frames.append(data[["ghi", "dni", "dhi", "temp_air", "wind_speed"]])
    return pd.concat(frames), {"code": code, "name": meta["name"], "latitude": meta["latitude"],
                               "longitude": meta["longitude"], "altitude": meta["elevation"]}


def pv_power(location, solpos, ghi, dni, dhi, temp_air, wind_speed):
    """dc power of the reference system: transpose to the panel plane (Hay-Davies),
    SAPM cell temperature, then PVWatts."""
    poa = pvlib.irradiance.get_total_irradiance(
        PV_TILT, PV_AZIMUTH, solpos["apparent_zenith"], solpos["azimuth"], dni, ghi, dhi,
        dni_extra=pvlib.irradiance.get_extra_radiation(ghi.index), model="haydavies")["poa_global"]
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    temp_cell = pvlib.temperature.sapm_cell(poa, temp_air, wind_speed, **params)
    return pvlib.pvsystem.pvwatts_dc(poa, temp_cell, PV_PDC0, PV_GAMMA)


def analyze(data, meta):
    """add clear-sky irradiance, k, closure, clear-sky flags and PV power to one station's 1-min data.
    df.attrs gets clearsky_alpha plus the station metadata (code, name, latitude, longitude, altitude)."""
    location = pvlib.location.Location(meta["latitude"], meta["longitude"], altitude=meta["altitude"])
    times = pd.date_range(data.index.min().floor("D"), data.index.max().ceil("D"),
                          freq="1min", inclusive="left", tz="UTC")
    df = data.reindex(times)  # regular 1-min grid, gaps as nan
    solpos = location.get_solarposition(times)
    clear = location.get_clearsky(times, solar_position=solpos)  # ghi, dni, dhi
    df["zenith"] = solpos["apparent_zenith"]
    for v in ["ghi", "dni", "dhi"]:
        df[f"{v}_cs"] = clear[v]

    day = (df["zenith"] < MAX_ZENITH) & (df["ghi_cs"] > MIN_GHI_CS)
    df["k"] = (df["ghi"] / df["ghi_cs"]).where(day)
    # closure: GHI should equal DNI cos(zenith) + DHI within a few percent (standard radiometer QC check)
    df["closure"] = ((df["dni"] * np.cos(np.radians(df["zenith"])) + df["dhi"]) / df["ghi"]).where(day)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        is_clear, _, alpha = pvlib.clearsky.detect_clearsky(df["ghi"].fillna(0), df["ghi_cs"], times,
                                                            return_components=True)
    df["is_clear"] = is_clear & day
    # alpha > 1 means the Ineichen clear sky (monthly turbidity climatology) runs low at this site
    df.attrs.update(meta)
    df.attrs["clearsky_alpha"] = alpha

    temp_air = df["temp_air"].fillna(20.0)
    wind = df["wind_speed"].fillna(1.0)
    df["power"] = pv_power(location, solpos, df["ghi"], df["dni"], df["dhi"], temp_air, wind)
    df["power_cs"] = pv_power(location, solpos, df["ghi_cs"], df["dni_cs"], df["dhi_cs"], temp_air, wind)
    df["k_power"] = (df["power"] / df["power_cs"]).where(day)
    return df


def load_station_1min(code):
    """(meta, df) for one station's YEAR at 1 min, tz-aware UTC index, columns as in analyze().
    meta: code, name, latitude, longitude, altitude. cached at CACHE_DIR / f"{code}_1min_{YEAR}.pkl"."""
    path = CACHE_DIR / f"{code}_1min_{YEAR}.pkl"
    if path.exists():
        cached = pd.read_pickle(path)
        df = cached["df"]
        df.attrs.update(cached["attrs"])  # attrs don't always survive pickling, so stored separately
        return cached["meta"], df
    data, meta = read_surfrad_year(code)
    df = analyze(data, meta)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"meta": meta, "attrs": dict(df.attrs), "df": df}, path)
    return meta, df


def load_all_stations(codes=STATIONS):
    """{code: (meta, df)} from load_station_1min for every code."""
    return {code: load_station_1min(code) for code in codes}


def to_10min(df):
    """average to the GOES cadence. k is the ratio of the averages (not the average of
    ratios), so it stays consistent with energy over the interval."""
    cols = ["ghi", "ghi_cs", "dni", "dni_cs", "dhi", "dhi_cs", "power", "power_cs", "zenith"]
    out = df[cols].resample(AVERAGE).mean()
    out["k"] = out["ghi"] / out["ghi_cs"]
    out["k_power"] = out["power"] / out["power_cs"]
    out["k_std_1min"] = df["k"].resample(AVERAGE).std()  # variability GOES can't resolve
    out["clear_fraction"] = df["is_clear"].resample(AVERAGE).mean()
    out["n_valid"] = df["k"].resample(AVERAGE).count()
    return out[(out["zenith"] < MAX_ZENITH) & (out["n_valid"] >= 8)]


def alpha_corrected_k(df_or_series, alpha):
    """k / alpha, so k = 1 is the station's own clear sky. takes a DataFrame with a k column or a
    k Series; alpha is a float or a Series aligned with it (e.g. k10["station"].map(alphas))."""
    k = df_or_series["k"] if isinstance(df_or_series, pd.DataFrame) else df_or_series
    return k / alpha


def solar_date(index, longitude):
    """local solar calendar date of each UTC timestamp (UTC shifted by longitude / 15 h), so a whole
    daylight period falls on one date even when sunset is after 00 UTC."""
    return (index + pd.to_timedelta(longitude / 15, unit="h")).tz_localize(None).normalize()


def daylight(df):
    """minutes where k is defined: apparent zenith < MAX_ZENITH and ghi_cs > MIN_GHI_CS."""
    return (df["zenith"] < MAX_ZENITH) & (df["ghi_cs"] > MIN_GHI_CS)


def stein_variability_index(ghi, ghi_cs):
    """Stein et al. (2012) variability index for one day or window of regular 1-min data:
    sum(sqrt(dGHI^2 + dt^2)) / sum(sqrt(dGHI_cs^2 + dt^2)), dt = 1 min. only steps where both
    minutes have measured and clear-sky GHI count; mask out night (nan) before calling.
    1 = smooth as clear sky; ~2+ = broken clouds. nan if no valid steps."""
    d = ghi.diff()
    d_cs = ghi_cs.where(ghi.notna()).diff()
    ok = d.notna() & d_cs.notna()
    if not ok.any():
        return np.nan
    dt = 1.0  # minutes
    return np.sqrt(d[ok] ** 2 + dt ** 2).sum() / np.sqrt(d_cs[ok] ** 2 + dt ** 2).sum()


def daily_variability_index(df, longitude=None, min_minutes=MIN_DAY_MINUTES):
    """Stein VI per local solar date over daylight minutes; nan where fewer than min_minutes valid
    1-min steps. longitude defaults to df.attrs["longitude"]."""
    longitude = df.attrs["longitude"] if longitude is None else longitude
    day = daylight(df)
    ghi, ghi_cs = df["ghi"].where(day), df["ghi_cs"].where(day)
    date = solar_date(df.index, longitude)
    valid = (ghi.diff().notna() & ghi_cs.where(ghi.notna()).diff().notna()).groupby(date).sum()
    vi = pd.concat([ghi, ghi_cs], axis=1, keys=["ghi", "ghi_cs"]).groupby(date).apply(
        lambda g: stein_variability_index(g["ghi"], g["ghi_cs"]))
    return vi.where(valid.reindex(vi.index) >= min_minutes).rename_axis("date").rename("vi")


def classify_intervals(k10, alpha):
    """sky class of each 10-min row (from to_10min): "overcast" if corrected k < OVERCAST_K, "clear" if
    corrected k >= CLEAR_K and k_std_1min <= CLEAR_K_STD, else "broken". alpha: float or aligned Series."""
    k = alpha_corrected_k(k10, alpha)
    sky = pd.Series("broken", index=k10.index, name="sky")
    sky[(k >= CLEAR_K) & (k10["k_std_1min"] <= CLEAR_K_STD)] = "clear"
    sky[k < OVERCAST_K] = "overcast"
    return sky


def classify_days(df_1min, alpha=None, longitude=None, min_minutes=MIN_DAY_MINUTES):
    """one row per local solar date (daylight minutes only) with
    k: daily sum(ghi) / sum(ghi_cs) over valid minutes, divided by alpha (energy-weighted mean k),
    vi: daily Stein variability index, n_minutes: valid daylight minutes, and
    day_type: "clear" (k >= CLEAR_DAY_K and vi < CLEAR_DAY_VI), "overcast" (k < OVERCAST_DAY_K and
    vi < OVERCAST_DAY_VI) or "variable". days with < min_minutes valid minutes are dropped.
    alpha / longitude default to df_1min.attrs."""
    alpha = df_1min.attrs["clearsky_alpha"] if alpha is None else alpha
    longitude = df_1min.attrs["longitude"] if longitude is None else longitude
    ok = daylight(df_1min) & df_1min["ghi"].notna()
    date = solar_date(df_1min.index, longitude)
    sums = df_1min[["ghi", "ghi_cs"]].where(ok, axis=0).groupby(date).sum()
    out = pd.DataFrame({"k": sums["ghi"] / sums["ghi_cs"] / alpha,
                        "vi": daily_variability_index(df_1min, longitude, min_minutes),
                        "n_minutes": ok.groupby(date).sum()}).rename_axis("date")
    out = out[(out["n_minutes"] >= min_minutes) & out["vi"].notna()].copy()
    out["day_type"] = "variable"
    out.loc[(out["k"] >= CLEAR_DAY_K) & (out["vi"] < CLEAR_DAY_VI), "day_type"] = "clear"
    out.loc[(out["k"] < OVERCAST_DAY_K) & (out["vi"] < OVERCAST_DAY_VI), "day_type"] = "overcast"
    return out
