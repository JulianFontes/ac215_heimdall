# SURFRAD 2025 exploratory analysis: findings

The terms, the full list of tests, every figure, and the key takeaways are in `src/heimdall/solar_forecasting/deep_dive/findings.md`. The PDF with the charts embedded is `artifacts/solar_variability_findings.pdf`.

Seven NOAA SURFRAD stations, all of 2025, 1-min data averaged to 10 min. "k" is the clear-sky index GHI / GHI_cs from the pvlib Ineichen model, and "corrected k" is k / alpha, where alpha is the station's `detect_clearsky` scale factor (BON 1.083, DRA 1.028, FPK 1.024, GWN 1.042, PSU 1.062, SXF 1.048, TBL 1.016). Figures are in `artifacts/figures/solar_forecasting/eda/` and tables in `data/stations/eda/`. Each script is in this folder and runs with `uv run python src/heimdall/solar_forecasting/eda/<script>.py`.

## How this analysis was built

These four checks sit under the station ranking. A high variability score can be a property of the place, a season, a clear-sky model error, a data gap, or the same weather measured twice. Each script answers one of those.

- **Clear-sky bias, before and after the alpha correction.** k = GHI / Ineichen clear-sky GHI. Ineichen is high or low by station, so raw k > 1 looks like cloud enhancement. Dividing by the station's `detect_clearsky` scale factor removes that bias before sky classes or a variability index are trusted. The monthly and zenith panels then ask whether one annual alpha is enough.
- **Monthly climate.** Variability can be a summer at a stormy site, or a feature of the site in every month. Monthly sky-class shares, median corrected k, the PV energy ratio and the daily variability index separate a season from a place.
- **Missing and flagged data.** A gap is not a weather event. The check separates missing rows and −9999.9 fills from genuine cloudy months, including the network-wide hole on 2025-03-09 (the US daylight-saving changeover) and the Fort Peck DNI outage, which would otherwise look like a dark week.
- **Station correlation.** Daily and 10-minute correlations, and a one-day west-to-east lag, test whether these sites share weather. If they move together, one deep dive would stand in for the network. If they do not, each station needs its own case, and spatial forecast skill has to come from imagery rather than from a neighbour.

## 1. Clear-sky bias (`clearsky_bias.py`)

**Most of the "above clear sky" time is model bias. Cloud enhancement is the smaller part.** Dividing k by alpha cuts the share of 10-min intervals with k > 1.05 from 36.4% to 10.0% at Bondville, and from 16–25% to 6.5–12.2% elsewhere. The alpha removes between 26% (TBL) and 73% (BON) of these intervals. Most of what is left is still smooth like a clear sky: 58–76% of the remaining intervals have 1-min std ≤ 0.05. That is leftover clear-sky scatter from day-to-day turbidity. It is not enhancement. Only **1.8% (DRA) to 3.4% (TBL)** of daylight intervals are both above 1.05 and variable, which is the real cloud-enhancement signature (`clearsky_bias_shares.png`).

The correction barely moves the sky classes. Clear shares drop by at most 1.6 points and overcast shares rise by at most 1.7 points (PSU 20.1% → 21.8%). Mean k drops by 0.02–0.06 (table `clearsky_bias_summary.csv`).

**The bias is seasonal at the northern Plains sites.** Relative to the annual alpha, the clear-sky level at Fort Peck runs 1.11 in February and 0.97 in September, a 14% swing. Sioux Falls reaches 1.07 in February. Desert Rock stays within 3% all year (`clearsky_bias_month_zenith.png`, left). The winter excess at FPK and SXF most likely comes from snow albedo and a low Linke-turbidity climatology. A monthly correction instead of the annual alpha takes FPK's k > 1.05 share from 11.9% down to 7.2%.

**The bias also depends on sun angle.** At most stations, clear-sky k climbs 2–6% above alpha at zenith 70–80°. But Goodwin Creek is the exception. It drops to about 0.95 there, which is questionable. It may be horizon obstruction or a pyranometer cosine-response problem. Check it (`clearsky_bias_month_zenith.png`, right; `clearsky_bias_zenith.csv`).

Figure captions:
- `clearsky_bias_shares.png`: for each station, the share of 10-min intervals with k > 1.05 using raw k, k / annual alpha, and k / monthly clear-sky level, plus the share that is variable cloud enhancement.
- `clearsky_bias_month_zenith.png`: the median clear-minute k divided by the annual alpha, as a station × month heatmap (left) and against solar zenith (right); 1 means the annual alpha fits.

## 2. Monthly climate (`monthly_climate.py`)

All numbers use corrected k and local solar months (table `monthly_climate.csv`).

**Seasonal vs steady.** The range of the monthly median 10-min k separates the stations:
- PSU is the most seasonal at 0.55 (0.41 in Dec, 0.96 in Aug).
- GWN is next at 0.48 (0.50 in Feb, 0.98 in Jul).
- FPK (0.36), BON (0.35) and SXF (0.30) are in the middle.
- TBL is fairly steady at 0.20.
- DRA is essentially constant at 0.04 (0.97–1.01), with 52–85% clear intervals every month.

The PV energy ratio follows the same pattern. PSU runs 0.39 (Dec) to 0.83, while DRA runs 0.76 to 1.00 (`monthly_climate_panels.png`; `monthly_climate_seasonality.csv`). The energy ratio uses the uncorrected Ineichen clear-sky power, so it reads a few percent high (DRA reaches 1.00 in June).

**Variability peaks in summer.** The median daily Stein VI peaks in July at 5 of 7 stations. SXF peaks in August and DRA in April (7.7, its spring storm season). The all-station median is 12.4 in July vs 3.3 in February. The single most variable station-month is GWN in July (VI 18.4), followed by GWN in May and June (15.2) and BON in July (15.0) (`monthly_climate_heatmaps.png`, bottom).

**The broken-cloud share peaks in winter.** FPK is 78% and SXF is 71% in December, with a median of 57% across stations in December (`monthly_climate_heatmaps.png`, top). VI scales with sun strength, so winter variability shows up small in VI. The share counts intervals regardless of amplitude. The December FPK/SXF shares are questionable: low sun and snow can push the 1-min k std above the 0.05 clear threshold on skies that are actually clear.

Figure captions:
- `monthly_climate_panels.png`: monthly clear/broken/overcast shares (stacked bars) with the median corrected k and PV energy ratio, one panel per station, ordered from most to least seasonal.
- `monthly_climate_heatmaps.png`: station × month heatmaps of the broken-interval share (top) and the median daily variability index (bottom).

## 3. Data gaps (`data_gaps.py`)

**Bad data means missing data. It does not mean flagged data.** Values that are present but carry a nonzero SURFRAD quality flag make up ≤ 0.01% of minutes at every station. Every loss is a missing row or a -9999.9 fill. Over the year, missing GHI in the k window (zenith < 80°) runs from 0.08% (PSU) to 1.86% (BON). DNI is missing more often than GHI: up to 2.9% of sun-up minutes at GWN (table `data_gaps_monthly.csv`).

**Affected months (> 5% of k-window minutes unusable)** (`data_gaps_heatmap.png`):
- **GHI** (affects k, VI and sky classes): TBL Jan 10.8%, FPK Dec 7.9%, TBL Mar 6.4%, BON May 5.2%.
- **Any component** (affects PV power and energy totals): the GHI months above, plus FPK Nov 18.0%, FPK Dec 11.3%, BON May 9.5%, GWN Apr 5.6% and BON Apr 5.5%.

**Longest gaps** (`data_gaps_timeline.png`; table `data_gaps_longest.csv`):
- FPK lost **DNI for 7 days (Nov 25 – Dec 1)** while GHI stayed fine. The k statistics are fine, but power and energy for that week are missing.
- TBL lost 48 h on Jan 4–5.
- GWN lost 26 h on Apr 22–23.
- SXF lost 26 h on Feb 5.
- **2025-03-09 is missing as a whole day at 6 of 7 stations** (all but PSU). That is the US daylight-saving changeover date, so it is probably a processing artifact on NOAA's side.

Figure captions:
- `data_gaps_heatmap.png`: the share of k-window minutes with unusable GHI (top) and with any of GHI/DNI/DHI unusable (bottom), by station and month; * marks > 5%.
- `data_gaps_timeline.png`: the daily share of k-window minutes with any component unusable, one row per station, with each station's longest gap labeled.

## 4. Station correlation (`station_correlation.py`)

**Stations mostly move independently on the same day.** The closest pair, BON–GWN (658 km), has the strongest daily-k correlation at r = 0.43 (0.26 at 10 min). The next strongest are BON–PSU (0.28), FPK–SXF (0.23) and GWN–PSU (0.20). The median over all 21 pairs is 0.05 daily and 0.03 at 10 min. Removing each station's 31-day rolling median (the anomaly) barely changes this (BON–GWN 0.39). A shared seasonal cycle is not what drives the correlation (`station_correlation_heatmap.png`).

An exponential fit gives e-folding distances of about 660 km (daily) and 510 km (10 min). Beyond about 1,000 km, r is within ±0.1 of zero. No pair is closer than 658 km, so the fit below that is an extrapolation (`station_correlation_distance.png`, left; `station_correlation_pairs.csv`). DRA and TBL are the most isolated. Their mean daily r with the other six is 0.03.

**Weather moves west to east, and the lag shows it.** Yesterday's anomaly at a western station correlates with today's at an eastern one. The mean r is 0.13 west→east vs 0.01 east→west. The best pairs are BON yesterday → PSU today (r = 0.38, double their same-day 0.18), SXF → BON (0.36), TBL → GWN (0.28) and FPK → SXF (0.28) (`station_correlation_distance.png`, right).

Figure captions:
- `station_correlation_heatmap.png`: Pearson r between every station pair for daily corrected k, daily k anomaly, and 10-min k at common times.
- `station_correlation_distance.png`: same-day r against great-circle distance with exponential fits (left), and same-day vs next-day anomaly correlation for the eight strongest west→east pairs (right).

## What this means

**For choosing stations:**
- The network is sparse enough that the seven stations are close to independent samples. Each one adds new sky conditions, and none is redundant.
- DRA is a near-permanent clear-sky reference: the best site for calibrating clear-sky models and the worst for learning clouds.
- PSU, GWN and BON give the most cloud and variability.
- FPK and SXF bring a seasonal clear-sky bias and questionable winter sky classes, so treat their Dec–Feb data with care.
- Check GWN's low-sun clear-sky drop before relying on its sunrise and sunset data.

**For forecasting:**
- Use a per-station clear-sky correction, ideally a monthly one at FPK and SXF. Without it, 26–73% of "enhancement" events are fake.
- Real enhancement is only 2–3% of intervals, but it is variable, so it matters for ramps more than for energy.
- Neighboring stations give almost no same-day skill at 10-min to daily scales, so spatial information has to come from imagery (GOES), not other ground stations.
- The west→east next-day lag (r up to 0.38) is a usable feature for day-ahead forecasts at BON, PSU and GWN.
- Mask the listed gap months when you compute monthly energy or k statistics, and drop 2025-03-09 network-wide.
