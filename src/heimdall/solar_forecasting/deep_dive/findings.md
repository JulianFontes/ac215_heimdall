# Where cloud variability justifies a 3D cloud product

SURFRAD, all of 2025. Seven ground stations measure sunlight every minute. The question is where clouds change fast enough that a 3D cloud product would improve a solar forecast. A 2D cloud fraction can already say "cloudy" or "clear." It cannot say whether the cloud over the sensor is thick, broken, or about to move.

The ranking tables are in `data/stations/selection/`, the deep-dive tables in `data/stations/deep_dive/`, the network-wide tables in `data/stations/eda/`, and the figures under `artifacts/figures/solar_forecasting/`. A PDF of this report, with the figures embedded, is `artifacts/solar_variability_findings.pdf`. Shared loading and the definitions below live in `src/heimdall/solar_forecasting/common.py`.

## What the words mean

- **GHI** (global horizontal irradiance) is the sunlight landing on a flat horizontal surface, in watts per square metre (W/m²). It includes the beam from the sun and the light scattered by air and clouds. A pyranometer at each station measures it every minute. This is the quantity the forecasts below try to predict.
- **DNI** (direct normal irradiance) is the beam from the sun's disk, measured on a surface pointed at the sun. **DHI** (diffuse horizontal irradiance) is the scattered light on a horizontal surface. The gap check uses all three. The cloud scores use GHI.
- **Clear-sky GHI** is a model of the GHI the station would record with no clouds, given where the sun is and how hazy the air usually is. It is not a measurement.
- **Ineichen** is the clear-sky model used here, as implemented in pvlib. It estimates clear-sky GHI from the solar zenith angle (how high the sun is) and a Linke turbidity, a single number for the usual haziness of the air at that place and month. pvlib supplies a turbidity climatology. If Ineichen runs low at a station, measured GHI looks "brighter than clear sky" on days that are actually clear.
- **k**, the clear-sky index, is measured GHI divided by the Ineichen clear-sky GHI. k = 1 means the measurement matches the model. k = 0.5 means the station is receiving half of the clear-sky value, usually because of cloud. k above 1 can mean the model is low, or that cloud edges briefly added extra light.
- **Alpha** is a scale factor for each station, from pvlib's `detect_clearsky`. It is the typical ratio of measured GHI to Ineichen GHI on minutes that look clear. Corrected k is k / alpha. The alphas are BON 1.083, DRA 1.028, FPK 1.024, GWN 1.042, PSU 1.062, SXF 1.048, TBL 1.016. An alpha of 1.083 means Ineichen runs about 8% low at Bondville, so dividing by alpha removes that bias before a day is called cloudy. Sky classes, day types, and the forecast climatology all use corrected k.
- **Stein variability index (VI)** is a daily score (Stein et al. 2012). It equals 1 on a day as smooth as clear sky. A larger VI means k jumped around during the day. Daily k is energy-weighted: the day's sum of GHI, divided by the day's sum of clear-sky GHI, divided by alpha. Days follow local solar time, not UTC.
- **Sky class** labels each 10-minute interval. Clear: corrected k is at least 0.9 and the standard deviation of the 1-minute k values inside the interval is at most 0.05. Overcast: corrected k is below 0.3. Broken: every other daylight interval.
- **Day type** labels a whole day, if the day has at least 60 minutes of usable daylight. Clear: daily corrected k is at least 0.85 and VI is below 1.5. Overcast: daily k is below 0.4 and VI is below 3. Variable: every other such day.
- **nRMSE** is the root-mean-square error divided by the mean measured GHI in that subset, written as a percent. **Skill** against climatology is 1 minus (forecast RMSE divided by climatology RMSE). Skill of 0 means the forecast ties the climatology. Negative skill means the forecast is worse.
- **Persistence** predicts that future GHI equals GHI now. **Smart persistence** carries the current corrected k forward and multiplies it by the clear-sky GHI at the future time, so the forecast still follows the sun's path. **Climatology** uses the median corrected k for that solar hour in the two neighbouring months, with the target month left out, times the clear-sky GHI at the target time.
- **GOES-19** is the US geostationary weather satellite. **Band 13** is its 10.3 µm infrared channel: the brightness temperature of the cloud top, or of the ground if the sky is clear. The **cloud mask** (product ABI-L2-ACMC) is the satellite's cloudy-or-not flag for each pixel. **Cloud fraction** here is the share of good-quality pixels inside a 50 km box around the station that the mask calls cloudy.

## What was tested

Each test uses the same 2025 files, the same clear-sky correction, and the same daylight screen (solar zenith under 80°, clear-sky GHI above 50 W/m²). Most scores are on 10-minute averages, which matches a GOES full-disk scan. The ramp curves and the within-interval standard deviation stay at 1 minute. The scripts are listed with the test.

- **All seven SURFRAD stations, calendar year 2025.** The stations are Bondville IL (BON), Desert Rock NV (DRA), Fort Peck MT (FPK), Goodwin Creek MS (GWN), Penn State PA (PSU), Sioux Falls SD (SXF), and Table Mountain CO (TBL). The public 1-minute files were already on disk. `common.py` reads them and does not download. One minute is the resolution of ramps and of the variability inside a 10-minute average.

- **Clear-sky bias, before and after alpha** (`eda/clearsky_bias.py`). Raw k above 1 looks like cloud enhancement. This test counts how many of those intervals disappear after dividing by alpha, how many of the rest are still smooth like clear sky, and whether one annual alpha fits every month and every sun angle.
- **Monthly climate** (`eda/monthly_climate.py`). For each station and month: sky-class shares, median corrected k, the PV energy ratio against the clear-sky plan, and the median daily VI. This separates a stormy season from a stormy place.
- **Missing and flagged data** (`eda/data_gaps.py`). Counts missing rows and −9999.9 fills separately from minutes that are present but carry a SURFRAD quality flag. A gap is not a cloudy hour.
- **Whether stations move together** (`eda/station_correlation.py`). Pearson correlation of daily corrected k, of the daily k anomaly after removing a 31-day rolling median, and of 10-minute k at shared times. Also a one-day lag from western stations to eastern ones.
- **Day-type cutoff** (`station_selection/calibrate_day_types.py`). The clear-day VI limit was tightened from 2.0 to 1.5 after looking at the days. Genuinely clear days already sit just above 1. Among days this rule calls clear, the median VI is 1.13 at Penn State, 1.10 at Goodwin Creek, and 1.03 at Table Mountain. On days `detect_clearsky` marks at least 90% clear, the 95th percentile of VI is 1.44. By eye, days with VI between 1.5 and 2 already have cloud passages, so a cutoff of 2.0 was calling those days clear. The tight cutoff is why most days land in "variable" (81–85% at the three sites below). That split is used later, and the circularity comes with it: a large forecast error on variable days partly restates that those days were defined as the unsettled ones.
- **Station ranking** (`station_selection/rank_stations.py`). Six measures, each turned into a rank (1 = most variable), then averaged. The six are median daily VI, median 1-minute k standard deviation, broken-interval share, variable-day share, the share of 10-minute power changes larger than 20% of inverter capacity, and the 1-hour RMSE of persisting k. Each one puts a different station first, so a single metric would have picked a different site. Ties in the mean rank are broken by the broken-cloud share. A station correlated above 0.7 in daily k with an earlier pick would have been skipped as a near-duplicate. No pair of the seven is that close (the maximum is Bondville–Goodwin Creek at 0.43). Goodwin Creek was a candidate because it leads daily VI (median 8.6). It is only fifth on broken-cloud share and on variable-day share. It reaches second because its mean rank is 3.33, and it beats Table Mountain on the tie-break (46.9% broken intervals against 46.0%). The deep dive is the top three. Penn State is alone at mean rank 1.50. Goodwin Creek, Table Mountain, and Bondville then sit in a cluster (3.33, 3.33, 3.50). Three sites cover three cloud regimes. Their daily k correlations are small (Penn State–Goodwin Creek 0.20, Goodwin Creek–Table Mountain 0.05, Penn State–Table Mountain −0.05).
- **When each chosen station is variable** (`deep_dive/seasonal_diurnal.py` and `deep_dive/days_and_ramps.py`). Month-by-hour medians of corrected k and of the 1-minute k standard deviation, monthly sky shares, a calendar of daily VI, and ramp exceedance at lags of 1, 10, and 60 minutes. A ramp is a change in reference PV power (5 kW, tilt 30°, azimuth 180°) larger than a given percent of capacity. The sun-only curve is clear-sky power recomputed at a fixed 20 °C and 1 m/s wind, because the stored clear-sky power uses measured weather and wind gusts made it jump by up to 8% of capacity in one minute. Six case days per station: the smoothest clear day, the darkest overcast day, and the highest-VI variable day in the warm season (April–September) and again in the cold season (October–March).
- **Forecast baselines, before any learned model** (`deep_dive/forecast_baselines.py`). On the 10-minute daylight series, for horizons from 10 minutes to 3 hours: persistence of GHI, smart persistence, month-hour climatology, and a least-squares blend of climatology and smart persistence fit with the target month left out. Desert Rock is scored on the same baselines as a clear-sky control. If smart persistence already matches climatology at 3 hours on variable days, that is the gap a 3D product has to beat.
- **GOES-19 on the same case days** (`deep_dive/goes_case_days.py`). CONUS band 13 plus the cloud mask, every matched daylight scan, in a 50 km box (about 380 pixels). Full-disk files are about 250 MB each, so the scans are the CONUS product after a size check (the files used were about 4–5 MB). There is no parallax correction. A 50 km box still covers the station if the cloud is placed a few kilometres off the ground point. The starting full-year charts from `explore_stations.py` (station map, k distributions, persistence, ramps, energy) sit in the same figures folder and are listed at the end. They use uncorrected 10-minute k with the sun more than 10° above the horizon.

## Which stations, and why

**Penn State is first, and it stays first if any one metric is dropped.** Mean rank 1.50. It is first on the 1-minute k standard deviation, on the broken-interval share (51%) and on the variable-day share (85% of 365 days). Median daily VI is 7.6, second only to Goodwin Creek. The 1-hour RMSE of persisting k is 0.24. The site label used in the ranking is a humid-continental Appalachian valley with persistent stratocumulus. Only 15 days meet the clear-day rule. The clear-day forecast numbers below rest on that small sample.

**Goodwin Creek is second because it leads daily variability.** But the other metrics do not all agree. Mean rank 3.33, tied with Table Mountain and placed ahead by the broken-cloud tie-break. Median daily VI is 8.6, the highest of the seven. It is third on 1-minute k variability, on ramps and on k-persistence error (1-hour RMSE 0.23). It is fifth on broken share (47%) and variable-day share (81% of 364 days). The site label is humid-subtropical Deep South convection. The month-by-hour section below shows that label in the data. The variability is a summer midday feature. It is not a year-round one.

**Table Mountain is third because the power actually moves, and because simple persistence fails.** Same mean rank, 3.33. It is first on the 10-minute ramp share (11.4% of daylight steps change PV power by more than 20% of capacity) and first on 1-hour k-persistence RMSE (0.28). Median daily VI is 7.3, broken share 46%, variable days 84% of 360 days. On the 1-minute k standard deviation and on the broken share it is only sixth, so it is not "the cloudiest site." The site label is semi-arid Front Range foothills with afternoon orographic convection. The diurnal section shows that afternoon shift.

**Bondville is close, and places 2–4 trade if one metric is removed.** Bondville's mean rank is 3.50, with a higher broken share (49%) than either site we kept, median daily VI 7.1, and 85% variable days. Penn State remains first in every leave-one-metric-out ranking. Dropping the ramp metric or the k-persistence metric puts Bondville second and drops Table Mountain out of the top three. Dropping daily VI drops Goodwin Creek to fourth (Bondville and Table Mountain then tie at 3.4, and Bondville wins on broken share). The top-three list is a stable first place plus a fragile pair.

The ranking chart is `artifacts/figures/solar_forecasting/selection/selection_summary.png`. Penn State leads the mean rank, Goodwin Creek and Table Mountain tie, and Bondville is a tenth of a rank behind them.

`artifacts/figures/solar_forecasting/selection/rank_heatmap.png`. Each variability metric puts a different station first, which is why the pick uses the mean rank.

## What simple forecasts miss

Errors below are for smart persistence of 10-minute GHI. nRMSE is the RMSE divided by the mean measured GHI in that subset. The day-type split is partly circular. "Variable" is whatever is left after the clear and overcast rules, so a large error on variable days partly restates that those days were defined as the unsettled ones. Overcast nRMSE is inflated by a small mean GHI. At 1 hour on overcast days the smart-persistence RMSE is only 57 W/m² at Penn State (nRMSE 50%), 58 W/m² at Goodwin Creek (50%) and 74 W/m² at Table Mountain (57%). The climatology nRMSE on those same hours is 242%, 337% and 359%.

**On variable days the 1-hour error is about 165–203 W/m². On clear days it is about 11–17 W/m².**

| Station | Clear days | 1 h smart persistence | Variable days | 1 h smart persistence |
| --- | --- | --- | --- | --- |
| Penn State | 15 | 1.8% (11 W/m²) | 312 | 39.5% (165 W/m²) |
| Goodwin Creek | 38 | 3.1% (17 W/m²) | 295 | 34.7% (169 W/m²) |
| Table Mountain | 47 | 2.2% (13 W/m²) | 303 | 43.4% (203 W/m²) |

The watt figures are the CSV RMSEs rounded to the nearest W/m² (11.3, 16.9, 12.6, 165.4, 168.7, 203.2). Table Mountain's variable-day error is the largest in watts, which matches its first-place rank on persistence error.

**At 3 hours on variable days, smart persistence has essentially no skill against the month-hour climatology.** Skill is 1 − RMSE / RMSE of climatology: +0.04 at Penn State (213 vs 222 W/m²; nRMSE 48.0% vs 50.0%), +0.02 at Goodwin Creek (217 vs 221 W/m²; 42.0% vs 42.8%), and −0.06 at Table Mountain (263 vs 248 W/m²; 54.3% vs 51.3%). Carrying the current k forward for three hours does not beat "this hour in the neighbouring months."

`artifacts/figures/solar_forecasting/deep_dive/comparison/summary_1h_nrmse_by_day_type.png`. At 1 hour, smart persistence is a few percent on clear days and about 35–43% on variable days at the three stations.

`artifacts/figures/solar_forecasting/deep_dive/comparison/skill_vs_horizon_stations.png`. Skill of smart persistence against climatology falls to about zero by 3 hours on variable days.

## What GOES-19 sees on the case days

Pooled over the three stations and all 1,110 matched scans, box cloud fraction and corrected k correlate at r = −0.84. Restricted to variable days the correlation falls to −0.38. The standard deviation of band-13 brightness temperature inside the 50 km box correlates with the within-10-minute k standard deviation at r = +0.41 pooled and only +0.07 on variable days. A 2D cloud amount tracks the daily clear-versus-cloudy contrast. It does not track the variability that the forecast misses.

Two case days show that split on the ground.

**Penn State, 8 April 2025 (warm-season variable day, VI 38.3).** From late morning through mid-afternoon the 50 km box stays essentially filled. Cloud fraction is at least 0.95 on most scans, and the median for the day is 1.00. Corrected k still runs from 0.38 to 1.08, with one early scan at 1.47 while the box cloud fraction is 1.00. A mask that says "fully cloudy" is not a GHI forecast.

**Goodwin Creek, 18 June 2025 (warm-season variable day, VI 36.6).** The day's median box cloud fraction is 0.09. A thicker deck in the morning does reach cloud fraction 0.75 and pulls k down to 0.18. The argument for a 3D product is the afternoon, when the box is only about a tenth cloudy. At 18:11 UTC, k is 0.33 with cloud fraction 0.12, and ten minutes later k is 1.03 with cloud fraction 0.11. Small cumulus inside the box move the ground irradiance from a third of clear sky back to clear sky without the cloud fraction changing.

`artifacts/figures/solar_forecasting/deep_dive/comparison/goes_vs_ground_k.png`. Cloud fraction tracks k across clear and cloudy scans, and the relationship weakens once the day is already variable.

`artifacts/figures/solar_forecasting/deep_dive/psu/goes_case_days_psu.png`. On 8 April the box is full while the ground k keeps swinging.

`artifacts/figures/solar_forecasting/deep_dive/gwn/goes_case_days_gwn.png`. On 18 June a low cloud fraction still accompanies a drop of k to about 0.3 and a recovery to 1.

## Goodwin Creek at low sun

Corrected clear-minute k at solar zenith 70–80° is 0.95 at Goodwin Creek (raw clear-minute median 0.986 divided by alpha 1.042). At the other six stations the same bin sits 2–6% above the annual alpha (from +2.1% at Bondville and Desert Rock to +5.7% at Table Mountain). Sunrise and sunset at Goodwin Creek should not be used until that drop is checked for shading or a cosine-response problem. The same bin is plotted in `artifacts/figures/solar_forecasting/eda/clearsky_bias_month_zenith.png`.

## When it is variable

Month-by-hour medians, monthly sky shares and ramp exceedance are computed for all three stations (12 months by local solar hours 6–17; ramp lags of 1, 10 and 60 minutes). A ramp is a change in reference PV power larger than a given percent of capacity. The sun-only curve is clear-sky power recomputed at a fixed 20 °C and 1 m/s wind, because the stored clear-sky power uses measured weather and wind gusts made it jump by up to 8% of capacity in one minute. With the weather held fixed, the sun alone stays under about 5% of capacity in 10 minutes, so a 20% exceedance is cloud, not geometry.

**Penn State is variable most of the year. The winter is dim, and it is not calm.** Median daily VI peaks at 14.1 in July and is still 3.9 in February, its quietest month. The energy ratio against the alpha-scaled clear-sky plan runs from 0.37 in December to 0.79 in July. January is 66% broken intervals. July and August drop the overcast share to 8% and 7% but stay about half broken. Across months, the median 1-minute k standard deviation is highest from local solar hour 11 to 14 (about 0.032–0.034). The noisiest cell is July at hour 9 (0.059). Summer month-hour cells are only modestly noisier than winter ones (median 0.031 against 0.027). At a 10-minute lag, 11.1% of daylight steps exceed a 20% capacity change, and that share stays between 10% and 12% in every season. At 60 minutes the share is 29%.

**Goodwin Creek is a summer-midday site.** Median daily VI is 18.4 in July, 15.2 in May and June, and 2.2 in January. June–August has no clear day (0 of 92). The energy ratio runs from 0.54 in February to 0.88 in July. June is 64% broken. The median winter month-hour cell has a 1-minute k standard deviation of 0.016. The summer median cell is 0.035. July at local solar hour 11 reaches 0.16, and June at hour 13 reaches 0.14. The 10-minute, 20%-of-capacity ramp rate is 15.5% of steps in June–August and 5.1% in December–February (11.0% for the year). At 60 minutes the year share is 30%.

**Table Mountain is steadier in energy, and more violent in the afternoon and at the hourly scale.** The energy ratio stays between 0.69 (May) and 0.81 (January). Median daily VI still peaks in July (11.8) and falls to 3.3 in October. Overcast intervals never exceed 21% (May). Broken intervals stay between 34% and 55% all year. The median 1-minute k standard deviation rises from 0.011 at local solar hour 7 to 0.028 at hour 13. July at hour 13 is the noisiest cell (0.088) and the median k in that cell is only 0.66, a cloudy afternoon rather than a bright noisy noon. The 10-minute ramp rate is 11.4% for the year and 11–12% in every season. At 60 minutes, 38% of daylight steps change power by more than 20% of capacity, the highest of the three stations, which is the ramp ranking showing up at the hourly lag.

`artifacts/figures/solar_forecasting/deep_dive/comparison/monthly_broken_vi.png`. Goodwin Creek's variability index peaks hardest in summer. Penn State's broken-cloud share stays high in winter. Table Mountain's energy and cloud shares move less from month to month.

`artifacts/figures/solar_forecasting/deep_dive/psu/diurnal_heatmaps.png`. Penn State's within-hour variability is spread through the middle of the day in most months.

`artifacts/figures/solar_forecasting/deep_dive/gwn/diurnal_heatmaps.png`. Goodwin Creek's 1-minute k standard deviation concentrates at midday in May–August.

`artifacts/figures/solar_forecasting/deep_dive/tbl/diurnal_heatmaps.png`. Table Mountain's noisy cells sit in the afternoon, especially July.

`artifacts/figures/solar_forecasting/deep_dive/gwn/ramp_exceedance.png`. Goodwin Creek's 10-minute ramps are a summer event. The clear-sky curve stays near 5% of capacity.

`artifacts/figures/solar_forecasting/deep_dive/tbl/ramp_exceedance.png`. Table Mountain's large swings show up at the 60-minute lag in every season.

## What the seven-station checks found

These are the network-wide tests. The longer write-up is `src/heimdall/solar_forecasting/eda/findings.md`.

- Dividing k by alpha cuts the share of 10-minute intervals with k > 1.05 from 36.4% to 10.0% at Bondville, and from 16–25% to 6.5–12.2% elsewhere. The alpha removes between 26% (TBL) and 73% (BON) of those intervals. Only 1.8% (DRA) to 3.4% (TBL) of daylight intervals are both above 1.05 and variable. That is the real cloud-enhancement share.
- The bias is seasonal at Fort Peck (clear-sky level 1.11 in February and 0.97 in September) and at Sioux Falls (1.07 in February). At zenith 70–80°, clear-sky k sits 2–6% above alpha at six stations. Goodwin Creek drops to about 0.95 there.
- Penn State is the most seasonal station: monthly median k runs from 0.41 in December to 0.96 in August, a range of 0.55. Desert Rock stays between 0.97 and 1.01. The all-station median daily VI is 12.4 in July and 3.3 in February. The single most variable station-month is Goodwin Creek in July (VI 18.4).
- Broken-cloud share peaks in winter (Fort Peck 78% and Sioux Falls 71% in December). Treat that as a low-sun and snow artifact: the 1-minute k standard deviation can cross 0.05 on skies that are actually clear.
- Present-but-flagged minutes are at most 0.01% everywhere. Missing GHI in the daylight window runs from 0.08% (Penn State) to 1.86% (Bondville). Fort Peck lost DNI for 7 days (25 November–1 December) while GHI stayed fine. 9 March 2025 is missing as a whole day at 6 of 7 stations (all but Penn State), the US daylight-saving changeover.
- Same-day station correlation is weak. The median over 21 pairs is r = 0.05. The strongest pair is Bondville–Goodwin Creek, r = 0.43 at 658 km. Beyond about 1,000 km, r is within ±0.1 of zero. Yesterday at a western station correlates with today at an eastern one. The best pair is Bondville yesterday to Penn State today, r = 0.38, double their same-day 0.18.

## Figures

Each bullet is one chart. The PDF embeds the image under the bullet. Paths are relative to the repository root.

### Ranking and day types

- `artifacts/figures/solar_forecasting/selection/selection_summary.png`. Penn State leads the mean rank. Goodwin Creek and Table Mountain tie. Bondville is a tenth of a rank behind them.
- `artifacts/figures/solar_forecasting/selection/rank_heatmap.png`. Each of the six metrics puts a different station first.
- `artifacts/figures/solar_forecasting/selection/metric_panels.png`. The six variability metrics side by side, with the highest station labelled on each panel.
- `artifacts/figures/solar_forecasting/selection/daily_k_correlation.png`. Daily corrected k is largely independent across stations. The largest off-diagonal correlation is 0.43.
- `artifacts/figures/solar_forecasting/selection/day_type_check/daily_vi_by_group.png`. Clear days sit near VI 1.0–1.4. Days above 1.5 already show cloud passages.
- `artifacts/figures/solar_forecasting/selection/day_type_check/boundary_high_k.png`. Days near the clear-day boundary.
- `artifacts/figures/solar_forecasting/selection/day_type_check/boundary_low_k.png`. Days near the overcast boundary.
- `artifacts/figures/solar_forecasting/selection/day_type_check/day_types_psu.png`. Penn State day types: clear days are smooth, variable days are broken, overcast days are dim.
- `artifacts/figures/solar_forecasting/selection/day_type_check/day_types_gwn.png`. The same day-type check at Goodwin Creek.
- `artifacts/figures/solar_forecasting/selection/day_type_check/day_types_tbl.png`. The same day-type check at Table Mountain.
- `artifacts/figures/solar_forecasting/selection/day_type_check/day_types_bon.png`. The same day-type check at Bondville.
- `artifacts/figures/solar_forecasting/selection/day_type_check/day_types_dra.png`. The same day-type check at Desert Rock, the clear reference.
- `artifacts/figures/solar_forecasting/selection/day_type_check/day_types_fpk.png`. The same day-type check at Fort Peck.
- `artifacts/figures/solar_forecasting/selection/day_type_check/day_types_sxf.png`. The same day-type check at Sioux Falls.

### All seven stations

- `artifacts/figures/solar_forecasting/eda/clearsky_bias_shares.png`. Most intervals with k > 1.05 are clear-sky model bias. The station alpha removes them. The variable cloud-enhancement share is the small remainder.
- `artifacts/figures/solar_forecasting/eda/clearsky_bias_month_zenith.png`. One annual alpha does not fit every month or every sun angle. Fort Peck swings by about 14% over the year. Goodwin Creek drops at high zenith.
- `artifacts/figures/solar_forecasting/eda/monthly_climate_panels.png`. Monthly clear, broken, and overcast shares, with median corrected k and the PV energy ratio. Stations are ordered from most seasonal to least.
- `artifacts/figures/solar_forecasting/eda/monthly_climate_heatmaps.png`. Broken-interval share (top) and median daily VI (bottom) by station and month. VI peaks in summer. Broken share peaks in winter.
- `artifacts/figures/solar_forecasting/eda/data_gaps_heatmap.png`. Share of daylight minutes with unusable GHI (top) and with any of GHI, DNI, or DHI unusable (bottom). A star marks months above 5%.
- `artifacts/figures/solar_forecasting/eda/data_gaps_timeline.png`. Daily share of daylight minutes with any component unusable. The longest gap is Fort Peck's 7-day DNI outage.
- `artifacts/figures/solar_forecasting/eda/station_correlation_heatmap.png`. Pairwise correlation of daily k, daily k anomaly, and 10-minute k. The stations mostly move independently.
- `artifacts/figures/solar_forecasting/eda/station_correlation_distance.png`. Same-day correlation falls off beyond about 1,000 km. The right panel is the west-to-east next-day lag.

### Deep dive: forecasts, seasons, and GOES

- `artifacts/figures/solar_forecasting/deep_dive/comparison/summary_1h_nrmse_by_day_type.png`. At 1 hour, smart persistence is a few percent on clear days and about 35–43% on variable days.
- `artifacts/figures/solar_forecasting/deep_dive/comparison/error_vs_horizon_stations.png`. Smart-persistence error grows with the horizon. At 1 hour it is a few percent on clear days and much larger on variable days.
- `artifacts/figures/solar_forecasting/deep_dive/comparison/skill_vs_horizon_stations.png`. On variable days, skill of smart persistence against climatology is near zero by 3 hours.
- `artifacts/figures/solar_forecasting/deep_dive/comparison/monthly_broken_vi.png`. Goodwin Creek's VI peaks hardest in summer. Penn State's broken-cloud share stays high in winter. Table Mountain moves less from month to month.
- `artifacts/figures/solar_forecasting/deep_dive/comparison/goes_vs_ground_k.png`. Cloud fraction tracks k across clear and cloudy scans (r = −0.84) and much less on variable days (r = −0.38).
- `artifacts/figures/solar_forecasting/deep_dive/psu/diurnal_heatmaps.png`. Penn State's within-hour variability is spread through the middle of the day in most months.
- `artifacts/figures/solar_forecasting/deep_dive/psu/monthly_sky_energy.png`. Penn State's sky mix and PV energy by month. Clouds cost the largest share of the clear-sky plan here.
- `artifacts/figures/solar_forecasting/deep_dive/psu/daily_vi_calendar.png`. Penn State has a variable sky on most days. Clear days are rare.
- `artifacts/figures/solar_forecasting/deep_dive/psu/ramp_exceedance.png`. About 11% of daylight minutes swing by more than 20% of capacity within 10 minutes, in every season. The sun-only curve stays under about 5%.
- `artifacts/figures/solar_forecasting/deep_dive/psu/case_days.png`. The six Penn State case days: measured GHI, clear-sky GHI, and corrected k.
- `artifacts/figures/solar_forecasting/deep_dive/psu/goes_case_days_psu.png`. On 8 April the 50 km box stays full while ground k keeps swinging.
- `artifacts/figures/solar_forecasting/deep_dive/psu/goes_bt13_strip_psu.png`. Three band-13 frames on 8 April. k goes from about 0.5 to 1.0 and back while the box stays cloudy.
- `artifacts/figures/solar_forecasting/deep_dive/psu/error_vs_horizon_psu.png`. Penn State forecast error against horizon, split by day type.
- `artifacts/figures/solar_forecasting/deep_dive/psu/error_by_day_type_and_sky_psu.png`. At 1 hour, the error is much larger on broken-cloud intervals than on clear ones.
- `artifacts/figures/solar_forecasting/deep_dive/psu/example_days_psu.png`. A 1-hour smart-persistence forecast tracks a clear day and lags the cloud passages on a variable day.
- `artifacts/figures/solar_forecasting/deep_dive/gwn/diurnal_heatmaps.png`. Goodwin Creek's 1-minute k standard deviation concentrates at midday in May–August.
- `artifacts/figures/solar_forecasting/deep_dive/gwn/monthly_sky_energy.png`. Goodwin Creek's sky mix and PV energy by month. June is the broken-cloud peak.
- `artifacts/figures/solar_forecasting/deep_dive/gwn/daily_vi_calendar.png`. Goodwin Creek's variable days pile up in summer. June–August has no clear day.
- `artifacts/figures/solar_forecasting/deep_dive/gwn/ramp_exceedance.png`. Goodwin Creek's 10-minute ramps are a summer event. The sun-only curve stays near 5% of capacity.
- `artifacts/figures/solar_forecasting/deep_dive/gwn/case_days.png`. The six Goodwin Creek case days.
- `artifacts/figures/solar_forecasting/deep_dive/gwn/goes_case_days_gwn.png`. On 18 June a low cloud fraction still accompanies a drop of k to about 0.3 and a recovery to 1.
- `artifacts/figures/solar_forecasting/deep_dive/gwn/goes_bt13_strip_gwn.png`. Three band-13 frames on 18 June. Small cumulus sit near the station while k swings and the box cloud fraction stays near 0.1.
- `artifacts/figures/solar_forecasting/deep_dive/gwn/error_vs_horizon_gwn.png`. Goodwin Creek forecast error against horizon, split by day type.
- `artifacts/figures/solar_forecasting/deep_dive/gwn/error_by_day_type_and_sky_gwn.png`. The 1-hour error by day type and sky class at Goodwin Creek.
- `artifacts/figures/solar_forecasting/deep_dive/gwn/example_days_gwn.png`. A clear day and a variable day at Goodwin Creek, with the 1-hour smart-persistence forecast.
- `artifacts/figures/solar_forecasting/deep_dive/tbl/diurnal_heatmaps.png`. Table Mountain's noisy cells sit in the afternoon, especially July.
- `artifacts/figures/solar_forecasting/deep_dive/tbl/monthly_sky_energy.png`. Table Mountain's energy ratio stays in a narrow band. Overcast intervals stay uncommon.
- `artifacts/figures/solar_forecasting/deep_dive/tbl/daily_vi_calendar.png`. Table Mountain's day-type calendar. Overcast days are rare (10 all year).
- `artifacts/figures/solar_forecasting/deep_dive/tbl/ramp_exceedance.png`. Table Mountain's large swings show up at the 60-minute lag in every season.
- `artifacts/figures/solar_forecasting/deep_dive/tbl/case_days.png`. The six Table Mountain case days.
- `artifacts/figures/solar_forecasting/deep_dive/tbl/goes_case_days_tbl.png`. On the variable days, cloud fraction moves only loosely with k.
- `artifacts/figures/solar_forecasting/deep_dive/tbl/goes_bt13_strip_tbl.png`. Three band-13 frames on 23 May. Broken cloud drifts over the station as k goes from about 0.55 to 0.99 and back.
- `artifacts/figures/solar_forecasting/deep_dive/tbl/error_vs_horizon_tbl.png`. Table Mountain forecast error against horizon, split by day type.
- `artifacts/figures/solar_forecasting/deep_dive/tbl/error_by_day_type_and_sky_tbl.png`. The 1-hour error by day type and sky class at Table Mountain.
- `artifacts/figures/solar_forecasting/deep_dive/tbl/example_days_tbl.png`. A clear day and a variable day at Table Mountain, with the 1-hour smart-persistence forecast.

### Starting charts from `explore_stations.py`

These use uncorrected 10-minute k for the full year, with the sun more than 10° above the horizon. Two of them are a single day, 9 July 2025.

- `artifacts/figures/solar_forecasting/station_map.png`. The seven SURFRAD stations in the GOES-19 view.
- `artifacts/figures/solar_forecasting/k_all_stations_2025-01-01_2025-12-31.png`. Clear-sky index over the year at all seven stations.
- `artifacts/figures/solar_forecasting/k_distribution_2025-01-01_2025-12-31.png`. Histogram of k at each station.
- `artifacts/figures/solar_forecasting/k_by_solar_hour_2025-01-01_2025-12-31.png`. k by time of day.
- `artifacts/figures/solar_forecasting/k_vs_zenith_2025-01-01_2025-12-31.png`. k against sun angle.
- `artifacts/figures/solar_forecasting/k_variability_2025-01-01_2025-12-31.png`. Variability of k inside each 10-minute average.
- `artifacts/figures/solar_forecasting/monthly_energy_2025-01-01_2025-12-31.png`. Monthly PV energy produced against the clear-sky plan.
- `artifacts/figures/solar_forecasting/persistence_2025-01-01_2025-12-31.png`. Persistence error and correlation of k against horizon.
- `artifacts/figures/solar_forecasting/power_ramps_2025-01-01_2025-12-31.png`. PV power ramps. Changes above the dashed line are from clouds, not from the sun's motion.
- `artifacts/figures/solar_forecasting/diffuse_fraction_2025-01-01_2025-12-31.png`. Diffuse fraction against the Erbs model.
- `artifacts/figures/solar_forecasting/cloud_enhancement_2025-01-01_2025-12-31.png`. How long k stays above clear sky, and how much of the time it does.
- `artifacts/figures/solar_forecasting/closure_2025-01-01_2025-12-31.png`. Whether GHI, DNI, and DHI close with each other.
- `artifacts/figures/solar_forecasting/ghi_vs_clearsky_2025-07-09.png`. Measured and clear-sky GHI on 9 July 2025.
- `artifacts/figures/solar_forecasting/bon_detail_2025-07-09.png`. Bondville on 9 July 2025: irradiance, k, PV power, and k from GHI versus k from power.

## Key takeaways

- Penn State is the most variable of the seven stations, and it stays first if any one metric is dropped. Goodwin Creek is second because its days swing the hardest, mostly at summer midday (July median VI 18.4, and no clear day in June–August). Table Mountain is third because the power jumps, especially on summer afternoons. Bondville is close enough that second and third place trade if one metric is removed.
- A 1-hour smart-persistence forecast is off by about 11–17 W/m² on clear days and about 165–203 W/m² on variable days (Penn State 165, Goodwin Creek 169, Table Mountain 203). At 3 hours on variable days, that forecast has essentially no skill against a month-hour climatology (skill +0.04, +0.02, and −0.06).
- GOES-19 cloud fraction over a 50 km box tracks the ground when clear days and cloudy days are pooled (r = −0.84 on 1,110 scans) and loses the swings on variable days (r = −0.38). On 8 April at Penn State the box stays full while k runs from 0.38 to 1.08. On 18 June at Goodwin Creek the box stays about a tenth cloudy while k goes from 0.33 to 1.03 in ten minutes.
- About 11% of daylight minutes, reference PV output changes by more than 20% of capacity within 10 minutes. With the weather held fixed, the sun alone stays under about 5% in that same 10 minutes.
- Most readings above clear sky are the Ineichen model running low, not extra light from cloud edges. Real cloud enhancement is 1.8–3.4% of daylight intervals. Divide by the station alpha before treating k > 1 as a cloud effect. At Fort Peck and Sioux Falls the bias is also seasonal.
- The seven stations do not share the same weather on the same day (median r = 0.05). A neighbouring ground station is a weak substitute for a satellite image. There is a usable west-to-east lag at one day (Bondville yesterday to Penn State today, r = 0.38).
- Do not use Goodwin Creek sunrise and sunset until the drop in corrected clear-minute k to 0.95 at zenith 70–80° is checked. Mask Fort Peck's DNI outage of 25 November–1 December, and drop 9 March 2025 at the six stations where that day is missing.
