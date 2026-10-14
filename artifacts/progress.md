# Solar variability study: progress

Last updated: Tue Oct 13, 8:32 PM

## Setup
- [x] Branch `tal`, shared loader, clear-sky bias correction, 1-minute caches for all 7 stations

## Ranking: done
- [x] Calibrate the clear / variable / overcast day cutoffs (clear-day cutoff tightened from 2.0 to 1.5)
- [x] Rank all 7 stations on 6 variability measures
- [x] Pick the top 3, with a written reason for each: **Penn State, Goodwin Creek, Table Mountain**
- [x] One-slide chart of the ranking (`artifacts/figures/solar_forecasting/selection/selection_summary.png`)

## General analysis: done
- [x] Clear-sky bias, before vs after correction
- [x] Seasonal patterns by month
- [x] Missing and flagged data
- [x] Which stations move together
- [x] Findings page with captioned charts (`src/heimdall/solar_forecasting/eda/findings.md`)

## Deep dives on Penn State, Goodwin Creek, Table Mountain: done
- [x] Seasons, days, ramps agent: when each station is variable (month by hour)
- [x] Seasons, days, ramps agent: day types, power ramps, six case days per station
- [x] Forecast baselines agent: how simple forecasts fail on variable days (`deep_dive/comparison/summary_1h_nrmse_by_day_type.png`)
- [x] GOES-19 agent: satellite view over each station on the case days (`deep_dive/comparison/goes_vs_ground_k.png`)
- [x] Findings page: why each station, what it shows

## Final
- [ ] Every script runs cleanly, charts reviewed
