# Optimizing Fire Station Placement Based on Street Characteristics Using a Decision Tree Travel Time Model

ASI

## Overview

We predict how long a London fire engine takes to reach an incident from the characteristics of the streets it drives. We then use those predictions to decide where new fire stations should go. Team: Ricky Zhao and Sarp Akalin (ASI), targeting CSEF and possibly ISEF. We use London Fire Brigade (LFB) data, which records the station each engine actually left from, its turnout and driving times, and London's closure of 10 fire stations in 2014.

**Research question:** Does placing fire stations with travel times learned from street characteristics reduce real response times more than placing them by distance?

**Hypothesis:** Station sites chosen with a street-aware decision tree travel time model will give lower average and 90th-percentile response times on held-out 2025 London Fire Brigade calls than sites chosen by road distance or by the Kolesar travel time formula (Kolesar et al., 1975). Because travel times from a station that does not exist yet cannot be observed, we also test whether the model predicts a real change: the response times recorded after London closed 10 fire stations in 2014.

## Prior work

Fire station placement and fire engine travel times have been studied for about 50 years. Our project builds on five lines of work.

- **Fire engine travel time and distance.** Kolesar, Walker and Hausner (1975) found that a fire engine's travel time grows with the square root of distance on short trips, while it is still speeding up, and in a straight line on long trips. Kolesar and Blum (1973) showed that the average response distance in an area depends mainly on the area's size and the number of available engines. The Kolesar formula, refit on London trips, is our main baseline and the parametric part of our semi-parametric model.
- **Emergency vehicle travel time models.** Budge, Ingolfsson and Zerom (2010) modeled Calgary ambulance travel times from distance and showed that the spread of travel times also changes with distance. Westgate et al. (2013) estimated ambulance travel times along road routes with a Bayesian model. Poulton et al. (2018) studied how London ambulances move under blue lights. Urfalı and Eymen (2025) predicted fire response times in Kayseri, Turkey, with XGBoost on about 7,400 incidents (MAE 1.67 minutes, R² 0.46). These models use distance, time and location rather than the characteristics of the streets driven, and none of them uses the learned model to choose new station sites.
- **Learning street speeds from trip times.** Zhan, Hasan and Ukkusuri (2013) estimated travel times on each street from taxi trip records that give only the start, the end and the total time. Our learned fire truck speed per street segment applies the same idea to fire engines, which drive differently from taxis under lights and sirens.
- **Station placement models.** The set covering model (Toregas et al., 1971) finds the fewest stations that reach every area within a time limit. The maximal covering model (Church and ReVelle, 1974) reaches as many calls as possible with a fixed number of stations, which is our repositioning goal. The p-median model minimizes average travel time, and Larson's hypercube queuing model (1974) adds the chance that the nearest unit is busy. All of them need a travel time from every candidate site to every demand point, usually road distance at an assumed speed. Our contribution is to supply those travel times from a model trained on real fire responses and tested against real outcomes.
- **Station closures and their effects.** On 9 January 2014 London closed 10 fire stations: Belsize, Bow, Clerkenwell, Downham, Kingsland, Knightsbridge, Silvertown, Southwark, Westminster and Woolwich (London Fire Brigade, 2014). Taylor (2017) measured afterwards where response times got worse, using a spatial survival model. We use the same event as a test: whether a model trained only on calls from before the closures predicts the change that happened. Placement decisions have real consequences. That is why we test our travel times on real outcomes before trusting the placements they produce.

## Project plan

The project has three linked models. The travel time model predicts how long a trip takes, the route model predicts which streets an engine will use, and the placement model uses both to decide where firehouses should go.

1. **Travel time model.** Predicts how long an engine takes to reach an incident from the characteristics of the streets on its route and the conditions at the time. It is a LightGBM model trained on London Fire Brigade trips without station IDs, so it can score sites that have no station yet.
2. **Route model.** Imitates the routes fire engines currently take, which are not always the shortest path. We will rebuild our route generation to match that behavior, then pass the predicted routes to the travel time model and the placement model.
3. **Placement model.** Chooses firehouse locations at a fine level of detail: individual streets, corners and one-way roads, not grid cells or neighborhoods. It combines graph theory, with the street network as a directed graph so one-way streets are respected, and a semi-parametric travel time model: a parametric part for well-understood effects (the Kolesar distance formula refit on London trips, plus speed limits), and the decision tree model for the rest.
4. **Repositioning goal.** Given a fixed number of firehouses, reposition them to reach as many fires as possible as quickly as possible, measured against London Fire Brigade's own attendance standards (below): average first-engine attendance time and the share of calls where the first engine arrives within 10 minutes. Demand is past incident density (2023–24 calls); incident prediction is left for future work. A second experiment adds 1 to 3 new stations instead, and both are compared with today's locations on real 2025 calls.

## London attendance standards

We score placements against the standards London Fire Brigade sets for itself (London Fire Brigade, 2022), not US standards such as NFPA 1710. They apply to London as a whole, and LFB also tries to meet them in each borough.

- **First engine.** Arrives within an average of 6 minutes, and within 10 minutes on more than 90% of calls.
- **Second engine.** Arrives within an average of 8 minutes, and within 12 minutes on more than 95% of calls.
- **How the time is counted.** From the moment control mobilises the engine to its arrival, so it includes turnout as well as driving. Our travel time model predicts driving only, so when scoring a placement we add each station's recorded turnout time to the predicted driving time.

## Methodology

The travel time model is being built on London Fire Brigade data, so there are no results to report yet. These are the steps of that build.

1. **Data.** LFB incident and mobilisation records (Open Government Licence), 2009 onward, which give the station each engine left from, its turnout time and its travel time. Supporting sources: LFB fire station locations, the OS NGD road network and OpenStreetMap drive network (see London street data), Open-Meteo hourly weather, traffic counts, and Google Maps travel times.
2. **Where each engine started.** LFB records the station each engine was sent from and whether it left from the station or from somewhere else. We keep the trips that started at a station, so every route has a known start, and flag implausible implied speeds.
3. **Street characteristics per route.** For each firehouse-to-incident route on the street graph we compute path length, average street width, travel lanes, posted speed, the share of the route on bus lanes, bike lanes, A roads, dual carriageways, one-way streets and bridges or tunnels, plus signals per km, turns per km and average block length.
4. **Context features.** Hour, day and month, rush hour and night flags, weather, traffic, recent call load on the same engine and borough, units sent, and Google Maps civilian travel times.
5. **Model.** A hybrid of gradient-boosted decision trees: a parametric travel time prior (the Kolesar distance formula refit on London trips, using road-network distance), plus four LightGBM models with different random seeds trained on the remaining error. The target is driving time only, from leaving the station to arriving, which LFB records separately from turnout. LFB's attendance standards count from mobilisation, so placements are scored on predicted driving time plus each station's recorded turnout time.
6. **Evaluation.** Train on 2023 to 2024 and test on 2025, so the test is fully in the future. Placement asks the model about places it has not seen, so we will also test it by leaving out one firehouse at a time (see Validation plan). We compare the model with a median baseline, straight-line distance, road-network time, the Kolesar formula, linear regression and a small neural net. An ablation study removes one feature group at a time (street characteristics, traffic, weather, Google Maps) and reports the change in MAE, and SHAP or partial dependence plots show the effects of width, lanes, signals and turns. Quantile LightGBM models give a prediction range for each trip.

## Novel aspects

All of these are planned and are what make the project new compared with the prior work above.

- **Street-level features for real emergency routes (planned).** Each real London Fire Brigade response is linked to the width, lanes, speed limits, bus lanes, signals and turns of its street route, not just its distance.
- **Learned fire truck speed per street segment (planned core).** We will fit a speed for every street segment from its characteristics so that the summed routes match recorded response times. That produces a map of how fast fire trucks really move on each street, and it can score any candidate station site, which a model that uses station IDs cannot. We check the fitted speeds against the 2025 test calls. Zhan et al. (2013) estimated street speeds this way from taxi trips; we have not found it done for fire engines.
- **Placement for the slow calls (planned).** London's standard requires the first engine within 10 minutes on more than 90% of calls, but placement methods usually minimize the average. We will optimize the 90th-percentile time using quantile models, and account for the local engine already being busy (see Busy engines measured from real data).
- **Tested on a real closure (planned core).** Placement studies usually trust their travel time estimates without checking them on a real change. We test whether our model predicts the response times recorded after London's 2014 closures, and compare it with distance-based estimates.
- **Busy engines measured from real data (planned core).** When a station's engine is already out on another call, LFB sends one from a different station, which arrives later. For every 2023–24 incident we compare the station ground the incident is in with the station the first engine actually came from, and count how often each station's own engine was unavailable, split by station, hour of day and season. Classic placement models, such as Larson's hypercube model (1974), take busy rates as inputs, usually as one assumed value or a rate estimated from workload. We feed each station's measured rate into placement, so a site's score includes the chance that its engine is busy and the extra time for the backup engine from the next nearest station. We then test whether this predicts 2025 attendance times better than assuming the local engine is always free, and whether it changes which sites the optimizer picks.
- **Ranking the 10 closed stations (planned).** Using the model trained only on calls from before 2014, we put each closed station back one at a time and predict, on 2025 calls, the seconds of average first-engine attendance time it would save and how many more calls would be reached within 6 and 10 minutes. The result is a ranked list of which closure costs London the most today. We check the ranking against the real change each closed station's area saw after January 2014. Taylor (2017) measured where response times worsened; we add a prediction of how much each station is worth and a check of that prediction against real data.
- **Which street features slow fire engines (planned).** From the learned speed on each street segment, we describe which street features go with slower or faster fire engine travel: traffic calming such as speed humps (TfL Cycling Infrastructure Database), bus lanes, carriageway width (OS NGD), 20 mph speed limits (OpenStreetMap), traffic signals and turns. We report a ranked table of seconds per kilometre with bootstrap confidence intervals and a map of London's slowest streets for fire engines. This describes what the model learned, not cause and effect. A slow 20 mph street may be slow because of the limit or because it is narrow and lined with parked cars, and ordinary response data cannot tell these apart. So we do not use these numbers to predict what changing a street would do. Testing that would need a real street change, such as London's 2020 low-traffic neighbourhoods or its 20 mph rollouts, which we leave for future work.

## Validation plan

Response times from a station that does not exist yet cannot be observed, so a placement scored only by our own model would look good almost by definition. These tests check the model against real outcomes.

- **Real station closures (main test).** Train the travel time model on London Fire Brigade calls from before 9 January 2014, predict response times after the 10 closures, and compare with the times recorded in 2014. Do the same with road distance and the Kolesar formula. The model that best predicts the real change is the one we use for placement. This differs from testing on ordinary incidents. An ordinary test only checks trips from stations that already exist, but placement asks what happens when the set of stations changes. After the closures, areas that lost their station were served from stations farther away, often along routes those crews rarely drove. That is the same kind of change our optimizer makes when it adds, removes or moves a station. Removing the 10 stations in the model and comparing its predictions with the times London actually recorded is the only test in this plan that checks that kind of change against real data.
- **New locations.** Leave-one-firehouse-out cross-validation: train without one firehouse's trips, test on them, and repeat for every firehouse. This is the accuracy that matters for scoring sites that have no station yet, and we report it next to the time-split result.
- **Several travel time models.** Score every placement (ours, distance-based and today's stations) under our model, road network time and the Kolesar formula. A placement counts as better only if it wins under all three.
- **Statistics.** Paired bootstrap confidence intervals, resampling calls, on the difference in average and 90th-percentile travel time between placements.
- **Robustness.** Repeat the placement comparison for rush hour and night, bad weather and clear weather, and each borough, and sweep over the number of new stations.
- **Equity.** Report who gains: the change in travel time by borough and by neighborhood median income.

## Route model (planned)

The route model predicts which streets an engine drives from its firehouse to an incident. We do not assume engines take the shortest distance or the fastest route at posted speed limits. We assume they take the quickest route for an emergency vehicle, using a speed for each street learned from real emergency trips, because with lights and sirens on, drivers favor wide main roads and avoid narrow or blocked streets.

- **Road graph.** Build a directed street graph from OpenStreetMap with osmnx, keeping one-way streets and the street characteristics we already compute for each segment.
- **Learned street speeds.** A model predicts each segment's emergency vehicle speed from its characteristics (width, lanes, road type, posted speed, bus lanes, signals, turns) and the time of day.
- **Routing.** For each trip, find the quickest path from the starting firehouse to the incident with Dijkstra's algorithm under the learned speeds, and add up the segment times.
- **Training.** Adjust the speed model until the summed route times match recorded travel times across the training trips, re-routing after each round because faster streets can change which route is quickest. The target is driving time only, from departure to arrival, so the time it takes crews to leave the station is excluded.
- **Validation data.** Testing route choice needs GPS tracks of real trips, and no open dataset has them. The options are a Freedom of Information request to the London Fire Brigade for engine GPS logs and the London Ambulance Service study "Modelling Metropolitan-area Ambulance Mobility under Blue Light Conditions", whose data is not public but whose findings we can compare with. We will file the request early in case it takes a while or needs narrowing. If no tracks arrive, we will claim only that our routes give accurate travel times, not that they match the routes engines drive.
- **Validation method.** Map-match each GPS track onto the road graph with Valhalla or the leuvenmapmatching package, route the same trip with our model, and measure route overlap: the share of the real route's length that our route also uses, plus how often the two routes are identical.
- **Baselines.** Score the shortest-distance route and the fastest route at posted speed limits the same way. The route model is only useful if it matches real routes more closely than both, since it is meant to copy real routing, not improve on it.
- **Check without GPS tracks.** If our routes' summed times match held-out recorded travel times, the routes are plausible, but that does not prove engines drove them.

## London street data (OS Data Hub)

London is our study city. Ordnance Survey's road network gives the street detail our features need. It comes from the OS National Geographic Database (OS NGD) through the OS Data Hub.

- **What it has.** For every London road segment: average and minimum carriageway width, bus lane and cycle lane presence, one-way direction, road type (single or dual carriageway, roundabout, slip road), tunnels, and the climb in each direction. OS also has turn, width and height restrictions, traffic calming such as speed bumps, and measured average speeds by time of day.
- **Gaps.** The lane count field is not filled in yet, and there is no parking lane data. Lane counts, speed limits and traffic signals come from OpenStreetMap, and TfL's free Cycling Infrastructure Database can cross-check cycle lanes and traffic calming.
- **Access.** Sign up for the OS Data Hub Premium plan and download Greater London's road links through the OS NGD API (Features). The plan includes £1,000 of free premium API use each month. London has a few hundred thousand road links, which should take a few thousand requests and fit inside the free allowance, but we will confirm the price per request on the Data Hub before downloading.
- **Fallback.** If the free allowance does not cover it, OS offers organisations a free three to six month Data Exploration Licence, and UK universities can get OS data through Digimap.
- **Terms.** Credit the data as "Contains OS data © Crown copyright and database rights", and check the Premium plan terms for how long downloaded data can be kept.

## References

Budge, S., Ingolfsson, A., & Zerom, D. (2010). Empirical analysis of ambulance travel times: The case of Calgary emergency medical services. *Management Science*, 56(4). https://doi.org/10.1287/mnsc.1090.1142

Church, R., & ReVelle, C. (1974). The maximal covering location problem. *Papers of the Regional Science Association*, 32, 101–118.

Kolesar, P., & Blum, E. H. (1973). Square root laws for fire engine response distances. *Management Science*, 19(12), 1368–1378. https://doi.org/10.1287/mnsc.19.12.1368

Kolesar, P., Walker, W., & Hausner, J. (1975). Determining the relation between fire engine travel times and travel distances in New York City. *Operations Research*, 23(4), 614–627. https://doi.org/10.1287/opre.23.4.614

Larson, R. C. (1974). A hypercube queuing model for facility location and redistricting in urban emergency services. *Computers & Operations Research*, 1(1), 67–95.

London Fire Brigade. (2014, January). Brigade will remain world class and Londoners will still be safe, says authority chief ahead of station closures [News release].

London Fire Brigade. (2022). Fire facts: Incident response times. London Datastore.

Poulton, M., et al. (2018). Modelling metropolitan-area ambulance mobility under blue light conditions. arXiv:1812.03181.

Taylor, B. M. (2017). Spatial modelling of emergency service response times. *Journal of the Royal Statistical Society Series A*, 180(2), 433–453.

Toregas, C., Swain, R., ReVelle, C., & Bergman, L. (1971). The location of emergency service facilities. *Operations Research*, 19(6), 1363–1373. https://doi.org/10.1287/opre.19.6.1363

Urfalı, T., & Eymen, A. (2025). Estimating fire response times and planning optimal routes using GIS and machine learning techniques. *Geomatics*, 5(4), 58. https://doi.org/10.3390/geomatics5040058

Westgate, B. S., Woodard, D. B., Matteson, D. S., & Henderson, S. G. (2013). Travel time estimation for ambulances using Bayesian data augmentation. *Annals of Applied Statistics*, 7(2), 1139–1161.

Zhan, X., Hasan, S., & Ukkusuri, S. V. (2013). Urban link travel time estimation using large-scale taxi data with partial information. *Transportation Research Part C*, 33.
