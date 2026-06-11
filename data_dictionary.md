# Data Dictionary

The final modeling dataset is stored in `data/final_analysis_dataset.xlsx`, sheet `Rsf`. This sheet contains 93 overtaking observations, including 73 completed overtaking maneuvers and 20 right-censored observations.

| Variable | Type | Unit / coding | Description |
|---|---|---|---|
| `T` | survival time | seconds | Overtaking duration used as the time-to-completion outcome. |
| `overtaking` | event indicator | `1` = completed maneuver; `0` = right-censored observation | Event status for survival modeling. |
| `D` | continuous | meters | Overtaking distance. |
| `ID` | continuous | meters | Initial longitudinal distance between the overtaking and overtaken vehicles. |
| `V1` | continuous | km/h | Initial speed of the overtaking vehicle. |
| `V2` | continuous | km/h | Initial speed of the overtaken vehicle. |
| `DV` | continuous | km/h | Initial speed difference, calculated as `V1 - V2`. Negative values indicate that the overtaking vehicle was initially slightly slower than the overtaken vehicle at the start of the maneuver. |
| `H` | continuous | meters | Lateral distance between the overtaking and overtaken vehicles. |
| `VL` | continuous | meters | Length of the overtaken vehicle. |
| `INVASION` | binary | `0` = no oncoming traffic; `1` = oncoming traffic | Indicator for the presence of oncoming traffic during the maneuver. |
| `overtaken vehicles` | categorical | `0` = passenger car; `1` = heavy vehicle; `2` = motorcycle | Vehicle type of the overtaken vehicle. |
| `overtaking vehicles` | categorical | `0` = passenger car; `1` = heavy vehicle | Vehicle type of the overtaking vehicle. No motorcycle was observed as an overtaking vehicle in the final dataset. |

The file `data/traffic_flow_data.xlsx` contains traffic-flow records used for descriptive speed-distribution and traffic-composition analyses. The main speed variable is `车速` in km/h. The workbook includes vehicle-type sheets for passenger cars, heavy vehicles, and motorcycles.
