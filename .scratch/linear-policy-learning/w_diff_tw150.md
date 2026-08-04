# W trajectory diff: 200 legacy episodes vs 200 repo episodes

Same seed integers, different worlds (ticket 13 replaced the legacy's two shared global streams with per-Episode PCG64 spawning). Compare the *shape* of each component's trajectory, never its values.

## Norms

- legacy `|W|`: 26.73 after episode 1 -> 2895 after episode 200

- repo `|W|`: 22.45 after episode 1 -> 1.575e+04 after episode 200

## Peak magnitude here against the legacy's

| i | feature | legacy peak | repo peak | repo/legacy |
| --- | --- | --- | --- | --- |
| 18 | `distance_cost/100` | 338.1 | 4668 | 13.81 |
| 19 | `earliness_cost/60` | 306.4 | 3780 | 12.33 |
| 11 | `earliness_bin1` | 174.1 | 1727 | 9.92 |
| 21 | `future_delay/2500` | 1305 | 1.057e+04 | 8.10 |
| 22 | `(future_delay/2500)^2` | 840.3 | 6622 | 7.88 |
| 15 | `congestion_signal` | 365.2 | 2506 | 6.86 |
| 6 | `time^2*clients_left^2` | 173 | 1075 | 6.21 |
| 23 | `overtime_cost/180` | 742.2 | 4578 | 6.17 |
| 12 | `earliness_bin2` | 369.7 | 1824 | 4.93 |
| 1 | `time_left` | 904.9 | 4439 | 4.91 |
| 5 | `time^2*clients_left` | 170.2 | 826 | 4.85 |
| 2 | `time_left^2` | 759.3 | 3665 | 4.83 |
| 16 | `late_count/13` | 1296 | 6234 | 4.81 |
| 4 | `clients_left^2*time` | 624.6 | 2941 | 4.71 |
| 9 | `depot_dist_max` | 1146 | 5005 | 4.37 |
| 3 | `clients_left^2` | 2422 | 1.04e+04 | 4.29 |
| 7 | `depot_dist_total` | 2200 | 9332 | 4.24 |
| 0 | `sqrt(clients_left)` | 1888 | 7581 | 4.02 |
| 8 | `depot_dist_min` | 568.1 | 2227 | 3.92 |
| 20 | `delay_cost/60` | 538.5 | 1640 | 3.05 |
| 14 | `mean_earliness_diff` | 1138 | 3364 | 2.96 |
| 10 | `earliness_bin0` | 176.4 | 465.8 | 2.64 |
| 17 | `zero_pad` | 0 | 0 | 1.00 |
| 13 | `earliness_bin3` | 0 | 3653 | inf |

Over the 22 components the legacy actually trains: median 4.84x, range 2.64x-13.81x.

## Diverging in ours, bounded in the legacy

No component diverges in ours while staying bounded in the legacy.

## Every component, both sides

| i | feature | legacy final | legacy peak | legacy growth | repo final | repo peak | repo growth | repo norm share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `sqrt(clients_left)` | 1052 | 1888 | 0.72 | 1920 | 7581 | 0.82 | 1.5% |
| 1 | `time_left` | 58.38 | 904.9 | 0.47 | -2207 | 4439 | 0.62 | 2.0% |
| 2 | `time_left^2` | 28.87 | 759.3 | 0.51 | -320.3 | 3665 | 0.73 | 0.0% |
| 3 | `clients_left^2` | 1495 | 2422 | 0.84 | 9137 | 1.04e+04 | 1.09 | 33.6% |
| 4 | `clients_left^2*time` | 504.1 | 624.6 | 0.94 | 2620 | 2941 | 1.39 | 2.8% |
| 5 | `time^2*clients_left` | 164.7 | 170.2 | 1.05 | 435.5 | 826 | 1.11 | 0.1% |
| 6 | `time^2*clients_left^2` | 157.9 | 173 | 0.98 | 965 | 1075 | 1.42 | 0.4% |
| 7 | `depot_dist_total` | 1220 | 2200 | 0.77 | 6717 | 9332 | 1.12 | 18.2% |
| 8 | `depot_dist_min` | -200.4 | 568.1 | 0.93 | -2134 | 2227 | 1.19 | 1.8% |
| 9 | `depot_dist_max` | 166.2 | 1146 | 0.48 | -3892 | 5005 | 0.95 | 6.1% |
| 10 | `earliness_bin0` | 1.429 | 176.4 | 0.85 | -179.4 | 465.8 | 1.54 | 0.0% |
| 11 | `earliness_bin1` | 171.4 | 174.1 | 0.98 | 1403 | 1727 | 1.04 | 0.8% |
| 12 | `earliness_bin2` | -177.7 | 369.7 | 0.66 | 941.2 | 1824 | 1.23 | 0.4% |
| 13 | `earliness_bin3` | 0 | 0 | 1.00 | 1477 | 3653 | 1.13 | 0.9% |
| 14 | `mean_earliness_diff` | -805.4 | 1138 | 1.04 | -2655 | 3364 | 1.28 | 2.8% |
| 15 | `congestion_signal` | 217.7 | 365.2 | 0.96 | -422 | 2506 | 0.97 | 0.1% |
| 16 | `late_count/13` | 923.3 | 1296 | 0.72 | -635 | 6234 | 0.94 | 0.2% |
| 17 | `zero_pad` | 0 | 0 | 1.00 | 0 | 0 | 1.00 | 0.0% |
| 18 | `distance_cost/100` | 225 | 338.1 | 1.43 | 3906 | 4668 | 1.73 | 6.1% |
| 19 | `earliness_cost/60` | 74.22 | 306.4 | 0.66 | 1928 | 3780 | 0.68 | 1.5% |
| 20 | `delay_cost/60` | 416.6 | 538.5 | 1.02 | -109.4 | 1640 | 1.03 | 0.0% |
| 21 | `future_delay/2500` | 1115 | 1305 | 0.87 | 6564 | 1.057e+04 | 1.36 | 17.4% |
| 22 | `(future_delay/2500)^2` | 295.4 | 840.3 | 0.36 | 1573 | 6622 | 0.79 | 1.0% |
| 23 | `overtime_cost/180` | -35.21 | 742.2 | 0.19 | -2452 | 4578 | 1.95 | 2.4% |

## Largest single-episode steps (repo)

| episode | \|dW\| | \|W\| after |
| --- | --- | --- |
| 43 | 1.795e+04 | 2.134e+04 |
| 195 | 1.68e+04 | 2.221e+04 |
| 87 | 1.451e+04 | 1.79e+04 |
| 114 | 1.44e+04 | 1.877e+04 |
| 25 | 1.224e+04 | 1.519e+04 |
| 88 | 1.174e+04 | 9912 |
| 154 | 1.076e+04 | 1.679e+04 |
| 115 | 9911 | 1.278e+04 |
| 139 | 9754 | 1.475e+04 |
| 152 | 9348 | 1.451e+04 |

## Largest single-episode steps (legacy)

| episode | \|dW\| | \|W\| after |
| --- | --- | --- |
| 2 | 1844 | 1865 |
| 50 | 1809 | 4249 |
| 35 | 1409 | 2700 |
| 49 | 1335 | 2575 |
| 53 | 1115 | 4437 |
| 44 | 1098 | 2192 |
| 54 | 1087 | 3710 |
| 61 | 1050 | 3605 |
| 51 | 1020 | 3573 |
| 148 | 984.4 | 2757 |
