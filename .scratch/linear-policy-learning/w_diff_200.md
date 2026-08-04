# W trajectory diff: 200 legacy episodes vs 200 repo episodes

Same seed integers, different worlds (ticket 13 replaced the legacy's two shared global streams with per-Episode PCG64 spawning). Compare the *shape* of each component's trajectory, never its values.

## Norms

- legacy `|W|`: 37.34 after episode 1 -> 3732 after episode 200

- repo `|W|`: 27.78 after episode 1 -> 7502 after episode 200

## Peak magnitude here against the legacy's

| i | feature | legacy peak | repo peak | repo/legacy |
| --- | --- | --- | --- | --- |
| 18 | `distance_cost/100` | 334.1 | 3270 | 9.79 |
| 5 | `time^2*clients_left` | 74.26 | 689.8 | 9.29 |
| 19 | `earliness_cost/60` | 309.5 | 2759 | 8.91 |
| 6 | `time^2*clients_left^2` | 90.61 | 748.8 | 8.26 |
| 21 | `future_delay/2500` | 1225 | 1.001e+04 | 8.17 |
| 22 | `(future_delay/2500)^2` | 1249 | 7428 | 5.95 |
| 16 | `late_count/13` | 1397 | 6958 | 4.98 |
| 23 | `overtime_cost/180` | 537 | 2671 | 4.97 |
| 20 | `delay_cost/60` | 468.9 | 2297 | 4.90 |
| 4 | `clients_left^2*time` | 366.8 | 1704 | 4.64 |
| 15 | `congestion_signal` | 551.2 | 2359 | 4.28 |
| 3 | `clients_left^2` | 2284 | 8352 | 3.66 |
| 7 | `depot_dist_total` | 2019 | 7164 | 3.55 |
| 0 | `sqrt(clients_left)` | 1849 | 5712 | 3.09 |
| 9 | `depot_dist_max` | 1287 | 3837 | 2.98 |
| 1 | `time_left` | 1230 | 3620 | 2.94 |
| 8 | `depot_dist_min` | 439.9 | 1295 | 2.94 |
| 14 | `mean_earliness_diff` | 1152 | 3091 | 2.68 |
| 2 | `time_left^2` | 1185 | 3102 | 2.62 |
| 10 | `earliness_bin0` | 391.6 | 960.7 | 2.45 |
| 12 | `earliness_bin2` | 581.6 | 1331 | 2.29 |
| 11 | `earliness_bin1` | 477.5 | 1060 | 2.22 |
| 17 | `zero_pad` | 0 | 0 | 1.00 |
| 13 | `earliness_bin3` | 0 | 2641 | inf |

Over the 22 components the legacy actually trains: median 3.97x, range 2.22x-9.79x.

## Diverging in ours, bounded in the legacy

No component diverges in ours while staying bounded in the legacy.

## Every component, both sides

| i | feature | legacy final | legacy peak | legacy growth | repo final | repo peak | repo growth | repo norm share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `sqrt(clients_left)` | 1570 | 1849 | 0.92 | 937.8 | 5712 | 0.93 | 1.6% |
| 1 | `time_left` | 743.3 | 1230 | 0.72 | -1037 | 3620 | 0.79 | 1.9% |
| 2 | `time_left^2` | 1011 | 1185 | 0.94 | 223.5 | 3102 | 1.02 | 0.1% |
| 3 | `clients_left^2` | 1929 | 2284 | 1.07 | 4616 | 8352 | 1.38 | 37.9% |
| 4 | `clients_left^2*time` | 293 | 366.8 | 1.06 | 1085 | 1704 | 1.25 | 2.1% |
| 5 | `time^2*clients_left` | 50.37 | 74.26 | 0.81 | 462.9 | 689.8 | 1.56 | 0.4% |
| 6 | `time^2*clients_left^2` | 78.06 | 90.61 | 1.17 | 619.4 | 748.8 | 1.82 | 0.7% |
| 7 | `depot_dist_total` | 1814 | 2019 | 1.09 | 3438 | 7164 | 1.22 | 21.0% |
| 8 | `depot_dist_min` | -152.2 | 439.9 | 1.04 | 396.8 | 1295 | 1.67 | 0.3% |
| 9 | `depot_dist_max` | 399.9 | 1287 | 0.43 | -562.4 | 3837 | 0.96 | 0.6% |
| 10 | `earliness_bin0` | 391.6 | 391.6 | 2.06 | 877.3 | 960.7 | 3.38 | 1.4% |
| 11 | `earliness_bin1` | 444.9 | 477.5 | 1.33 | 447 | 1060 | 1.71 | 0.4% |
| 12 | `earliness_bin2` | 506.3 | 581.6 | 1.05 | 616.3 | 1331 | 0.99 | 0.7% |
| 13 | `earliness_bin3` | 0 | 0 | 1.00 | 273.2 | 2641 | 0.84 | 0.1% |
| 14 | `mean_earliness_diff` | -158.7 | 1152 | 0.42 | -338.2 | 3091 | 1.17 | 0.2% |
| 15 | `congestion_signal` | 303.5 | 551.2 | 0.82 | 447.3 | 2359 | 1.36 | 0.4% |
| 16 | `late_count/13` | 968.2 | 1397 | 0.83 | 111.5 | 6958 | 1.19 | 0.0% |
| 17 | `zero_pad` | 0 | 0 | 1.00 | 0 | 0 | 1.00 | 0.0% |
| 18 | `distance_cost/100` | 313.4 | 334.1 | 1.50 | 2028 | 3270 | 1.53 | 7.3% |
| 19 | `earliness_cost/60` | 186.9 | 309.5 | 1.05 | 1079 | 2759 | 1.73 | 2.1% |
| 20 | `delay_cost/60` | 215.9 | 468.9 | 0.87 | -28.07 | 2297 | 1.47 | 0.0% |
| 21 | `future_delay/2500` | 650.4 | 1225 | 0.77 | 3287 | 1.001e+04 | 1.44 | 19.2% |
| 22 | `(future_delay/2500)^2` | 491.2 | 1249 | 0.78 | -996.6 | 7428 | 1.44 | 1.8% |
| 23 | `overtime_cost/180` | -306.7 | 537 | 0.79 | -268.6 | 2671 | 0.73 | 0.1% |

## Largest single-episode steps (repo)

| episode | \|dW\| | \|W\| after |
| --- | --- | --- |
| 154 | 1.823e+04 | 2.081e+04 |
| 128 | 1.429e+04 | 1.737e+04 |
| 51 | 1.27e+04 | 1.656e+04 |
| 39 | 9587 | 1.298e+04 |
| 52 | 9582 | 9486 |
| 129 | 9213 | 1.098e+04 |
| 155 | 9184 | 1.483e+04 |
| 96 | 9163 | 1.198e+04 |
| 97 | 7890 | 6960 |
| 40 | 5944 | 8448 |

## Largest single-episode steps (legacy)

| episode | \|dW\| | \|W\| after |
| --- | --- | --- |
| 25 | 1926 | 3944 |
| 39 | 1798 | 4488 |
| 2 | 1709 | 1734 |
| 40 | 1695 | 3056 |
| 58 | 1610 | 2983 |
| 57 | 1546 | 4067 |
| 26 | 1205 | 2909 |
| 60 | 1201 | 4072 |
| 117 | 1006 | 3746 |
| 162 | 995.4 | 2863 |
