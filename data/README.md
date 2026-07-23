# Instance data (not committed)

This directory holds the Chengdu instance data. Everything here except this file is gitignored — tests never depend on these files (they use the small committed fixture under `tests/fixtures/` instead).

## Expected files

| File | Size | Content |
|---|---|---|
| `link.csv` | ~0.4 MB | Road network arcs: `Link`, `Node_Start`, `Node_End`, coordinates, `Length` (meters) |
| `speed[601]_[0].csv` | ~23 MB | Historical speeds per link, day 601, morning period |
| `speed[601]_[1].csv` | ~23 MB | Historical speeds per link, day 601, afternoon period |
| `all_shortest_paths.csv` | ~907 MB | Precomputed shortest paths from every network node to every node (the `ShortestPathCache` source) |

Additional days (`speed[602..715]_[0/1].csv`) follow the same naming scheme. The full Chengdu experiment (`experiments/chengdu/config.yaml`) reads all 44 archive days (601–630 and 701–714, both periods) plus `link.csv` and `all_shortest_paths.csv`.

## Where to get them

The dataset is the Chengdu road-network and historical speed archive kept locally in the `Mega city` folder (one level above this repository) — there is no public download; ask the maintainer for the archive. The default experiment config reads the files from that parent folder directly (`data_dir: ../../..`), so nothing needs to be copied here. To run against copies under `data/` instead, place them here and point `data_dir` at this directory:

```powershell
Copy-Item "..\link.csv","..\all_shortest_paths.csv","..\speed[*]_[*].csv" -Destination data\
```

A future `DataSource` implementation will read these from a database instead (see ADR-0002); the on-disk layout here only matters for the CSV `DataSource`.
