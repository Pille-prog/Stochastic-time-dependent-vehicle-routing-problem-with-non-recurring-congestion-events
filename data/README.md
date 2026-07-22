# Instance data (not committed)

This directory holds the Chengdu instance data. Everything here except this file is gitignored — tests never depend on these files (they use the small committed fixture under `tests/fixtures/` instead).

## Expected files

| File | Size | Content |
|---|---|---|
| `link.csv` | ~0.4 MB | Road network arcs: `Link`, `Node_Start`, `Node_End`, coordinates, `Length` (meters) |
| `speed[601]_[0].csv` | ~23 MB | Historical speeds per link, day 601, morning period |
| `speed[601]_[1].csv` | ~23 MB | Historical speeds per link, day 601, afternoon period |

Additional days (`speed[602..715]_[0/1].csv`) follow the same naming scheme and can be placed here when an experiment needs them.

## Where to get them

The dataset is the Chengdu road-network and historical speed archive kept locally in the `Mega city` folder (one level above this repository). Copy the files listed above into this directory:

```powershell
Copy-Item "..\link.csv","..\speed[601]_[0].csv","..\speed[601]_[1].csv" -Destination data\
```

A future `DataSource` implementation will read these from a database instead (see ADR-0002); the on-disk layout here only matters for the CSV `DataSource`.
