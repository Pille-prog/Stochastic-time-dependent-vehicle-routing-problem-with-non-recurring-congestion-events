"""Extract a deterministic sub-network fixture from the real Chengdu dataset.

The committed mini fixture (default arguments) powers the fast test suite:

    uv run python scripts/make_fixture.py

A larger, gitignored variant for local legacy runs (ticket 04) can be produced
with e.g. ``--target-nodes 260 --out data/fixture_large --days 601 602``.

Determinism: node selection grows a breadth-first ball around the depot
(node 0) over the undirected view of the graph, visiting neighbours in sorted
order, then keeps the strongly connected component containing the depot. No
randomness is involved anywhere, so the same inputs always produce the same
fixture byte for byte.

Output files mirror the real dataset's names and schemas exactly (the legacy
script resolves those names against its working directory). Node ids are
renumbered to a contiguous 0..N-1 range (depot stays 0); original ``Link`` ids
and the fixture→Chengdu node mapping are preserved for provenance.

``all_shortest_paths.csv`` is recomputed on the sub-network: arc weight =
length_km / (mean day-601 speed / 60), i.e. average travel time in minutes.
NOTE: these values are self-consistent within the fixture but are NOT the
legacy file's values — the real ``all_shortest_paths.csv`` was precomputed
from 44-day aggregated speeds. Tickets that compare against legacy outputs
must not assume equivalence (see ticket 03 comments).
"""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import networkx as nx
import pandas as pd

# Mirrors the legacy `Minute_start >= 300` filter. There is deliberately no upper
# cut: the legacy keeps periods past the 780 horizon end (vehicles finish trips,
# and speed interpolation reads periods, beyond it).
MIN_PERIOD_START_MINUTE = 300


def period_start_minute(period: str) -> int:
    """Minutes since 03:00 of a Period's start, as the legacy pipeline defines it."""
    start = period.split("-")[0]
    hour, minute = map(int, start.split(":"))
    return (hour - 3) * 60 + minute


def select_nodes(link: pd.DataFrame, target_nodes: int, min_nodes: int) -> list[int]:
    """Grow a BFS ball around the depot and keep the depot's strongly connected component."""
    graph = nx.DiGraph()
    graph.add_edges_from(zip(link["Node_Start"], link["Node_End"], strict=True))
    undirected = graph.to_undirected()

    # Grow well past the target: the SCC step below can discard a large share
    # of a BFS ball (one-way streets), so we oversample before trimming.
    ball_budget = 4 * target_nodes
    ball = [0]
    seen = {0}
    frontier = [0]
    while frontier and len(ball) < ball_budget:
        next_frontier: list[int] = []
        for node in frontier:
            for neighbour in sorted(undirected.neighbors(node)):
                if neighbour not in seen:
                    seen.add(neighbour)
                    ball.append(neighbour)
                    next_frontier.append(neighbour)
        frontier = next_frontier

    # Take the smallest BFS-ordered prefix whose depot SCC reaches the target.
    for prefix_size in range(min(target_nodes, len(ball)), len(ball) + 1):
        candidates = ball[:prefix_size]
        component = strongly_connected_with_depot(graph, candidates)
        if len(component) >= min_nodes:
            return sorted(component)
    raise SystemExit(
        f"could not find a strongly connected component of >= {min_nodes} nodes around the depot"
    )


def strongly_connected_with_depot(graph: nx.DiGraph, candidates: list[int]) -> set[int]:
    induced = graph.subgraph(candidates)
    for component in nx.strongly_connected_components(induced):
        if 0 in component:
            return set(component)
    return set()


def build_average_travel_time_graph(
    mini_link: pd.DataFrame, speeds: list[pd.DataFrame]
) -> nx.DiGraph:
    """Arc weights = length_km / (mean_speed / 60), matching the legacy data_average semantics."""
    mean_speed = (
        pd.concat(speeds, ignore_index=True).groupby("Link")["Speed"].mean().rename("MeanSpeed")
    )
    weighted = mini_link.merge(mean_speed, on="Link")
    graph = nx.DiGraph()
    # Row order matters: on duplicate arcs the last row wins, as in the legacy read_network.
    for row in weighted.itertuples():
        length_km = row.Length / 1000
        minutes = length_km / (row.MeanSpeed / 60)
        graph.add_edge(row.Node_Start, row.Node_End, weight=minutes, length=length_km)
    return graph


def compute_all_shortest_paths(graph: nx.DiGraph, n_nodes: int) -> pd.DataFrame:
    rows = []
    for source in range(n_nodes):
        _, node_paths = nx.single_source_dijkstra(graph, source, weight="weight")
        for client in range(n_nodes):
            path = node_paths[client]
            time = sum(graph[a][b]["weight"] for a, b in pairwise(path))
            length = sum(graph[a][b]["length"] for a, b in pairwise(path))
            rows.append(
                {
                    "Node": source,
                    "Client": client,
                    "ShortestPath": "->".join(map(str, path)),
                    "AverageTime": round(time, 6),
                    "Length": round(length, 6),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--source", type=Path, default=repo_root.parent)
    parser.add_argument("--out", type=Path, default=repo_root / "tests/fixtures/chengdu_mini")
    parser.add_argument("--days", type=int, nargs="+", default=[601])
    parser.add_argument("--target-nodes", type=int, default=45)
    parser.add_argument("--min-nodes", type=int, default=30)
    args = parser.parse_args()

    link = pd.read_csv(args.source / "link.csv")
    chosen = select_nodes(link, args.target_nodes, args.min_nodes)
    relabel = {old: new for new, old in enumerate(chosen)}

    mini_link = link[link["Node_Start"].isin(relabel) & link["Node_End"].isin(relabel)].copy()
    mini_link["Node_Start"] = mini_link["Node_Start"].map(relabel)
    mini_link["Node_End"] = mini_link["Node_End"].map(relabel)
    link_ids = set(mini_link["Link"])

    args.out.mkdir(parents=True, exist_ok=True)
    mini_link.to_csv(args.out / "link.csv", index=False)

    mini_speeds = []
    for day in args.days:
        for half in (0, 1):
            speed = pd.read_csv(args.source / f"speed[{day}]_[{half}].csv")
            minutes = speed["Period"].map(period_start_minute)
            mini = speed[(minutes >= MIN_PERIOD_START_MINUTE) & speed["Link"].isin(link_ids)].copy()
            mini["Speed"] = mini["Speed"].round(3)
            mini.to_csv(args.out / f"speed[{day}]_[{half}].csv", index=False)
            mini_speeds.append(mini)

    graph = build_average_travel_time_graph(mini_link, mini_speeds)
    paths = compute_all_shortest_paths(graph, len(chosen))
    paths.to_csv(args.out / "all_shortest_paths.csv", index=False)

    node_map = pd.DataFrame({"fixture_node": range(len(chosen)), "chengdu_node": chosen})
    node_map.to_csv(args.out / "node_map.csv", index=False)

    total = sum(f.stat().st_size for f in args.out.iterdir() if f.suffix == ".csv")
    print(
        f"fixture: {len(chosen)} nodes, {len(mini_link)} arcs, "
        f"days {args.days}, {total} bytes -> {args.out}"
    )


if __name__ == "__main__":
    main()
