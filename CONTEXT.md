# STDVRP Orchestrator

Simulation and policy-optimization laboratory for the Stochastic Time-Dependent Vehicle Routing Problem (STDVRP) with non-recurring congestion events. The generic domain (network, simulation, policies) is instantiable for a specific problem; Chengdu is the first instance. Vocabulary follows Powell's sequential decision analytics framework; code identifiers are in English.

## Language

### Sequential decision core (Powell)

**State**:
The information available to make a decision at a point in simulated time: vehicle positions, pending clients, current velocities, elapsed time. "Vehicle position" is not one field: `last_node_reached` is only the last node a vehicle reached — it can be strictly mid-arc, having merely driven through that node on the way somewhere else — while `vehicle_standing` is the separate fact of whether the vehicle is actually standing there (parked, serving, or holding) rather than travelling past it (ticket 04, simulator-correctness, B1a/B1b, ADR-0005). The depot is not only the fleet's starting point — it is an interior node on 6.8% of cached shortest paths — so `last_node_reached == depot` is true for vehicles genuinely parked *and* for vehicles merely passing through; only `vehicle_standing` tells them apart.
_Avoid_: snapshot, context; treating `last_node_reached == depot` as "the vehicle is home" — that also needs `vehicle_standing`

**Policy**:
A rule that maps a State to a decision (which client each vehicle serves next). The first axis of variation: static, dynamic, Monte Carlo, Q-learning variants implement one interface. Every Policy is bound by the observability rule: it may read this Episode's State, its time windows and the static EpisodeGeometry (an offline historical prior, not an observation of this Episode), never the live congestion or velocity field the simulator itself sees (ticket 04, neural-policy, ADR-0006).
_Avoid_: strategy, agent, algorithm

**Model**:
The sequential decision model in Powell's sense: owns the transition function that advances the State given a decision and exogenous information (velocities, congestion events). It is the simulator the Policy interacts with.
_Avoid_: environment (reserved by RL literature for this very concept — never use it for data containers), engine

**Trainer**:
Runs training and evaluation episodes over the Model to fit and compare Policies.
_Avoid_: training_and_testing, runner

### Problem data

**RoadNetwork**:
The directed graph of nodes and arcs (links) of the instance's road network, with static attributes (coordinates, lengths).
_Avoid_: environment, graph data

**TrafficHistory**:
Historical speed observations per arc and time interval, used to derive time-dependent stochastic travel times.
_Avoid_: velocities data, environment

**DataSource**:
The boundary through which RoadNetwork and TrafficHistory are loaded. CSV files today; a database later. Only the origin of the data varies — the domain model does not.
_Avoid_: loader, reader

**TravelTimeModel**:
Derives stochastic time-dependent travel times (interpolated speeds, deviations) from TrafficHistory.
_Avoid_: DataCalculations

**CongestionGenerator**:
Generates non-recurring congestion events during an episode (by radius, by arc probability, bounded variants). The second axis of variation: one interface, several implementations.
_Avoid_: unexpected event creator

**Client**:
A demand point with a location on the RoadNetwork and a time window, to be served by a vehicle within the horizon.
_Avoid_: customer, node (a Client sits on a node but is not the node)

**EpisodeDemand**:
What one Episode must serve: the Clients drawn for it (in draw order) and the vehicle fleet size, both produced by the ClientGenerator from the episode seed.
_Avoid_: client list, instance (reserved for the problem instance)

**ShortestPathCache**:
Precomputed shortest paths from network nodes to Clients.
_Avoid_: shortest_path_memory

### Simulation

**Episode**:
One complete simulated run over the time horizon: clients are generated, congestion events occur, vehicles execute a Policy, costs are accumulated.
_Avoid_: iteration, run

**Horizon**:
The simulated time interval an Episode runs within, from `horizon_start_minute` to `episode_end_minute` (the hard stop). Not the same as when overtime starts: `shift_end_minute` is the vehicles' shift end, a separate, earlier clock that decisions and events can and do run past (ticket 02, simulator-correctness, B12) — with `horizon_start_minute` = 300 and `shift_end_minute` = 780, Episodes demonstrably run to 1148.
_Avoid_: time window (reserved for Clients); using "horizon end" to mean the shift end

**Unserved Client**:
A Client the Episode ends without ever having assigned a vehicle to. While the Episode is still running this is provisional — the Client is merely pending, and a vehicle could still reach it before its window closes. Once the Episode terminates it is final: the Client is abandoned, a different outcome from one served after its window closed (late, but served — see Client). An abandoned Client is priced at termination against the fixed reference clock `max(episode_end_minute, tau_episode)`, never the live `tau_episode` alone (ticket 03, simulator-correctness, B3, ADR-0004): a pending Client's window might still be met, but an abandoned one's never will be, so its price cannot depend on *when* the Episode happened to stop.
_Avoid_: late (reserved for a Client actually served past its window); treating "unserved" as one outcome — pending (Episode still running) and abandoned (Episode terminated) are priced by two different formulas for exactly this reason
