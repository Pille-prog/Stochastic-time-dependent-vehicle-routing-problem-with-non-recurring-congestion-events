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

**Reference card**:
A completed Policy's frozen per-seed costs — the fixed opponent every later run in an effort is compared against, seed by seed (ticket 01, neural-policy). Never recomputed once frozen: a moving comparison target would make "did it improve" unanswerable. Carries two disjoint seed sets for two disjoint purposes — `evaluation_seeds` (what a training run's live report reads from, ticket 07) select checkpoints and hyperparameters and are therefore contaminated for a verdict by construction; `test_seeds` are the held-out verdict set, touched only when an effort is ready to answer "does it win".
_Avoid_: baseline (ambiguous with the Policy it was captured from — the card is the frozen *measurement*, not the Policy itself); benchmark

**Approximator**:
What maps an observation to a decision's value. The variation point *inside* a Policy — linear weights or a neural network — with the decision rule unchanged: the Policy still takes the same argmin over the same feasible actions. Two Policies differing only in Approximator are the comparison this lab is built to make.
_Avoid_: model (reserved for Powell's Model, the simulator); network (only one Approximator is a network); estimator (reserved for how the Approximator is *fitted*, which varies independently)

**Myopic base**:
The projected cost of assigning one vehicle to one Client, computed from the Episode's time windows and the static EpisodeGeometry prior, and added to the Approximator's output from outside its parameters. It is what a Policy decides with when it has learned nothing at all (ticket 15, neural-policy, ADR-0010).
_Avoid_: warm start (a warm start is a point training moves away from; nothing moves away from this — the distinction is the whole reason it exists); immediate cost (it is a projection from an offline prior, not what the simulator will charge for the leg); post-decision state (Powell's term, reserved for the state *after* a decision and *before* the exogenous information arrives — this is a cost projection, not a state)

**Residual approximator**:
The only learned term when a Policy is decomposed into a Myopic base plus a correction: what the Approximator adds on top of the base, rather than the whole value it would otherwise have to reconstruct.
_Avoid_: correction, delta; advantage (reserved by RL literature for the value of an action minus the value of its state — a different decomposition against a different reference)

**Null policy**:
The Policy an Approximator produces before anything has been learned — the floor a trained Policy must clear before "it learned" means anything. Not a fixed object: with a Residual approximator the null *is* the Myopic base, a competent dispatcher rather than a random or nearest-Client one, which makes the same phrase a much stronger claim. Every reported improvement therefore names the null it was measured against (ticket 08, neural-policy).
_Avoid_: random policy (it has never been random here); "the untrained network" without naming which base it sits on — two nulls of the same architecture are not the same opponent

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
