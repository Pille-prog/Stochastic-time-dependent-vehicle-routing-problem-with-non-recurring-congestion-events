# STDVRP Orchestrator

Simulation and policy-optimization laboratory for the Stochastic Time-Dependent Vehicle Routing Problem (STDVRP) with non-recurring congestion events. The generic domain (network, simulation, policies) is instantiable for a specific problem; Chengdu is the first instance. Vocabulary follows Powell's sequential decision analytics framework; code identifiers are in English.

## Language

### Sequential decision core (Powell)

**State**:
The information available to make a decision at a point in simulated time: vehicle positions, pending clients, current velocities, elapsed time.
_Avoid_: snapshot, context

**Policy**:
A rule that maps a State to a decision (which client each vehicle serves next). The first axis of variation: static, dynamic, Monte Carlo, Q-learning variants implement one interface.
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

**ShortestPathCache**:
Precomputed shortest paths from network nodes to Clients.
_Avoid_: shortest_path_memory

### Simulation

**Episode**:
One complete simulated run over the time horizon: clients are generated, congestion events occur, vehicles execute a Policy, costs are accumulated.
_Avoid_: iteration, run

**Horizon**:
The simulated time interval (start minute, end minute) within which all decisions and events happen.
_Avoid_: time window (reserved for Clients)
