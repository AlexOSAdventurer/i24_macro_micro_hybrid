"""Check mass conservation for the no-masks LWR case."""
import os, sys, json
sys.path.insert(0, "/home/richarwa/SecondSSD/I24/i24_macroscopic")

from simulation import Simulation, GroundTruthStore

def total_mass(network):
    return sum(c.mass for road in network.roads.values()
               for c in road.cells.values())

def boundary_mass(network):
    """Mass on cells with no inflow OR no outflow (these get overwritten each step)."""
    return sum(c.mass for road in network.roads.values()
               for c in road.cells.values()
               if len(c.inflow_connections) == 0 or len(c.outflow_connections) == 0)

with open("/home/richarwa/SecondSSD/I24/i24_macroscopic/i24_motion_to_dataset.json") as f:
    config = json.load(f)

sim = Simulation.from_json(
    json_path=os.path.join(config["storage_locations"]["simulation_dataset"], "network.json"),
    time_resolution=config["time_step"],
    origin_time=config["time_origin"],
    min_cell_length=100.0,
)
gt = GroundTruthStore.from_parquet(
    os.path.join(config["storage_locations"]["simulation_dataset"], "micro.parquet"),
    os.path.join(config["storage_locations"]["simulation_dataset"], "macro.parquet"),
)
sim.initialize_from_ground_truth(gt, time_value=config["time_origin"])

# No bridge / no masks.
print(f"{'step':>4} {'M_total':>14} {'M_interior':>14} {'M_boundary':>14} {'dM_total':>12} {'dM_interior':>12}")
prev_total = total_mass(sim.network)
prev_interior = prev_total - boundary_mass(sim.network)
print(f"{0:>4} {prev_total:>14.4f} {prev_interior:>14.4f} {boundary_mass(sim.network):>14.4f} {'-':>12} {'-':>12}")

for i in range(1, 11):
    sim.step()
    M = total_mass(sim.network)
    Mb = boundary_mass(sim.network)
    Mi = M - Mb
    print(f"{i:>4} {M:>14.4f} {Mi:>14.4f} {Mb:>14.4f} {M - prev_total:>12.4f} {Mi - prev_interior:>12.4f}")
    prev_total = M
    prev_interior = Mi
