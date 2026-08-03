import sys
sys.path.append("..")
import simulation
from simulation import TriangularFD, Network, Road
from functools import partial
import pandas as pd

class Component:
    def __init__(self, s: float, velocity: float):
        self.s = s
        self.velocity = velocity

class ConstantRegion(Component):
    def __init__(self, s: float, region_length: float, density: float):
        super().__init__(s, 0.0)
        self.region_length = region_length
        self.density = density

class Mask(Component):
    def __init__(self, rear: Component, front: Component):
        super().__init(0.0, 0.0)
        self.rear = rear
        self.front = front

    @property
    def s(self):
        length = self.rear.region_length if (isinstance(self.rear, ConstantRegion)) else 0.0
        return self.rear.s + length

    @property
    def velocity(self):
        return self.rear.velocity

    @property
    def region_length(self):
        return self.front.s - self.s

class SilentBoundary(Component):
    def __init__(self, s: float, velocity: float = 0.0):
        super().__init__(s, 0.0)

class MovingBoundary(Component):
    def __init__(self, s: float, boundary_init_function: callable):
        super().__init__(s, 0.0)
        self.density = 0.0
        self.boundary_init_function = partial(boundary_init_function, self)

class WaveFront(Component):
    def __init__(self, s: float, velocity: float):
        super().__init__(s, velocity)

class Solver:
    def __init__(self, fd: TriangularFD, initial_state: list[Component] =[]):
        self.current_state = sorted(initial_state, key=lambda v: v.s)
        self.t = 0.0
        self.breakpoints = {}
        self.fd = fd
        for entry in self.current_state:
            if (isinstance(entry, MovingBoundary)):
                entry.boundary_init_function(self)
        self.resolve_contacts() # Create wave fronts based off the initial conditions.

    def register_breakpoint(self, callback: callable, t: float):
        if t not in self.breakpoints:
            self.breakpoints[t] = []
        self.breakpoints[t].append(callback)

    def time_to_contact(self, left_component: Component, right_component: Component, eps=1e-4):
        if (isinstance(left_component, ConstantRegion) or isinstance(right_component, ConstantRegion)):
            return float('inf')
        s_difference = right_component.s - left_component.s
        closing_rate = left_component.velocity - right_component.velocity
        time_to_contact = s_difference / closing_rate if (closing_rate > eps) else float('inf')
        return time_to_contact

    def get_no_constant_regions(self):
        result = []
        for i in range(len(self.current_state)):
            if (not isinstance(self.current_state[i], ConstantRegion)):
                result.append(i)
        return result

    def get_contacts_in_time_region(self, dt: float, eps: float = 1e-4):
        results = []
        no_constant_regions = self.get_no_constant_regions()
        for j in range(len(no_constant_regions) - 1):
            left_component_index = no_constant_regions[j]
            right_component_index = no_constant_regions[j + 1]
            left_component = self.current_state[left_component_index]
            right_component = self.current_state[right_component_index]
            current_ttc = self.time_to_contact(left_component, right_component)
            if (current_ttc <= (dt - eps)):
                results.append(current_ttc)
        breakpoint_times = [t for t in sorted(list(self.breakpoints.keys())) if (t > self.t) and ((t - self.t - dt - eps) <= 0.0)]
        for breakpoint in breakpoint_times:
            results.append(breakpoint - self.t)
        return results
        
    def determine_next_breakpoint(self, dt):
        contacts = sorted(self.get_contacts_in_time_region(dt))
        if (len(contacts) == 0):
            return float('inf')
        return contacts[0]
    
    def advance(self, dt: float = 0.001):
        ttc = self.determine_next_breakpoint(dt)
        while (ttc <= dt):
            self.advance_fronts(ttc)
            self.t += ttc
            self.resolve_breakpoints()
            self.resolve_contacts()
            dt -= ttc
            ttc = self.determine_next_breakpoint(dt)
        self.advance_fronts(dt)
        self.t += dt

    # We assume there are no contacts in this temporal region
    def advance_fronts(self, dt, ignore_contacts=False):
        contacts = self.get_no_constant_regions()
        for contact_index in contacts:
            constant_region_left_index = contact_index - 1
            constant_region_right_index = contact_index + 1
            contact = self.current_state[contact_index]
            #print("Contact ", contact)
            if (constant_region_left_index >= 0):
                constant_region_left = self.current_state[constant_region_left_index]
                #print("Constant_region_left ", constant_region_left)
                if (isinstance(constant_region_left, ConstantRegion)):
                    constant_region_left.region_length += (contact.velocity * dt)
                else:
                    raise Exception("Left region is not a constant region!")

            if (constant_region_right_index < len(self.current_state)):
                constant_region_right = self.current_state[constant_region_right_index]
                #print("Constant_region_right ", constant_region_right)
                if (isinstance(constant_region_right, ConstantRegion)):
                    constant_region_right.s += (contact.velocity * dt)
                    constant_region_right.region_length -= (contact.velocity * dt)
                else:
                    raise Exception("Right region is not a constant region!")
            contact.s += (contact.velocity * dt)
            #print("-------")
 
    # All contacts within eps distance will be resolved
    def resolve_contacts(self, eps: float = 1e-8):
        # First, any constant regions that straddle a silent or moving boundary need to be split, with the split at the boundary.
        # This should only happen at the very beginning of simulation run.
        regions_to_add = []
        boundaries = [(i, b) for i, b in enumerate(self.current_state) if isinstance(b, (SilentBoundary, MovingBoundary))]
        for i in range(len(self.current_state)):
            component = self.current_state[i]
            if (isinstance(component, ConstantRegion)):
                for (j, b) in boundaries:
                    if (component.s < b.s) and ((component.s + component.region_length) > b.s):
                        original_length = component.region_length
                        component.region_length = (b.s - component.s)
                        new_component = ConstantRegion(s=b.s, region_length=original_length - component.region_length, density=component.density)
                        new_component_index = j + 1 + len(regions_to_add)
                        regions_to_add.append((new_component_index, new_component))

        for (index, component) in regions_to_add:
            self.current_state.insert(index, component)

        # Second, remove all components lying within a mask.
        # This should only happen at the very beginning of simulation run.
        components_to_delete = []
        masks = [m for m in self.current_state if isinstance(m, Mask)]
        for i in range(len(self.current_state)):
            component = self.current_state[i]
            for mask in masks:
                if ((component.s >= mask.s) and (component.s <= (mask.s + mask.region_length)) and (component != mask.rear) and (component != mask.front)):
                    components_to_delete.append(i)
        self.current_state = [c for i, c in enumerate(self.current_state) if i not in components_to_delete]

        # Third, we mark identical constant regions and resolve them by replacing them with a single larger region.
        currently_building_larger_region = False
        current_start_i = 0
        current_end_i = 0
        regions_to_delete = []
        regions_to_add = []
        """
        print("Resolving contacts")
        for i, c in enumerate(self.current_state):
            if (isinstance(c, (SilentBoundary, MovingBoundary))):
                print("Boundary ", i, c.s, c.velocity)
            elif (isinstance(c, (ConstantRegion))):
                print("Constant region ", i, c.s, c.region_length, c.density)
            elif (isinstance(c, (WaveFront))):
                print("Wave ", i, c.s, c.velocity)
        """
        for i in range(len(self.current_state)):
            component = self.current_state[i]
            merge_list = []
            if (isinstance(component, ConstantRegion)):
                if currently_building_larger_region:
                    start_region = self.current_state[current_start_i]
                    if (abs(start_region.density - component.density) < eps):
                        current_end_i = i
                    elif (current_start_i != current_end_i):
                        # Mark merge here, reset start_i and end_i
                        merge_list = [i for i in range(current_start_i, current_end_i + 1)]
                        current_start_i = i
                        current_end_i = i
                else:
                    currently_building_larger_region = True
                    current_start_i = i
                    current_end_i = i
            else:
                currently_building_larger_region = False
                if (isinstance(self.current_state[current_start_i], ConstantRegion)):
                    merge_list = [i for i in range(current_start_i, current_end_i + 1)]
            if (len(merge_list) > 0):
                for r in merge_list:
                    regions_to_delete.append(r + len(regions_to_add))

                density = self.current_state[merge_list[0]].density
                start_s = min([self.current_state[r].s for r in merge_list])
                region_length = sum([self.current_state[r].region_length for r in merge_list])
                new_region_index = max(merge_list) + len(regions_to_add) + 1
                regions_to_add.append((new_region_index, ConstantRegion(s=start_s, region_length=region_length, density=density)))

        #print("regions_to_add ", regions_to_add)
        #print("regions_to_delete ", regions_to_delete)

        for (index, component) in regions_to_add:
            self.current_state.insert(index, component)

        self.current_state = [c for i, c in enumerate(self.current_state) if i not in regions_to_delete]
        """
        for i, c in enumerate(self.current_state):
            if (isinstance(c, (SilentBoundary, MovingBoundary))):
                print("Boundary ", i, c.s, c.velocity)
            elif (isinstance(c, (ConstantRegion))):
                print("Constant region ", i, c.s, c.region_length, c.density)
            elif (isinstance(c, (WaveFront))):
                print("Wave ", i, c.s, c.velocity)
        """
        
        # Fourth, we mark colliding wave fronts and solve them by deleting them and their contained intermediate region.
        # If one of the fronts is actually a boundary (Silent or Moving), we won't delete the boundary.
        fronts_to_delete = []
        for i in range(len(self.current_state) - 2):
            j = i + 1
            k = i + 2
            left_front = self.current_state[i]
            intermediate = self.current_state[j]
            right_front = self.current_state[k]
            if (isinstance(left_front, (SilentBoundary, MovingBoundary, WaveFront)) and isinstance(right_front, (SilentBoundary, MovingBoundary, WaveFront))):
                distance = abs(right_front.s - left_front.s)
                closing_rate = left_front.velocity - right_front.velocity
                if (distance < eps) and (closing_rate > eps):
                    if (isinstance(left_front, WaveFront)):
                        fronts_to_delete.append(i)
                    if (isinstance(right_front, WaveFront)):
                        fronts_to_delete.append(k)
                    fronts_to_delete.append(j) # We always delete their contained intermediate region
        #print("Deleting fronts ", fronts_to_delete)
        self.current_state = [c for i, c in enumerate(self.current_state) if i not in fronts_to_delete]
        # Fifth, we mark colliding constant regions and generate an appropriate wave front between them.
        # Moving Boundaries count as a constant region
        components_to_insert = []
        for i in range(len(self.current_state) - 1):
            left_region = self.current_state[i]
            right_region = self.current_state[i + 1]
            two_constant_regions = isinstance(left_region, ConstantRegion) and isinstance(right_region, ConstantRegion)
            left_moving_boundary_right_constant_region = isinstance(left_region, MovingBoundary) and isinstance(right_region, ConstantRegion)
            right_moving_boundary_left_constant_region = isinstance(left_region, ConstantRegion) and isinstance(right_region, MovingBoundary)
            contacting = False
            if two_constant_regions:
                contacting = (abs(left_region.s + left_region.region_length - right_region.s) < eps)
            elif left_moving_boundary_right_constant_region:
                contacting = (abs(left_region.s - right_region.s) < eps)
            elif right_moving_boundary_left_constant_region:
                contacting = (abs(left_region.s + left_region.region_length - right_region.s) < eps)
            must_resolve = contacting and (abs(left_region.density - right_region.density) > eps)
            if must_resolve:
                if (two_constant_regions or left_moving_boundary_right_constant_region or right_moving_boundary_left_constant_region):
                    left_density = left_region.density
                    right_density = right_region.density

                    contact_discontinuity = ((left_density <= self.fd.rho_c) and (right_density <= self.fd.rho_c)) or ((left_density >= self.fd.rho_c) and (right_density >= self.fd.rho_c))
                    shock = (left_density < self.fd.rho_c) and (right_density > self.fd.rho_c)
                    # Contact discontinuity or shock? We insert a corresponding wave front here to their characteristic speed
                    if contact_discontinuity or shock:
                        velocity = self.fd.shock_speed(left_density, right_density)
                        moving_boundary_should_replace_wave = False
                        if (isinstance(left_region, MovingBoundary)):
                            moving_boundary_should_replace_wave = (velocity <= left_region.velocity)
                        elif (isinstance(right_region, MovingBoundary)):
                            moving_boundary_should_replace_wave = (velocity >= right_region.velocity)
                        #print("Contact discontinuity or shock: ", left_density, right_density)
                        if not moving_boundary_should_replace_wave:
                            if (isinstance(left_region, MovingBoundary)):
                                rear_placement_index = i + 1 + len(components_to_insert)
                                new_region = ConstantRegion(s=right_region.s, region_length=0.0, density=left_density)
                                components_to_insert.append((rear_placement_index, new_region))
                            front_placement_index = i + 1 + len(components_to_insert)
                            
                            new_front = WaveFront(right_region.s, velocity)
                            components_to_insert.append((front_placement_index, new_front))
                            if (isinstance(right_region, MovingBoundary)):
                                rear_placement_index = i + 1 + len(components_to_insert)
                                new_region = ConstantRegion(s=right_region.s, region_length=0.0, density=right_density)
                                components_to_insert.append((rear_placement_index, new_region))

                    # Rarefaction shock? We insert a two wave fronts here, one that moves backwards, one that moves forwards, and an intermediate p_c state between
                    elif (left_density > self.fd.rho_c) and (right_density < self.fd.rho_c):
                        if (not isinstance(left_region, MovingBoundary)):
                            left_front_index = i + 1 + len(components_to_insert)
                            left_front_component = WaveFront(right_region.s, -self.fd.w)
                            components_to_insert.append((left_front_index, left_front_component))

                        intermediate_state_index = i + 1 + len(components_to_insert)
                        intermediate_component = ConstantRegion(right_region.s, 0.0, self.fd.rho_c)
                        components_to_insert.append((intermediate_state_index, intermediate_component))

                        if (not isinstance(right_region, MovingBoundary)):
                            right_front_index = i + 1 + len(components_to_insert)
                            right_front_component = WaveFront(right_region.s, self.fd.v_f)
                            components_to_insert.append((right_front_index, right_front_component))
                    else:
                        print(left_region, right_region, left_density, right_density)
                        raise Exception("Case is occurring that doesn't fit into any of these!")
        #print("Inserting components ", components_to_insert)
        for (index, component) in components_to_insert:
            self.current_state.insert(index, component)

    # All breakpoints within eps will be resolved
    def resolve_breakpoints(self, eps: float = 1e-8):
        breakpoint_times = sorted(list(self.breakpoints.keys()))
        valid_breakpoint_times = [t for t in breakpoint_times if abs(t - self.t) < eps]
        for t in valid_breakpoint_times:
            breakpoint_callbacks = self.breakpoints[t]
            for callback in breakpoint_callbacks:
                callback(self)

    def get_constant_region_mass(self):
        mass = 0.0
        for entry in self.current_state:
            if isinstance(entry, ConstantRegion):
                mass += (entry.region_length * entry.density)
        return mass

    def get_current_net_flux(self):
        constant_region_and_boundaries = [entry for entry in self.current_state if isinstance(entry, (ConstantRegion, MovingBoundary))]
        first_region = constant_region_and_boundaries[0]
        last_region = constant_region_and_boundaries[-1]
        return self.fd._flow(first_region.density) - self.fd._flow(last_region.density)

    def generate_density_field(self):
        assert(isinstance(self.current_state[0], (SilentBoundary, MovingBoundary)) and isinstance(self.current_state[-1], (SilentBoundary, MovingBoundary))), "Must be contained as boundaries!"
        density = []
        position = []
        length = []
        current_time = []
        for entry in self.current_state:
            if (isinstance(entry, ConstantRegion)):
                density.append(entry.density)
                position.append(entry.s)
                length.append(entry.region_length)
                current_time.append(self.t)
        return pd.DataFrame(data={
            "time": current_time,
            "density": density,
            "position": position,
            "length": length
        })

    @classmethod
    def generate_solver_from_macro_data(cls, network_path: str, macro_data_path: str, fd: TriangularFD, t: float = 0.0, road_id: str = "1", lane_id: int = -1, min_s: float = 0.0, max_s: float = float('inf'), additional_components: list[Component] = []) -> "Solver":
        network = Network.from_json(network_path)
        road = network.roads[road_id]
        lane_cells = road.cells_for_lane(lane_id)
        macro_data = pd.read_parquet(macro_data_path)
        macro_data = macro_data[(macro_data["time"] == t) & (macro_data["road_id"] == road_id)]
        components = [SilentBoundary(s=min_s)]
        for cell in lane_cells:
            macro_data_overlaps = macro_data[macro_data["cell_id"] == cell.cell_id]
            assert(len(macro_data_overlaps) == 1), f"{macro_data_overlaps}\n{cell}"
            components.append(ConstantRegion(cell.start_s, cell.length, macro_data_overlaps.iloc[0]["density"]))
        components.append(SilentBoundary(s=max_s))

        components = components + additional_components
        return Solver(fd, components)


def return_demo():
    fd = simulation.TriangularFD(v_f=50.0, rho_j=0.13, w=6.0)
    initial_states = []
    initial_states.append(SilentBoundary(s=0.0))
    initial_states.append(ConstantRegion(s=0.0, density=0.10, region_length=100.0))
    initial_states.append(ConstantRegion(s=100.0, density=fd.rho_j, region_length=100.0))
    def boundary_init_function(boundary: MovingBoundary, solver: Solver):
        boundary.velocity = fd.v_f
        boundary.density = fd.rho_j
        def breakpoint(solver):
            boundary.density = 0.05 * fd.rho_c
        solver.register_breakpoint(breakpoint, 10.0)
    initial_states.append(MovingBoundary(s=200.0, boundary_init_function=boundary_init_function))
    solver = Solver(initial_state=initial_states, fd=fd)
    return solver
