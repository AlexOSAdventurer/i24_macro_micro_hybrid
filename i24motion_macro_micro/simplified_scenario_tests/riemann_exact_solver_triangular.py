import sys
sys.path.append("..")
from simulation import TriangularFD
class Component:
    def __init__(self, s: float, velocity: float):
        self.s = s
        self.velocity = velocity

class ConstantRegion(Component):
    def __init__(self, s: float, region_length: float, density: float):
        super().__init__(s, 0.0)
        self.region_length = region_length
        self.density = density

class SilentBoundary(Component):
    def __init__(self, s: float):
        super().__init__(s, 0.0)

class MovingBoundary(Component):
    def __init__(self, s: float, velocity: float, density: float, boundary_init_function: callable):
        super().__init__(s, velocity)
        self.density = density
        self.boundary_init_function = boundary_init_function

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
            return None
        s_difference = right_component.s - left_component.s
        closing_rate = left_component.velocity - right_component.velocity
        time_to_contact = s_difference / closing_rate if (closing_rate > eps) else None
        return time_to_contact

    def get_no_constant_regions(self):
        result = []
        for i in range(len(self.current_state)):
            if (not isinstance(self.current_state[i], ConstantRegion)):
                result.append(i)
        return result

    def get_contacts_in_time_region(self, dt, eps=1e-4):
        results = []
        no_constant_regions = self.get_no_constant_regions()
        for j in range(len(no_constant_regions) - 1):
            left_component_index = no_constant_regions[j]
            right_component_index = no_constant_regions[j + 1]
            left_component = self.current_state[left_component_index]
            right_component = self.current_state[right_component_index]
            current_ttc = self.time_to_contact(left_component, right_component)
            if (current_ttc <= (dt - eps)):
                results.append(((left_component_index, right_component_index), current_ttc))
        return results
        
    def determine_next_breakpoint(self, dt):
        contacts = sorted(self.get_contacts_in_time_region(dt), key=lambda e: e[1])
        if (len(contacts) == 0):
            return (None, float('inf'))
        return contacts[0]
    
    def advance(self, dt=0.001):
        _, ttc = self.determine_next_breakpoint(dt)
        while (ttc <= dt):
            self.advance_fronts(ttc)
            self.resolve_breakpoints()
            self.resolve_contacts()
            dt -= ttc
            _, ttc = self.determine_next_breakpoint(dt)
        self.advance_fronts(dt)

    # We assume there are no contacts in this temporal region
    def advance_fronts(self, dt, ignore_contacts=False):
        contacts = self.get_no_constant_regions(dt)
        for contact_index in contacts:
            constant_region_left_index = contact_index - 1
            constant_region_right_index = contact_index + 1
            contact = self.current_state[contact_index]
            if (constant_region_left_index >= 0):
                constant_region_left = self.current_state[constant_region_left_index]
                if (isinstance(constant_region_left, ConstantRegion)):
                    constant_region_left.region_length += (contact.velocity * dt)
                else:
                    raise Exception("Left region is not a constant region!")

            if (constant_region_right_index < len(self.current_state)):
                constant_region_right = self.current_state[constant_region_right_index]
                if (isinstance(constant_region_right, ConstantRegion)):
                    constant_region_right.s += (contact.velocity * dt)
                    constant_region_right.region_length -= (contact.velocity * dt)
                else:
                    raise Exception("Right region is not a constant region!")
 
    # All contacts within eps distance will be resolved
    def resolve_contacts(self, eps=1e-4):
        # First, we mark colliding wave fronts and solve them by deleting them and their contained intermediate region.
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
                        fronts_to_delete.append(i)
        self.current_state = [c for i, c in enumerate(self.current_state) if i not in fronts_to_delete]
        # Second, we mark colliding constant regions and generate an appropriate wave front between them.
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
            if contacting:
                if (two_constant_regions or left_moving_boundary_right_constant_region or right_moving_boundary_left_constant_region):
                    left_density = left_region.density
                    right_density = right_region.density

                    # Contact discontinuity? We insert a corresponding wave front here to their characteristic speed
                    if ((left_density < self.fd.rho_c) and (right_density < self.fd.rho_c)) or ((left_density >= self.fd.rho_c) and (right_density >= self.fd.rho_c)):
                        placement_index = i + 1 + len(components_to_insert)
                        velocity = self.fd.shock_speed(left_density, right_density)
                        new_component = WaveFront(right_region.s, velocity)
                        components_to_insert.append((placement_index, new_component))

                    # Shock discontinuty? We insert a corresponding wave front here that moves backwards
                    elif (left_density < self.fd.rho_c) and (right_density > self.fd.rho_c):
                        placement_index = i + 1 + len(components_to_insert)
                        velocity = self.fd.shock_speed(left_density, right_density)
                        new_component = WaveFront(right_region.s, velocity)
                        components_to_insert.append((placement_index, new_component))

                    # Rarefaction shock? We insert a two wave fronts here, one that moves backwards, one that moves forwards, and an intermediate p_c state between
                    elif (left_density > self.fd.rho_c) and (right_density < self.fd.rho_c):
                        left_front_index = i + 1 + len(components_to_insert)
                        intermediate_state_index = i + 2 + len(components_to_insert)
                        right_front_index = i + 3 + len(components_to_insert)
                        left_front_component = WaveFront(right_region.s, -self.fd.w)
                        intermediate_component = ConstantRegion(right_region.s, 0.0, self.fd.rho_c)
                        right_front_component = WaveFront(right_region.s, self.fd.v_f)
                        components_to_insert((left_front_index, left_front_component))
                        components_to_insert((intermediate_state_index, intermediate_component))
                        components_to_insert((right_front_index, right_front_component))
                    else:
                        raise Exception("Case is occurring that doesn't fit into any of these!")
        for (index, component) in components_to_insert:
            self.current_state.insert(index, component)

    # All breakpoints within eps will be resolved
    def resolve_breakpoints(self, eps=1e-4):
        breakpoint_times = sorted(list(self.breakpoints.keys()))
        valid_breakpoint_times = [t for t in breakpoint_times if abs(t - self.t) < eps]
        for t in valid_breakpoint_times:
            breakpoint_callbacks = self.breakpoints[t]
            for callback in breakpoint_callbacks:
                callback(self)