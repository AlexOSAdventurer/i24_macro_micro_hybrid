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
    def __init__(self, s: float, velocity: float, density: float, boundary_function: function):
        super().__init__(s, velocity)
        self.density = density
        self.boundary_function = boundary_function

class WaveFront(Component):
    def __init__(self, s: float, velocity: float):
        super().__init__(s, velocity)

class Solver:
    def __init__(self, initial_state: list[Component] =[]):
        self.current_state = sorted(initial_state, key=lambda v: v.s)
        self.t = 0.0
        self.breakpoints = {}

    def register_breakpoint(self, callback: function, t: float):
        if t not in self.breakpoints:
            self.breakpoints[t] = []
        self.breakpoints[t].append(callback)

    def time_to_contact(self, left_component: Component, right_component: Component, eps=1e-4):
        if (isinstance(left_component, ConstantRegion) or isinstance(right_component, ConstantRegion)):
            return None
        s_difference = right_component.s - left_component.s
        closing_rate = left_component.velocity - right_component.velocity
        time_to_contact = s_difference / closing_rate if (abs(closing_rate) > eps) else None
        return time_to_contact

    def get_no_constant_regions(self):
        result = []
        for i in range(len(self.current_state)):
            if (not isinstance(self.current_state[i], ConstantRegion)):
                result.append(i)
        return result

    def determine_next_breakpoint(self):
        pair = None
        pair_ttc = None
        no_constant_regions = self.get_no_constant_regions()
        for j in range(len(no_constant_regions) - 1):
            left_component_index = no_constant_regions[j]
            right_component_index = no_constant_regions[j + 1]
            left_component = self.current_state[left_component_index]
            right_component = self.current_state[right_component_index]
            current_ttc = self.time_to_contact(left_component, right_component)
            if (pair is None) or ((current_ttc is not None) and (pair_ttc > current_ttc)):
                pair = (left_component_index, right_component_index)
                pair_ttc = current_ttc
        return pair, pair_ttc

    def advance(self, dt=0.001):
        breakpoint_components, ttc = self.determine_next_breakpoint()
        if (ttc > dt):
            self.advance_fronts()
        else:
            self.resolve_contacts()

    def advance_fronts(self):
        pass

    def resolve_contacts(self):
        pass