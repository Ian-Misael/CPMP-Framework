from instances.generators.instance_generator import InstanceGenerator
from solvers.FRG import FRGSolver
from cpmp.layout import Layout
import copy

class UniformFRGGenerator(InstanceGenerator):
    def __init__(self, H, S, seed):
        super().__init__(H=H, S=S, N=S*(H-2), seed=seed)
        self.solver = FRGSolver()
    
    def generate_instances(self, amount):           
        count = 0
        while count < amount:
            stacks = self.generate_stacks(self.H, self.S, self.N, sorted=False)
            lay = Layout(stacks, self.H)

            steps = self.solver.solve_from_layout(lay, self.H, 100000, return_steps=True)

            for step in reversed(steps):
                lay.move(*step)
                if self.add_instance(copy.deepcopy(lay.stacks)): count += 1
                
        return self.instances[:amount]