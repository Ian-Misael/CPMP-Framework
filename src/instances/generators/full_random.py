from instances.generators.instance_generator import InstanceGenerator

class FullRandomGenerator(InstanceGenerator):
    def generate_instances(self, amount):
        while len(self.instances) < amount:
            stacks = self.generate_stacks(self.H, self.S, self.N, sorted=False)
            self.add_instance(stacks)
        return self.instances