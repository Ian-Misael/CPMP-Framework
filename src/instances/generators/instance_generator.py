from cpmp.layout import Layout
import random
from abc import ABC, abstractmethod

class InstanceGenerator(ABC):
    def __init__(self, H, S, N, seed):
        self.H = H
        self.S = S
        self.N = N
        self.seed = seed
        random.seed(seed)

        self.instances = []
        self.instance_set = set()

    def generate_stacks(self, H, S, N, sorted):
        stacks = []
        for _ in range(S):
            stacks.append([])

        available_numbers = list(range(1, N + 1))
        if not sorted:
            random.shuffle(available_numbers)

        for j in range(N):
            s = random.randint(0, S - 1)
            while len(stacks[s]) == H:
                s = random.randint(0, S - 1)
            
            if sorted:
                g = available_numbers.pop()
            else:
                g = available_numbers.pop()
                
            stacks[s].append(g)

        return stacks
    
    def add_instance(self, instance):
        instance_hash = tuple(tuple(stack) for stack in instance)
        if tuple(instance_hash) in self.instance_set: return False

        layout = Layout(instance, self.H)
        if layout.is_sorted(): return False
        
        self.instances.append(instance)
        self.instance_set.add(instance_hash)
        return True

    @abstractmethod
    def generate_instances(self, amount):
        pass