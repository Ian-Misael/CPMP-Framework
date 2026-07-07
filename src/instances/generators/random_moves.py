from instances.generators.instance_generator import InstanceGenerator
import random
    
class RandomMovesGenerator(InstanceGenerator):
    def __init__(self, H, S, N, r, seed):
        super().__init__(H, S, N, seed)
        self.r = r

    def generate_instances(self, amount):
        current_len = len(self.instances)
        while len(self.instances) < amount + current_len:
            stacks = self.generate_stacks(self.H, self.S, self.N, sorted=True)
            stacks = random_moves(stacks, self.H, self.r)
            self.add_instance(stacks)
        return self.instances
    
def random_moves(stacks, H, r):
    last_move = (None, None)
    moves_made = 0
    
    while moves_made < r:
        # 1. Elegir un origen que no esté vacío
        valid_origins = [i for i, s in enumerate(stacks) if len(s) > 0]
        if not valid_origins:
            break  # No hay movimientos posibles
            
        origin_idx = random.choice(valid_origins)
        
        # 2. Elegir un destino que no esté lleno y no sea el origen
        valid_destinations = [
            i for i, s in enumerate(stacks) 
            if i != origin_idx and len(s) < H
        ]
        
        if not valid_destinations:
            continue # Reintentar con otro origen
            
        dest_idx = random.choice(valid_destinations)
        
        # 3. Validar que no anule el movimiento anterior
        # El inverso de (a, b) es (b, a)
        if (dest_idx, origin_idx) == last_move:
            continue
            
        # Ejecutar el movimiento
        container = stacks[origin_idx].pop()
        stacks[dest_idx].append(container)
        
        # Registrar rastro
        last_move = (origin_idx, dest_idx)
        moves_made += 1

    return stacks