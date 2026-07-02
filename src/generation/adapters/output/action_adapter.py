from generation.adapters.output.output_adapter import OutputAdapter
import numpy as np

class ActionAdapter(OutputAdapter):
    def __init__(self, S_max=10):
        super().__init__({
            "Y": np.int32
        })
        self.S_max = S_max
    
    def output_2_vec(self, moves_costs):
        # Inicializar el vector con ceros (0 indica acción no óptima o no factible)
        Y = np.zeros(self.S_max * (self.S_max - 1), dtype=np.int32)

        # 1. Encontrar el costo mínimo absoluto en el batch actual
        min_cost = min(cost for move, cost in moves_costs)

        # 2. Asignar 1 solo a los movimientos que empaten con el costo mínimo
        for move, cost in moves_costs:
            if cost == min_cost:
                src, dst = move
                idx = src * (self.S_max - 1) + (dst - int(dst > src))
                Y[idx] = 1

        return Y
    
    def add(self, output_data):
        self.data['Y'].append(output_data)