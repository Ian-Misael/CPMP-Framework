from generation.adapters.output.output_adapter import OutputAdapter
import numpy as np

class SoftAdapter(OutputAdapter):
    # Se añade la temperatura como parámetro opcional para controlar la "agresividad"
    def __init__(self, S_max=10, temperature=1.0):
        super().__init__({
            "Y": np.float32  # CRÍTICO: Cambiado a float32 porque ahora retornamos probabilidades
        })
        self.S_max = S_max
        self.temperature = temperature
    
    def output_2_vec(self, moves_costs):
        # Inicializar el vector con ceros (las acciones no factibles tendrán probabilidad 0)
        Y = np.zeros(self.S_max * (self.S_max - 1), dtype=np.float32)

        # 1. Separar los movimientos de los costos
        moves = [mc[0] for mc in moves_costs]
        costs = np.array([mc[1] for mc in moves_costs], dtype=np.float32)

        # 2. Calcular Z-score
        mu = np.mean(costs)
        sigma = np.std(costs)
        epsilon = 1e-8  # Evita la división por cero si todos los costos son idénticos
        
        z_scores = (costs - mu) / (sigma + epsilon)

        # 3. Softmax Negativo con Temperatura (para minimización)
        # Se invierte el signo de z_scores para premiar los costos más bajos
        logits = -z_scores / self.temperature
        
        # Truco de estabilidad numérica: restar el máximo logit antes de la exponencial
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # 4. Asignar las probabilidades calculadas a los índices correspondientes
        for move, prob in zip(moves, probs):
            src, dst = move
            idx = src * (self.S_max - 1) + (dst - int(dst > src))
            Y[idx] = prob

        return Y
    
    def add(self, output_data):
        self.data['Y'].append(output_data)