from instances.generators.instance_generator import InstanceGenerator
import random
from itertools import combinations

class BadPlacedUniformGenerator(InstanceGenerator):
    def __init__(self, H, S, seed):
        super().__init__(H=H, S=S, N=S*(H-2), seed=seed)
    
    def _distribute_bounded(self, total: int, bins: int, maxima: list[int]) -> list[int]:
        """
        Muestrea uniformemente una composicion de `total` en `bins` partes
        donde la parte i <= maxima[i]. Rejection sampling sobre stars-and-bars.
        """
        if sum(maxima) < total:
            raise ValueError("Imposible distribuir: suma de maximos insuficiente.")
        if total == 0:
            return [0] * bins
        while True:
            cuts = sorted(random.sample(range(total + bins - 1), bins - 1))
            cuts = [-1] + cuts + [total + bins - 1]
            result = [cuts[i + 1] - cuts[i] - 1 for i in range(bins)]
            if all(result[i] <= maxima[i] for i in range(bins)):
                return result
    
    
    def _build_stack(self, values: list[int], n_good: int, n_bad: int) -> list[int]:
        """
        Construye un stack con los valores dados y LB1 = n_bad exacto.
        Muestrea uniformemente sobre todas las permutaciones validas de `values`.
    
        Estructura resultante (indice 0 = fondo, -1 = tope):
            [ base decreciente (n_good) | primer_malo >= tope_base | libres (n_bad-1) ]
    
        Metodo: enumerar todos los subconjuntos posibles para la base, ponderarlos
        por el numero de primeros-malos validos que generan, y muestrear
        proporcionalmente. Esto garantiza uniformidad estricta.
        """
        if n_bad == 0:
            return sorted(values, reverse=True)
    
        N = len(values)
        vals_sorted = sorted(values)
        idx_set_cache = [set(combo) for combo in combinations(range(N), n_good)]
    
        # Para cada subconjunto base posible, calcular cuantos primeros-malos validos hay
        subsets = []
        weights = []
        for idx_set in idx_set_cache:
            base_vals = [vals_sorted[i] for i in idx_set]
            base_top = min(base_vals)
            rest_vals = [vals_sorted[i] for i in range(N) if i not in idx_set]
            k = sum(1 for v in rest_vals if v >= base_top)
            if k > 0:
                subsets.append((base_vals, rest_vals))
                weights.append(k)
    
        # Muestrear subconjunto base ponderado por numero de primeros-malos validos
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)
        cumulative = 0.0
        chosen_base_vals, chosen_rest = subsets[-1]  # fallback
        for (base_vals, rest_vals), w in zip(subsets, weights):
            cumulative += w
            if r <= cumulative:
                chosen_base_vals = base_vals
                chosen_rest = rest_vals
                break
    
        # Elegir primer malo uniformemente entre los candidatos validos
        base_top = min(chosen_base_vals)
        candidates = [v for v in chosen_rest if v >= base_top]
        first_bad = random.choice(candidates)
    
        # Malos libres: cualquier permutacion
        free_bad = [v for v in chosen_rest if v != first_bad]
        random.shuffle(free_bad)
    
        base = sorted(chosen_base_vals, reverse=True)
        return base + [first_bad] + free_bad

    def compute_lb(self, stacks: list[list[int]]) -> int:
        """
        Calcula LB: numero de contenedores mal ubicados.
    
        El prefijo bueno de cada stack es el mas largo desde la base con
        orden estrictamente decreciente. Todo lo que queda encima es malo.
        """
        total = 0
        for stack in stacks:
            if len(stack) <= 1:
                continue
            good = 1
            for j in range(len(stack) - 1):
                if stack[j] > stack[j + 1]:
                    good += 1
                else:
                    break
            total += len(stack) - good
        return total
    
    def generate_instance(self, LB) -> list[list[int]]:
        N = self.S * (self.H - 2)
        if self.H < 2:
            raise ValueError("H debe ser >= 2.")
        if LB < 0 or LB > N:
            raise ValueError(f"LB debe estar en [0, {N}].")
    
        # Paso 1: distribuir N contenedores entre S stacks (max H-1 por stack).
        # Repetir hasta obtener una distribucion que permita alcanzar LB exacto.
        max_size = [self.H - 1] * self.S
        stack_sizes = self._distribute_bounded(N, self.S, max_size)
        for _ in range(100000):
            if sum(max(0, s - 1) for s in stack_sizes) >= LB:
                break
            stack_sizes = self._distribute_bounded(N, self.S, max_size)
        else:
            raise RuntimeError(
                f"No se encontro distribucion de tamanos compatible con LB={LB}. "
                f"Intenta con S o H mas grandes."
            )
    
        # Paso 2: distribuir LB malos entre stacks (max s-1 malos por stack de tamano s).
        max_bad = [max(0, s - 1) for s in stack_sizes]
        bad_per_stack = self._distribute_bounded(LB, self.S, max_bad)
    
        # Paso 3: repartir los N valores [0..N-1] aleatoriamente entre stacks
        # y construir cada uno respetando su estructura (n_good, n_bad).
        all_values = list(range(N))
        random.shuffle(all_values)
    
        stacks = []
        idx = 0
        for i in range(self.S):
            size = stack_sizes[i]
            n_bad = bad_per_stack[i]
            n_good = size - n_bad
    
            if size == 0:
                stacks.append([])
                continue
    
            values = all_values[idx: idx + size]
            idx += size
            stacks.append(self._build_stack(values, n_good, n_bad))
    
        return stacks
    
    def generate_instances(self, amount):
        for LB in range(self.S, self.S * (self.H - 3) + 1):
            if LB == self.S * (self.H - 3):
                curr_amount = amount - len(self.instances)
            else:
                curr_amount = amount // (self.S * (self.H - 3) - self.S)
                
            count = 0
            while count < curr_amount:
                stacks = self.generate_instance(LB)
                if self.add_instance(stacks): count += 1
                
        return self.instances