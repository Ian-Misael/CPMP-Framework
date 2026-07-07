import numpy as np
import copy
from scipy.optimize import nnls
from instances.generators.instance_generator import InstanceGenerator
from instances.generators.random_moves import random_moves
from solvers.FRG import FRGSolver
from cpmp.layout import Layout
from utils.utils import distribuir_suma_exacta
import random

class UniformEmpiricalGenerator(InstanceGenerator):
    def __init__(self, H, S, solver, seed):
        super().__init__(H=H, S=S, N=S*(H-2), seed=seed)
        self.solver = solver
        self.empirical_counts = {}  # { k: {costo: cantidad_ocurrencias} }
        self.U = 0                  # Upper Bound
        self.k_anchors = []         # Lista de valores k descubiertos

    def generate_instances(self, amount):
        print("Iniciando fase de Burn-in (Descubrimiento)...")
        self._burn_in()
        self.U = max(1, self.U) # Por seguridad, U nunca puede ser 0
        print(f"Burn-in finalizado. Upper Bound (U) detectado: {self.U}")

        # 1. Definir los Bins exactamente uniformes
        cuotas_iniciales = distribuir_suma_exacta(np.ones(self.U), amount)
        
        # Mapeamos costo -> cantidad faltante
        bins_faltantes = {costo: int(cuotas_iniciales[costo - 1]) for costo in range(1, self.U + 1)}
        instancias_aceptadas = 0

        print(f"Iniciando recolección dirigida (NNLS) para {amount} instancias...")
        while instancias_aceptadas < amount:
            k_probs = self._calcular_pesos_nnls(bins_faltantes)
            k_elegido = random.choices(self.k_anchors, weights=k_probs, k=1)[0]
            
            instancia, costo_real = self._generar_y_evaluar(k_elegido)
            self._registrar_resultado(k_elegido, costo_real)
            
            if costo_real in bins_faltantes and bins_faltantes[costo_real] > 0:
                if self.add_instance(instancia):
                    bins_faltantes[costo_real] -= 1
                    instancias_aceptadas += 1
                    
                    if instancias_aceptadas % max(1, amount // 10) == 0:
                        print(f"Progreso: {instancias_aceptadas}/{amount} instancias.")
        
        return self.instances[:amount]

    def _burn_in(self, batch_size=100, stagnation_patience=3, threshold=0.5):
        k_seq = self._fibonacci_gen()
        historial_promedios = []
        
        for k in k_seq:
            self.k_anchors.append(k)
            self.empirical_counts[k] = {}
            
            costos_batch = []
            for _ in range(batch_size):
                _, costo = self._generar_y_evaluar(k)
                self._registrar_resultado(k, costo)
                costos_batch.append(costo)
                
            promedio_actual = np.mean(costos_batch)
            historial_promedios.append(promedio_actual)
            
            # Decisión de diseño: U se basa en el promedio máximo, no en el costo absoluto
            self.U = int(round(max(historial_promedios)))
            
            if len(historial_promedios) >= stagnation_patience + 2:
                promedio_reciente = np.mean(historial_promedios[-stagnation_patience:])
                mejor_promedio_historico = max(historial_promedios[:-stagnation_patience])
                
                crecimiento_del_promedio = promedio_reciente - mejor_promedio_historico
                
                print(f"[Burn-in] k={k:<4} | Promedio={promedio_actual:.2f} | U_actual={self.U} | Crec. Prom={crecimiento_del_promedio:.2f}")
                
                if crecimiento_del_promedio <= threshold:
                    break
            else:
                print(f"[Burn-in] k={k:<4} | Promedio={promedio_actual:.2f} | U_actual={self.U}")

    def _calcular_pesos_nnls(self, bins_faltantes):
        """
        Arma la matriz P y el vector b actuales, y resuelve con NNLS para
        saber qué valores de k nos convienen más en este instante.
        """
        # Vector b (lo que nos falta)
        b = np.array([bins_faltantes.get(c, 0) for c in range(1, self.U + 1)], dtype=float)
        
        if np.sum(b) == 0:
            return [1.0 / len(self.k_anchors)] * len(self.k_anchors)

        # Matriz P (probabilidad de obtener el costo c dado k)
        # Filas = k_anchors, Columnas = costos del 1 al U
        P = np.zeros((len(self.k_anchors), self.U))
        
        for i, k in enumerate(self.k_anchors):
            total_hits = sum(self.empirical_counts[k].values())
            if total_hits == 0: continue
            
            for c in range(1, self.U + 1):
                P[i, c - 1] = self.empirical_counts[k].get(c, 0) / total_hits
                
        # Resolver NNLS (P.T * pesos = b)
        pesos, _ = nnls(P.T, b)
        
        # Normalizar para usar como probabilidades
        suma_pesos = np.sum(pesos)
        if suma_pesos <= 1e-8:
            # Fallback si NNLS da 0 (ej. pedimos un costo que nunca hemos visto)
            # Damos probabilidad uniforme a todos los k
            return [1.0 / len(self.k_anchors)] * len(self.k_anchors)
            
        return pesos / suma_pesos

    def _generar_y_evaluar(self, k):
        """
        Parte de un estado ordenado, aplica k movimientos aleatorios,
        lo resuelve y devuelve la instancia (stacks) y su costo real.
        """
        # 1. Generar estado resuelto/ordenado
        stacks = self.generate_stacks(self.H, self.S, self.N, sorted=True)
        
        # 2. Aplicar k movimientos aleatorios válidos
        stacks = random_moves(stacks, self.H, k) 
        lay = Layout(stacks, self.H)
        
        # 3. Resolver para obtener el costo exacto
        # Solo necesitamos el número de pasos, no la trayectoria
        cost = self._solve(lay)
        self.solver.reset()
        
        return copy.deepcopy(lay.stacks), cost

    def _registrar_resultado(self, k, costo):
        if costo not in self.empirical_counts[k]:
            self.empirical_counts[k][costo] = 0
        self.empirical_counts[k][costo] += 1

    def _fibonacci_gen(self):
        a, b = 1, 2
        while True:
            yield a
            a, b = b, a + b

    def _solve(self, layout):
        moves = self.get_feasible_moves(layout)

        lay_copies = []
        for (i, j) in moves:
            lay_copy = copy.deepcopy(layout)
            lay_copy.move(i, j)
            lay_copy.steps = 0
            lay_copies.append(lay_copy)

        results = self.solver.solve_from_layouts(lay_copies, self.H, 1000)
        min_cost = float('inf')

        for solved, cost in results:
            if not solved: 
                continue

            if cost + 1 < min_cost:
                min_cost = cost + 1
                
        return min_cost

    def get_feasible_moves(self, layout):
        moves = []
        num_stacks = len(layout.stacks)

        for i in range(num_stacks):
            if len(layout.stacks[i]) > 0:
                for j in range(num_stacks):
                    if i != j and len(layout.stacks[j]) < layout.H:
                        moves.append((i, j))

        return moves